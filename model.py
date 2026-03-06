"""
model.py — Multi-task ADR risk prediction network with Monte Carlo Dropout
  - Input:  feature vector (26 dims from data_gen + optional CATE features)
  - Output: ADR probability (sigmoid) + severity score (linear)
  - Uncertainty: 50 MC Dropout forward passes → aleatoric + epistemic decomposition
  - Calibration: post-hoc temperature scaling (Guo et al. 2017)

Temperature scaling:
  The model tends to become overconfident as FL rounds increase — it assigns
  high-confidence predictions that exceed true accuracy. Temperature scaling
  fixes this without touching model weights: a single scalar T (learned on a
  held-out calibration split) divides logits before sigmoid.

    p_cal = sigmoid(logit / T)

  T > 1  →  softens overconfident predictions toward 0.5  (our case)
  T < 1  →  sharpens underconfident predictions
  T = 1  →  no change (identity)

  Crucially, temperature scaling is rank-preserving, so AUROC and F1 are
  completely unaffected — only ECE improves.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

INPUT_DIM  = 26   # 6 clinical + 20 drug one-hot
HIDDEN     = [128, 64, 32]
DROPOUT_P  = 0.1
MC_PASSES  = 50


class ADRNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, dropout_p: float = DROPOUT_P):
        super().__init__()
        dims = [input_dim] + HIDDEN
        layers = []
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),   # MC Dropout: stays active at inference
            ]
        self.backbone = nn.Sequential(*layers)

        # Two heads
        self.adr_head      = nn.Linear(HIDDEN[-1], 1)   # sigmoid → probability
        self.severity_head = nn.Linear(HIDDEN[-1], 1)   # linear  → severity score

    def forward(self, x):
        h = self.backbone(x)
        adr_prob = torch.sigmoid(self.adr_head(h)).squeeze(-1)
        severity = self.severity_head(h).squeeze(-1)
        return adr_prob, severity


def train_local(model: ADRNet, X: np.ndarray, y: np.ndarray,
                epochs: int = 10, lr: float = 1e-3, batch_size: int = 64,
                verbose: bool = False) -> dict:
    """Train model on local hospital data. Returns final train loss."""
    model.train()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset  = TensorDataset(X_t, y_t)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    bce_loss  = nn.BCELoss()

    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            adr_prob, _ = model(xb)
            loss = bce_loss(adr_prob, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        total_loss = epoch_loss / len(loader)
        if verbose:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss:.4f}")

    return {"loss": total_loss}


def get_weights(model: ADRNet) -> list[np.ndarray]:
    """Extract model weights as list of numpy arrays (for FL)."""
    return [p.data.cpu().numpy() for p in model.parameters()]


def set_weights(model: ADRNet, weights: list[np.ndarray]):
    """Load weights from list of numpy arrays (for FL)."""
    for p, w in zip(model.parameters(), weights):
        p.data = torch.tensor(w, dtype=torch.float32)


def mc_predict(model: ADRNet, X: np.ndarray, n_passes: int = MC_PASSES,
               temperature: float = 1.0) -> dict:
    """
    Monte Carlo Dropout inference with optional temperature scaling.

    temperature: learned calibration scalar (1.0 = no scaling).
      Applied as: p_cal = sigmoid(logit(p_raw) / T)
      Since we don't store raw logits across MC passes, we recover the logit
      from the mean prediction and re-scale, preserving uncertainty structure.

    Following Kendall & Gal (2017):
      σ²_epistemic  = variance of per-pass means     (model ignorance)
      σ²_aleatoric  = mean of per-pass p*(1-p)       (data noise, Bernoulli)
      σ²_total      = σ²_epistemic + σ²_aleatoric
    """
    model.train()   # keep dropout ACTIVE during inference — this is MC Dropout
    X_t = torch.tensor(X, dtype=torch.float32)

    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            prob, _ = model(X_t)
            preds.append(prob.cpu().numpy())

    preds = np.stack(preds, axis=0)          # (n_passes, N)

    # Apply temperature scaling per-pass before averaging
    # logit(p) = log(p / (1-p));  then re-sigmoid with T
    if abs(temperature - 1.0) > 1e-6:
        eps = 1e-6
        preds_clipped = np.clip(preds, eps, 1.0 - eps)
        logits = np.log(preds_clipped / (1.0 - preds_clipped))
        preds  = 1.0 / (1.0 + np.exp(-logits / temperature))

    mean_pred = preds.mean(axis=0)           # (N,) — ADR probability
    epistemic = preds.var(axis=0)            # variance across passes
    aleatoric = (preds * (1 - preds)).mean(axis=0)
    total_unc = epistemic + aleatoric

    return {
        "risk_score":   mean_pred,
        "epistemic":    epistemic,
        "aleatoric":    aleatoric,
        "total_unc":    total_unc,
        "ci_lower":     np.clip(mean_pred - 2 * np.sqrt(total_unc), 0, 1),
        "ci_upper":     np.clip(mean_pred + 2 * np.sqrt(total_unc), 0, 1),
    }


# ── Temperature Scaling ──────────────────────────────────────────────────────

def fit_temperature(model: ADRNet, X_val: np.ndarray, y_val: np.ndarray,
                    n_passes: int = 20) -> float:
    """
    Fit a temperature scalar T on a held-out calibration set by minimising
    NLL (negative log-likelihood) — the standard objective from Guo et al. 2017.

    Uses scipy.optimize.minimize_scalar for a clean 1-D search over T ∈ [0.1, 10].

    Returns the optimal temperature T* (float).
    A well-calibrated model typically has T* in [1.2, 2.5] after FL training.
    """
    from scipy.optimize import minimize_scalar

    # Get raw (uncalibrated) mean predictions from MC passes
    result = mc_predict(model, X_val, n_passes=n_passes, temperature=1.0)
    probs  = result["risk_score"]                       # (N,) ∈ (0,1)

    eps    = 1e-6
    probs  = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(probs / (1.0 - probs))              # recover logits

    y = y_val.astype(np.float32)

    def nll(T):
        """NLL of Bernoulli model with temperature-scaled logits."""
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, eps, 1.0 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_star = float(result.x)
    return T_star


def _auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Area Under the Precision-Recall Curve via trapezoidal integration.
    sklearn's average_precision_score uses a step interpolation (pessimistic);
    we use the trapezoidal rule for a smoother, standard AUPRC.
    """
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns decreasing recall — reverse for trapz
    return float(np.trapezoid(precision[::-1], recall[::-1]))


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) — Guo et al. 2017.
    Bins predictions by confidence, measures |accuracy − confidence| per bin,
    weighted by bin size.

    ECE = Σ_b (|B_b| / N) · |acc(B_b) − conf(B_b)|

    A perfectly calibrated model scores ECE = 0.
    Values below 0.05 are considered well-calibrated for clinical AI.
    """
    bins      = np.linspace(0.0, 1.0, n_bins + 1)
    ece_total = 0.0
    N         = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece_total += (mask.sum() / N) * abs(acc - conf)

    return float(ece_total)


def evaluate(model: ADRNet, X: np.ndarray, y: np.ndarray,
             temperature: float = 1.0) -> dict:
    """Evaluate model using MC Dropout predictions.

    temperature: calibration scalar from fit_temperature() — pass 1.0 for
                 uncalibrated evaluation (default).

    Returns AUROC, AUPRC, F1, FPR, ECE, and uncertainty decomposition.
    """
    result = mc_predict(model, X, temperature=temperature)
    preds  = result["risk_score"]
    binary = (preds >= 0.5).astype(int)

    has_both_classes = len(np.unique(y)) > 1

    auroc = roc_auc_score(y, preds) if has_both_classes else 0.0
    auprc = _auprc(y, preds)        if has_both_classes else 0.0
    ece   = _ece(y, preds)

    fp   = ((binary == 1) & (y == 0)).sum()
    tn   = ((binary == 0) & (y == 0)).sum()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    tp   = ((binary == 1) & (y == 1)).sum()
    fn   = ((binary == 0) & (y == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "auroc":            round(float(auroc), 4),
        "auprc":            round(float(auprc), 4),
        "f1":               round(float(f1), 4),
        "fpr":              round(float(fpr), 4),
        "ece":              round(float(ece),  4),
        "mean_epistemic":   round(float(result["epistemic"].mean()), 5),
        "mean_aleatoric":   round(float(result["aleatoric"].mean()), 5),
    }
"""
model.py — Multi-task ADR risk prediction network with Monte Carlo Dropout
  - Input:  feature vector (26 dims from data_gen + optional CATE features)
  - Output: ADR probability (sigmoid) + severity score (linear)
  - Uncertainty: 50 MC Dropout forward passes → aleatoric + epistemic decomposition
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


def mc_predict(model: ADRNet, X: np.ndarray, n_passes: int = MC_PASSES) -> dict:
    """
    Monte Carlo Dropout inference.
    Returns mean prediction, aleatoric uncertainty, epistemic uncertainty.

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

    mean_pred    = preds.mean(axis=0)         # (N,) — ADR probability
    epistemic    = preds.var(axis=0)          # variance across passes
    aleatoric    = (preds * (1 - preds)).mean(axis=0)  # mean Bernoulli variance
    total_unc    = epistemic + aleatoric

    return {
        "risk_score":   mean_pred,
        "epistemic":    epistemic,
        "aleatoric":    aleatoric,
        "total_unc":    total_unc,
        "ci_lower":     np.clip(mean_pred - 2 * np.sqrt(total_unc), 0, 1),
        "ci_upper":     np.clip(mean_pred + 2 * np.sqrt(total_unc), 0, 1),
    }


def evaluate(model: ADRNet, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate model using MC Dropout predictions."""
    result = mc_predict(model, X)
    preds  = result["risk_score"]
    binary = (preds >= 0.5).astype(int)

    auroc = roc_auc_score(y, preds) if len(np.unique(y)) > 1 else 0.0
    fp    = ((binary == 1) & (y == 0)).sum()
    tn    = ((binary == 0) & (y == 0)).sum()
    fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    tp    = ((binary == 1) & (y == 1)).sum()
    fn    = ((binary == 0) & (y == 1)).sum()
    fp_   = fp
    prec  = tp / (tp + fp_) if (tp + fp_) > 0 else 0.0
    rec   = tp / (tp + fn)  if (tp + fn)  > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "auroc":     round(float(auroc), 4),
        "f1":        round(float(f1), 4),
        "fpr":       round(float(fpr), 4),
        "mean_epistemic": round(float(result["epistemic"].mean()), 5),
        "mean_aleatoric": round(float(result["aleatoric"].mean()), 5),
    }
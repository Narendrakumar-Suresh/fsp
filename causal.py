"""
causal.py — Causal inference pipeline
  1. Propensity Score Matching to de-bias ADR labels per drug
  2. T-Learner for CATE estimation (per-drug treatment effect)
  3. Returns CATE features to augment the risk model input
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# ── Propensity Score Matching ────────────────────────────────────────────────

def estimate_propensity(X_confounders: np.ndarray, treatment: np.ndarray) -> np.ndarray:
    """
    Estimate P(T=1 | X) via logistic regression.
    X_confounders: patient features (age, conditions, labs — NOT drugs)
    treatment:     binary array, 1 = patient takes the target drug
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_confounders)
    lr = LogisticRegression(max_iter=500, C=1.0)

    # Need both classes present
    if len(np.unique(treatment)) < 2:
        return np.full(len(treatment), 0.5)

    lr.fit(X_scaled, treatment)
    propensity = lr.predict_proba(X_scaled)[:, 1]
    return propensity


def propensity_match(X: np.ndarray, treatment: np.ndarray,
                     propensity: np.ndarray, caliper: float = 0.2) -> tuple:
    """
    Nearest-neighbour PSM. Returns (matched_treated_idx, matched_control_idx).
    Caliper = 0.2 × std(logit(propensity)) as per standard practice.
    """
    logit_p = np.log(np.clip(propensity, 1e-6, 1 - 1e-6) /
                     (1 - np.clip(propensity, 1e-6, 1 - 1e-6)))
    sigma   = logit_p.std() * caliper

    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    matched_t, matched_c = [], []
    used_controls = set()

    for ti in treated_idx:
        dists = np.abs(logit_p[ti] - logit_p[control_idx])
        # Sort controls by distance, pick closest unused within caliper
        sorted_c = control_idx[np.argsort(dists)]
        for ci in sorted_c:
            if ci not in used_controls and abs(logit_p[ti] - logit_p[ci]) <= sigma:
                matched_t.append(ti)
                matched_c.append(ci)
                used_controls.add(ci)
                break

    return np.array(matched_t), np.array(matched_c)


# ── T-Learner CATE Estimation ────────────────────────────────────────────────

class TLearner:
    """
    T-Learner: trains separate outcome models on treated and control groups.
    CATE(x) = μ₁(x) − μ₀(x)
    """
    def __init__(self):
        self.mu1 = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        self.mu0 = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        self._fitted = False

    def fit(self, X: np.ndarray, treatment: np.ndarray, outcome: np.ndarray):
        treated  = treatment == 1
        control  = treatment == 0
        if treated.sum() >= 5 and control.sum() >= 5:
            self.mu1.fit(X[treated], outcome[treated])
            self.mu0.fit(X[control], outcome[control])
            self._fitted = True

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(X))
        return self.mu1.predict(X) - self.mu0.predict(X)


# ── Full Causal Pipeline ─────────────────────────────────────────────────────

# Top drugs to compute CATE for (keeps compute manageable)
CATE_DRUGS = ["warfarin", "lisinopril", "spironolactone", "aspirin", "digoxin"]


def run_causal_pipeline(df: pd.DataFrame,
                         X_confounders: np.ndarray,
                         y: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    For each drug in CATE_DRUGS:
      1. Define treatment = patient takes this drug
      2. Estimate propensity scores
      3. Fit T-Learner on matched cohort
      4. Predict CATE for all patients

    Returns:
      cate_features: (N, len(CATE_DRUGS)) — augments risk model input
      report:        dict with per-drug attribution summary
    """
    from data_gen import DRUGS

    N = len(df)
    cate_features = np.zeros((N, len(CATE_DRUGS)), dtype=np.float32)
    report = {}

    for j, drug in enumerate(CATE_DRUGS):
        # Binary treatment indicator
        treatment = np.array([
            1 if drug in row.split("|") else 0
            for row in df["drugs"]
        ], dtype=int)

        n_treated = treatment.sum()
        if n_treated < 10 or (len(treatment) - n_treated) < 10:
            report[drug] = {"cate_mean": 0.0, "n_treated": int(n_treated), "status": "skipped"}
            continue

        # Propensity score
        propensity = estimate_propensity(X_confounders, treatment)

        # PSM matching
        t_idx, c_idx = propensity_match(X_confounders, treatment, propensity)

        if len(t_idx) < 5:
            report[drug] = {"cate_mean": 0.0, "n_treated": int(n_treated),
                            "n_matched": 0, "status": "too_few_matches"}
            continue

        # Build matched dataset for T-Learner
        match_idx  = np.concatenate([t_idx, c_idx])
        X_matched  = X_confounders[match_idx]
        T_matched  = treatment[match_idx]
        y_matched  = y[match_idx]

        # Fit T-Learner
        learner = TLearner()
        learner.fit(X_matched, T_matched, y_matched)

        # Predict CATE for all patients
        cate_all = learner.predict_cate(X_confounders)
        cate_features[:, j] = cate_all.astype(np.float32)

        report[drug] = {
            "cate_mean":   round(float(cate_all[treatment == 1].mean()), 4),
            "cate_std":    round(float(cate_all[treatment == 1].std()), 4),
            "n_treated":   int(n_treated),
            "n_matched":   int(len(t_idx)),
            "status":      "ok",
        }

    return cate_features, report
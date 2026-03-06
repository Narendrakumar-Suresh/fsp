"""
attribution.py — Drug attribution within a polypharmacy combination
=================================================================
Answers the question:
  "For THIS patient on {warfarin, digoxin, furosemide, aspirin, lisinopril},
   WHICH drug (or pair) is responsible for the predicted ADR?"

Two complementary methods:
  1. Leave-One-Out (LOO) Counterfactuals
     Remove each drug in turn and measure risk delta.
     Fast, interpretable, clinically meaningful.

  2. Shapley Values (cooperative game theory)
     Each drug is a "player". Its Shapley value = average marginal
     contribution across all possible drug orderings/subsets.
     Theoretically sound (satisfies efficiency, symmetry, dummy properties).
     O(2^N) exact or O(T*N) via Monte Carlo sampling.

  3. Synergy Detection
     Pairwise / triple interaction: does the pair {A,B} cause more harm
     than the sum of individual effects? Detects non-linear combinations.

Output per patient:
  {
    "drugs":          ["warfarin", "aspirin", "digoxin", ...],
    "risk_score":     0.83,
    "epistemic":      0.012,
    "aleatoric":      0.21,
    "loo_attributions": {"warfarin": +0.31, "aspirin": +0.18, ...},
    "shapley_values":   {"warfarin": +0.29, "aspirin": +0.15, ...},
    "top_culprit":      "warfarin",
    "top_pair":         ("warfarin", "aspirin"),
    "synergy_score":    0.24,  # risk(pair) - risk(A alone) - risk(B alone)
    "alert_level":      "HIGH",
    "explanation":      "Warfarin is the primary driver (+0.31 risk increase). ..."
  }
"""

import itertools
import numpy as np
import pandas as pd
from typing import NamedTuple

from model import ADRNet, mc_predict, get_weights, set_weights
from data_gen import DRUGS


# ── Patient record for attribution ──────────────────────────────────────────

class PatientRecord(NamedTuple):
    patient_id: str
    feature_vector: np.ndarray   # (D,) normalized feature vector
    drug_indices: list[int]      # indices into DRUGS list
    drug_names: list[str]        # drug names


def build_patient_record(patient_id: str,
                          x: np.ndarray,
                          drug_names: list[str],
                          scaler=None) -> PatientRecord:
    """
    Build a PatientRecord from a feature vector and drug list.
    x should be the unnormalized feature vector; scaler will normalize it.
    """
    drug_indices = [DRUGS.index(d) for d in drug_names if d in DRUGS]
    drug_names_clean = [DRUGS[i] for i in drug_indices]
    x_norm = scaler.transform(x.reshape(1, -1))[0] if scaler else x
    return PatientRecord(patient_id, x_norm, drug_indices, drug_names_clean)


# ── Core scoring function ────────────────────────────────────────────────────

def _score_with_drugs(model: ADRNet,
                       base_x: np.ndarray,
                       active_drug_indices: list[int],
                       drug_offset: int = 6,
                       n_passes: int = 20,
                       temperature: float = 1.0) -> dict:
    x = base_x.copy()
    x[drug_offset: drug_offset + len(DRUGS)] = 0.0
    for idx in active_drug_indices:
        x[drug_offset + idx] = 1.0

    result = mc_predict(model, x.reshape(1, -1), n_passes=n_passes,
                        temperature=temperature)
    return {
        "risk":       float(result["risk_score"][0]),
        "epistemic":  float(result["epistemic"][0]),
        "aleatoric":  float(result["aleatoric"][0]),
    }


# ── Leave-One-Out Attribution ────────────────────────────────────────────────

def leave_one_out(model: ADRNet,
                   patient: PatientRecord,
                   drug_offset: int = 6,
                   n_passes: int = 20,
                   temperature: float = 1.0) -> dict:
    """
    For each drug in the patient's regimen, compute:
      delta_i = risk(full regimen) - risk(regimen without drug_i)

    Positive delta = this drug contributes to ADR risk.
    Negative delta = this drug is actually protective in context.
    """
    full = _score_with_drugs(model, patient.feature_vector,
                              patient.drug_indices, drug_offset, n_passes,
                              temperature=temperature)
    full_risk = full["risk"]

    attributions = {}
    for i, (idx, name) in enumerate(zip(patient.drug_indices, patient.drug_names)):
        reduced = [d for j, d in enumerate(patient.drug_indices) if j != i]
        reduced_score = _score_with_drugs(
            model, patient.feature_vector, reduced, drug_offset, n_passes,
            temperature=temperature
        )
        attributions[name] = round(full_risk - reduced_score["risk"], 4)

    return {"full_risk": full_risk, "loo": attributions,
            "epistemic": full["epistemic"], "aleatoric": full["aleatoric"]}


# ── Shapley Value Attribution ────────────────────────────────────────────────

def shapley_values(model: ADRNet,
                    patient: PatientRecord,
                    drug_offset: int = 6,
                    n_passes: int = 10,
                    max_exact_n: int = 10,
                    n_samples: int = 256,
                    temperature: float = 1.0) -> dict:
    """
    Compute Shapley values for each drug.

    For N ≤ max_exact_n: exact computation over all 2^N subsets.
    For N > max_exact_n: Monte Carlo sampling of random orderings.

    Shapley value for drug i = average marginal contribution of i
    across all orderings of all drugs.
    """
    drugs = patient.drug_indices
    names = patient.drug_names
    N = len(drugs)

    # Cache for subset scores to avoid recomputation
    _cache = {}

    def score_subset(subset_tuple: tuple) -> float:
        if subset_tuple in _cache:
            return _cache[subset_tuple]
        result = _score_with_drugs(model, patient.feature_vector,
                                    list(subset_tuple), drug_offset, n_passes,
                                    temperature=temperature)
        _cache[subset_tuple] = result["risk"]
        return result["risk"]

    shapley = {name: 0.0 for name in names}

    if N <= max_exact_n:
        # Exact: iterate all subsets
        for i, (drug_i, name_i) in enumerate(zip(drugs, names)):
            others = [d for j, d in enumerate(drugs) if j != i]
            total_weight = 0.0
            for r in range(len(others) + 1):
                weight = 1.0 / (N * _comb(N - 1, r))
                for subset in itertools.combinations(others, r):
                    s_without = tuple(sorted(subset))
                    s_with    = tuple(sorted(subset + (drug_i,)))
                    marginal  = score_subset(s_with) - score_subset(s_without)
                    shapley[name_i] += weight * marginal
                    total_weight += weight
    else:
        # Monte Carlo sampling of random permutations
        rng = np.random.default_rng(42)
        counts = {name: 0 for name in names}
        for _ in range(n_samples):
            perm = rng.permutation(N).tolist()
            running = []
            prev_score = score_subset(())
            for pos in perm:
                drug_i = drugs[pos]
                name_i = names[pos]
                running.append(drug_i)
                cur_score = score_subset(tuple(sorted(running)))
                shapley[name_i] += cur_score - prev_score
                counts[name_i] += 1
                prev_score = cur_score
        shapley = {k: v / counts[k] for k, v in shapley.items()}

    return {k: round(v, 4) for k, v in shapley.items()}


def _comb(n: int, k: int) -> int:
    """Binomial coefficient C(n,k)."""
    from math import comb
    return comb(n, k)


# ── Synergy / Interaction Detection ─────────────────────────────────────────

def detect_synergies(model: ADRNet,
                      patient: PatientRecord,
                      drug_offset: int = 6,
                      n_passes: int = 10,
                      top_k: int = 3,
                      temperature: float = 1.0) -> list[dict]:
    """
    Detect pairwise and triple synergies: combinations where the joint
    effect exceeds the sum of individual effects.

    synergy(A,B) = risk({A,B}) - risk({A}) - risk({B}) + risk({})
    (Interaction contrast, analogous to second-order ANOVA interaction)
    """
    drugs = patient.drug_indices
    names = patient.drug_names
    N = len(drugs)

    baseline = _score_with_drugs(model, patient.feature_vector, [], drug_offset, n_passes, temperature=temperature)["risk"]

    individual = {}
    for idx, name in zip(drugs, names):
        individual[name] = _score_with_drugs(
            model, patient.feature_vector, [idx], drug_offset, n_passes
        )["risk"]

    synergies = []

    # Pairwise
    for i, j in itertools.combinations(range(N), 2):
        pair_risk = _score_with_drugs(
            model, patient.feature_vector,
            [drugs[i], drugs[j]], drug_offset, n_passes
        )["risk"]
        synergy = pair_risk - individual[names[i]] - individual[names[j]] + baseline
        synergies.append({
            "drugs": (names[i], names[j]),
            "order": 2,
            "joint_risk": round(pair_risk, 4),
            "synergy_score": round(synergy, 4),
        })

    # Triple (only if N <= 8 to keep compute reasonable)
    if N <= 8:
        for i, j, k in itertools.combinations(range(N), 3):
            triple_risk = _score_with_drugs(
                model, patient.feature_vector,
                [drugs[i], drugs[j], drugs[k]], drug_offset, n_passes
            )["risk"]
            # Approximate three-way synergy (over-inclusion correction)
            pij = _score_with_drugs(model, patient.feature_vector,
                                     [drugs[i], drugs[j]], drug_offset, n_passes)["risk"]
            pik = _score_with_drugs(model, patient.feature_vector,
                                     [drugs[i], drugs[k]], drug_offset, n_passes)["risk"]
            pjk = _score_with_drugs(model, patient.feature_vector,
                                     [drugs[j], drugs[k]], drug_offset, n_passes)["risk"]
            synergy3 = (triple_risk
                        - individual[names[i]] - individual[names[j]] - individual[names[k]]
                        + (pij + pik + pjk) / 3.0
                        - baseline)
            synergies.append({
                "drugs": (names[i], names[j], names[k]),
                "order": 3,
                "joint_risk": round(triple_risk, 4),
                "synergy_score": round(synergy3, 4),
            })

    synergies.sort(key=lambda x: x["synergy_score"], reverse=True)
    return synergies[:top_k]


# ── Full Attribution Report ──────────────────────────────────────────────────

def attribute_patient(model: ADRNet,
                       patient: PatientRecord,
                       drug_offset: int = 6,
                       use_shapley: bool = True,
                       n_mc_passes: int = 20,
                       temperature: float = 1.0) -> dict:
    """
    Run full attribution pipeline for one patient and return a structured report.

    Returns a dict with:
      - risk_score, uncertainty
      - loo_attributions (per drug)
      - shapley_values (per drug, if use_shapley=True)
      - synergies (top drug pairs/triples)
      - top_culprit, top_pair
      - alert_level: "LOW" / "MODERATE" / "HIGH" / "CRITICAL"
      - explanation: human-readable string
    """
    loo_result  = leave_one_out(model, patient, drug_offset, n_mc_passes,
                                temperature=temperature)
    loo_attr    = loo_result["loo"]
    full_risk   = loo_result["full_risk"]
    epistemic   = loo_result["epistemic"]
    aleatoric   = loo_result["aleatoric"]

    shap_attr = {}
    if use_shapley and len(patient.drug_names) <= 12:
        shap_attr = shapley_values(model, patient, drug_offset,
                                    n_passes=max(5, n_mc_passes // 4),
                                    temperature=temperature)

    synergies = detect_synergies(model, patient, drug_offset,
                                  n_passes=max(5, n_mc_passes // 4),
                                  temperature=temperature)

    # ── Determine top culprit ─────────────────────────────────────────────
    top_culprit = max(loo_attr, key=loo_attr.get) if loo_attr else None

    top_pair = None
    top_synergy_score = 0.0
    if synergies:
        best = synergies[0]
        if best["order"] == 2 and best["synergy_score"] > 0.05:
            top_pair = best["drugs"]
            top_synergy_score = best["synergy_score"]

    # ── Alert level ───────────────────────────────────────────────────────
    if full_risk >= 0.80:
        alert_level = "CRITICAL"
    elif full_risk >= 0.60:
        alert_level = "HIGH"
    elif full_risk >= 0.40:
        alert_level = "MODERATE"
    else:
        alert_level = "LOW"

    # ── Human-readable explanation ────────────────────────────────────────
    conf_pct = int((1 - epistemic / max(epistemic + aleatoric, 1e-6)) * 100)
    lines = [
        f"ADR risk: {full_risk:.0%} ({alert_level}) | Confidence: {conf_pct}%",
    ]

    if top_culprit and loo_attr[top_culprit] > 0.02:
        lines.append(
            f"Primary driver: {top_culprit.upper()} "
            f"(removing it reduces risk by {loo_attr[top_culprit]:+.0%})"
        )

    if top_pair:
        lines.append(
            f"Dangerous synergy: {top_pair[0].upper()} + {top_pair[1].upper()} "
            f"(synergy score: {top_synergy_score:+.3f})"
        )

    # Protective drugs
    protective = {k: v for k, v in loo_attr.items() if v < -0.02}
    if protective:
        best_prot = min(protective, key=protective.get)
        lines.append(
            f"Protective drug: {best_prot.upper()} "
            f"(removing it increases risk by {-protective[best_prot]:.0%})"
        )

    if epistemic > 0.02:
        lines.append(
            f"⚠ High model uncertainty (σ²_epi={epistemic:.4f}) — "
            f"consider additional data before acting on this alert."
        )

    return {
        "patient_id":       patient.patient_id,
        "drugs":            patient.drug_names,
        "risk_score":       round(full_risk, 4),
        "epistemic":        round(epistemic, 5),
        "aleatoric":        round(aleatoric, 5),
        "loo_attributions": loo_attr,
        "shapley_values":   shap_attr,
        "synergies":        synergies,
        "top_culprit":      top_culprit,
        "top_pair":         top_pair,
        "alert_level":      alert_level,
        "explanation":      " | ".join(lines),
    }


def batch_attribute(model: ADRNet,
                     X: np.ndarray,
                     df: pd.DataFrame,
                     scaler=None,
                     top_n: int = 10,
                     risk_threshold: float = 0.60,
                     n_mc_passes: int = 20,
                     temperature: float = 1.0) -> list[dict]:
    """
    Run attribution for the top_n highest-risk patients above risk_threshold.
    Returns list of attribution reports sorted by risk score descending.
    temperature: calibration scalar from fit_temperature() — applied to all
                 mc_predict calls within attribution.
    """
    from model import mc_predict

    # First pass: get risk scores for all patients cheaply (5 passes)
    all_results = mc_predict(model, X, n_passes=5, temperature=temperature)
    risk_scores = all_results["risk_score"]

    # Pick top_n above threshold
    high_risk_idx = np.where(risk_scores >= risk_threshold)[0]
    if len(high_risk_idx) == 0:
        high_risk_idx = np.argsort(risk_scores)[-top_n:]
    high_risk_idx = sorted(high_risk_idx,
                            key=lambda i: -risk_scores[i])[:top_n]

    reports = []
    for i in high_risk_idx:
        row = df.iloc[i]
        drug_names = [d for d in row["drugs"].split("|") if d in DRUGS]
        if not drug_names:
            continue

        patient = PatientRecord(
            patient_id=str(row.get("patient_id", f"P{i}")),
            feature_vector=X[i],
            drug_indices=[DRUGS.index(d) for d in drug_names],
            drug_names=drug_names,
        )
        report = attribute_patient(model, patient,
                                    n_mc_passes=n_mc_passes,
                                    temperature=temperature)
        reports.append(report)

    return reports


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_attribution_report(report: dict, console=None):
    """Print a formatted attribution report to console."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    if console is None:
        console = Console()

    level_color = {"LOW": "green", "MODERATE": "yellow",
                   "HIGH": "red", "CRITICAL": "bold red"}
    color = level_color.get(report["alert_level"], "white")

    console.print(Panel(
        f"[bold]Patient:[/bold] {report['patient_id']}\n"
        f"[bold]Drugs:[/bold]  {', '.join(report['drugs'])}\n\n"
        f"[bold]Risk Score:[/bold]  [{color}]{report['risk_score']:.0%}[/{color}]  "
        f"[{color}]{report['alert_level']}[/{color}]\n"
        f"[bold]σ²_epi:[/bold]      {report['epistemic']:.5f}  "
        f"[bold]σ²_ale:[/bold] {report['aleatoric']:.5f}\n\n"
        f"[italic]{report['explanation']}[/italic]",
        title="[bold cyan]ADR Attribution Report",
        border_style=color
    ))

    # LOO table
    t = Table(title="Leave-One-Out Attribution", show_header=True)
    t.add_column("Drug",          style="cyan")
    t.add_column("Risk Δ (LOO)",  justify="right")
    t.add_column("Shapley",       justify="right")
    t.add_column("Role",          justify="left")

    loo = report["loo_attributions"]
    shap = report.get("shapley_values", {})

    for drug in sorted(loo, key=lambda d: -loo[d]):
        delta = loo[drug]
        sv    = shap.get(drug, "—")
        role  = ("⚠ Culprit" if delta > 0.1
                 else "↑ Risk"   if delta > 0.02
                 else "↓ Prot."  if delta < -0.02
                 else "Neutral")
        sv_str = f"{sv:+.4f}" if isinstance(sv, float) else sv
        color_ = "red" if delta > 0.05 else "green" if delta < -0.02 else "white"
        t.add_row(drug, f"[{color_}]{delta:+.4f}[/{color_}]", sv_str, role)
    console.print(t)

    # Synergy table
    if report["synergies"]:
        s = Table(title="Top Drug Synergies / Interactions", show_header=True)
        s.add_column("Combination",    style="yellow")
        s.add_column("Order",          justify="center")
        s.add_column("Joint Risk",     justify="right")
        s.add_column("Synergy Score",  justify="right")
        for syn in report["synergies"]:
            drugs_str = " + ".join(syn["drugs"])
            color_ = "red" if syn["synergy_score"] > 0.1 else "yellow"
            s.add_row(drugs_str, str(syn["order"]),
                      f"{syn['joint_risk']:.3f}",
                      f"[{color_}]{syn['synergy_score']:+.4f}[/{color_}]")
        console.print(s)
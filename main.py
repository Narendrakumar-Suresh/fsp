"""
run_faers.py — FPS Pipeline with FAERS Data
============================================
Drop-in replacement for run.py that uses real FAERS quarterly data
instead of the synthetic generator.

FAERS data setup:
  1. Download from: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html
  2. Unzip each quarter into separate folders, e.g.:
       faers_data/2024q1/
       faers_data/2024q2/
       faers_data/2024q3/
  3. Each folder must contain the ascii/ subfolder with DEMO*.txt, DRUG*.txt, REAC*.txt

  If FAERS folders are not found, the pipeline automatically falls back
  to the synthetic FAERS-compatible generator so you can still test it.

Usage:
  # With real FAERS data:
  python run_faers.py --faers-dirs faers_data/2024q1 faers_data/2024q2 faers_data/2024q3

  # With synthetic fallback (no download needed):
  python run_faers.py --synthetic

  # Auto-detect (tries real data first, falls back to synthetic):
  python run_faers.py
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# ── Dual output: terminal + results file ─────────────────────────────────────
_RESULTS_FILE = "fps_faers_results.txt"
_file_handle  = open(_RESULTS_FILE, "w", encoding="utf-8")
_file_console = Console(file=_file_handle, highlight=False, markup=True, width=110)

class _DualConsole:
    def __init__(self):
        self._term = Console()
        self._file = _file_console

    def print(self, *args, **kwargs):
        self._term.print(*args, **kwargs)
        self._file.print(*args, **kwargs)

    def rule(self, *args, **kwargs):
        self._term.rule(*args, **kwargs)
        self._file.rule(*args, **kwargs)

console = _DualConsole()


# ── Step 1: Load FAERS Data ───────────────────────────────────────────────────

def step1_load_faers(faers_dirs: list[str] | None = None, force_synthetic: bool = False):
    """
    Load FAERS data from quarterly directories, one per hospital shard.
    Falls back to synthetic FAERS-compatible data if real data unavailable.

    Key differences from synthetic data_gen:
      - Real FAERS has messy brand names  → normalizer maps them to canonical
      - Real FAERS has no lab values      → creatinine/hba1c/potassium imputed
      - Real FAERS n_conditions = 0       → causal pipeline uses age/drugs only
      - Each quarter = one hospital shard → natural non-IID split by time period
    """
    console.rule("[bold cyan]Step 1: Loading FAERS Patient Data")
    from data_faers import (load_faers_multi_quarter, generate_faers_synthetic,
                             extract_features_faers)

    using_real = False

    if not force_synthetic and faers_dirs:
        try:
            console.print(f"  Attempting to load {len(faers_dirs)} FAERS quarter(s)...")
            df_all = load_faers_multi_quarter(faers_dirs)
            using_real = True
            console.print(f"  [green]✓[/] Loaded real FAERS data")
        except Exception as e:
            console.print(f"  [yellow]⚠ FAERS load failed: {e}[/]")
            console.print(f"  [yellow]  Falling back to synthetic FAERS-compatible data[/]")

    if not using_real:
        console.print("  [dim]Using synthetic FAERS-compatible data (no real FAERS dirs provided)[/]")
        df_all = generate_faers_synthetic(n_cases=1200, n_quarters=3)

    # Assign hospital_id per unique quarter/shard if not already set
    if "hospital_id" not in df_all.columns:
        df_all["hospital_id"] = 0

    X_all, y_all = extract_features_faers(df_all)

    data_source = "Real FAERS" if using_real else "Synthetic FAERS-compatible"
    console.print(f"  [green]✓[/] Data source:   {data_source}")
    console.print(f"  [green]✓[/] Total patients: {len(df_all)}")
    console.print(f"  [green]✓[/] ADR rate:       {y_all.mean():.1%}")
    console.print(f"  [green]✓[/] Feature dims:   {X_all.shape[1]}")

    # Per-shard summary
    shards = []
    hospital_ids = sorted(df_all["hospital_id"].unique())
    for hid in hospital_ids:
        df_h = df_all[df_all["hospital_id"] == hid].reset_index(drop=True)
        X_h, y_h = extract_features_faers(df_h)
        shards.append((df_h, X_h, y_h))
        label = f"Quarter {hid}" if using_real else f"Hospital {hid}"
        console.print(f"    {label}: {len(df_h)} patients, ADR rate {y_h.mean():.1%}")

    if len(shards) < 2:
        console.print(
            "  [yellow]⚠ Only 1 shard found. FL needs ≥2 hospitals. "
            "Splitting single shard into 3 virtual hospitals.[/]"
        )
        df_all_reset = df_all.reset_index(drop=True)
        n = len(df_all_reset)
        chunk = n // 3
        shards = []
        for hid in range(3):
            start = hid * chunk
            end   = (hid + 1) * chunk if hid < 2 else n
            df_h  = df_all_reset.iloc[start:end].reset_index(drop=True)
            df_h["hospital_id"] = hid
            X_h, y_h = extract_features_faers(df_h)
            shards.append((df_h, X_h, y_h))
            console.print(f"    Virtual Hospital {hid}: {len(df_h)} patients")

    # ── FAERS-specific data quality notes ────────────────────────────────────
    console.print()
    console.print("  [dim]FAERS data notes:[/]")
    console.print("  [dim]  · Lab values (creatinine, HbA1c, K⁺) imputed at population means[/]")
    console.print("  [dim]  · Comorbidity count = 0 (not in FAERS schema)[/]")
    console.print("  [dim]  · Drug names normalised: brand → generic via DRUG_NORM map[/]")
    console.print("  [dim]  · ADR labels: any mapped reaction term = adr_occurred=1[/]")

    return df_all, X_all, y_all, shards, using_real


# ── Step 2: Causal Inference + Normalisation ──────────────────────────────────

def step2_causal(df_all, X_all, y_all, shards):
    """
    Same causal pipeline as run.py.
    Note: X_confounders uses only the first 6 features (age, n_conditions,
    n_drugs, creatinine, hba1c, potassium). For FAERS data, n_conditions=0
    and labs are imputed, so the PSM is driven mainly by age and n_drugs.
    This is a known limitation vs. real EHR data with full lab panels.
    """
    console.rule("[bold cyan]Step 2: Causal Inference — Propensity Matching + T-Learner")
    from causal import run_causal_pipeline
    from data_faers import extract_features_faers

    X_confounders = X_all[:, :6]   # age, n_conditions, n_drugs, creatinine, hba1c, potassium
    cate_features, report = run_causal_pipeline(df_all, X_confounders, y_all)

    t = Table(title="Per-Drug Causal Attribution (CATE)", show_header=True)
    t.add_column("Drug",      style="cyan")
    t.add_column("Status",    style="green")
    t.add_column("N Treated", justify="right")
    t.add_column("N Matched", justify="right")
    t.add_column("Mean CATE", justify="right", style="yellow")
    t.add_column("Std CATE",  justify="right")

    for drug, info in report.items():
        t.add_row(
            drug, info["status"],
            str(info["n_treated"]),
            str(info.get("n_matched", "-")),
            f"{info['cate_mean']:+.4f}" if info["status"] == "ok" else "-",
            f"{info.get('cate_std', 0):.4f}" if info["status"] == "ok" else "-",
        )
    console.print(t)

    # Augment features with CATE
    X_all_aug_raw = np.hstack([X_all, cate_features])
    scaler = StandardScaler()
    scaler.fit(X_all_aug_raw)
    X_all_aug = scaler.transform(X_all_aug_raw)

    aug_shards = []
    for hid, (df_h, X_h, y_h) in enumerate(shards):
        mask = df_all["hospital_id"] == df_h["hospital_id"].iloc[0]
        cate_h = cate_features[mask.values]
        # Pad/trim if size mismatch (can happen at shard boundaries)
        if len(cate_h) != len(X_h):
            cate_h = cate_features[df_all["hospital_id"] == hid]
        X_h_aug = scaler.transform(np.hstack([X_h, cate_h]))
        aug_shards.append((df_h, X_h_aug, y_h))

    console.print(f"  [green]✓[/] Augmented dims: {X_all_aug.shape[1]} (base {X_all.shape[1]} + 5 CATE)")
    console.print(f"  [green]✓[/] StandardScaler applied — zero-mean, unit-variance")
    return X_all_aug, aug_shards, cate_features, scaler


# ── Steps 3–7: Reuse from run.py unchanged ───────────────────────────────────

def step3_local_baseline(aug_shards):
    console.rule("[bold cyan]Step 3: Baseline — Local-Only Training (No Federation)")
    from model import ADRNet, train_local, evaluate

    local_metrics = []
    for hid, (df_h, X_h, y_h) in enumerate(aug_shards):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
        m = ADRNet(input_dim=X_h.shape[1])
        train_local(m, X_tr, y_tr, epochs=30)
        metrics = evaluate(m, X_te, y_te)
        local_metrics.append(metrics)
        console.print(
            f"  Hospital {hid}: AUROC={metrics['auroc']:.4f}  "
            f"AUPRC={metrics['auprc']:.4f}  F1={metrics['f1']:.4f}  "
            f"FPR={metrics['fpr']:.4f}  ECE={metrics['ece']:.4f}")

    avg = {k: round(np.mean([m[k] for m in local_metrics]), 4) for k in local_metrics[0]}
    console.print(
        f"  [yellow]Average local:[/]  AUROC={avg['auroc']:.4f}  "
        f"AUPRC={avg['auprc']:.4f}  F1={avg['f1']:.4f}  "
        f"FPR={avg['fpr']:.4f}  ECE={avg['ece']:.4f}")
    return avg


def step4_federated(aug_shards, fl_rounds: int = 20):
    console.rule(
        f"[bold cyan]Step 4: Federated Learning — {fl_rounds} rounds, "
        f"{len(aug_shards)} hospital clients (custom FedProx)")
    from model import evaluate
    from fl_client import run_federated

    input_dim = aug_shards[0][1].shape[1]
    splits = []
    for df_h, X_h, y_h in aug_shards:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
        splits.append((X_tr, y_tr, X_te, y_te))

    X_te_all = np.vstack([s[2] for s in splits])
    y_te_all = np.concatenate([s[3] for s in splits])

    def evaluate_fn(rnd, global_model):
        metrics = evaluate(global_model, X_te_all, y_te_all)
        console.print(
            f"  Round {rnd:2d}: AUROC={metrics['auroc']:.4f}  "
            f"AUPRC={metrics['auprc']:.4f}  F1={metrics['f1']:.4f}  "
            f"FPR={metrics['fpr']:.4f}  ECE={metrics['ece']:.4f}  "
            f"σ²_epi={metrics['mean_epistemic']:.5f}  "
            f"σ²_ale={metrics['mean_aleatoric']:.5f}")
        return metrics

    round_metrics, global_model = run_federated(
        splits=splits, input_dim=input_dim,
        fl_rounds=fl_rounds, local_epochs=5,
        proximal_mu=0.1, evaluate_fn=evaluate_fn)

    return round_metrics, global_model


def step5_final_evaluation(aug_shards, round_metrics, global_model):
    console.rule("[bold cyan]Step 5: Temperature Calibration + Final Results")
    from model import evaluate, fit_temperature

    splits = []
    for df_h, X_h, y_h in aug_shards:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
        splits.append((X_tr, y_tr, X_te, y_te))

    X_te_all = np.vstack([s[2] for s in splits])
    y_te_all = np.concatenate([s[3] for s in splits])

    from sklearn.model_selection import train_test_split as tts
    X_cal, X_rep, y_cal, y_rep = tts(
        X_te_all, y_te_all, test_size=0.5, random_state=0, stratify=y_te_all)

    T_star       = fit_temperature(global_model, X_cal, y_cal, n_passes=20)
    metrics_raw  = evaluate(global_model, X_rep, y_rep, temperature=1.0)
    metrics_cal  = evaluate(global_model, X_rep, y_rep, temperature=T_star)

    console.print()
    console.print(
        f"  [bold]Temperature scaling:[/]  T* = [yellow]{T_star:.4f}[/yellow]  "
        f"({'softening' if T_star > 1 else 'sharpening'})")
    console.print(
        f"  ECE before: [red]{metrics_raw['ece']:.4f}[/red]  →  "
        f"ECE after: [green]{metrics_cal['ece']:.4f}[/green]")
    console.print(
        f"  AUROC: {metrics_raw['auroc']:.4f} → {metrics_cal['auroc']:.4f}  "
        f"[dim](rank-preserving ✓)[/dim]")

    console.print()
    console.print(Panel(
        f"[bold]Final Round {round_metrics[-1][0]} — Global Federated Model (calibrated)[/bold]\n\n"
        f"  AUROC          : [green]{metrics_cal['auroc']:.4f}[/green]\n"
        f"  AUPRC          : [green]{metrics_cal['auprc']:.4f}[/green]\n"
        f"  F1 Score       : [green]{metrics_cal['f1']:.4f}[/green]\n"
        f"  False Pos Rate : {metrics_cal['fpr']:.4f}\n"
        f"  Calib. ECE     : [green]{metrics_cal['ece']:.4f}[/green]  [dim](T*={T_star:.3f})[/dim]\n\n"
        f"[bold]Uncertainty (MC Dropout, T=50 passes):[/bold]\n"
        f"  σ²_epistemic   : {metrics_cal['mean_epistemic']:.5f}  (model ignorance)\n"
        f"  σ²_aleatoric   : {metrics_cal['mean_aleatoric']:.5f}  (data noise)\n",
        title="[bold cyan]FPS Results (FAERS)", border_style="cyan"))

    return round_metrics[0][1], metrics_cal, T_star


def step6_comparison(local_avg, fl_round1, fl_final):
    console.rule("[bold cyan]Step 6: Comparison — Local vs FL Round 1 vs FL Final")

    t = Table(title="FPS Results (FAERS Data)", show_header=True,
              header_style="bold magenta")
    t.add_column("Metric",          style="cyan", min_width=18)
    t.add_column("Local Only",      justify="center")
    t.add_column("FL Round 1",      justify="center")
    t.add_column("FL Final (Ours)", justify="center", style="bold green")

    def fmt(v): return f"{v:.4f}"
    rows = [
        ("AUROC",         "auroc"), ("AUPRC",      "auprc"),
        ("F1 Score",      "f1"),    ("False Pos Rate", "fpr"),
        ("Calib. ECE",    "ece"),
    ]
    for label, key in rows:
        t.add_row(label, fmt(local_avg[key]), fmt(fl_round1[key]), fmt(fl_final[key]))

    t.add_row("Privacy", "[red]No[/red]", "[green]Yes[/green]", "[green]Yes[/green]")
    console.print(t)

    auroc_gain = fl_final["auroc"] - local_avg["auroc"]
    color = "green" if auroc_gain > 0 else "red"
    console.print(f"\n  AUROC gain (FL Final vs Local): [{color}]{auroc_gain:+.4f}[/{color}]")

    if auroc_gain > 0:
        console.print(Panel(
            "[bold green]✓ Federated learning outperforms local-only training on FAERS data.[/]\n"
            "Privacy-preserving · Causal CATE features · MC Dropout uncertainty",
            border_style="green"))
    else:
        console.print(Panel(
            f"[yellow]⚠ FL Final AUROC ({fl_final['auroc']:.4f}) did not beat "
            f"local ({local_avg['auroc']:.4f}).\n"
            f"With real FAERS data this is expected if lab imputation reduces signal quality.\n"
            f"Adding real lab values would likely improve results.[/yellow]",
            border_style="yellow"))


def step7_attribution(aug_shards, df_all, global_model, temperature: float = 1.0):
    console.rule("[bold cyan]Step 7: Drug Attribution — Identifying Culprit Drugs")
    from attribution import batch_attribute, print_attribution_report

    splits = []
    for df_h, X_h, y_h in aug_shards:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
        splits.append((X_tr, y_tr, X_te, y_te))

    X_te_all = np.vstack([s[2] for s in splits])
    df_test_parts = []
    for df_h, X_h, y_h in aug_shards:
        _, df_te_h = train_test_split(df_h, test_size=0.2, random_state=42, stratify=y_h)
        df_test_parts.append(df_te_h)
    df_test = pd.concat(df_test_parts, ignore_index=True)

    console.print(f"  Running attribution on top high-risk patients "
                  f"(threshold >= 60%, T*={temperature:.3f})...")

    reports = batch_attribute(
        model=global_model, X=X_te_all, df=df_test,
        top_n=5, risk_threshold=0.60,
        n_mc_passes=15, temperature=temperature)

    if not reports:
        console.print("  [yellow]No patients above 60% — lowering threshold to 40%[/]")
        reports = batch_attribute(
            global_model, X_te_all, df_test,
            top_n=5, risk_threshold=0.40,
            n_mc_passes=15, temperature=temperature)

    console.print(f"\n  [green]✓[/] Attribution complete for {len(reports)} high-risk patients\n")
    for report in reports:
        print_attribution_report(report, console=console)

    return reports


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="FPS pipeline with FAERS data")
    parser.add_argument(
        "--faers-dirs", nargs="+", default=None,
        metavar="DIR",
        help="Paths to unzipped FAERS quarterly folders (one per hospital shard). "
             "E.g.: --faers-dirs faers_data/2024q1 faers_data/2024q2 faers_data/2024q3")
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Force use of synthetic FAERS-compatible data (skip real FAERS load)")
    parser.add_argument(
        "--fl-rounds", type=int, default=20,
        help="Number of federated learning rounds (default: 20)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    console.print(Panel(
        "[bold]Federated Polypharmacy Safety (FPS) — FAERS Edition[/bold]\n"
        "Privacy-preserving · Causal · Uncertainty-aware · Temperature Calibrated\n"
        f"Data: {'Synthetic FAERS-compatible' if args.synthetic else 'Real FAERS (with synthetic fallback)'}",
        title="FPS Demo", border_style="blue"))

    df_all, X_all, y_all, shards, using_real = step1_load_faers(
        faers_dirs=args.faers_dirs,
        force_synthetic=args.synthetic)

    X_all_aug, aug_shards, cate_feats, sc = step2_causal(df_all, X_all, y_all, shards)
    local_avg                              = step3_local_baseline(aug_shards)
    round_metrics, global_model            = step4_federated(aug_shards, fl_rounds=args.fl_rounds)
    fl_round1, fl_final, T_star            = step5_final_evaluation(
                                                aug_shards, round_metrics, global_model)
    step6_comparison(local_avg, fl_round1, fl_final)
    step7_attribution(aug_shards, df_all, global_model, temperature=T_star)

    _file_handle.flush()
    _file_handle.close()
    print(f"\n  Results saved to: {_RESULTS_FILE}")
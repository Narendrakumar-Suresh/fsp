"""
run.py — FPS Proof of Concept
Runs the full pipeline end-to-end and prints results to terminal + fps_results.txt.

Pipeline:
  1. Generate synthetic patients (3 hospital shards)
  2. Causal inference → CATE features per drug
  3. StandardScaler normalization (critical for FL convergence)
  4. Baseline: local-only training per hospital
  5. Federated learning — custom FedProx loop (no deprecated Flower API)
  6. Comparison table: local vs FL
  7. Drug attribution — Shapley + LOO culprit identification

Usage:
  uv run run.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# ── Dual output: terminal + fps_results.txt ──────────────────────────────────
_RESULTS_FILE = "fps_results.txt"
_file_handle  = open(_RESULTS_FILE, "w", encoding="utf-8")
_file_console = Console(file=_file_handle, highlight=False, markup=True, width=110)

class _DualConsole:
    """Writes to both terminal and results file simultaneously."""
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


# ── Step 1: Generate Data ────────────────────────────────────────────────────

def step1_generate():
    console.rule("[bold cyan]Step 1: Generating Synthetic Patient Data")
    from data_gen import generate_all, extract_features

    df_all = generate_all(n_per_hospital=400)
    X_all, y_all = extract_features(df_all)

    console.print(f"  [green]✓[/] Total patients: {len(df_all)}")
    console.print(f"  [green]✓[/] ADR rate:       {y_all.mean():.1%}")
    console.print(f"  [green]✓[/] Feature dims:   {X_all.shape[1]}")

    shards = []
    for hid in range(3):
        df_h = df_all[df_all["hospital_id"] == hid].reset_index(drop=True)
        X_h, y_h = extract_features(df_h)
        shards.append((df_h, X_h, y_h))
        console.print(f"    Hospital {hid}: {len(df_h)} patients, ADR rate {y_h.mean():.1%}")

    return df_all, X_all, y_all, shards


# ── Step 2: Causal Inference + Normalization ─────────────────────────────────

def step2_causal(df_all, X_all, y_all, shards):
    console.rule("[bold cyan]Step 2: Causal Inference — Propensity Matching + T-Learner")
    from causal import run_causal_pipeline

    X_confounders = X_all[:, :6]
    cate_features, report = run_causal_pipeline(df_all, X_confounders, y_all)

    t = Table(title="Per-Drug Causal Attribution (CATE)", show_header=True)
    t.add_column("Drug",       style="cyan")
    t.add_column("Status",     style="green")
    t.add_column("N Treated",  justify="right")
    t.add_column("N Matched",  justify="right")
    t.add_column("Mean CATE",  justify="right", style="yellow")
    t.add_column("Std CATE",   justify="right")

    for drug, info in report.items():
        t.add_row(
            drug,
            info["status"],
            str(info["n_treated"]),
            str(info.get("n_matched", "-")),
            f"{info['cate_mean']:+.4f}" if info["status"] == "ok" else "-",
            f"{info.get('cate_std', 0):.4f}" if info["status"] == "ok" else "-",
        )
    console.print(t)

    # Augment raw features with CATE
    X_all_aug_raw = np.hstack([X_all, cate_features])

    # ── StandardScaler: fit on combined, apply per shard ─────────────────────
    # This is non-negotiable for FL convergence: without it high-magnitude
    # clinical features (age ~70) swamp drug interaction signals (~0.06).
    scaler = StandardScaler()
    scaler.fit(X_all_aug_raw)
    X_all_aug = scaler.transform(X_all_aug_raw)

    aug_shards = []
    for hid in range(3):
        df_h, X_h, y_h = shards[hid]
        mask   = df_all["hospital_id"] == hid
        cate_h = cate_features[mask.values]
        X_h_aug_raw = np.hstack([X_h, cate_h])
        X_h_aug = scaler.transform(X_h_aug_raw)
        aug_shards.append((df_h, X_h_aug, y_h))

    console.print(f"  [green]✓[/] Augmented dims: {X_all_aug.shape[1]} (base 26 + 5 CATE)")
    console.print(f"  [green]✓[/] StandardScaler applied — zero-mean, unit-variance")
    return X_all_aug, aug_shards, cate_features, scaler


# ── Step 3: Baseline — Local-Only Training ───────────────────────────────────

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
            f"AUPRC={metrics['auprc']:.4f}  "
            f"F1={metrics['f1']:.4f}  FPR={metrics['fpr']:.4f}  "
            f"ECE={metrics['ece']:.4f}")

    avg = {k: round(np.mean([m[k] for m in local_metrics]), 4) for k in local_metrics[0]}
    console.print(
        f"  [yellow]Average local:[/]  AUROC={avg['auroc']:.4f}  "
        f"AUPRC={avg['auprc']:.4f}  F1={avg['f1']:.4f}  "
        f"FPR={avg['fpr']:.4f}  ECE={avg['ece']:.4f}")
    return avg


# ── Step 4: Federated Learning — custom FedProx, no deprecated Flower API ────

def step4_federated(aug_shards, fl_rounds: int = 20):
    console.rule(
        f"[bold cyan]Step 4: Federated Learning — {fl_rounds} rounds, "
        f"3 virtual hospital clients (custom FedProx)")
    from model import ADRNet, evaluate
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
            f"AUPRC={metrics['auprc']:.4f}  "
            f"F1={metrics['f1']:.4f}  FPR={metrics['fpr']:.4f}  "
            f"ECE={metrics['ece']:.4f}  "
            f"σ²_epi={metrics['mean_epistemic']:.5f}  "
            f"σ²_ale={metrics['mean_aleatoric']:.5f}"
        )
        return metrics

    round_metrics, global_model = run_federated(
        splits=splits,
        input_dim=input_dim,
        fl_rounds=fl_rounds,
        local_epochs=5,
        proximal_mu=0.1,    # mild proximal term — stabilises non-IID without over-constraining
        evaluate_fn=evaluate_fn,
    )

    return round_metrics, global_model


# ── Step 5: Temperature Calibration + Final Evaluation ──────────────────────

def step5_final_evaluation(aug_shards, round_metrics, global_model):
    console.rule("[bold cyan]Step 5: Temperature Calibration + Final Results")
    from model import evaluate, fit_temperature

    final_round, final_metrics = round_metrics[-1]
    first_round, first_metrics = round_metrics[0]

    # Build cal/test splits from aug_shards (same seed = same splits as FL)
    splits = []
    for df_h, X_h, y_h in aug_shards:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
        splits.append((X_tr, y_tr, X_te, y_te))

    X_te_all = np.vstack([s[2] for s in splits])
    y_te_all = np.concatenate([s[3] for s in splits])

    # ── Calibration: carve 50% of test set as cal set, keep 50% for reporting
    # In a real system you'd use a dedicated held-out set; here we split the
    # existing test set to keep the pipeline self-contained.
    from sklearn.model_selection import train_test_split as tts
    X_cal, X_rep, y_cal, y_rep = tts(
        X_te_all, y_te_all, test_size=0.5, random_state=0, stratify=y_te_all)

    # Fit temperature on cal split
    T_star = fit_temperature(global_model, X_cal, y_cal, n_passes=20)

    # Evaluate on reporting split — uncalibrated vs calibrated
    metrics_raw  = evaluate(global_model, X_rep, y_rep, temperature=1.0)
    metrics_cal  = evaluate(global_model, X_rep, y_rep, temperature=T_star)

    console.print()
    console.print(
        f"  [bold]Temperature scaling:[/bold]  "
        f"T* = [yellow]{T_star:.4f}[/yellow]  "
        f"({'softening overconfident predictions' if T_star > 1 else 'sharpening underconfident predictions'})"
    )
    console.print(
        f"  ECE before calibration : [red]{metrics_raw['ece']:.4f}[/red]  →  "
        f"ECE after calibration  : [green]{metrics_cal['ece']:.4f}[/green]"
    )
    console.print(
        f"  AUROC unchanged        : {metrics_raw['auroc']:.4f}  →  {metrics_cal['auroc']:.4f}  "
        f"[dim](rank-preserving ✓)[/dim]"
    )

    auroc_color = "green" if metrics_cal["auroc"] >= 0.70 else \
                  "yellow" if metrics_cal["auroc"] >= 0.60 else "red"
    ece_color   = "green" if metrics_cal["ece"] <= 0.05 else \
                  "yellow" if metrics_cal["ece"] <= 0.10 else "red"

    console.print()
    console.print(Panel(
        f"[bold]Final Round {final_round} — Global Federated Model (calibrated)[/bold]\n\n"
        f"  AUROC          : [{auroc_color}]{metrics_cal['auroc']:.4f}[/{auroc_color}]\n"
        f"  AUPRC          : [green]{metrics_cal['auprc']:.4f}[/green]\n"
        f"  F1 Score       : [green]{metrics_cal['f1']:.4f}[/green]\n"
        f"  False Pos Rate : [green]{metrics_cal['fpr']:.4f}[/green]\n"
        f"  Calib. ECE     : [{ece_color}]{metrics_cal['ece']:.4f}[/{ece_color}]"
        f"  [dim](T*={T_star:.3f})[/dim]\n\n"
        f"[bold]Uncertainty (MC Dropout, T=50 passes):[/bold]\n"
        f"  σ²_epistemic   : {metrics_cal['mean_epistemic']:.5f}  (model ignorance)\n"
        f"  σ²_aleatoric   : {metrics_cal['mean_aleatoric']:.5f}  (data noise)\n",
        title="[bold cyan]FPS Results",
        border_style="cyan"
    ))

    # Return calibrated metrics + T* for downstream use
    return first_metrics, metrics_cal, T_star


# ── Step 6: Comparison Table ─────────────────────────────────────────────────

def step6_comparison(local_avg, fl_round1, fl_final):
    console.rule("[bold cyan]Step 6: Comparison — Local vs FL Round 1 vs FL Final")

    t = Table(title="FPS Proof-of-Concept Results",
              show_header=True, header_style="bold magenta")
    t.add_column("Metric",           style="cyan", min_width=18)
    t.add_column("Local Only",       justify="center")
    t.add_column("FL Round 1",       justify="center")
    t.add_column("FL Final (Ours)",  justify="center", style="bold green")

    def fmt(v): return f"{v:.4f}"

    t.add_row("AUROC",
              fmt(local_avg['auroc']),    fmt(fl_round1['auroc']),    fmt(fl_final['auroc']))
    t.add_row("AUPRC",
              fmt(local_avg['auprc']),    fmt(fl_round1['auprc']),    fmt(fl_final['auprc']))
    t.add_row("F1 Score",
              fmt(local_avg['f1']),       fmt(fl_round1['f1']),       fmt(fl_final['f1']))
    t.add_row("False Pos Rate",
              fmt(local_avg['fpr']),      fmt(fl_round1['fpr']),      fmt(fl_final['fpr']))
    t.add_row("Calib. ECE",
              fmt(local_avg['ece']),      fmt(fl_round1['ece']),      fmt(fl_final['ece']))
    t.add_row("σ² Epistemic",
              fmt(local_avg['mean_epistemic']),
              fmt(fl_round1['mean_epistemic']),
              fmt(fl_final['mean_epistemic']))
    t.add_row("σ² Aleatoric",
              fmt(local_avg['mean_aleatoric']),
              fmt(fl_round1['mean_aleatoric']),
              fmt(fl_final['mean_aleatoric']))
    t.add_row("Privacy",     "[red]No[/red]",    "[green]Yes[/green]", "[green]Yes[/green]")
    t.add_row("Causal CATE", "[green]Yes[/green]","[green]Yes[/green]","[green]Yes[/green]")

    console.print(t)

    auroc_gain = fl_final['auroc'] - local_avg['auroc']
    fpr_drop   = local_avg['fpr']  - fl_final['fpr']

    gain_color = "green" if auroc_gain > 0 else "red"
    drop_color = "green" if fpr_drop  > 0 else "red"

    console.print()
    console.print(
        f"  [bold]AUROC gain  (FL Final vs Local):[/] "
        f"[{gain_color}]{auroc_gain:+.4f}[/{gain_color}]")
    console.print(
        f"  [bold]FPR  drop   (FL Final vs Local):[/] "
        f"[{drop_color}]{fpr_drop:+.4f}[/{drop_color}]")
    console.print()

    # Only claim success if FL actually beats local
    if auroc_gain > 0:
        console.print(Panel(
            "[bold green]✓ Federated learning outperforms local-only training.[/bold green]\n"
            "Privacy-preserving · Causal CATE features · MC Dropout uncertainty",
            border_style="green"))
    else:
        console.print(Panel(
            f"[bold yellow]⚠ FL Final AUROC ({fl_final['auroc']:.4f}) did not beat "
            f"local ({local_avg['auroc']:.4f}) — check training config.[/bold yellow]",
            border_style="yellow"))


# ── Step 7: Drug Attribution ─────────────────────────────────────────────────

def step7_attribution(aug_shards, df_all, global_model, temperature: float = 1.0):
    console.rule("[bold cyan]Step 7: Drug Attribution — Identifying Culprit Drugs")
    from attribution import batch_attribute, print_attribution_report

    splits = []
    for df_h, X_h, y_h in aug_shards:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_h, y_h, test_size=0.2, random_state=42, stratify=y_h)
        splits.append((X_tr, y_tr, X_te, y_te))

    X_te_all = np.vstack([s[2] for s in splits])
    y_te_all = np.concatenate([s[3] for s in splits])

    df_test_parts = []
    for hid, (df_h, X_h, y_h) in enumerate(aug_shards):
        _, df_te_h = train_test_split(
            df_h, test_size=0.2, random_state=42, stratify=y_h)
        df_test_parts.append(df_te_h)
    df_test = pd.concat(df_test_parts, ignore_index=True)

    console.print("  Running attribution on top high-risk patients "
                  f"(threshold >= 60%, temperature T*={temperature:.3f})...")

    reports = batch_attribute(
        model=global_model, X=X_te_all, df=df_test,
        top_n=5, risk_threshold=0.60, n_mc_passes=15,
        temperature=temperature)

    if not reports:
        console.print("  [yellow]No patients above 60% — lowering threshold to 40%[/]")
        reports = batch_attribute(
            global_model, X_te_all, df_test,
            top_n=5, risk_threshold=0.40, n_mc_passes=15,
            temperature=temperature)

    console.print(f"\n  [green]✓[/] Attribution complete for {len(reports)} high-risk patients\n")
    for report in reports:
        print_attribution_report(report, console=console)

    return reports


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print(Panel(
        "[bold]Federated Polypharmacy Safety (FPS)[/bold]\n"
        "Privacy-preserving · Causal · Uncertainty-aware · Temperature Calibrated",
        title="FPS Demo", border_style="blue"))

    df_all, X_all, y_all, shards          = step1_generate()
    X_all_aug, aug_shards, cate_feats, sc  = step2_causal(df_all, X_all, y_all, shards)
    local_avg                              = step3_local_baseline(aug_shards)
    round_metrics, global_model            = step4_federated(aug_shards, fl_rounds=20)
    fl_round1, fl_final, T_star            = step5_final_evaluation(
                                                aug_shards, round_metrics, global_model)
    step6_comparison(local_avg, fl_round1, fl_final)
    step7_attribution(aug_shards, df_all, global_model, temperature=T_star)

    _file_handle.flush()
    _file_handle.close()
    print(f"\n  Results saved to: {_RESULTS_FILE}")
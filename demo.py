"""
FPS Demo — Real-time drug combination risk assessment
Drop into your project folder alongside the other files and run:
    python demo.py
"""

import warnings

warnings.filterwarnings("ignore")
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/mnt/user-data/uploads")

import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from attribution import PatientRecord, detect_synergies, leave_one_out
from causal import run_causal_pipeline
from data_gen import DRUGS, HIGH_RISK_TRIPLES, extract_features, generate_all
from model import ADRNet, fit_temperature, mc_predict, train_local

console.print(
    Panel(
        "[bold]FPS — Federated Polypharmacy Safety[/bold]\n"
        "Real-time Drug Combination Risk Assessment\n"
        "[dim]Privacy-preserving · Causal · Uncertainty-aware[/dim]",
        border_style="cyan",
    )
)

# ── Model / scaler paths ───────────────────────────────────────────────────
MODEL_PATH = "fps_model.pt"
SCALER_PATH = "fps_scaler.pkl"

# ── Load or train model ────────────────────────────────────────────────────
console.print("\n[bold cyan]Loading...[/bold cyan] ", end="")

# Always need data to build the feature matrix (determines input_dim)
df_all = generate_all(n_per_hospital=1500)
X_all, y_all = extract_features(df_all)
cate_features, _ = run_causal_pipeline(df_all, X_all[:, :6], y_all)
X_aug_raw = np.hstack([X_all, cate_features])

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    # ── Existing model found — load weights + scaler ───────────────────────
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    X_aug = scaler.transform(X_aug_raw)
    _, X_te, _, y_te = train_test_split(
        X_aug, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    model = ADRNet(input_dim=X_aug.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    T_star = fit_temperature(model, X_te, y_te, n_passes=20)
    console.print(f"[green]Model loaded! (T*={T_star:.3f})[/green]\n")

else:
    # ── No saved model — train from scratch then save ──────────────────────
    scaler = StandardScaler()
    X_aug = scaler.fit_transform(X_aug_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_aug, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    model = ADRNet(input_dim=X_aug.shape[1])
    train_local(model, X_tr, y_tr, epochs=150)
    T_star = fit_temperature(model, X_te, y_te, n_passes=20)

    torch.save(model.state_dict(), MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    console.print(f"[green]Ready! (T*={T_star:.3f})[/green]\n")

# ── Main loop ──────────────────────────────────────────────────────────────
console.print("[bold]Available drugs:[/bold]")
for i, d in enumerate(DRUGS):
    console.print(f"  {i + 1:2d}. {d}", end="   ")
    if (i + 1) % 4 == 0:
        console.print()
console.print("\n")

while True:
    console.rule("[bold cyan]New Patient")
    raw = input("\nEnter drugs (comma-separated, or 'q' to quit): ").strip().lower()
    if raw == "q":
        break

    drug_names = [d.strip() for d in raw.split(",") if d.strip() in DRUGS]
    unknown = [
        d.strip() for d in raw.split(",") if d.strip() not in DRUGS and d.strip()
    ]

    if unknown:
        console.print(f"[yellow]Unknown drugs ignored: {', '.join(unknown)}[/yellow]")
    if len(drug_names) < 2:
        console.print("[red]Need at least 2 valid drugs. Try again.[/red]")
        continue

    try:
        age = int(input("Patient age [default 68]: ").strip() or "68")
        creatinine = float(input("Creatinine mg/dL [default 1.4]: ").strip() or "1.4")
        n_conds = int(input("Number of comorbidities [default 3]: ").strip() or "3")
    except ValueError:
        age, creatinine, n_conds = 68, 1.4, 3

    console.print(
        f"\n[bold]Assessing:[/bold] {', '.join(drug_names)} | age={age} | creatinine={creatinine}\n"
    )

    # Build feature vector
    x_basic = np.array(
        [age, n_conds, len(drug_names), creatinine, 6.8, 4.3], dtype=np.float32
    )
    drug_ohe = np.zeros(len(DRUGS), dtype=np.float32)
    drug_idx = []
    for d in drug_names:
        i = DRUGS.index(d)
        drug_ohe[i] = 1.0
        drug_idx.append(i)
    x_raw = np.concatenate([x_basic, drug_ohe, np.zeros(5, dtype=np.float32)])
    x_norm = scaler.transform(x_raw.reshape(1, -1))[0]

    patient = PatientRecord("PATIENT", x_norm, drug_idx, drug_names)

    # MC Dropout inference
    console.print("[dim]Running 50 MC Dropout passes...[/dim]")
    result = mc_predict(model, x_norm.reshape(1, -1), n_passes=50, temperature=T_star)
    risk = float(result["risk_score"][0])
    epi = float(result["epistemic"][0])
    ale = float(result["aleatoric"][0])
    ci_lo, ci_hi = float(result["ci_lower"][0]), float(result["ci_upper"][0])

    alert = (
        "CRITICAL"
        if risk >= 0.8
        else "HIGH"
        if risk >= 0.6
        else "MODERATE"
        if risk >= 0.4
        else "LOW"
    )
    color = {
        "CRITICAL": "bold red",
        "HIGH": "red",
        "MODERATE": "yellow",
        "LOW": "green",
    }[alert]

    # Detected known triples
    drug_set = set(drug_names)
    active_triples = [
        f"{a}+{b}+{c}" for a, b, c in HIGH_RISK_TRIPLES if {a, b, c}.issubset(drug_set)
    ]
    triple_line = (
        f"\n[bold red]⚠ HIGH-RISK TRIPLE: {', '.join(active_triples)}[/bold red]"
        if active_triples
        else ""
    )

    console.print(
        Panel(
            f"[bold]Regimen:[/bold] {', '.join(drug_names)}{triple_line}\n\n"
            f"[bold]ADR Risk Score:[/bold]  [{color}]{risk:.1%}[/{color}]  [{color}]{alert}[/{color}]\n"
            f"[bold]Uncertainty :[/bold]\n"
            f"  σ²_epistemic = {epi:.5f}  (model ignorance — reducible)\n"
            f"  σ²_aleatoric = {ale:.5f}  (data noise — irreducible)\n"
            f"  σ²_total     = {epi + ale:.5f}",
            title="[bold cyan]FPS Risk Assessment",
            border_style=color.split()[-1],
        )
    )

    # LOO attribution
    console.print("[dim]Computing Leave-One-Out attribution...[/dim]")
    loo = leave_one_out(model, patient, n_passes=20, temperature=T_star)["loo"]

    t = Table(title="Which drug is driving the risk?", show_header=True)
    t.add_column("Drug", style="cyan")
    t.add_column("Remove it → risk shifts by", justify="right")
    t.add_column("Role")
    for drug in sorted(loo, key=lambda d: -loo[d]):
        delta = loo[drug]
        role = (
            "⚠ PRIMARY CULPRIT"
            if delta > 0.10
            else "↑ Adds Risk"
            if delta > 0.02
            else "↓ Protective"
            if delta < -0.02
            else "Neutral"
        )
        c = (
            "bold red"
            if delta > 0.10
            else "red"
            if delta > 0.02
            else "green"
            if delta < -0.02
            else "white"
        )
        t.add_row(drug, f"[{c}]{delta:+.1%}[/{c}]", role)
    console.print(t)

    # Synergy
    console.print("[dim]Detecting synergies...[/dim]")
    synergies = detect_synergies(
        model, patient, n_passes=10, top_k=3, temperature=T_star
    )
    if synergies:
        s = Table(title="Drug Interaction Synergies", show_header=True)
        s.add_column("Combination", style="yellow")
        s.add_column("Order", justify="center")
        s.add_column("Joint Risk", justify="right")
        s.add_column("Synergy Score", justify="right")
        for syn in synergies:
            c = (
                "bold red"
                if syn["synergy_score"] > 0.1
                else "red"
                if syn["synergy_score"] > 0.05
                else "yellow"
            )
            s.add_row(
                " + ".join(syn["drugs"]),
                str(syn["order"]),
                f"{syn['joint_risk']:.1%}",
                f"[{c}]{syn['synergy_score']:+.4f}[/{c}]",
            )
        console.print(s)

    top = max(loo, key=loo.get)
    conf = int((1 - epi / max(epi + ale, 1e-6)) * 100)
    console.print(
        Panel(
            f"[bold]Primary culprit:[/bold] [red]{top.upper()}[/red]\n"
            f"[bold]Confidence:[/bold] {conf}%  "
            f"({'Act on alert' if epi < 0.02 else 'Gather more data before acting'})\n\n"
            f"[dim]Patient data never leaves the hospital — only model weights travel.[/dim]",
            title="[bold]Clinical Summary[/bold]",
            border_style="cyan",
        )
    )

console.print("\n[bold green]Demo complete.[/bold green]")

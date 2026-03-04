# Federated Polypharmacy Safety (FPS)

> Privacy-preserving ADR detection for patients on 5+ medications — federated learning, causal inference, and drug-level attribution in one pipeline.

---

## Overview

Adverse Drug Reactions (ADRs) from polypharmacy are a leading cause of preventable hospital admissions. Pairwise drug interaction checkers miss the combinatorial explosions that occur when patients take 5–15 medications simultaneously. FPS addresses this with three complementary ideas:

1. **Federated Learning** — hospitals collaboratively train a shared model without ever sharing patient data
2. **Causal Inference** — CATE features (Conditional Average Treatment Effects) per drug, estimated via propensity score matching + T-Learner, augment the neural network input
3. **Drug Attribution** — once the model flags a high-risk patient, Shapley values and leave-one-out counterfactuals identify *which drug* in the regimen is the culprit and *which pairs* are synergistically dangerous

### Key Results (on synthetic data)

| Metric | Local Only | FL Final |
|---|---|---|
| AUROC | 0.8201 | **0.8480** |
| F1 Score | 0.7204 | **0.7407** |
| Privacy | ✗ | ✓ |
| Causal CATE | ✓ | ✓ |

The FL model converges monotonically over 20 rounds, rising from 0.646 (Round 1) to 0.848 (Round 20), while aleatoric uncertainty falls from 0.250 → 0.066 as the global model learns to separate signal from noise across all three hospital shards.

---

## Architecture

```
Hospital 0 ──┐
Hospital 1 ──┼──► Local Training (FedProx) ──► FedAvg Aggregation ──► Global Model
Hospital 2 ──┘         ▲                                                     │
                        └─────── Updated global weights ◄────────────────────┘
                                    (20 rounds)

Global Model ──► MC Dropout (T=50) ──► Risk Score + σ²_epistemic + σ²_aleatoric
                                                │
                                        High-risk patients
                                                │
                              ┌─────────────────┼──────────────────┐
                         LOO Δ risk        Shapley values      Synergy scores
                     (remove each drug)  (marginal contrib.)  (pairwise/triple)
```

**Non-IID data**: Each hospital shard has different demographics (age mean shifts by 4 years per hospital), different condition prevalence, and different ADR rates (39.8% / 48.5% / 48.2%). This is a realistic simulation of real-world federated data heterogeneity.

---

## File Structure

```
fps/
├── run.py              # Main pipeline — run this
├── data_gen.py         # Synthetic patient generator (3 hospital shards)
├── causal.py           # Propensity matching + T-Learner CATE estimation
├── model.py            # ADRNet — multi-task neural net with MC Dropout
├── fl_client.py        # FedAvg + FedProx simulation (no external FL framework)
├── attribution.py      # Shapley values, LOO counterfactuals, synergy detection
├── data_faers.py       # Real FAERS data loader (FDA adverse event reports)
├── pyproject.toml      # Dependencies
└── fps_results.txt     # Output from last run (auto-generated)
```

---

## Pipeline Steps

### Step 1 — Synthetic Patient Generation (`data_gen.py`)

Generates 1,200 patients across 3 hospitals (400 each). Each patient has:
- Age, sex, 1–4 chronic conditions (diabetes, hypertension, CKD, heart failure, AFib, osteoarthritis)
- 5–10 drugs from a set of 20 common chronic-disease medications
- Lab values: creatinine, HbA1c, potassium
- ADR label derived from a deterministic risk function encoding known pairwise interactions and high-risk triples

**High-risk triples encoded in the data:**

| Triple | Mechanism |
|---|---|
| warfarin + aspirin + clopidogrel | Triple anticoagulation → haemorrhage |
| lisinopril + spironolactone + aspirin | Hyperkalemia + AKI |
| digoxin + furosemide + hydrochlorothiazide | Hypokalemia → digoxin toxicity |
| metformin + furosemide + hydrochlorothiazide | Lactic acidosis risk |
| warfarin + azithromycin + omeprazole | INR spike |

### Step 2 — Causal Inference (`causal.py`)

For 5 index drugs (warfarin, lisinopril, spironolactone, aspirin, digoxin):

1. **Propensity Score Matching** — logistic regression estimates P(drug | clinical features), nearest-neighbour matching with caliper = 0.2 × σ(logit propensity) creates balanced treated/control cohorts
2. **T-Learner** — separate gradient-boosted regressors fit on treated vs control, CATE = μ₁(x) − μ₀(x)
3. CATE features appended to input (26 → 31 dims)

### Step 3 — Feature Normalisation

`StandardScaler` fitted on all-hospital combined data, applied per shard. Without this, clinical features (age ~70) dominate gradients by ~60× over drug interaction signals (~0.06), causing the FL model to collapse.

### Step 4 — Local Baseline

Per-hospital models trained independently for 30 epochs. This is the ceiling that federation should beat — hospitals can't share data so each model only sees 320 training patients.

### Step 5 — Federated Learning (`fl_client.py`)

Custom FedProx implementation — no external FL framework dependency.

**FedProx objective** (Li et al., 2020):

```
min_w  F_k(w)  +  (μ/2) · ||w − w_global||²
```

- `μ = 0.1` — mild proximal term stabilises non-IID training without over-constraining local updates
- 20 rounds, 5 local epochs per round, 3 clients
- FedAvg aggregation weighted by dataset size

### Step 6 — ADRNet + MC Dropout (`model.py`)

```
Input (31 dims) → Linear(128) → ReLU → Dropout(0.1)
               → Linear(64)  → ReLU → Dropout(0.1)
               → Linear(32)  → ReLU → Dropout(0.1)
               → ADR head: sigmoid → probability
               → Severity head: linear → severity score
```

**MC Dropout uncertainty decomposition** (Kendall & Gal, 2017):
- `σ²_epistemic` = variance across T=50 forward passes (model ignorance — reducible with more data)
- `σ²_aleatoric` = mean Bernoulli variance (irreducible data noise)

### Step 7 — Drug Attribution (`attribution.py`)

For the top-N highest-risk patients (threshold ≥ 60%):

**Leave-One-Out (LOO):**
```
Δᵢ = risk(full regimen) − risk(regimen without drug_i)
```
Positive Δ = drug contributes to risk. Negative Δ = drug is protective.

**Shapley Values:**
Exact computation for N ≤ 10 drugs (all 2^N subsets), Monte Carlo sampling for N > 10. Distributes risk credit fairly across all drugs using cooperative game theory.

**Synergy Detection:**
```
synergy(A, B) = risk({A,B}) − risk({A}) − risk({B}) + risk({})
```
Identifies non-additive combinations where drugs are collectively more dangerous than the sum of their individual effects.

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd fps

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install torch numpy pandas scikit-learn rich econml dowhy
```

**Requirements:** Python ≥ 3.11

---

## Usage

```bash
# Run the full pipeline
uv run run.py

# Or directly
python run.py
```

Results are printed to terminal and saved to `fps_results.txt`.

**Expected runtime:** ~3–5 minutes on CPU (dominated by causal inference in Step 2 and 20 FL rounds in Step 5).

---

## Using Real FAERS Data

The pipeline supports real FDA Adverse Event Reporting System data via `data_faers.py`. Download quarterly FAERS files from [FDA FAERS](https://fis.fda.gov/extensions/FPD-QDE-FAERS/):

```python
from data_faers import load_faers_multi_quarter

df = load_faers_multi_quarter([
    "faers_2023Q4/",
    "faers_2024Q1/",
    "faers_2024Q2/",
])
# Each quarter becomes one hospital shard — natural non-IID from temporal variation
```

The loader handles:
- Drug name normalisation (1,000+ brand/generic variants → canonical names)
- ADR term standardisation
- Missing lab value imputation
- Filtering to polypharmacy cases (5+ drugs)

---

## Attribution Example

```
Patient: H1_P0196
Drugs: spironolactone, aspirin, omeprazole, metformin, glipizide,
       hydrochlorothiazide, metoprolol, digoxin, furosemide, clopidogrel

Risk Score: 99%  CRITICAL
σ²_epistemic: 0.00002   σ²_aleatoric: 0.00561

Top synergy: DIGOXIN + FUROSEMIDE  (score: +0.350)
```

Digoxin + furosemide is one of the most clinically dangerous known combinations: furosemide causes potassium wasting, and hypokalemia potentiates digoxin toxicity. The model discovers this from statistical patterns alone, without being told the mechanism.

---

## Limitations

- **Synthetic data**: Drug interaction signals are engineered into `data_gen.py`. The model partially rediscovers what was built in. Real FAERS data provides stronger validation.
- **LOO saturation**: At 99–100% risk scores, removing any single drug barely shifts the prediction — the synergy scores are more informative than individual LOO deltas in these cases.
- **Epistemic uncertainty at saturation**: σ²_epistemic < 0.0001 on CRITICAL patients is unrealistically low. A model should express more uncertainty about extreme 10-drug combinations.
- **Synthetic federation**: All three "hospitals" run on one machine. Real deployment requires differential privacy (e.g. DP-SGD), secure aggregation, and communication compression.

---

## References

1. McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg)
2. Li et al. (2020) — *Federated Optimization in Heterogeneous Networks* (FedProx)
3. Kendall & Gal (2017) — *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* (MC Dropout decomposition)
4. Künzel et al. (2019) — *Metalearners for Estimating Heterogeneous Treatment Effects* (T-Learner)
5. Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions* (Shapley values)

---

## Scope

**In scope:** Chronic polypharmacy patients (5+ medications), privacy-preserving federated training, causal attribution, uncertainty estimation, drug-induced harm modelling.

**Out of scope:** Drug discovery, genomic/pharmacogenomic factors, automated prescribing decisions, real-time clinical deployment.
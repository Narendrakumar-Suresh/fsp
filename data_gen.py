"""
data_gen.py — Synthetic chronic-disease polypharmacy patient generator
Produces realistic-ish medication histories + ADR labels for 3 hospital shards.
"""

import numpy as np
import pandas as pd

# 20 common chronic-disease drugs with known interaction risk pairs
DRUGS = [
    "metformin", "lisinopril", "amlodipine", "atorvastatin", "aspirin",
    "warfarin",  "metoprolol", "furosemide", "spironolactone", "digoxin",
    "omeprazole","insulin",    "glipizide",  "losartan",       "hydrochlorothiazide",
    "clopidogrel","allopurinol","colchicine", "prednisolone",   "azithromycin",
]

# High-risk triples: (drug_a, drug_b, drug_c) → strongly increases ADR probability
HIGH_RISK_TRIPLES = [
    ("warfarin",      "aspirin",        "clopidogrel"),   # triple anticoag → bleed
    ("lisinopril",    "spironolactone", "aspirin"),       # hyperkalemia + AKI
    ("digoxin",       "furosemide",     "hydrochlorothiazide"),  # hypokalemia → toxicity
    ("metformin",     "furosemide",     "hydrochlorothiazide"),  # lactic acidosis risk
    ("warfarin",      "azithromycin",   "omeprazole"),    # INR spike
]

CONDITIONS = ["diabetes_t2", "hypertension", "ckd", "heart_failure", "afib", "osteoarthritis"]

rng = np.random.default_rng(42)


def _assign_drugs(n_drugs: int, condition_flags: np.ndarray,
                  local_rng: np.random.Generator) -> list[str]:
    """Pick drugs biased by conditions. Uses per-hospital RNG to avoid entanglement."""
    weights = np.ones(len(DRUGS))
    if condition_flags[0]:  # diabetes
        for d in ["metformin", "insulin", "glipizide"]:
            weights[DRUGS.index(d)] += 3
    if condition_flags[1]:  # hypertension
        for d in ["lisinopril", "amlodipine", "losartan", "hydrochlorothiazide", "metoprolol"]:
            weights[DRUGS.index(d)] += 3
    if condition_flags[2]:  # ckd
        for d in ["furosemide", "spironolactone", "allopurinol"]:
            weights[DRUGS.index(d)] += 2
    if condition_flags[3]:  # heart failure
        for d in ["furosemide", "digoxin", "spironolactone", "metoprolol"]:
            weights[DRUGS.index(d)] += 3
    if condition_flags[4]:  # afib
        for d in ["warfarin", "clopidogrel", "digoxin", "metoprolol"]:
            weights[DRUGS.index(d)] += 3
    weights /= weights.sum()
    chosen = local_rng.choice(DRUGS, size=n_drugs, replace=False, p=weights)
    return list(chosen)


def _compute_adr_probability(drugs: list[str], age: int, creatinine: float,
                              n_conditions: int) -> float:
    """Deterministic risk score from drug combinations + patient factors."""
    base = 0.05 + (age - 50) * 0.005 + (creatinine - 1.0) * 0.05 + n_conditions * 0.03

    # Pairwise known interactions
    drug_set = set(drugs)
    if {"warfarin", "aspirin"}.issubset(drug_set):        base += 0.15
    if {"warfarin", "clopidogrel"}.issubset(drug_set):    base += 0.18
    if {"lisinopril", "spironolactone"}.issubset(drug_set): base += 0.10
    if {"digoxin", "furosemide"}.issubset(drug_set):      base += 0.12
    if {"metformin", "furosemide"}.issubset(drug_set):    base += 0.08

    # High-order triple interactions — this is the point of the whole project
    for (a, b, c) in HIGH_RISK_TRIPLES:
        if {a, b, c}.issubset(drug_set):
            base += 0.25  # substantial bump for triple

    return float(np.clip(base, 0.0, 1.0))


def generate_hospital(n_patients: int = 400, hospital_id: int = 0,
                       seed_offset: int = 0) -> pd.DataFrame:
    """Generate one hospital's patient shard."""
    local_rng = np.random.default_rng(42 + hospital_id * 100 + seed_offset)

    # Different hospitals have different patient demographics (non-IID)
    age_mean = 62 + hospital_id * 4   # hospital 0: younger, hospital 2: older
    condition_bias = 0.3 + hospital_id * 0.1

    records = []
    for i in range(n_patients):
        age = int(np.clip(local_rng.normal(age_mean, 10), 40, 90))
        sex = local_rng.choice(["M", "F"])

        # Conditions
        n_conds = local_rng.integers(1, 5)
        condition_flags = (local_rng.random(len(CONDITIONS)) < condition_bias).astype(int)
        condition_flags[:n_conds] = 1
        conditions = [c for c, f in zip(CONDITIONS, condition_flags) if f]

        # Labs
        creatinine = float(np.clip(local_rng.normal(1.2 + condition_flags[2] * 0.8, 0.3), 0.6, 4.5))
        hba1c      = float(np.clip(local_rng.normal(6.5 + condition_flags[0] * 1.5, 0.8), 4.5, 12.0))
        potassium  = float(np.clip(local_rng.normal(4.2, 0.5), 3.0, 6.5))

        # Drug regimen: 5-10 drugs (polypharmacy by definition)
        n_drugs = local_rng.integers(5, 11)
        drugs = _assign_drugs(n_drugs, condition_flags, local_rng)
        doses = [local_rng.choice([5, 10, 25, 50, 100, 250, 500, 1000]) for _ in drugs]

        # ADR probability + label
        p_adr = _compute_adr_probability(drugs, age, creatinine, len(conditions))
        adr_occurred = int(local_rng.random() < p_adr)
        adr_types = []
        if adr_occurred:
            candidates = ["bleeding", "aki", "hyperkalemia", "lactic_acidosis",
                          "hypoglycemia", "bradycardia", "liver_toxicity"]
            adr_types = list(local_rng.choice(candidates,
                                               size=local_rng.integers(1, 3),
                                               replace=False))

        records.append({
            "patient_id":   f"H{hospital_id}_P{i:04d}",
            "hospital_id":  hospital_id,
            "age":          age,
            "sex":          sex,
            "conditions":   "|".join(conditions),
            "n_conditions": len(conditions),
            "drugs":        "|".join(drugs),
            "n_drugs":      len(drugs),
            "doses":        "|".join(map(str, doses)),
            "creatinine":   round(creatinine, 2),
            "hba1c":        round(hba1c, 2),
            "potassium":    round(potassium, 2),
            "adr_occurred": adr_occurred,
            "adr_types":    "|".join(adr_types),
            "true_risk":    round(p_adr, 4),
        })

    return pd.DataFrame(records)


def generate_all(n_per_hospital: int = 400) -> pd.DataFrame:
    """Generate data for 3 hospitals, return combined DataFrame."""
    shards = [generate_hospital(n_per_hospital, hospital_id=i) for i in range(3)]
    df = pd.concat(shards, ignore_index=True)
    return df


def extract_features(df: pd.DataFrame):
    """
    Convert raw patient DataFrame into a numeric feature matrix X and label y.
    Features: age, n_conditions, n_drugs, creatinine, hba1c, potassium,
              one-hot for each drug (20 dims) = 26 total features
    """
    import numpy as np

    X_basic = df[["age", "n_conditions", "n_drugs", "creatinine", "hba1c", "potassium"]].values.astype(np.float32)

    # Drug one-hot
    drug_ohe = np.zeros((len(df), len(DRUGS)), dtype=np.float32)
    for i, drug_str in enumerate(df["drugs"]):
        for drug in drug_str.split("|"):
            if drug in DRUGS:
                drug_ohe[i, DRUGS.index(drug)] = 1.0

    X = np.hstack([X_basic, drug_ohe])  # (N, 26)
    y = df["adr_occurred"].values.astype(np.float32)
    return X, y


if __name__ == "__main__":
    df = generate_all()
    print(f"Generated {len(df)} patients across 3 hospitals")
    print(f"ADR rate: {df['adr_occurred'].mean():.1%}")
    print(df.head(3).to_string())
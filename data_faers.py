"""
data_faers.py — FDA Adverse Event Reporting System (FAERS) data loader
FAERS is publicly available at: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

Directory structure expected (after unzipping a quarterly download):
  faers_data/
    ascii/
      DEMO24Q1.txt   — patient demographics
      DRUG24Q1.txt   — drug records
      REAC24Q1.txt   — adverse reactions
      OUTC24Q1.txt   — outcomes
      THER24Q1.txt   — therapy dates (optional)

Each row in DRUG maps to a case via primaryid.
One case can have many drugs (polypharmacy) and many reactions.

This module:
  1. Loads and joins the four core tables
  2. Normalizes drug names (FDA preferred names)
  3. Filters to chronic polypharmacy cases (5+ drugs)
  4. Outputs a DataFrame compatible with the rest of the FPS pipeline
  5. Includes a fallback synthetic-FAERS generator for development/testing
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path


# ── Known drug name normalization map (extend as needed) ────────────────────
# Maps messy brand/generic names → canonical lowercase names used in DRUGS list
DRUG_NORM = {
    # warfarin family
    "warfarin sodium": "warfarin", "coumadin": "warfarin", "jantoven": "warfarin",
    # aspirin
    "acetylsalicylic acid": "aspirin", "asa": "aspirin", "ecotrin": "aspirin",
    "bayer aspirin": "aspirin",
    # lisinopril
    "lisinopril": "lisinopril", "zestril": "lisinopril", "prinivil": "lisinopril",
    # metformin
    "metformin hydrochloride": "metformin", "glucophage": "metformin",
    # digoxin
    "digoxin": "digoxin", "lanoxin": "digoxin",
    # furosemide
    "furosemide": "furosemide", "lasix": "furosemide",
    # spironolactone
    "spironolactone": "spironolactone", "aldactone": "spironolactone",
    # metoprolol
    "metoprolol succinate": "metoprolol", "metoprolol tartrate": "metoprolol",
    "lopressor": "metoprolol", "toprol": "metoprolol",
    # amlodipine
    "amlodipine besylate": "amlodipine", "norvasc": "amlodipine",
    # atorvastatin
    "atorvastatin calcium": "atorvastatin", "lipitor": "atorvastatin",
    # clopidogrel
    "clopidogrel bisulfate": "clopidogrel", "plavix": "clopidogrel",
    # losartan
    "losartan potassium": "losartan", "cozaar": "losartan",
    # hydrochlorothiazide
    "hydrochlorothiazide": "hydrochlorothiazide", "hctz": "hydrochlorothiazide",
    "microzide": "hydrochlorothiazide",
    # omeprazole
    "omeprazole": "omeprazole", "prilosec": "omeprazole",
    # insulin variants
    "insulin glargine": "insulin", "insulin aspart": "insulin",
    "insulin lispro": "insulin", "lantus": "insulin", "humalog": "insulin",
    "novolog": "insulin", "levemir": "insulin",
    # glipizide
    "glipizide": "glipizide", "glucotrol": "glipizide",
    # allopurinol
    "allopurinol": "allopurinol", "zyloprim": "allopurinol",
    # colchicine
    "colchicine": "colchicine", "colcrys": "colchicine",
    # prednisolone
    "prednisolone": "prednisolone", "prednisone": "prednisolone",
    "methylprednisolone": "prednisolone",
    # azithromycin
    "azithromycin": "azithromycin", "zithromax": "azithromycin", "z-pak": "azithromycin",
}

# ADR terms that map to our internal label categories
ADR_MAP = {
    "haemorrhage": "bleeding", "hemorrhage": "bleeding", "bleeding": "bleeding",
    "gastrointestinal haemorrhage": "bleeding", "epistaxis": "bleeding",
    "acute kidney injury": "aki", "renal failure": "aki",
    "renal impairment": "aki", "nephrotoxicity": "aki",
    "hyperkalaemia": "hyperkalemia", "hyperkalemia": "hyperkalemia",
    "lactic acidosis": "lactic_acidosis",
    "hypoglycaemia": "hypoglycemia", "hypoglycemia": "hypoglycemia",
    "bradycardia": "bradycardia", "heart rate decreased": "bradycardia",
    "hepatotoxicity": "liver_toxicity", "liver injury": "liver_toxicity",
    "hepatic failure": "liver_toxicity", "jaundice": "liver_toxicity",
}

KNOWN_DRUGS = [
    "metformin", "lisinopril", "amlodipine", "atorvastatin", "aspirin",
    "warfarin", "metoprolol", "furosemide", "spironolactone", "digoxin",
    "omeprazole", "insulin", "glipizide", "losartan", "hydrochlorothiazide",
    "clopidogrel", "allopurinol", "colchicine", "prednisolone", "azithromycin",
]


# ── Drug name normalizer ─────────────────────────────────────────────────────

def normalize_drug(raw_name: str) -> str | None:
    """
    Normalize a raw FAERS drug name to canonical form.
    Returns None if name can't be mapped to a known drug.
    """
    if not isinstance(raw_name, str):
        return None
    cleaned = raw_name.strip().lower()
    cleaned = re.sub(r"\s+\d+.*$", "", cleaned)   # strip dosage suffix
    cleaned = re.sub(r"\(.*?\)", "", cleaned).strip()  # strip parenthetical

    # Direct lookup
    if cleaned in DRUG_NORM:
        return DRUG_NORM[cleaned]

    # Partial match: check if any canonical name appears in the string
    for canonical in KNOWN_DRUGS:
        if canonical in cleaned:
            return canonical

    return None


# ── FAERS file loader ────────────────────────────────────────────────────────

def _find_faers_files(data_dir: str) -> dict:
    """Locate FAERS ASCII files (handles various naming conventions)."""
    data_dir = Path(data_dir)
    result = {}

    patterns = {
        "demo": ["DEMO*.txt", "demo*.txt"],
        "drug": ["DRUG*.txt", "drug*.txt"],
        "reac": ["REAC*.txt", "reac*.txt"],
        "outc": ["OUTC*.txt", "outc*.txt"],
    }

    for key, pats in patterns.items():
        for pat in pats:
            found = list(data_dir.rglob(pat))
            if found:
                result[key] = found[0]
                break

    return result


def load_faers_quarter(data_dir: str) -> pd.DataFrame:
    """
    Load one quarter of FAERS data and return a patient-level DataFrame
    compatible with the FPS pipeline.

    Parameters
    ----------
    data_dir : str
        Path to unzipped FAERS quarter folder (containing ascii/ subfolder)

    Returns
    -------
    pd.DataFrame with columns:
        patient_id, age, sex, n_drugs, drugs ("|"-joined), adr_occurred,
        adr_types ("|"-joined), n_conditions (0, not available in FAERS)
    """
    files = _find_faers_files(data_dir)
    missing = [k for k in ["demo", "drug", "reac"] if k not in files]
    if missing:
        raise FileNotFoundError(
            f"Missing FAERS files for: {missing}\n"
            f"Found: {list(files.keys())} in {data_dir}\n"
            f"Download from: https://fis.fda.gov/extensions/FPD-QDE-FAERS/"
        )

    # ── Load tables ──────────────────────────────────────────────────────────
    sep = "$"  # FAERS uses $ as delimiter

    demo = pd.read_csv(files["demo"], sep=sep, dtype=str, on_bad_lines="skip",
                       usecols=lambda c: c.upper() in
                       ["PRIMARYID", "CASEID", "AGE", "AGE_COD", "SEX"])
    demo.columns = demo.columns.str.upper()

    drug = pd.read_csv(files["drug"], sep=sep, dtype=str, on_bad_lines="skip",
                       usecols=lambda c: c.upper() in
                       ["PRIMARYID", "DRUGNAME", "PROD_AI", "ROLE_COD"])
    drug.columns = drug.columns.str.upper()

    reac = pd.read_csv(files["reac"], sep=sep, dtype=str, on_bad_lines="skip",
                       usecols=lambda c: c.upper() in ["PRIMARYID", "PT"])
    reac.columns = reac.columns.str.upper()

    # ── Normalise ages to years ──────────────────────────────────────────────
    def age_to_years(row):
        try:
            val = float(row.get("AGE", ""))
            cod = str(row.get("AGE_COD", "YR")).upper()
            if cod in ("YR", "YEAR"):    return val
            if cod in ("MON", "MONTH"): return val / 12
            if cod in ("DY", "DAY"):    return val / 365
            if cod in ("DEC",):         return val * 10
            return val
        except (ValueError, TypeError):
            return np.nan
    demo["age_years"] = demo.apply(age_to_years, axis=1)

    # ── Normalise drug names ─────────────────────────────────────────────────
    # Prefer PROD_AI (active ingredient) over DRUGNAME (brand name)
    drug["canon"] = drug["PROD_AI"].apply(normalize_drug)
    mask = drug["canon"].isna()
    drug.loc[mask, "canon"] = drug.loc[mask, "DRUGNAME"].apply(normalize_drug)

    # Keep only drugs in our known list and primary/secondary roles
    drug_clean = drug[
        drug["canon"].notna() &
        drug.get("ROLE_COD", pd.Series(dtype=str)).isin(["PS", "SS", "C", "I", ""])
    ].copy()

    # Group drugs per case
    drug_grp = (
        drug_clean.groupby("PRIMARYID")["canon"]
        .apply(lambda x: list(set(x)))
        .reset_index()
        .rename(columns={"canon": "drug_list"})
    )

    # ── Map reactions to ADR types ───────────────────────────────────────────
    def map_adr(pt_term: str) -> str | None:
        if not isinstance(pt_term, str):
            return None
        low = pt_term.lower()
        for key, val in ADR_MAP.items():
            if key in low:
                return val
        return None

    reac["adr_type"] = reac["PT"].apply(map_adr)
    reac_known = reac[reac["adr_type"].notna()]
    reac_grp = (
        reac_known.groupby("PRIMARYID")["adr_type"]
        .apply(lambda x: list(set(x)))
        .reset_index()
        .rename(columns={"adr_type": "adr_types"})
    )

    # ── Join everything ──────────────────────────────────────────────────────
    df = (
        demo[["PRIMARYID", "age_years", "SEX"]]
        .merge(drug_grp, on="PRIMARYID", how="inner")
        .merge(reac_grp, on="PRIMARYID", how="left")
    )

    df["adr_types"] = df["adr_types"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df["adr_occurred"] = df["adr_types"].apply(lambda x: int(len(x) > 0))
    df["adr_types_str"] = df["adr_types"].apply(lambda x: "|".join(x))

    # ── Filter to polypharmacy cases (5+ known drugs) ────────────────────────
    df["n_drugs"] = df["drug_list"].apply(len)
    df = df[df["n_drugs"] >= 5].copy()

    # ── Clip age to reasonable range ─────────────────────────────────────────
    df["age"] = df["age_years"].clip(18, 100).fillna(65).astype(int)
    df["sex"] = df["SEX"].map({"M": "M", "F": "F"}).fillna("M")

    # ── Build output DataFrame ───────────────────────────────────────────────
    out = pd.DataFrame({
        "patient_id":   "FAERS_" + df["PRIMARYID"].astype(str),
        "hospital_id":  0,  # single shard from one quarter
        "age":          df["age"].values,
        "sex":          df["sex"].values,
        "conditions":   "",
        "n_conditions": 0,
        "drugs":        df["drug_list"].apply("|".join).values,
        "n_drugs":      df["n_drugs"].values,
        "doses":        "",
        "creatinine":   1.2,
        "hba1c":        6.5,
        "potassium":    4.2,
        "adr_occurred": df["adr_occurred"].values,
        "adr_types":    df["adr_types_str"].values,
        "true_risk":    df["adr_occurred"].values.astype(float),
    })
    return out.reset_index(drop=True)


def load_faers_multi_quarter(data_dirs: list[str],
                              hospital_labels: list[int] | None = None) -> pd.DataFrame:
    """
    Load multiple FAERS quarters and treat each as a separate hospital shard.
    This naturally creates non-IID splits (different time periods = different
    patient populations and drug availability).

    Parameters
    ----------
    data_dirs : list of paths to quarterly FAERS folders
    hospital_labels : optional list of int IDs (defaults to 0,1,2,...)
    """
    dfs = []
    for i, d in enumerate(data_dirs):
        try:
            shard = load_faers_quarter(d)
            hid = hospital_labels[i] if hospital_labels else i
            shard["hospital_id"] = hid
            dfs.append(shard)
            print(f"  ✓ Loaded FAERS quarter {i}: {len(shard)} polypharmacy cases "
                  f"(ADR rate {shard['adr_occurred'].mean():.1%})")
        except FileNotFoundError as e:
            print(f"  ✗ Skipped {d}: {e}")

    if not dfs:
        raise RuntimeError("No FAERS data loaded. Check your data paths.")
    return pd.concat(dfs, ignore_index=True)


# ── Synthetic FAERS-compatible generator (for development / CI) ─────────────

def generate_faers_synthetic(n_cases: int = 1200,
                              n_quarters: int = 3,
                              seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic DataFrame that mirrors real FAERS structure.
    Used for development when real FAERS data isn't available.
    The drug names match FAERS-style messy inputs so the normalizer runs.
    """
    from data_gen import generate_all
    # Reuse existing generator but tag it as "FAERS-style"
    df = generate_all(n_per_hospital=n_cases // n_quarters)
    df["patient_id"] = "FAERS_SYN_" + df["patient_id"]
    # Simulate messy brand names in the drug column (as FAERS would have)
    brand_map = {
        "warfarin": "warfarin sodium",
        "aspirin": "acetylsalicylic acid",
        "lisinopril": "lisinopril",
        "metformin": "metformin hydrochloride",
        "digoxin": "lanoxin",
    }
    def messify(drug_str):
        drugs = drug_str.split("|")
        return "|".join(brand_map.get(d, d) for d in drugs)
    df["drugs_raw"] = df["drugs"].apply(messify)
    return df


# ── Feature extraction for FAERS data ────────────────────────────────────────

def extract_features_faers(df: pd.DataFrame):
    """
    Feature extraction for FAERS data.
    Similar to data_gen.extract_features but handles missing lab values
    and maps drug names through the normalizer.
    """
    import numpy as np
    from data_gen import DRUGS

    # Basic clinical features (FAERS has age/sex, no lab values — impute)
    sex_enc = (df["sex"] == "F").astype(float)
    X_basic = np.column_stack([
        df["age"].values.astype(np.float32),
        df["n_conditions"].values.astype(np.float32),
        df["n_drugs"].values.astype(np.float32),
        df.get("creatinine", pd.Series([1.2] * len(df))).values.astype(np.float32),
        df.get("hba1c", pd.Series([6.5] * len(df))).values.astype(np.float32),
        df.get("potassium", pd.Series([4.2] * len(df))).values.astype(np.float32),
    ])

    # Drug one-hot: normalize drug names before lookup
    drug_ohe = np.zeros((len(df), len(DRUGS)), dtype=np.float32)
    for i, drug_str in enumerate(df["drugs"]):
        for raw in drug_str.split("|"):
            canon = normalize_drug(raw) or raw.strip().lower()
            if canon in DRUGS:
                drug_ohe[i, DRUGS.index(canon)] = 1.0

    X = np.hstack([X_basic, drug_ohe])
    y = df["adr_occurred"].values.astype(np.float32)
    return X, y
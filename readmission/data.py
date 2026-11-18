import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
EXCLUDED_DISCHARGES = {11, 13, 14, 19, 20, 21}
NUMERIC_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
DRUG_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
CATEGORICAL_COLUMNS = [
    "race", "gender", "age", "weight", "admission_type_id",
    "discharge_disposition_id", "admission_source_id", "payer_code",
    "medical_specialty", "diag_1", "diag_2", "diag_3", "max_glu_serum",
    "A1Cresult", *DRUG_COLUMNS, "change", "diabetesMed",
]
RAW_FEATURES = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


def download_data(directory="data/raw"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "diabetic_data.csv"
    if target.exists():
        return target
    print("Downloading the public UCI dataset...", flush=True)
    with urlopen(DATA_URL, timeout=90) as response:
        archive_bytes = response.read()
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        for name in ("diabetic_data.csv", "IDS_mapping.csv"):
            (directory / name).write_bytes(archive.read(name))
    (directory / "download_metadata.json").write_text(json.dumps({
        "source": DATA_URL,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "license": "CC BY 4.0",
        "dataset_doi": "10.24432/C5230J",
    }, indent=2) + "\n")
    return target


def load_data(path):
    df = pd.read_csv(path, keep_default_na=False, na_values=["?", ""], low_memory=False)
    required = set(RAW_FEATURES + ["encounter_id", "patient_nbr", "readmitted"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[["encounter_id", "patient_nbr", "readmitted"]].isna().any().any():
        raise ValueError("Identifiers and target must not contain missing values.")
    if df.encounter_id.duplicated().any():
        raise ValueError("Duplicate encounter IDs found; inspect the input data.")
    if not set(df.readmitted.unique()) <= {"<30", ">30", "NO"}:
        raise ValueError("Unexpected readmitted label; expected <30, >30, or NO.")
    return df


def clean_cohort(df):
    discharge = pd.to_numeric(df.discharge_disposition_id, errors="raise")
    return df.loc[~discharge.isin(EXCLUDED_DISCHARGES)].copy().reset_index(drop=True)


def diagnosis_group(value):
    if pd.isna(value) or str(value).strip() in {"", "?"}:
        return "Missing"
    try:
        code = float(value)
    except (ValueError, TypeError):
        return "Other"
    if 250 <= code < 251:
        return "Diabetes"
    for low, high, special, label in [
        (390, 460, 785, "Circulatory"),
        (460, 520, 786, "Respiratory"),
        (520, 580, 787, "Digestive"),
        (580, 630, 788, "Genitourinary"),
        (800, 1000, -1, "Injury"),
        (710, 740, -1, "Musculoskeletal"),
        (140, 240, -1, "Neoplasms"),
    ]:
        if low <= code < high or (special >= 0 and special <= code < special + 1):
            return label
    return "Other"


def make_features(df):
    X = df[RAW_FEATURES].copy()
    X = X.replace({"?": np.nan, "": np.nan})
    X = X.drop(columns=["weight", "examide", "citoglipton"])
    for col in NUMERIC_COLUMNS:
        X[col] = pd.to_numeric(X[col], errors="raise")
    for col in ("diag_1", "diag_2", "diag_3"):
        X[col] = X[col].map(diagnosis_group)
    X["total_prior_visits"] = X[
        ["number_outpatient", "number_emergency", "number_inpatient"]
    ].sum(axis=1, min_count=3)
    drugs = [c for c in DRUG_COLUMNS if c in X]
    X["num_medication_changes"] = X[drugs].isin(["Up", "Down"]).sum(axis=1)
    numeric = NUMERIC_COLUMNS + ["total_prior_visits", "num_medication_changes"]
    for col in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        X[col] = pd.to_numeric(X[col], errors="raise").astype("Int64")
    for col in X.columns.difference(numeric):
        X[col] = X[col].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return X


def make_splits(X, y, groups, seed=42, folds=5):
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(outer.split(X, y, groups))
    if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
        raise AssertionError("Patient leakage in holdout split")
    inner = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
    cv = list(inner.split(X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]))
    train_groups = groups.iloc[train_idx]
    for fit_idx, val_idx in cv:
        if set(train_groups.iloc[fit_idx]) & set(train_groups.iloc[val_idx]):
            raise AssertionError("Patient leakage in cross-validation")
        if y.iloc[train_idx].iloc[val_idx].nunique() != 2:
            raise ValueError("A fold lacks a class; use more patients or fewer folds.")
    return train_idx, test_idx, cv

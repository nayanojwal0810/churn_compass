"""Load, validate, and split the Telco customer churn dataset."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
REQUIRED_COLUMNS = {
    ID_COLUMN,
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    TARGET_COLUMN,
}


@dataclass(frozen=True)
class DatasetBundle:
    """Validated dataset and a reproducible development/test split."""

    raw: pd.DataFrame
    cleaned: pd.DataFrame
    quality: pd.DataFrame
    X_development: pd.DataFrame
    X_test: pd.DataFrame
    y_development: pd.Series
    y_test: pd.Series
    development_index: pd.Index
    test_index: pd.Index
    dataset_sha256: str


def _quality_row(check: str, value: Any, status: str, details: str = "") -> dict[str, Any]:
    return {"check": check, "value": value, "status": status, "details": details}


def download_dataset(data_path: str | Path) -> Path:
    """Download the canonical sample only when the requested file is absent.

    The file's SHA-256 is verified before it is written to disk. Existing files are
    never silently overwritten; a checksum mismatch is reported by validation.
    """
    path = Path(data_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(DATASET_URL, headers={"User-Agent": "ChurnCompass/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = response.read()
    except Exception as exc:  # pragma: no cover - network-dependent error path
        raise RuntimeError(
            f"Could not download the dataset from {DATASET_URL}. "
            "Check the internet connection or pass --data-path to a local CSV."
        ) from exc

    actual_hash = sha256(payload).hexdigest()
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(
            "Downloaded dataset checksum does not match the expected canonical file. "
            f"Expected {EXPECTED_SHA256}; got {actual_hash}. The file was not saved."
        )
    path.write_bytes(payload)
    return path


def ensure_dataset_available(data_path: str | Path, allow_download: bool = False) -> Path:
    path = Path(data_path)
    if path.exists():
        return path
    if allow_download:
        return download_dataset(path)
    raise FileNotFoundError(
        f"Dataset not found: {path}. Place the Telco CSV at this path or pass --data-path."
    )


def load_and_validate_data(
    data_path: str | Path,
    test_size: float = 0.20,
    random_state: int = 20261014,
    verify_canonical_checksum: bool = False,
) -> DatasetBundle:
    """Validate the canonical schema and create a stratified holdout.

    The 11 whitespace-only TotalCharges values in the canonical sample occur only
    for zero-tenure customers and are normalized to 0.00. Missing cumulative charges
    for positive-tenure customers are a hard error, not an imputation opportunity.
    """
    data_path = ensure_dataset_available(data_path)
    file_hash = sha256(data_path.read_bytes()).hexdigest()
    if verify_canonical_checksum and file_hash != EXPECTED_SHA256:
        raise ValueError(
            "Dataset checksum mismatch. Use the canonical source file or disable the "
            "canonical-source check only when deliberately using a documented alternative."
        )

    raw = pd.read_csv(data_path, dtype={"TotalCharges": "string"})
    missing_columns = sorted(REQUIRED_COLUMNS - set(raw.columns))
    unexpected_columns = sorted(set(raw.columns) - REQUIRED_COLUMNS)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if unexpected_columns:
        raise ValueError(f"Dataset contains unexpected columns: {unexpected_columns}")
    if raw.empty:
        raise ValueError("Dataset is empty.")
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be greater than 0 and less than 0.5.")

    df = raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    categorical_columns = [
        column for column in df.columns
        if column not in {"TotalCharges", "tenure", "SeniorCitizen", "MonthlyCharges"}
    ]
    for column in categorical_columns:
        cleaned_values = df[column].astype("string").str.strip()
        df[column] = cleaned_values.mask(cleaned_values.eq(""), pd.NA).astype(object)

    blank_total = int(df["TotalCharges"].isna().sum())
    zero_tenure = df["tenure"].eq(0)
    blank_zero_tenure = int((df["TotalCharges"].isna() & zero_tenure).sum())
    blank_positive_tenure = blank_total - blank_zero_tenure
    if blank_positive_tenure:
        raise ValueError(
            "TotalCharges is missing for customer(s) with positive tenure; the pipeline "
            "will not silently invent cumulative charges."
        )
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    if df[TARGET_COLUMN].isna().any() or not set(df[TARGET_COLUMN].unique()).issubset({"Yes", "No"}):
        raise ValueError("Churn must be complete and contain only 'Yes' and 'No'.")
    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Churn target must contain both classes for model evaluation.")
    if df[ID_COLUMN].isna().any() or df[ID_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("Customer identifiers contain missing or blank values.")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Customer identifiers are not unique.")

    numeric_columns = ["tenure", "SeniorCitizen", "MonthlyCharges", "TotalCharges"]
    if df[numeric_columns].isna().any().any():
        invalid = df[numeric_columns].columns[df[numeric_columns].isna().any()].tolist()
        raise ValueError(f"Required numeric fields contain missing or invalid values: {invalid}")
    if not set(df["SeniorCitizen"].unique()).issubset({0, 1}):
        raise ValueError("SeniorCitizen must contain only 0 and 1.")
    if not set(df["tenure"].unique()).issubset(set(range(73))):
        raise ValueError("Tenure must be a whole number between 0 and 72 months.")
    if (df[["MonthlyCharges", "TotalCharges"]] < 0).any().any():
        raise ValueError("Negative charges found; inspect the source before modelling.")
    if (df["TotalCharges"].eq(0) & df["tenure"].gt(0)).sum() > 0:
        # Zero total charges with positive tenure is suspicious, but not auto-corrected.
        zero_total_positive_tenure = int((df["TotalCharges"].eq(0) & df["tenure"].gt(0)).sum())
    else:
        zero_total_positive_tenure = 0

    if not 1000 <= len(df):
        raise ValueError(f"Dataset is unexpectedly small: {len(df)} rows.")

    duplicate_rows = int(raw.duplicated().sum())
    raw_missing_total = int(raw.isna().sum().sum())
    categorical_missing_total = int(df[categorical_columns].isna().sum().sum())
    quality_rows = [
        _quality_row("row_count", len(df), "PASS"),
        _quality_row("column_count", len(raw.columns), "PASS"),
        _quality_row("unexpected_columns", len(unexpected_columns), "PASS", "None"),
        _quality_row("raw_missing_cells_pandas", raw_missing_total, "INFO", "Pandas-recognized nulls in the raw CSV before normalization."),
        _quality_row("blank_categorical_cells_after_trim", categorical_missing_total, "REVIEW" if categorical_missing_total else "PASS", "Whitespace-only category values are normalized to missing and handled by the fitted imputer."),
        _quality_row("duplicate_rows", duplicate_rows, "PASS" if duplicate_rows == 0 else "FAIL"),
        _quality_row("duplicate_customer_ids", int(df[ID_COLUMN].duplicated().sum()), "PASS"),
        _quality_row("unique_customer_ids", int(df[ID_COLUMN].nunique()), "PASS"),
        _quality_row("churn_yes_count", int(df[TARGET_COLUMN].eq("Yes").sum()), "INFO"),
        _quality_row("churn_prevalence", float(df[TARGET_COLUMN].eq("Yes").mean()), "INFO", "Reference prevalence for interpreting average precision."),
        _quality_row("blank_total_charges_after_trim", blank_total, "REVIEW" if blank_total else "PASS", "Blank values occur only at zero tenure and are set to 0.00."),
        _quality_row("blank_total_charges_positive_tenure", blank_positive_tenure, "PASS"),
        _quality_row("zero_total_charges_positive_tenure", zero_total_positive_tenure, "REVIEW" if zero_total_positive_tenure else "PASS", "Flagged for review; values are retained, not imputed."),
        _quality_row("invalid_senior_citizen_values", int((~df["SeniorCitizen"].isin([0, 1])).sum()), "PASS"),
        _quality_row("invalid_tenure_values", int((~df["tenure"].isin(range(73))).sum()), "PASS"),
        _quality_row("negative_monthly_charges", int(df["MonthlyCharges"].lt(0).sum()), "PASS"),
        _quality_row("negative_total_charges", int(df["TotalCharges"].lt(0).sum()), "PASS"),
        _quality_row("model_features_exclude_customer_id", True, "PASS", "customerID is retained for output joins but excluded from model features."),
        _quality_row("target_excluded_from_features", True, "PASS", "Churn is separated before preprocessing and model fitting."),
        _quality_row("dataset_sha256", file_hash, "INFO", "SHA-256 of the source CSV used for this run."),
    ]
    quality = pd.DataFrame(quality_rows)
    if (quality["status"] == "FAIL").any():
        raise ValueError("Dataset validation failed; inspect the validation output.")

    feature_columns = [column for column in df.columns if column not in {TARGET_COLUMN, ID_COLUMN}]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].eq("Yes").astype("int8")
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return DatasetBundle(
        raw=raw,
        cleaned=df,
        quality=quality,
        X_development=X_development,
        X_test=X_test,
        y_development=y_development,
        y_test=y_test,
        development_index=X_development.index,
        test_index=X_test.index,
        dataset_sha256=file_hash,
    )

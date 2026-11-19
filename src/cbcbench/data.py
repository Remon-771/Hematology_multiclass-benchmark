"""Data loading, identifier / missingness handling and the leakage-safe train/test split (notebook §1-§3)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (ABS_DIFF_CANDIDATES, DATA_PATH, DL_VAL_SIZE, HIGH_MISSING_THRESH, ID_NAMES, PCT_DIFF_CANDIDATES,
                     RENAME, SEED, TARGET_COL, TEST_SIZE)


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Read the CSV and standardise column names only (no value is changed)."""
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Expected data/hematology_cbc_dataset.csv")
    df_raw = pd.read_csv(path)
    return df_raw.rename(columns={k: v for k, v in RENAME.items() if k in df_raw.columns})


def abs_diff_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in ABS_DIFF_CANDIDATES if c in df.columns]


def pct_diff_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in PCT_DIFF_CANDIDATES if c in df.columns]


def identifier_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c != TARGET_COL and (c.strip().lower() in ID_NAMES or c.strip().lower().endswith("_id"))]


def missing_percent(df: pd.DataFrame):
    return (df.drop(columns=[TARGET_COL]).isna().mean() * 100).sort_values(ascending=False).round(2)


def high_missing_columns(df: pd.DataFrame, thresh: float = HIGH_MISSING_THRESH) -> list[str]:
    m = missing_percent(df)
    return m[m > thresh].index.tolist()


@dataclass
class Split:
    """Everything produced by the stratified split. The test portion is touched only by 09/10."""

    feature_names: list[str]
    class_names: list[str]
    label_encoder: LabelEncoder
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: object
    y_test: object
    X_tr: pd.DataFrame            # deep-learning fit-train sub-split (training data only)
    X_val: pd.DataFrame           # deep-learning early-stopping validation (training data only)
    y_tr: object
    y_val: object
    id_cols: list[str] = field(default_factory=list)
    dropped_missing: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def make_split(df_raw: pd.DataFrame, seed: int = SEED) -> Split:
    id_cols = identifier_columns(df_raw)
    cols_to_drop_missing = high_missing_columns(df_raw)
    drop_cols = id_cols + cols_to_drop_missing
    df = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns]).copy()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = X.columns.tolist()

    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=seed)
    # A further split of the training portion only, used for early-stopping the deep models.
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=DL_VAL_SIZE, stratify=y_train, random_state=seed)
    return Split(feature_names, class_names, label_encoder, X_train, X_test, y_train, y_test, X_tr, X_val, y_tr, y_val,
                 id_cols, cols_to_drop_missing)

"""Leakage-controlled preprocessing factories (notebook §4).

Median imputation (+ missing-indicator flags) and standard scaling are wrapped, together with SMOTE, in an
``imblearn`` pipeline so that every statistic is re-computed on the training fold only and SMOTE is skipped at
prediction time.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED, USE_MISSING_INDICATORS


def numeric_steps():
    return SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=USE_MISSING_INDICATORS)),
        ("scaler", StandardScaler()),
    ])


def make_preprocessor(feature_names):
    """Fresh, unfitted preprocessor. A new instance is created on every call so that nothing fitted is reused
    across folds or across the ablation analyses."""
    return ColumnTransformer(transformers=[("num", numeric_steps(), list(feature_names))], remainder="drop")


def make_full_pipeline(model, feature_names, use_smote: bool = True):
    """Unfitted pipeline: preprocessing -> (SMOTE) -> model."""
    steps = [("preprocess", make_preprocessor(feature_names))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=SEED, k_neighbors=5)))
    steps.append(("model", model))
    return ImbPipeline(steps=steps)

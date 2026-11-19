"""Cross-validation schemes, scorers, bootstrap confidence intervals (notebook §6, §9, §16)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import SEED


def make_cv_scheme(n_splits: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def roc_auc_ovr_scorer():
    try:
        return make_scorer(roc_auc_score, average="macro", multi_class="ovr", response_method="predict_proba")
    except TypeError:  # older scikit-learn
        return make_scorer(roc_auc_score, average="macro", multi_class="ovr", needs_proba=True)


def get_scoring() -> dict:
    return {"accuracy": "accuracy", "macro_f1": make_scorer(f1_score, average="macro"),
            "weighted_f1": make_scorer(f1_score, average="weighted"), "roc_auc_ovr": roc_auc_ovr_scorer()}


def cv_macro_f1(pipe, X, y, cv) -> float:
    """Mean five-fold macro-F1 on training data; the Optuna objective."""
    scores = cross_validate(pipe, X, y, cv=cv, scoring=make_scorer(f1_score, average="macro"), n_jobs=-1)
    return scores["test_score"].mean()


def summarise_scores(name: str, scores: dict) -> dict:
    return {"Model": name, "CV_MacroF1_mean": scores["test_macro_f1"].mean(), "CV_MacroF1_std": scores["test_macro_f1"].std(),
            "CV_Accuracy_mean": scores["test_accuracy"].mean(), "CV_WeightedF1_mean": scores["test_weighted_f1"].mean(),
            "CV_ROC_AUC_mean": scores["test_roc_auc_ovr"].mean()}


def multiclass_metrics(y_true, preds, probs) -> dict:
    return {"Accuracy": accuracy_score(y_true, preds), "MacroF1": f1_score(y_true, preds, average="macro"),
            "WeightedF1": f1_score(y_true, preds, average="weighted"),
            "ROC_AUC": roc_auc_score(y_true, probs, average="macro", multi_class="ovr")}


def bootstrap_metrics(y_test, test_preds, test_probs, n_boot: int, seed: int = SEED):
    """Percentile bootstrap of accuracy / macro-F1 / OvR ROC-AUC on the held-out test set."""
    rng = np.random.default_rng(seed)
    boot_acc, boot_f1, boot_auc = [], [], []
    y_test_arr = np.asarray(y_test)
    n = len(y_test_arr)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb, prb = y_test_arr[idx], test_preds[idx], test_probs[idx]
        boot_acc.append(accuracy_score(yb, pb))
        boot_f1.append(f1_score(yb, pb, average="macro"))
        try:
            boot_auc.append(roc_auc_score(yb, prb, average="macro", multi_class="ovr"))
        except ValueError:
            pass
    return boot_acc, boot_f1, boot_auc


def ci(arr):
    return np.percentile(arr, [2.5, 97.5])


def bootstrap_table(point, boots) -> pd.DataFrame:
    """``point`` = (acc, macro_f1, auc); ``boots`` = (boot_acc, boot_f1, boot_auc)."""
    acc, f1, auc_ = point
    b_acc, b_f1, b_auc = boots
    return pd.DataFrame({"Metric": ["Accuracy", "Macro-F1", "ROC-AUC (OVR)"],
                         "Point_estimate": [acc, f1, auc_],
                         "95%_CI_low": [ci(b_acc)[0], ci(b_f1)[0], ci(b_auc)[0]],
                         "95%_CI_high": [ci(b_acc)[1], ci(b_f1)[1], ci(b_auc)[1]]}).round(4)

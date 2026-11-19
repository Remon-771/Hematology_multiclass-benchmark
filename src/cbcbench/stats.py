"""Friedman test with Nemenyi post-hoc comparison of training-CV macro-F1 (notebook §11)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

from .config import HYBRID_NAME


def build_fold_score_matrix(base_names, fitted_cv_details):
    """One fold-score entry per model slot: the tuned version where tuned, otherwise the default version,
    plus the stacking ensemble (a model's default and tuned versions are never counted twice)."""
    candidate_slots = list(base_names) + [HYBRID_NAME]
    model_order, matrix, seen = [], [], set()
    for name in candidate_slots:
        key = name + " (tuned)" if (name + " (tuned)") in fitted_cv_details else name
        if key in seen or key not in fitted_cv_details:
            continue
        seen.add(key)
        model_order.append(key)
        matrix.append(fitted_cv_details[key]["test_macro_f1"])
    return model_order, np.array(matrix)


def friedman(fold_score_matrix):
    stat, p = friedmanchisquare(*fold_score_matrix)
    return stat, p


def nemenyi(fold_score_matrix, model_order):
    import scikit_posthocs as sp

    pvals = sp.posthoc_nemenyi_friedman(fold_score_matrix.T)
    pvals.index = model_order
    pvals.columns = model_order
    return pvals


def average_ranks(fold_score_matrix, model_order) -> pd.Series:
    ranks_per_fold = np.array([rankdata(-fold_score_matrix[:, f]) for f in range(fold_score_matrix.shape[1])])
    return pd.Series(ranks_per_fold.mean(axis=0), index=model_order).sort_values()

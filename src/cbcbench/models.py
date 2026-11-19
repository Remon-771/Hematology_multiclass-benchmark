"""Classical / ensemble model zoo, Optuna search spaces and diverse-candidate selection (notebook §5, §7)."""
from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import HYBRID_NAME, SEED

try:
    from catboost import CatBoostClassifier
    CATBOOST_OK = True
except ImportError:  # pragma: no cover
    CATBOOST_OK = False


def get_base_models() -> dict:
    """Seventeen classical models (sixteen scikit-learn / boosting + CatBoost) with default hyper-parameters."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    models = {
        "Logistic Regression":             LogisticRegression(max_iter=2000, random_state=SEED),
        "SVM (RBF)":                       SVC(kernel="rbf", probability=True, random_state=SEED),
        "K-Nearest Neighbors":             KNeighborsClassifier(n_neighbors=15),
        "Decision Tree":                   DecisionTreeClassifier(random_state=SEED),
        "Random Forest":                   RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
        "Extra Trees":                     ExtraTreesClassifier(n_estimators=400, random_state=SEED, n_jobs=-1),
        "Gradient Boosting":               GradientBoostingClassifier(random_state=SEED),
        "AdaBoost":                        AdaBoostClassifier(random_state=SEED),
        "Bagging":                         BaggingClassifier(random_state=SEED, n_jobs=-1),
        "Hist Gradient Boosting":          HistGradientBoostingClassifier(random_state=SEED),
        "Naive Bayes":                     GaussianNB(),
        "Linear Discriminant Analysis":    LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
        "MLP (sklearn)":                   MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=SEED),
        "LightGBM":                        LGBMClassifier(random_state=SEED, verbosity=-1),
        "XGBoost":                         XGBClassifier(random_state=SEED, eval_metric="mlogloss", verbosity=0),
    }
    if CATBOOST_OK:
        models["CatBoost"] = CatBoostClassifier(random_state=SEED, verbose=False, allow_writing_files=False)
    return models


# Models that are not tuned: either not meaningfully improved by hyper-parameter search at this scale, or
# (Gradient Boosting) dominated on speed by an equally accurate alternative already in the pool.
NON_TUNED = {"K-Nearest Neighbors", "Naive Bayes", "Linear Discriminant Analysis",
             "Quadratic Discriminant Analysis", "MLP (sklearn)", "Gradient Boosting"}

# Algorithm families used only to enforce diversity among the three models chosen for tuning/stacking.
MODEL_FAMILY = {
    "LightGBM": "boosting", "XGBoost": "boosting", "CatBoost": "boosting",
    "Hist Gradient Boosting": "boosting", "Gradient Boosting": "boosting", "AdaBoost": "boosting",
    "Random Forest": "bagged_trees", "Extra Trees": "bagged_trees", "Bagging": "bagged_trees",
    "Decision Tree": "single_tree",
    "Logistic Regression": "linear", "SVM (RBF)": "kernel",
    "K-Nearest Neighbors": "instance_based", "MLP (sklearn)": "neural",
    "Naive Bayes": "generative", "Linear Discriminant Analysis": "generative",
    "Quadratic Discriminant Analysis": "generative",
}

TREE_MODELS = {"LightGBM", "XGBoost", "CatBoost", "Random Forest", "Extra Trees", "Gradient Boosting",
               "Hist Gradient Boosting", "Decision Tree", "Bagging", "AdaBoost"}


def pick_diverse_top_k(ranked_models, k: int = 3):
    """Walk the CV ranking best-to-worst, taking at most one model per algorithm family, until k models are
    chosen. Falls back to filling remaining slots from the ranking if the pool of distinct families is < k."""
    chosen, used_families = [], set()
    for m in ranked_models:
        fam = MODEL_FAMILY.get(m, m)
        if fam not in used_families:
            chosen.append(m)
            used_families.add(fam)
        if len(chosen) == k:
            return chosen
    for m in ranked_models:
        if m not in chosen:
            chosen.append(m)
        if len(chosen) == k:
            break
    return chosen


def build_model_from_trial(name, trial, base_models=None):
    """Instantiate ``name`` with hyper-parameters drawn from an Optuna trial (or a ``FrozenTrial``)."""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    if name == "LightGBM":
        return LGBMClassifier(random_state=SEED, verbosity=-1,
            n_estimators=trial.suggest_int("n_estimators", 200, 900), num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True), max_depth=trial.suggest_int("max_depth", 3, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True), reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    if name == "XGBoost":
        return XGBClassifier(random_state=SEED, eval_metric="mlogloss", verbosity=0,
            n_estimators=trial.suggest_int("n_estimators", 200, 900), max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True), subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0), reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
    if name == "CatBoost" and CATBOOST_OK:
        return CatBoostClassifier(random_state=SEED, verbose=False, allow_writing_files=False, iterations=trial.suggest_int("iterations", 200, 900),
            depth=trial.suggest_int("depth", 4, 10), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0))
    if name == "Random Forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, n_estimators=trial.suggest_int("n_estimators", 200, 900),
            max_depth=trial.suggest_int("max_depth", 5, 30), min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 6))
    if name == "Extra Trees":
        return ExtraTreesClassifier(random_state=SEED, n_jobs=-1, n_estimators=trial.suggest_int("n_estimators", 200, 900),
            max_depth=trial.suggest_int("max_depth", 5, 30), min_samples_split=trial.suggest_int("min_samples_split", 2, 10))
    if name == "Hist Gradient Boosting":
        return HistGradientBoostingClassifier(random_state=SEED, max_iter=trial.suggest_int("max_iter", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 15), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 10, log=True))
    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=SEED, n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0))
    if name == "Bagging":
        return BaggingClassifier(random_state=SEED, n_jobs=-1, n_estimators=trial.suggest_int("n_estimators", 50, 400),
            max_samples=trial.suggest_float("max_samples", 0.5, 1.0))
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=SEED, max_depth=trial.suggest_int("max_depth", 3, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20), min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10))
    if name == "AdaBoost":
        return AdaBoostClassifier(random_state=SEED, n_estimators=trial.suggest_int("n_estimators", 50, 400),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 2.0, log=True))
    if name == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True, random_state=SEED, C=trial.suggest_float("C", 0.1, 50, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True))
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=SEED, C=trial.suggest_float("C", 1e-3, 10, log=True))
    return (base_models or get_base_models())[name]


class FrozenTrial:
    """Replays a stored ``study.best_params`` dict through ``build_model_from_trial``."""

    def __init__(self, params):
        self.p = params

    def suggest_int(self, k, *a, **kw):
        return self.p[k]

    def suggest_float(self, k, *a, **kw):
        return self.p[k]


def build_tuned_models(best_params: dict, base_models=None) -> dict:
    return {name: build_model_from_trial(name, FrozenTrial(p), base_models) for name, p in best_params.items()}


def build_stacking(tuned_models: dict, top_candidates: list[str]) -> StackingClassifier:
    """Logistic-regression meta-learner on out-of-fold predictions of the tuned base learners (notebook §8)."""
    estimators = [(name.replace(" ", "_"), tuned_models[name]) for name in top_candidates]
    return StackingClassifier(estimators=estimators,
                              final_estimator=LogisticRegression(max_iter=2000, random_state=SEED),
                              cv=3, n_jobs=-1, passthrough=False)


__all__ = ["get_base_models", "NON_TUNED", "MODEL_FAMILY", "TREE_MODELS", "pick_diverse_top_k", "build_model_from_trial",
           "FrozenTrial", "build_tuned_models", "build_stacking", "HYBRID_NAME", "CATBOOST_OK"]

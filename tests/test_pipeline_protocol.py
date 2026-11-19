import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cbcbench.config import SEED
from cbcbench.data import load_raw, make_split
from cbcbench.models import FrozenTrial, pick_diverse_top_k
from cbcbench.preprocessing import make_full_pipeline, make_preprocessor


@pytest.fixture(scope="module")
def split():
    return make_split(load_raw())


def test_split_sizes_and_disjointness(split):
    assert split.X_train.shape == (10094, 23) and split.X_test.shape == (2524, 23)
    assert split.X_tr.shape == (8579, 23) and split.X_val.shape == (1515, 23)
    assert set(split.X_train.index).isdisjoint(split.X_test.index)
    assert set(split.X_tr.index).isdisjoint(split.X_val.index)
    assert set(split.X_tr.index) | set(split.X_val.index) == set(split.X_train.index)  # DL split uses training data only


def test_split_is_stratified(split):
    for y in (split.y_train, split.y_test):
        shares = np.bincount(y) / len(y)
        assert np.allclose(shares, np.bincount(np.r_[split.y_train, split.y_test]) / 12618, atol=0.002)


def test_preprocessor_fits_on_training_only(split):
    prep = make_preprocessor(split.feature_names).fit(split.X_train)
    imputer = prep.named_transformers_["num"].named_steps["imputer"]
    assert np.allclose(imputer.statistics_, split.X_train.median().values)
    assert not np.allclose(imputer.statistics_, pd.concat([split.X_train, split.X_test]).median().values)


def test_pipeline_skips_smote_at_predict(split):
    pipe = make_full_pipeline(LogisticRegression(max_iter=500, random_state=SEED), split.feature_names, use_smote=True)
    pipe.fit(split.X_train, split.y_train)
    probs = pipe.predict_proba(split.X_test)
    assert probs.shape == (len(split.X_test), 5) and np.allclose(probs.sum(axis=1), 1)
    assert 0.9 < (pipe.predict(split.X_test) == split.y_test).mean() <= 1


def test_diverse_candidate_selection():
    ranked = ["Hist Gradient Boosting", "XGBoost", "LightGBM", "Bagging", "Random Forest", "Decision Tree"]
    assert pick_diverse_top_k(ranked, 3) == ["Hist Gradient Boosting", "Bagging", "Decision Tree"]


def test_frozen_trial_replays_params():
    pytest.importorskip("lightgbm"); pytest.importorskip("xgboost")
    from cbcbench.models import build_model_from_trial
    m = build_model_from_trial("Hist Gradient Boosting", FrozenTrial(
        {"max_iter": 325, "max_depth": 15, "learning_rate": 0.0122, "l2_regularization": 0.089}))
    assert m.max_iter == 325 and m.max_depth == 15

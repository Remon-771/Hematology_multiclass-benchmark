"""Consistency of the committed study outputs with the dataset and with each other."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbcbench.data import load_raw, make_split

R = Path(__file__).resolve().parents[1]
res = R / "results"


def test_table1_matches_dataset():
    t = pd.read_csv(res / "audit" / "Table1_class_distribution.csv", index_col=0)
    counts = load_raw()["Disorder_Class"].value_counts()
    assert t["Count"].to_dict() == counts.to_dict() and t["Count"].sum() == 12618


def test_test_support_matches_split():
    sp = make_split(load_raw())
    rep = pd.read_csv(res / "final_evaluation" / "Table5_test_classification_report.csv", index_col=0)
    for i, c in enumerate(sp.class_names):
        assert rep.loc[c, "support"] == (sp.y_test == i).sum()
    assert rep.loc["macro avg", "support"] == 2524


def test_headline_numbers_consistent():
    rep = pd.read_csv(res / "final_evaluation" / "Table5_test_classification_report.csv", index_col=0)
    ci = pd.read_csv(res / "final_evaluation" / "Table6_bootstrap_CI.csv")
    acc = ci.set_index("Metric").loc["Accuracy"]
    assert abs(rep.loc["accuracy", "precision"] - acc["Point_estimate"]) < 1e-3
    assert (ci["95%_CI_low"] <= ci["Point_estimate"]).all() and (ci["Point_estimate"] <= ci["95%_CI_high"]).all()


def test_selected_model_is_top_of_leaderboard_and_hpo_file_valid():
    lb = pd.read_csv(res / "model_selection" / "Table4_unified_leaderboard.csv")
    assert lb.iloc[0]["Model"] == "Hist Gradient Boosting (tuned)"
    assert lb["Selection_Score_MacroF1"].is_monotonic_decreasing
    hp = json.loads((res / "model_selection" / "best_hyperparameters.json").read_text())
    assert set(hp) == {"Hist Gradient Boosting", "Bagging", "Decision Tree"}


def test_saved_model_reproduces_table5():
    """Needs the library versions in requirements/model-artifact.txt (pickles are version-sensitive)."""
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("imblearn")
    try:
        pipe = joblib.load(R / "models" / "final_model_pipeline.joblib")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"saved pipeline not loadable with the installed library versions ({type(e).__name__}); "
                    "see requirements/model-artifact.txt or regenerate with scripts/09_final_test_once.py")
    sp = make_split(load_raw())
    acc = (pipe.predict(sp.X_test) == sp.y_test).mean()
    assert abs(acc - 0.9814) < 5e-4

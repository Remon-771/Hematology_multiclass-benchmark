"""Notebook §8: stacking ensemble of the three tuned models, logistic-regression meta-learner (Table 2b)."""
import time

import pandas as pd
from sklearn.model_selection import cross_validate

from _common import base_parser, load_split, setup
from cbcbench.config import HYBRID_NAME
from cbcbench.evaluation import get_scoring, make_cv_scheme
from cbcbench.models import build_stacking, build_tuned_models
from cbcbench.preprocessing import make_full_pipeline


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out = setup(args)
    _, split = load_split(args)
    tu = out.cache_load("tuning")
    tuned = build_tuned_models(tu["best_params"])
    hybrid = build_stacking(tuned, tu["top_candidates"])
    t0 = time.time()
    s = cross_validate(make_full_pipeline(hybrid, split.feature_names, True), split.X_train, split.y_train,
                       cv=make_cv_scheme(5), scoring=get_scoring(), n_jobs=-1, error_score="raise")
    print(f"[{time.time()-t0:6.1f}s] {HYBRID_NAME:20s} MacroF1={s['test_macro_f1'].mean():.4f} (+/- {s['test_macro_f1'].std():.4f})")
    pd.DataFrame([{"Model": HYBRID_NAME, "CV_MacroF1_mean": s["test_macro_f1"].mean(), "CV_MacroF1_std": s["test_macro_f1"].std(),
                   "CV_Accuracy_mean": s["test_accuracy"].mean(), "Base_learners": ", ".join(tu["top_candidates"])}]
                 ).to_csv(out.table("model_selection", "Table2b_stacking_cv.csv"), index=False)
    out.cache_save("stacking", {"hybrid_scores": s})


if __name__ == "__main__":
    main()

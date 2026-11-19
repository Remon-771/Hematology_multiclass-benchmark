"""Notebook §9: sensitivity checks on training data only - repeated CV (Table 2c), SMOTE vs none (Table S7), isotonic calibration."""
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from _common import base_parser, load_split, setup
from cbcbench.config import HYBRID_NAME, SEED
from cbcbench.evaluation import get_scoring, make_cv_scheme
from cbcbench.models import build_stacking, build_tuned_models
from cbcbench.preprocessing import make_full_pipeline, make_preprocessor


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out = setup(args)
    _, split = load_split(args)
    tu, st = out.cache_load("tuning"), out.cache_load("stacking")
    tuned = build_tuned_models(tu["best_params"])
    tuned[HYBRID_NAME] = hybrid = build_stacking(tuned, tu["top_candidates"])
    scoring, cv = get_scoring(), make_cv_scheme(5)
    fn = split.feature_names

    # 9a. repeated 5x3 cross-validation
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows = []
    for name in tu["top_candidates"] + [HYBRID_NAME]:
        s = cross_validate(make_full_pipeline(tuned[name], fn, True), split.X_train, split.y_train, cv=rcv, scoring=scoring, n_jobs=-1)
        rows.append({"Model": name, "RepeatedCV_MacroF1_mean": s["test_macro_f1"].mean(), "RepeatedCV_MacroF1_std": s["test_macro_f1"].std()})
    r = pd.DataFrame(rows).sort_values("RepeatedCV_MacroF1_mean", ascending=False)
    r.to_csv(out.table("model_selection", "Table2c_repeated_cv_stability.csv"), index=False)
    print(r)

    # 9b. SMOTE vs no oversampling for the stacking ensemble
    smote_scores = st["hybrid_scores"]
    plain = cross_validate(ImbPipeline(steps=[("preprocess", make_preprocessor(fn)), ("model", clone(hybrid))]),
                           split.X_train, split.y_train, cv=cv, scoring=scoring, n_jobs=-1)
    o = pd.DataFrame([
        {"Strategy": "SMOTE", "CV_MacroF1_mean": smote_scores["test_macro_f1"].mean(), "CV_Accuracy_mean": smote_scores["test_accuracy"].mean()},
        {"Strategy": "No oversampling", "CV_MacroF1_mean": plain["test_macro_f1"].mean(), "CV_Accuracy_mean": plain["test_accuracy"].mean()},
    ]).sort_values("CV_MacroF1_mean", ascending=False)
    o.to_csv(out.table("model_selection", "TableS7_oversampling_comparison.csv"), index=False)
    print(o)

    # 9c. isotonic calibration of the ensemble (printed only, as in the notebook)
    calib = ImbPipeline(steps=[("preprocess", make_preprocessor(fn)), ("smote", SMOTE(random_state=SEED, k_neighbors=5)),
                               ("model", CalibratedClassifierCV(clone(hybrid), method="isotonic", cv=3))])
    cs = cross_validate(calib, split.X_train, split.y_train, cv=cv, scoring=scoring, n_jobs=-1)
    print(f"Isotonic-calibrated ensemble (train-only CV): MacroF1={cs['test_macro_f1'].mean():.4f} | "
          f"Uncalibrated: MacroF1={smote_scores['test_macro_f1'].mean():.4f}")


if __name__ == "__main__":
    main()

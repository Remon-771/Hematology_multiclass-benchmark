"""Notebook §3-§6: stratified 80/20 split (test set locked) and 5-fold CV screening of 17 classical models (Table 2)."""
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from _common import base_parser, load_split, setup
from cbcbench.evaluation import get_scoring, make_cv_scheme
from cbcbench.models import get_base_models
from cbcbench.preprocessing import make_full_pipeline


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out = setup(args)
    _, split = load_split(args)
    print(f"Feature set ({len(split.feature_names)}): {split.feature_names}\nClasses: {split.class_names}")
    print("Train:", split.X_train.shape, "| Test (held out, untouched here):", split.X_test.shape)
    print("  -> deep-learning sub-split: fit-train", split.X_tr.shape, ", early-stopping validation", split.X_val.shape)

    cv, scoring = make_cv_scheme(5), get_scoring()
    base_models = get_base_models()
    print(f"{len(base_models)} classical models registered.")
    rows, details = [], {}
    for name, model in base_models.items():
        t0 = time.time()
        pipe = make_full_pipeline(model, split.feature_names, use_smote=True)
        try:
            s = cross_validate(pipe, split.X_train, split.y_train, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
            row = {"Model": name, "CV_Accuracy_mean": s["test_accuracy"].mean(), "CV_Accuracy_std": s["test_accuracy"].std(),
                   "CV_MacroF1_mean": s["test_macro_f1"].mean(), "CV_MacroF1_std": s["test_macro_f1"].std(),
                   "CV_WeightedF1_mean": s["test_weighted_f1"].mean(), "CV_ROC_AUC_mean": s["test_roc_auc_ovr"].mean(),
                   "Fit_time_s": time.time() - t0}
            details[name] = s
        except Exception as e:  # noqa: BLE001
            row = {"Model": name, "CV_Accuracy_mean": np.nan, "CV_Accuracy_std": np.nan, "CV_MacroF1_mean": np.nan,
                   "CV_MacroF1_std": np.nan, "CV_WeightedF1_mean": np.nan, "CV_ROC_AUC_mean": np.nan,
                   "Fit_time_s": time.time() - t0, "Error": str(e)}
        rows.append(row)
        print(f"[{time.time()-t0:6.1f}s] {name:35s} MacroF1={row.get('CV_MacroF1_mean', float('nan')):.4f}")
    df = pd.DataFrame(rows).sort_values("CV_MacroF1_mean", ascending=False).reset_index(drop=True)
    df.to_csv(out.table("model_selection", "Table2_CV_model_screening.csv"), index=False)
    out.cache_save("screening", {"cv_results": rows, "fitted_cv_details": details, "base_names": list(base_models)})


if __name__ == "__main__":
    main()

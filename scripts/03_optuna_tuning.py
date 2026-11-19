"""Notebook §7: Optuna tuning (training data only) of three algorithmically diverse candidates (Table 2d)."""
import json

import optuna
import pandas as pd
from sklearn.model_selection import cross_validate

from _common import base_parser, load_split, setup
from cbcbench.config import SEED, TOP_K
from cbcbench.evaluation import cv_macro_f1, get_scoring, make_cv_scheme
from cbcbench.models import FrozenTrial, NON_TUNED, build_model_from_trial, get_base_models, pick_diverse_top_k
from cbcbench.preprocessing import make_full_pipeline


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out = setup(args)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _, split = load_split(args)
    sc = out.cache_load("screening")
    ranked = pd.DataFrame(sc["cv_results"]).sort_values("CV_MacroF1_mean", ascending=False)["Model"]
    top_candidates = pick_diverse_top_k([m for m in ranked if m not in NON_TUNED], TOP_K)
    print("Tuning (one candidate per algorithm family where possible):", top_candidates)

    cv, scoring = make_cv_scheme(5), get_scoring()
    base_models = get_base_models()
    best_params, tuned_models = {}, {}
    for name in top_candidates:
        def objective(trial, name=name):
            return cv_macro_f1(make_full_pipeline(build_model_from_trial(name, trial, base_models), split.feature_names, True),
                               split.X_train, split.y_train, cv)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
        best_params[name] = study.best_params
        tuned_models[name] = build_model_from_trial(name, FrozenTrial(study.best_params), base_models)
        print(f"{name}: best CV Macro-F1 = {study.best_value:.4f}  params = {study.best_params}")
    with open(out.table("model_selection", "best_hyperparameters.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    # Re-cross-validate each tuned model once to obtain the per-fold scores reused by §11 and §12.
    details, summary = {}, []
    for name in top_candidates:
        s = cross_validate(make_full_pipeline(tuned_models[name], split.feature_names, True), split.X_train, split.y_train,
                           cv=cv, scoring=scoring, n_jobs=-1)
        details[name + " (tuned)"] = s
        summary.append({"Model": name + " (tuned)", "CV_MacroF1_mean": s["test_macro_f1"].mean(),
                        "CV_MacroF1_std": s["test_macro_f1"].std(), "CV_Accuracy_mean": s["test_accuracy"].mean(),
                        "CV_WeightedF1_mean": s["test_weighted_f1"].mean(), "CV_ROC_AUC_mean": s["test_roc_auc_ovr"].mean()})
        print(f"{name} (tuned): Macro-F1 = {s['test_macro_f1'].mean():.4f} (+/- {s['test_macro_f1'].std():.4f})")
    pd.DataFrame(summary).to_csv(out.table("model_selection", "Table2d_tuned_model_cv.csv"), index=False)
    out.cache_save("tuning", {"top_candidates": top_candidates, "best_params": best_params, "tuned_cv_details": details})


if __name__ == "__main__":
    main()

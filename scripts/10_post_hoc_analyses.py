"""Notebook §14, §15, §18, §20-§21: WBC-ablation sensitivity (Table 5c), shallow-tree reference (Table 5d),
learning curve / DL history, SHAP and permutation importance (Table 10). Secondary analyses; not used for model selection."""
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from _common import base_parser, load_split, setup
from cbcbench import final_eval
from cbcbench.config import HYBRID_NAME, SEED
from cbcbench.data import abs_diff_columns
from cbcbench.models import build_stacking, build_tuned_models, get_base_models
from cbcbench.preprocessing import make_full_pipeline


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out = setup(args)
    df_raw, split = load_split(args)
    final = out.cache_load("final")
    best_name, m = final["best_name"], final["metrics"]
    print("Post-hoc analyses for:", best_name)

    # §14 WBC / WBC-family ablation
    fam = final_eval.wbc_family(split.feature_names, split.X_train, abs_diff_columns(df_raw))
    print(f"WBC-correlated feature family ({len(fam)} of {len(split.feature_names)}): {fam}")
    rep_full = final_eval.classification_report_table(split.y_test, final["preds"], split.class_names)
    rep_wbc, m_wbc = final_eval.refit_and_evaluate(["WBC"], "no-wbc", split, final, cfg)
    rep_fam, m_fam = final_eval.refit_and_evaluate(fam, "no-wbc-family", split, final, cfg)
    print(f"Full: Acc={m['accuracy']:.4f} F1={m['macro_f1']:.4f} AUC={m['roc_auc']:.4f}")
    print(f"WBC removed: Acc={m_wbc['Accuracy']:.4f} F1={m_wbc['MacroF1']:.4f} AUC={m_wbc['ROC_AUC']:.4f}")
    print(f"WBC family removed: Acc={m_fam['Accuracy']:.4f} F1={m_fam['MacroF1']:.4f} AUC={m_fam['ROC_AUC']:.4f}")
    from sklearn.metrics import classification_report
    rep_full_d = classification_report(split.y_test, final["preds"], target_names=split.class_names, output_dict=True)
    cmp_ = pd.DataFrame([{"Class": c, "F1_full_features": round(rep_full_d[c]["f1-score"], 4),
                          "F1_WBC_removed": round(rep_wbc[c]["f1-score"], 4),
                          "F1_WBC_family_removed": round(rep_fam[c]["f1-score"], 4)} for c in split.class_names])
    cmp_["Delta_family_vs_full"] = (cmp_["F1_WBC_family_removed"] - cmp_["F1_full_features"]).round(4)
    cmp_.to_csv(out.table("final_evaluation", "Table5c_WBC_sensitivity_by_class.csv"), index=False)
    print(cmp_.to_string(index=False))

    # §15 interpretable reference model
    tree = make_full_pipeline(DecisionTreeClassifier(max_depth=4, random_state=SEED), split.feature_names, True)
    tree.fit(split.X_train, split.y_train)
    tp = tree.predict(split.X_test)
    pd.DataFrame([
        {"Model": "Shallow Decision Tree (depth=4)", "Accuracy": accuracy_score(split.y_test, tp),
         "MacroF1": f1_score(split.y_test, tp, average="macro"), "WeightedF1": f1_score(split.y_test, tp, average="weighted")},
        {"Model": best_name, "Accuracy": m["accuracy"], "MacroF1": m["macro_f1"], "WeightedF1": m["weighted_f1"]},
    ]).to_csv(out.table("final_evaluation", "Table5d_interpretable_baseline_comparison.csv"), index=False)

    # §18 learning curve (single tuned model: 5 folds x 6 sizes; ensemble: 3 folds x 4 sizes) / DL training history
    base_models = get_base_models()
    tuned = build_tuned_models(out.cache_load("tuning")["best_params"], base_models)
    tuned[HYBRID_NAME] = build_stacking(tuned, out.cache_load("tuning")["top_candidates"])
    lc_model = tuned.get(final["clean_name"], base_models.get(final["clean_name"]))
    if lc_model is not None and not final["is_dl"]:
        final_eval.learning_curve_figure(lc_model, split, final["clean_name"], best_name == HYBRID_NAME, out.fig("Fig_learning_curve.png"))
    hist = out.cache_load("deep")["dl_histories"]
    if hist:
        final_eval.dl_history_figure(hist, out.fig("Fig_dl_training_history.png"))

    # §20 feature importance
    final_eval.shap_summary(final, split, out.fig("Fig_shap_summary.png"))
    perm = final_eval.permutation_table(final, split, best_name, out.fig("Fig_permutation_importance.png"))
    perm.to_csv(out.table("final_evaluation", "Table10_permutation_importance.csv"), index=False)
    print(perm.head(10).to_string())


if __name__ == "__main__":
    main()

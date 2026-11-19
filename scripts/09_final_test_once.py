"""Notebook §12-§13, §16, §17, §19: refit the selected model on the full training split and evaluate the LOCKED test set once.

Writes Tables 5, 5b, 5e, 6, 8, the confusion-matrix / ROC / PR / calibration figures and the final model pipeline.
Run only after scripts 02-08 have fixed the model and hyper-parameters from training-data evidence.
"""
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

from _common import base_parser, load_split, setup
from cbcbench import deep, evaluation, final_eval
from cbcbench.models import build_stacking, build_tuned_models, get_base_models
from cbcbench.config import HYBRID_NAME
from cbcbench.preprocessing import make_full_pipeline


def main():
    p = base_parser(__doc__)
    p.add_argument("--confirm-final-test", action="store_true",
                   help="required: confirms the model/hyper-parameters are locked from training-data evidence")
    p.add_argument("--model", default=None, help="override the model chosen by 08_model_selection.py (e.g. 'Hist Gradient Boosting (tuned)')")
    args = p.parse_args()
    if not args.confirm_final_test:
        p.error("Add --confirm-final-test only AFTER model/hyper-parameters are locked from training-data CV.")
    cfg, out = setup(args)
    _, split = load_split(args)
    tu, sel = out.cache_load("tuning"), out.cache_load("selection")
    best_name = args.model or sel["best_model"]
    base_models = get_base_models()
    tuned = build_tuned_models(tu["best_params"], base_models)
    tuned[HYBRID_NAME] = build_stacking(tuned, tu["top_candidates"])

    final = final_eval.fit_final_model(best_name, split, cfg, tuned, base_models)
    preds, probs, y_test = final["preds"], final["probs"], split.y_test
    print("Model evaluated on the held-out test set:", best_name)
    m = final_eval.headline_metrics(y_test, preds, probs)
    print(f"  Accuracy={m['accuracy']:.4f}  Macro-F1={m['macro_f1']:.4f}  Weighted-F1={m['weighted_f1']:.4f}  ROC-AUC(OVR)={m['roc_auc']:.4f}")

    rep = final_eval.classification_report_table(y_test, preds, split.class_names)
    rep.to_csv(out.table("final_evaluation", "Table5_test_classification_report.csv"))
    final_eval.plot_confusion(y_test, preds, split.class_names, out.fig("Fig_confusion_matrix.png"))
    pr = final_eval.roc_pr_and_ap(y_test, probs, split.class_names, best_name, out.fig("Fig_roc_curves.png"), out.fig("Fig_pr_curves.png"))
    pr.to_csv(out.table("final_evaluation", "Table5b_average_precision_per_class.csv"), index=False)

    boots = evaluation.bootstrap_metrics(y_test, preds, probs, cfg.n_boot)
    evaluation.bootstrap_table((m["accuracy"], m["macro_f1"], m["roc_auc"]), boots
                               ).to_csv(out.table("final_evaluation", "Table6_bootstrap_CI.csv"), index=False)
    final_eval.calibration_and_brier(y_test, probs, split.class_names, best_name, out.fig("Fig_calibration_curves.png")
                                     ).to_csv(out.table("final_evaluation", "Table5e_brier_scores.csv"), index=False)

    if not final["is_dl"]:
        joblib.dump(final["obj"], out.model("final_model_pipeline.joblib"))

    # §19 McNemar test on test-set predictions of the two top-ranked leaderboard entries (characterisation only).
    top2 = sel["leaderboard"].head(2)["Model"].tolist()

    def preds_for(name):
        if name == best_name:
            return preds
        clean = name.replace(" (tuned)", "")
        mdl = tuned.get(clean, base_models.get(clean))
        if mdl is None:
            return None
        pipe = make_full_pipeline(mdl, split.feature_names, True).fit(split.X_train, split.y_train)
        return pipe.predict(split.X_test)

    a, b = preds_for(top2[0]), (preds_for(top2[1]) if len(top2) > 1 else None)
    if a is not None and b is not None:
        ca, cb = a == y_test, b == y_test
        b01, b10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
        r = mcnemar([[0, b01], [b10, 0]], exact=(b01 + b10 < 25))
        print(f"McNemar test, {top2[0]} vs {top2[1]}: statistic={r.statistic:.4f}, p={r.pvalue:.4g}")
        pd.DataFrame([{"Model_A": top2[0], "Model_B": top2[1], "statistic": r.statistic, "p_value": r.pvalue}]
                     ).to_csv(out.table("final_evaluation", "Table8_mcnemar_test.csv"), index=False)
    out.cache_save("final", {**final, "best_name": best_name, "metrics": m, "boots": boots})


if __name__ == "__main__":
    main()

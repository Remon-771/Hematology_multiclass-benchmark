"""Notebook §12 + §21: unified training-data leaderboard (Table 4), final-model selection rule, leaderboard figure."""
import json

import pandas as pd

from _common import base_parser, setup
from cbcbench.config import HYBRID_NAME
from cbcbench.final_eval import leaderboard_figure


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out = setup(args)
    sc, tu, st = out.cache_load("screening"), out.cache_load("tuning"), out.cache_load("stacking")
    dp, ss = out.cache_load("deep"), out.cache_load("stats")
    details = {**sc["fitted_cv_details"], **tu["tuned_cv_details"], HYBRID_NAME: st["hybrid_scores"]}
    top = tu["top_candidates"]
    est = "5-fold CV (training data)"

    def row(name, d):
        return {"Model": name, "Selection_Score_MacroF1": d["test_macro_f1"].mean(), "Accuracy": d["test_accuracy"].mean(),
                "WeightedF1": d["test_weighted_f1"].mean(), "ROC_AUC": d["test_roc_auc_ovr"].mean(), "Estimate_type": est}

    tuned_rows = [row(n + " (tuned)", details[n + " (tuned)"]) for n in top]
    stack_row = [row(HYBRID_NAME, details[HYBRID_NAME])]
    untuned = [{"Model": r["Model"], "Selection_Score_MacroF1": r["CV_MacroF1_mean"], "Accuracy": r["CV_Accuracy_mean"],
                "WeightedF1": r["CV_WeightedF1_mean"], "ROC_AUC": r["CV_ROC_AUC_mean"], "Estimate_type": est}
               for r in sc["cv_results"] if r["Model"] not in top]
    dl_rows = [{"Model": r["Model"], "Selection_Score_MacroF1": r["Val_MacroF1"], "Accuracy": r["Val_Accuracy"],
                "WeightedF1": r["Val_WeightedF1"], "ROC_AUC": r["Val_ROC_AUC"], "Estimate_type": "Validation split (training data)"}
               for r in dp["dl_val_rows"]]
    lb = pd.concat([pd.DataFrame(tuned_rows), pd.DataFrame(stack_row), pd.DataFrame(untuned), pd.DataFrame(dl_rows)],
                   ignore_index=True).sort_values("Selection_Score_MacroF1", ascending=False).reset_index(drop=True)
    lb.to_csv(out.table("model_selection", "Table4_unified_leaderboard.csv"), index=False)

    best, note = lb.iloc[0]["Model"], "Highest training-data selection score."
    if best == HYBRID_NAME:
        singles = [m for m in ss["model_order"] if m != HYBRID_NAME]
        best_single = max(singles, key=lambda m: details[m]["test_macro_f1"].mean())
        nem = ss["nemenyi"]
        tie = (not ss["friedman_significant"]) or (nem is not None and nem.loc[HYBRID_NAME, best_single] >= 0.05)
        if tie:
            best = best_single
            pn = f"Nemenyi p={nem.loc[HYBRID_NAME, best_single]:.3f}" if nem is not None else "Friedman test not significant / Nemenyi unavailable"
            note = (f"The stacking ensemble ranked first nominally, but its training-CV macro-F1 was not statistically "
                    f"distinguishable from {best_single} ({pn}). {best_single} is retained for parsimony.")
    print(note); print("Final model selected using training-data evidence only:", best)
    print(lb.head(10).to_string())
    json.dump({"best_model": best, "selection_note": note, "top_candidates": top},
              open(out.table("model_selection", "selected_model.json"), "w"), indent=2)
    leaderboard_figure(lb, {r["Model"] for r in dp["dl_val_rows"]}, out.fig("Fig_model_leaderboard.png"))
    out.cache_save("selection", {"best_model": best, "note": note, "leaderboard": lb})


if __name__ == "__main__":
    main()

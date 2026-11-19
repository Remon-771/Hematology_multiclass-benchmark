"""Notebook §11: Friedman test and Nemenyi post-hoc comparison of training-CV macro-F1 (Table 7)."""
import pandas as pd

from _common import base_parser, setup
from cbcbench import stats
from cbcbench.config import HYBRID_NAME


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out = setup(args)
    sc, tu, st = out.cache_load("screening"), out.cache_load("tuning"), out.cache_load("stacking")
    details = {**sc["fitted_cv_details"], **tu["tuned_cv_details"], HYBRID_NAME: st["hybrid_scores"]}
    order, M = stats.build_fold_score_matrix(sc["base_names"], details)
    chi, p = stats.friedman(M)
    significant = bool(p < 0.05)
    print(f"Friedman test across {len(order)} model candidates, {M.shape[1]} folds: chi-square={chi:.4f}, p={p:.4g}")
    pd.DataFrame([{"chi_square": chi, "p_value": p, "n_models": len(order), "n_folds": M.shape[1]}]
                 ).to_csv(out.table("model_selection", "Table7_friedman_test.csv"), index=False)

    nem, ranks = None, None
    try:
        import scikit_posthocs as sp
        import matplotlib.pyplot as plt
    except ImportError:
        print("scikit-posthocs is not installed (pip install scikit-posthocs): Nemenyi outputs skipped; model selection "
              "falls back to the conservative rule in 08_model_selection.py.")
        sp = None
    if sp is not None and significant:
        nem = stats.nemenyi(M, order)
        nem.to_csv(out.table("model_selection", "Table7b_nemenyi_posthoc_pvalues.csv"))
        ranks = stats.average_ranks(M, order)
        ranks.to_frame("Avg_CV_Rank").to_csv(out.table("model_selection", "Table7c_avg_cv_ranks.csv"))
        n_sig = int(((nem < 0.05).values.sum() - len(order)) / 2)
        print(f"{n_sig} of {len(order) * (len(order) - 1) // 2} pairwise comparisons differ at alpha=0.05 (Nemenyi).")
        plt.figure(figsize=(10, 8)); sp.sign_plot(nem)
        plt.title("Nemenyi post-hoc pairwise comparison (alpha=0.05)"); plt.tight_layout()
        plt.savefig(out.fig("Fig_nemenyi_heatmap.png"), bbox_inches="tight", dpi=300); plt.close()
        try:
            plt.figure(figsize=(10, 6)); sp.critical_difference_diagram(ranks, nem)
            plt.title(f"Critical-difference diagram (Nemenyi, alpha=0.05, n={M.shape[1]} folds)"); plt.tight_layout()
            plt.savefig(out.fig("Fig_critical_difference_diagram.png"), bbox_inches="tight", dpi=300); plt.close()
        except AttributeError:
            print("Installed scikit-posthocs lacks critical_difference_diagram().")
    out.cache_save("stats", {"model_order": order, "friedman_significant": significant, "nemenyi": nem, "avg_ranks": ranks})


if __name__ == "__main__":
    main()

# Reproducibility map

Every committed output, the command that regenerates it and the notebook section it comes from. Script outputs land in `outputs/` and map to the committed location as shown in the last column.

| Output | Script | Notebook § | Committed location |
|:--|:--|:--|:--|
| Table 1 class distribution | `01_data_audit.py` | 2.2 | `results/audit/Table1_class_distribution.csv` |
| Tables S2–S6 (missingness, descriptives, consistency, WBC recoverability, separability) | `01_data_audit.py` | 2.3–2.7 | `results/audit/TableS*.csv` |
| Table 2 CV screening (17 models) | `02_cv_model_screening.py` | 6 | `results/model_selection/Table2_CV_model_screening.csv` |
| Best hyper-parameters, Table 2d | `03_optuna_tuning.py` | 7 | `results/model_selection/best_hyperparameters.json`, `Table2d_*.csv` |
| Table 2b stacking | `04_stacking.py` | 8 | `results/model_selection/Table2b_stacking_cv.csv` |
| Tables 2c, S7 | `05_training_sensitivity.py` | 9 | `results/model_selection/` |
| Table 3 deep-model validation | `06_deep_learning.py` | 10 | `results/model_selection/Table3_DL_validation_scores.csv` |
| Table 7 Friedman (+7b/7c Nemenyi, optional) | `07_statistical_comparison.py` | 11 | `results/model_selection/Table7_friedman_test.csv` |
| Table 4 leaderboard, leaderboard figure | `08_model_selection.py` | 12, 21 | `results/model_selection/Table4_*.csv`, `figures/final/Fig_model_leaderboard.png` |
| Tables 5, 5b, 5e, 6, 8; confusion/ROC/PR/calibration figures; final model | `09_final_test_once.py` | 13, 16, 17, 19 | `results/final_evaluation/`, `figures/final/`, `models/final_model_pipeline.joblib` |
| Tables 5c, 5d, 10; SHAP, permutation, learning-curve, DL-history figures | `10_post_hoc_analyses.py` | 14, 15, 18, 20 | `results/final_evaluation/`, `figures/final/` |
| Methodology diagram | manually composed | — | `figures/final/fig1_methodology_diagram.png` |

Scripts `02`–`08` share intermediate state through `outputs/_cache/` (git-ignored), so run them in order or use `11_run_all_pipeline.py`.

`python scripts/12_release_audit.py --compare outputs` compares regenerated tables with the committed ones (tolerance 5e-3 for numeric columns; timing columns are ignored). Meaningful only after a full-scale run: a `--fast-debug` run uses fewer Optuna trials / DL epochs / bootstraps and is expected to differ in Tables 2d, 3, 4, 5b, 5c, 7, 8, 10.

## Not in the committed results
Table 7b/7c and Nemenyi/critical-difference figures (need `scikit-posthocs`), TabNet rows in Tables 3/4 (need `pytorch-tabnet`) — the original run did not have these packages. Table S1 (identifier audit) is only written when an identifier-like column exists; none does. Figure 1's BorderlineSMOTE/ADASYN and partial-dependence components are not implemented in the study notebook.

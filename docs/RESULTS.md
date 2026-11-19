# Results

All numbers are read from the committed files in `results/`. Training-data estimates are 5-fold CV unless stated.

## Screening and selection (training data)
Top of Table 4 (unified leaderboard, macro-F1): Hist Gradient Boosting (tuned) 0.9859 · Stacking Ensemble 0.9855 · Gradient Boosting 0.9847 · XGBoost 0.9838 · LightGBM 0.9836 · FT-Transformer 0.9830 (validation split) · Bagging (tuned) 0.9820 · CatBoost 0.9809. Lowest: Naive Bayes 0.6291, QDA 0.3330.

| Check | Result |
|:--|:--|
| Tuned CV macro-F1 (Table 2d) | HistGB 0.9859 ± 0.0030 · Bagging 0.9820 ± 0.0015 · Decision Tree 0.9792 ± 0.0021 |
| Stacking CV macro-F1 (Table 2b) | 0.9855 ± 0.0023 |
| Repeated 5×3 CV (Table 2c) | HistGB 0.9858 · Stacking 0.9855 · Bagging 0.9813 · Decision Tree 0.9790 |
| SMOTE vs none, stacking (Table S7) | 0.9855 vs 0.9851 |
| Deep models, validation split (Table 3) | FT-Transformer 0.9830 · Tabular ResNet-MLP 0.9708 |
| Friedman test (Table 7) | χ² = 81.88, p = 1.77e-10 (18 candidates, 5 folds) |

## Locked test set (n = 2,524) — Hist Gradient Boosting (tuned)
Accuracy 0.9814 [0.9758, 0.9865] · macro-F1 0.9825 [0.9774, 0.9873] · weighted-F1 0.9814 · ROC-AUC 0.9996 [0.9994, 0.9998] (Tables 5, 6).

| Class | Precision | Recall | F1 | Support |
|:--|--:|--:|--:|--:|
| Microcytic Hypochromic Anaemia | 0.9682 | 0.9666 | 0.9674 | 599 |
| Mild-Moderate Leukocytosis | 0.9960 | 0.9940 | 0.9950 | 500 |
| Normal / Non-specific | 0.9684 | 0.9665 | 0.9675 | 508 |
| Severe Leukocytosis Flag | 1.0000 | 1.0000 | 1.0000 | 423 |
| Thrombocytopaenia | 0.9799 | 0.9858 | 0.9828 | 494 |

Macro-average AP 0.9986 (Table 5b); macro-average Brier score 0.0051 (Table 5e). McNemar test vs the stacking ensemble: statistic 5.0, p = 0.424 (Table 8).

## WBC-ablation sensitivity (Table 5c)
| Model variant | Accuracy | Macro-F1 | ROC-AUC |
|:--|--:|--:|--:|
| All 23 features | 0.9814 | 0.9825 | 0.9996 |
| WBC removed | 0.9735 | 0.9743 | 0.9994 |
| WBC family removed (WBC + 5 absolute differential counts) | 0.8249 | 0.8124 | 0.9635 |

Per-class F1 change when the family is removed: Anaemia −0.026, Mild-Moderate Leukocytosis −0.225, Normal −0.212, Severe Leukocytosis Flag −0.331, Thrombocytopaenia −0.057. The result should be read together with the label-definition audit (Table S6).

## Interpretable reference (Table 5d)
Depth-4 decision tree: accuracy 0.9556, macro-F1 0.9564, versus 0.9814 / 0.9825 for the selected model.

## Feature importance (Table 10, permutation, macro-F1 decrease)
WBC 0.414 · Platelet 0.256 · MCV 0.154 · MCH 0.108 · Neutrophil % 0.062 · Neutrophil count 0.021. SHAP summary: `figures/final/Fig_shap_summary.png`.

## Figures
`figures/final/`: `Fig_confusion_matrix`, `Fig_roc_curves`, `Fig_pr_curves`, `Fig_calibration_curves`, `Fig_model_leaderboard`, `Fig_learning_curve`, `Fig_dl_training_history`, `Fig_shap_summary`, `Fig_permutation_importance`, and `fig1_methodology_diagram`.

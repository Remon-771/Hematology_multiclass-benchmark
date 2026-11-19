# Classification benchmark methodology

## Models
* **Classical (17):** Logistic Regression, SVM (RBF), k-NN (k = 15), Decision Tree, Random Forest, Extra Trees, Gradient Boosting, AdaBoost, Bagging, Hist Gradient Boosting, Gaussian Naive Bayes, LDA, QDA, sklearn MLP, LightGBM, XGBoost, CatBoost — default hyper-parameters (`cbcbench/models.py`).
* **Tuning:** one candidate per algorithm family from the CV ranking (family map in `models.py`; excluded from tuning: k-NN, NB, LDA, QDA, MLP, Gradient Boosting). Selected: Hist Gradient Boosting, Bagging, Decision Tree. Optuna TPE sampler (seed 42), 30 trials each; best parameters in `results/model_selection/best_hyperparameters.json`.
* **Stacking:** the three tuned models + logistic-regression meta-learner on 3-fold out-of-fold predictions.
* **Deep tabular:** FT-Transformer (d_token 64, 3 layers, 8 heads), residual MLP ("Tabular ResNet-MLP", 4 blocks, width 128), TabNet (optional dependency). AdamW, ReduceLROnPlateau, gradient clipping, early stopping (patience 12; max 100 epochs).

## Metrics
Accuracy, macro-F1 (primary), weighted-F1, one-vs-rest macro ROC-AUC; per-class precision/recall/F1, average precision, Brier score; 1,000-resample percentile bootstrap CIs; calibration curves.

## Model-selection rule
Highest training-data macro-F1 wins, unless it is the stacking ensemble and it is statistically indistinguishable from the best single model (Friedman/Nemenyi), in which case the single model is retained for parsimony (`scripts/08_model_selection.py`). Outcome: Hist Gradient Boosting (tuned).

## Statistical tests
Friedman test (18 candidates × 5 folds); Nemenyi post-hoc and critical-difference diagram (when scikit-posthocs is installed); McNemar test on test-set predictions of the two top-ranked models (characterisation only).

## Commands
See the README and [REPRODUCIBILITY_MAP.md](REPRODUCIBILITY_MAP.md).

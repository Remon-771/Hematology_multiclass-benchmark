# Dataset auditing and leakage-controlled validation protocol

## Descriptive audits (`scripts/01_data_audit.py`)
Identifier columns, class balance, missingness, descriptive statistics, exact duplicates, internal arithmetic consistency, WBC recoverability from differential counts, and single-feature one-vs-rest AUC per class. Output: `results/audit/`.

## Split and leakage controls
1. **Test locking.** A stratified 80/20 split (`random_state=42`) is made on the raw data *before* any statistic is computed: 10,094 training and 2,524 test records. The test set is first read in `scripts/09_final_test_once.py`.
2. **Fold-local preprocessing.** `SimpleImputer(median, add_indicator=True)` → `StandardScaler` → `SMOTE(k_neighbors=5)` are steps of an `imblearn.pipeline.Pipeline`, refitted from scratch in each training fold and never refitted on held-out folds or on the test set; SMOTE is skipped at prediction time.
3. **Training-only model development.** Screening, Optuna tuning (30 trials, macro-F1, 5-fold CV), stacking, repeated CV, the Friedman test and the final-model rule use only the training split.
4. **Deep models.** A further stratified 85/15 split of the training portion (8,579 fit / 1,515 early-stopping validation) is used; the preprocessor and SMOTE are fitted on the 8,579 records only. Deep models are ranked on that single validation split, which is a different (weaker) estimate than five-fold CV and is reported as such.
5. **Post-hoc analyses.** Everything that touches the test set after selection (bootstrap CIs, calibration, McNemar, WBC ablation, permutation importance) is labelled secondary and is never used to choose a model.

## Label-definition sensitivity
Because WBC defines one class by threshold and is encoded redundantly by the absolute differential counts, the final model is refitted without `WBC` and without the whole WBC family (`WBC`, the five absolute counts; identified by name and by |r| > 0.90 with WBC in the training split). Results: Table 5c.

## Data integrity
`scripts/12_release_audit.py` checks the dataset SHA-256, shape, label set, results inventory and `FILE_MANIFEST_SHA256.txt`.

# Dependencies

| File | Purpose |
|:--|:--|
| `base.txt` | Dataset checks, audit script `01`, the leakage-controlled pipeline, tests. |
| `full.txt` | Everything needed to regenerate **all** tables and figures: LightGBM, XGBoost, CatBoost, Optuna, SHAP, statsmodels, scikit-posthocs, PyTorch, TabNet, JupyterLab, pytest. |

Recommended: Python 3.11-3.13 and `pip install -e '.[full,test]'` from the repository root. A GPU is optional (the
deep models run on CPU, only more slowly); the study outputs were produced on Kaggle (Python 3.13.16, CUDA GPU).

**Optional packages that gate outputs.** `pytorch-tabnet` adds TabNet to the deep-learning comparison;
`scikit-posthocs` adds the Nemenyi post-hoc test and critical-difference diagram (Tables 7b/7c). These two
packages were *not* installed in the run that produced the committed `results/`, so those outputs are not included
there (see `docs/REPRODUCIBILITY_MAP.md`).

**Exact-reproduction environment.** The committed outputs were generated with scikit-learn 1.6.x (the saved pipeline
reports 1.6.1). With scikit-learn >= 1.7-1.9, Quadratic Discriminant Analysis raises a "covariance matrix is not
full rank" error on this collinear feature set (HCT, MCH and MCHC are arithmetic functions of other columns), so its
row in Tables 2/4 is NaN with an `Error` note and the Friedman test uses 17 instead of 18 candidates. Other
screening scores differ from the committed tables by <= 3e-4 in a smoke test with scikit-learn 1.9. For the closest
reproduction install `requirements/model-artifact.txt` (scikit-learn 1.6.1, imbalanced-learn 0.13.0) before the rest
of `requirements/full.txt`.

**Saved model artifact.** `models/final_model_pipeline.joblib` is a pickled scikit-learn/imbalanced-learn pipeline.
Pickles are version-sensitive: it was verified to load and reproduce Table 5 with `scikit-learn==1.6.1` and
`imbalanced-learn==0.13.0` (`requirements/model-artifact.txt`). With newer versions, regenerate it with
`python scripts/09_final_test_once.py --confirm-final-test`.

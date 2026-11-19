# Contributing

Contributions are welcome. Open a pull request with a concise description, the changed source files, and tests or
documentation for any analytical change.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/12_release_audit.py
```

Rules that protect the evaluation protocol: never fit imputation, scaling, oversampling or hyper-parameters on the
test split; keep model selection in scripts `02`-`08` (training data only); the locked test set is touched only in
scripts `09` and `10`. Keep original CBC observations separate from derived outputs, and share only synthetic or
aggregate examples in public issues. Dataset revisions require versioning and updated checksums.

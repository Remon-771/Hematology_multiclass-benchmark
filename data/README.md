# Data files

- **`hematology_cbc_dataset.csv`** — the original, unmodified source file: 12,618 records, 23 numerical CBC columns and one `Disorder_Class` label per record.
- **`data_dictionary.csv`** — description, conventional unit, type, missingness and observed range for each of the 24 source columns.

The source column headers are preserved exactly, including the misspelt `platelat` and `Lymphoeyte %`. `cbcbench.data.load_raw` standardises names **in memory only** (`platelat → Platelet`, `Lymphoeyte % → Lymphocyte %`, `mpv → MPV`, `Neutrophil → Neutrophil count`, …); no value is changed.

**SHA-256 fingerprint**

```text
129e781b4cd33b834a7f4c36ca14406c5a1038496a88c3bab3744033a3650504  data/hematology_cbc_dataset.csv
```

Verify with `python scripts/12_release_audit.py`. The dataset is also distributed on Kaggle as `alaminremon/hematology-cbc-dataset`, the location the study notebook was run against.

Units in the dictionary are the conventional reporting units for each parameter; they are not stated in the source file and should be confirmed by the data owners.

"""Notebook §1-§2: load data, identifier / missingness / consistency / label-definition audit (Tables 1, S2-S6)."""
from _common import base_parser, load_split, setup
from cbcbench import audit
from cbcbench.config import TARGET_COL
from cbcbench.data import abs_diff_columns, high_missing_columns, identifier_columns, sha256


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out = setup(args)
    from cbcbench.data import load_raw
    df_raw = load_raw(args.data)
    print("Dataset file:", args.data.name, "| shape:", df_raw.shape, "| sha256:", sha256(args.data))
    id_cols = identifier_columns(df_raw)
    drop_missing = high_missing_columns(df_raw)
    print("Identifier-like columns excluded from the feature set:", id_cols)
    print("Columns with >90% missing values (excluded):", drop_missing)

    s1 = audit.identifier_audit(df_raw, id_cols)
    if len(s1):
        s1.to_csv(out.table("audit", "TableS1_identifier_audit.csv"), index=False)
    audit.class_distribution(df_raw).to_csv(out.table("audit", "Table1_class_distribution.csv"))
    audit.missingness_table(df_raw).to_csv(out.table("audit", "TableS2_missingness.csv"))
    audit.descriptive_statistics(df_raw, id_cols).to_csv(out.table("audit", "TableS3_descriptive_statistics.csv"))
    print(f"Exact-duplicate records: {audit.n_exact_duplicates(df_raw, id_cols)} of {len(df_raw)}")
    audit.internal_consistency(df_raw).to_csv(out.table("audit", "TableS4_internal_consistency.csv"), index=False)
    audit.wbc_recoverability(df_raw).to_csv(out.table("audit", "TableS5_WBC_recoverability.csv"), index=False)
    sep = audit.single_feature_separability(df_raw, id_cols, drop_missing)
    sep.to_csv(out.table("audit", "TableS6_single_feature_separability.csv"), index=False)
    near = sep[sep["Single_feature_AUC"] >= 0.999]
    print(near.to_string(index=False) if len(near) else "No class is separable by a single feature at AUC >= 0.999.")


if __name__ == "__main__":
    main()

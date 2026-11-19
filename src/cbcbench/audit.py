"""Data description and quality audit (notebook §2). Every function returns a DataFrame; the script writes it."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score

from .config import TARGET_COL
from .data import missing_percent, pct_diff_columns


def identifier_audit(df_raw, id_cols) -> pd.DataFrame:
    """Does a record serial number carry class information? (empty when no identifier-like column exists)."""
    rows = []
    for c in id_cols:
        if pd.api.types.is_numeric_dtype(df_raw[c]):
            groups = [g[c].dropna().values for _, g in df_raw.groupby(TARGET_COL)]
            h, p = kruskal(*groups)
            filled = df_raw[c].fillna(df_raw[c].median())
            aucs = {cls: max(roc_auc_score((df_raw[TARGET_COL] == cls).astype(int), filled),
                             1 - roc_auc_score((df_raw[TARGET_COL] == cls).astype(int), filled))
                    for cls in df_raw[TARGET_COL].unique()}
            rows.append({"Column": c, "Kruskal_H": round(h, 3), "Kruskal_p": p,
                         "Max_single_class_AUC": round(max(aucs.values()), 4)})
    return pd.DataFrame(rows)


def class_distribution(df_raw) -> pd.DataFrame:
    counts = df_raw[TARGET_COL].value_counts()
    return pd.DataFrame({"Count": counts, "Percent": (counts / len(df_raw) * 100).round(2)})


def missingness_table(df_raw) -> pd.DataFrame:
    return missing_percent(df_raw).to_frame("Missing_%")


def descriptive_statistics(df_raw, id_cols) -> pd.DataFrame:
    feat_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns if c not in id_cols]
    desc = df_raw[feat_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3)
    desc["Missing_%"] = missing_percent(df_raw).reindex(desc.index).values
    return desc


def n_exact_duplicates(df_raw, id_cols) -> int:
    return int(df_raw.drop(columns=id_cols).duplicated().sum())


def _rel_err(a, b):
    m = a.notna() & b.notna() & (b != 0)
    return float(np.median(np.abs(a[m] - b[m]) / np.abs(b[m])))


def internal_consistency(d) -> pd.DataFrame:
    """Analysers report some quantities as arithmetic functions of others (e.g. HCT = RBC x MCV / 10)."""
    rows = []
    if {"RBC", "MCV", "HCT"}.issubset(d.columns):
        imp = d["RBC"] * d["MCV"] / 10
        rows.append({"Identity": "HCT = RBC x MCV / 10", "Pearson_r": round(d["HCT"].corr(imp), 3), "Median_rel_error": round(_rel_err(d["HCT"], imp), 4)})
    if {"Haemoglobin", "RBC", "MCH"}.issubset(d.columns):
        imp = d["Haemoglobin"] / d["RBC"] * 10
        rows.append({"Identity": "MCH = Hb / RBC x 10", "Pearson_r": round(d["MCH"].corr(imp), 3), "Median_rel_error": round(_rel_err(d["MCH"], imp), 4)})
    if {"Haemoglobin", "HCT", "MCHC"}.issubset(d.columns):
        imp = d["Haemoglobin"] / d["HCT"] * 100
        rows.append({"Identity": "MCHC = Hb / HCT x 100", "Pearson_r": round(d["MCHC"].corr(imp), 3), "Median_rel_error": round(_rel_err(d["MCHC"], imp), 4)})
    pct = pct_diff_columns(d)
    if pct:
        s = d[pct].sum(axis=1, min_count=len(pct))
        rows.append({"Identity": "Sum of differential percentages (complete rows)", "Pearson_r": np.nan,
                     "Median_rel_error": round(float(np.median(np.abs(s.dropna() - 100) / 100)), 4) if s.notna().any() else np.nan})
    return pd.DataFrame(rows)


def wbc_recoverability(d) -> pd.DataFrame:
    """Implied WBC = absolute count / percentage x 100."""
    pairs = [("Neutrophil count", "Neutrophil %"), ("Lymphocyte count", "Lymphocyte %"), ("Monocyte count", "Monocyte %"),
             ("Eosinophil count", "Eosinophil %"), ("Basophil count", "Basophil %")]
    rows = []
    for a, p in pairs:
        if a in d.columns and p in d.columns and "WBC" in d.columns:
            implied = d[a] / (d[p] / 100)
            m = implied.replace([np.inf, -np.inf], np.nan).notna() & d["WBC"].notna()
            rows.append({"Absolute_count": a, "Percentage": p,
                         "Pearson_r_with_WBC_of_count": round(d[a].corr(d["WBC"]), 3),
                         "Pearson_r_implied_WBC_vs_WBC": round(implied[m].corr(d["WBC"][m]), 3),
                         "Median_rel_error_implied_WBC": round(_rel_err(implied[m], d["WBC"][m]), 4)})
    return pd.DataFrame(rows)


def single_feature_separability(df_raw, id_cols, cols_to_drop_missing) -> pd.DataFrame:
    """One-vs-rest ROC-AUC of every individual feature (direction-free); top-3 per class."""
    numeric_cols = [c for c in df_raw.select_dtypes(include=[np.number]).columns
                    if c not in id_cols and c not in cols_to_drop_missing]
    rows = []
    for cls in sorted(df_raw[TARGET_COL].unique()):
        y_bin = (df_raw[TARGET_COL] == cls).astype(int)
        scored = []
        for col in numeric_cols:
            m = df_raw[col].notna()
            if m.sum() < 50 or y_bin[m].nunique() < 2:
                continue
            a = roc_auc_score(y_bin[m], df_raw.loc[m, col])
            scored.append((col, max(a, 1 - a)))
        scored.sort(key=lambda t: -t[1])
        for rank, (col, a) in enumerate(scored[:3], 1):
            rows.append({"Class": cls, "Rank": rank, "Feature": col, "Single_feature_AUC": round(a, 4)})
    return pd.DataFrame(rows)

"""Release audit: dataset fingerprint, results inventory, file manifest and (optionally) comparison with regenerated outputs.

    python scripts/12_release_audit.py                    # integrity checks
    python scripts/12_release_audit.py --write-manifest   # rewrite FILE_MANIFEST_SHA256.txt
    python scripts/12_release_audit.py --compare outputs  # regenerated tables vs committed results/
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from cbcbench.config import DATA_SHA256, N_RAW_COLUMNS, N_RECORDS, TARGET_COL  # noqa: E402
from cbcbench.data import sha256  # noqa: E402

EXPECTED_TABLES = {
    "audit": ["Table1_class_distribution.csv", "TableS2_missingness.csv", "TableS3_descriptive_statistics.csv",
              "TableS4_internal_consistency.csv", "TableS5_WBC_recoverability.csv", "TableS6_single_feature_separability.csv"],
    "model_selection": ["Table2_CV_model_screening.csv", "Table2b_stacking_cv.csv", "Table2c_repeated_cv_stability.csv",
                        "Table2d_tuned_model_cv.csv", "Table3_DL_validation_scores.csv", "Table4_unified_leaderboard.csv",
                        "Table7_friedman_test.csv", "TableS7_oversampling_comparison.csv", "best_hyperparameters.json"],
    "final_evaluation": ["Table5_test_classification_report.csv", "Table5b_average_precision_per_class.csv",
                         "Table5c_WBC_sensitivity_by_class.csv", "Table5d_interpretable_baseline_comparison.csv",
                         "Table5e_brier_scores.csv", "Table6_bootstrap_CI.csv", "Table8_mcnemar_test.csv",
                         "Table10_permutation_importance.csv"],
}
EXPECTED_FIGURES = ["fig1_methodology_diagram.png", "Fig_calibration_curves.png", "Fig_confusion_matrix.png",
                    "Fig_dl_training_history.png", "Fig_learning_curve.png", "Fig_model_leaderboard.png",
                    "Fig_permutation_importance.png", "Fig_pr_curves.png", "Fig_roc_curves.png", "Fig_shap_summary.png"]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "outputs", ".venv", ".ipynb_checkpoints", "notebook_run"}
IGNORE_COLS = {"Fit_time_s"}


def manifest_files():
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and not (set(rel.parts) & SKIP_DIRS) and rel.name != "FILE_MANIFEST_SHA256.txt" \
                and not rel.name.endswith((".pyc", ".egg-info")) and "egg-info" not in str(rel):
            yield rel


def check_data(errors):
    path = ROOT / "data" / "hematology_cbc_dataset.csv"
    if sha256(path) != DATA_SHA256:
        errors.append("dataset SHA-256 mismatch")
    df = pd.read_csv(path)
    if df.shape != (N_RECORDS, N_RAW_COLUMNS):
        errors.append(f"unexpected dataset shape {df.shape}")
    if df[TARGET_COL].nunique() != 5:
        errors.append("expected 5 classes")
    print(f"[data] shape={df.shape} classes={df[TARGET_COL].nunique()} sha256 OK" if not errors else "[data] problems found")


def check_inventory(errors):
    for group, files in EXPECTED_TABLES.items():
        for f in files:
            if not (ROOT / "results" / group / f).exists():
                errors.append(f"missing results/{group}/{f}")
    for f in EXPECTED_FIGURES:
        if not (ROOT / "figures" / "final" / f).exists():
            errors.append(f"missing figures/final/{f}")
    if not (ROOT / "models" / "final_model_pipeline.joblib").exists():
        errors.append("missing models/final_model_pipeline.joblib")
    print("[inventory] checked", sum(map(len, EXPECTED_TABLES.values())), "tables,", len(EXPECTED_FIGURES), "figures")


def check_manifest(errors):
    mf = ROOT / "FILE_MANIFEST_SHA256.txt"
    if not mf.exists():
        errors.append("FILE_MANIFEST_SHA256.txt missing")
        return
    listed = {}
    for line in mf.read_text().splitlines():
        h, _, name = line.partition("  ")
        listed[name] = h
    actual = {str(r): None for r in manifest_files()}
    for name in set(listed) - set(actual):
        errors.append(f"manifest lists missing file {name}")
    for name in set(actual) - set(listed):
        errors.append(f"file not in manifest: {name}")
    for name, h in listed.items():
        if name in actual and sha256(ROOT / name) != h:
            errors.append(f"checksum mismatch: {name}")
    print(f"[manifest] {len(listed)} entries verified")


def write_manifest():
    lines = [f"{sha256(ROOT / r)}  {r}" for r in manifest_files()]
    (ROOT / "FILE_MANIFEST_SHA256.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote FILE_MANIFEST_SHA256.txt ({len(lines)} files)")


def compare(out_dir: Path, tol=5e-3):
    n_ok = n_diff = n_missing = 0
    for group, files in EXPECTED_TABLES.items():
        for f in files:
            new, ref = out_dir / group / f, ROOT / "results" / group / f
            if not new.exists():
                n_missing += 1
                print(f"  not regenerated: {group}/{f}")
                continue
            if f.endswith(".json"):
                same = new.read_text().strip() == ref.read_text().strip()
                n_ok += same; n_diff += (not same)
                print(f"  {'same' if same else 'DIFF'}: {group}/{f}")
                continue
            a, b = pd.read_csv(new), pd.read_csv(ref)
            cols = [c for c in b.columns if c not in IGNORE_COLS and c in a.columns]
            if a.shape[0] != b.shape[0] or set(b.columns) - IGNORE_COLS - set(a.columns):
                n_diff += 1; print(f"  DIFF (shape/columns): {group}/{f}"); continue
            worst = 0.0
            for c in cols:
                if pd.api.types.is_numeric_dtype(b[c]):
                    worst = max(worst, float((a[c] - b[c]).abs().max()))
                elif not a[c].astype(str).equals(b[c].astype(str)):
                    worst = max(worst, 1.0)
            ok = worst <= tol
            n_ok += ok; n_diff += (not ok)
            print(f"  {'ok  ' if ok else 'DIFF'}: {group}/{f}  max|delta|={worst:.4g}")
    print(f"[compare] {n_ok} within tolerance, {n_diff} differ, {n_missing} not regenerated")
    return n_diff == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--compare", type=Path, default=None, help="outputs directory written by the pipeline scripts")
    a = ap.parse_args()
    if a.write_manifest:
        write_manifest()
        return
    errors = []
    check_data(errors); check_inventory(errors); check_manifest(errors)
    if a.compare:
        compare(a.compare)
    if errors:
        print("AUDIT FAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()

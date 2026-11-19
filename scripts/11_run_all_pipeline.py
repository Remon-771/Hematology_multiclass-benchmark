"""Run scripts 01-10 in order (full reproduction of every table and figure). Use --fast-debug for a smoke test."""
import subprocess
import sys
from pathlib import Path

from _common import base_parser

STEPS = ["01_data_audit.py", "02_cv_model_screening.py", "03_optuna_tuning.py", "04_stacking.py",
         "05_training_sensitivity.py", "06_deep_learning.py", "07_statistical_comparison.py",
         "08_model_selection.py", "09_final_test_once.py", "10_post_hoc_analyses.py"]


def main():
    p = base_parser(__doc__)
    p.add_argument("--confirm-final-test", action="store_true", help="required to run step 09 (locked test evaluation)")
    p.add_argument("--from-step", type=int, default=1)
    args = p.parse_args()
    if not args.confirm_final_test:
        p.error("Add --confirm-final-test to run the full pipeline including the locked test-set evaluation (step 09).")
    here = Path(__file__).parent
    for i, s in enumerate(STEPS, 1):
        if i < args.from_step:
            continue
        cmd = [sys.executable, str(here / s), "--data", str(args.data), "--out", str(args.out)]
        if args.fast_debug:
            cmd.append("--fast-debug")
        if s.startswith("09"):
            cmd.append("--confirm-final-test")
        print(f"\n===== [{i}/{len(STEPS)}] {s} =====", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

"""Shared helpers for the numbered pipeline scripts (works with or without ``pip install -e .``)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbcbench.config import DATA_PATH, DEFAULT_OUT, configure_matplotlib, quiet_warnings, run_config, set_global_seed  # noqa: E402
from cbcbench.outputs import Out  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", type=Path, default=DATA_PATH, help="CSV file (default: data/hematology_cbc_dataset.csv)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root (default: outputs/)")
    p.add_argument("--fast-debug", action="store_true",
                   help="reduced-scale smoke test (5 Optuna trials, 50 bootstraps, 3 DL epochs); NOT for paper results")
    return p


def setup(args):
    quiet_warnings()
    configure_matplotlib()
    set_global_seed()
    return run_config(True if args.fast_debug else None), Out(args.out)


def load_split(args):
    from cbcbench.data import load_raw, make_split
    df_raw = load_raw(args.data)
    return df_raw, make_split(df_raw)

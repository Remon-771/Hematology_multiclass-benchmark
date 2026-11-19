"""Global configuration, identical to Section 0 / Section 1 of the study notebook."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "hematology_cbc_dataset.csv"
DEFAULT_OUT = REPO_ROOT / "outputs"
# SHA-256 of the distributed source CSV (see data/README.md and docs/DATA_GOVERNANCE.md).
DATA_SHA256 = "129e781b4cd33b834a7f4c36ca14406c5a1038496a88c3bab3744033a3650504"

TARGET_COL = "Disorder_Class"
N_RECORDS = 12618
N_RAW_COLUMNS = 24

# Column names are standardised (spelling / unit-explicit names) without altering any value.
RENAME = {
    "Lymphoeyte %": "Lymphocyte %",
    "platelat": "Platelet",
    "mpv": "MPV",
    "rETICULOCYTE": "Reticulocyte",
    "Neutrophil": "Neutrophil count",
    "Lymphocyte": "Lymphocyte count",
    "Monocyte": "Monocyte count",
    "Eosinophil": "Eosinophil count",
    "Basophil": "Basophil count",
}
ABS_DIFF_CANDIDATES = ["Neutrophil count", "Lymphocyte count", "Monocyte count", "Eosinophil count", "Basophil count"]
PCT_DIFF_CANDIDATES = ["Neutrophil %", "Lymphocyte %", "Monocyte %", "Eosinophil %", "Basophil %"]
ID_NAMES = {"s", "id", "sl", "sl.", "sno", "s.no", "serial", "index", "patient_id", "patientid", "mrn"}
HIGH_MISSING_THRESH = 90.0

TEST_SIZE = 0.20          # stratified train/test split, performed before any preprocessing
DL_VAL_SIZE = 0.15        # early-stopping split carved out of the *training* portion only
TOP_K = 3                 # number of tuned / stacked candidates
WBC_CORR_THRESHOLD = 0.90
HYBRID_NAME = "Stacking Ensemble"
USE_MISSING_INDICATORS = True   # median imputation with missing-indicator flags (fit inside every CV fold)


@dataclass(frozen=True)
class RunConfig:
    """Run-scale configuration. ``fast_debug=True`` is only a smoke-test scale (never used for the paper)."""

    fast_debug: bool = False

    @property
    def n_trials(self) -> int:       # Optuna trials per tuned model
        return 5 if self.fast_debug else 30

    @property
    def n_boot(self) -> int:         # bootstrap resamples
        return 50 if self.fast_debug else 1000

    @property
    def dl_epochs(self) -> int:      # maximum epochs for the deep models
        return 3 if self.fast_debug else 100

    @property
    def tabnet_epochs(self) -> int:
        return 5 if self.fast_debug else 150


def run_config(fast_debug: bool | None = None) -> RunConfig:
    """``CBC_FAST_DEBUG=1`` in the environment also enables the smoke-test scale."""
    if fast_debug is None:
        fast_debug = os.environ.get("CBC_FAST_DEBUG", "0") == "1"
    return RunConfig(fast_debug=bool(fast_debug))


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "figure.autolayout": True,
    })
    try:
        import seaborn as sns

        sns.set_palette("colorblind")
    except ImportError:
        pass


def quiet_warnings() -> None:
    import warnings

    os.environ["PYTHONWARNINGS"] = "ignore"   # also silences joblib/loky worker processes
    warnings.filterwarnings("ignore")

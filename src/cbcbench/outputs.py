"""Output-directory layout and inter-script cache.

Scripts write to ``<out>/{audit,model_selection,final_evaluation,figures,models,_cache}``. The committed copies of
the paper outputs live in ``results/{audit,model_selection,final_evaluation}``, ``figures/final`` and ``models``.
"""
from __future__ import annotations

from pathlib import Path

import joblib

GROUPS = ("audit", "model_selection", "final_evaluation")


class Out:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        for sub in (*GROUPS, "figures", "models", "_cache"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def table(self, group: str, name: str) -> Path:
        assert group in GROUPS, group
        return self.root / group / name

    def fig(self, name: str) -> Path:
        return self.root / "figures" / name

    def model(self, name: str) -> Path:
        return self.root / "models" / name

    def cache_save(self, name: str, obj) -> None:
        joblib.dump(obj, self.root / "_cache" / f"{name}.joblib")

    def cache_load(self, name: str):
        p = self.root / "_cache" / f"{name}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run the earlier pipeline scripts first (see docs/REPRODUCIBILITY_MAP.md).")
        return joblib.load(p)

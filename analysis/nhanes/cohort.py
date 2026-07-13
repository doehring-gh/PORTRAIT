"""Cohort loader for NHANES cycle-G cardiometabolic panel.

Builds the 12-feature adult panel, the missingness-based label (>=50%
features missing then imputed), the complete-row reference, and the age-band x sex strata.
Public NHANES data only; nothing from this loader is written to disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Public NHANES XPT directory. Set NHANES_DIR to your local download; defaults to ./data/NHANES.
# Download recipe + expected file hashes are in the repo (see reproduce.sh / README).
NHANES = os.environ.get("NHANES_DIR", os.path.join(os.getcwd(), "data", "NHANES"))

# 12-feature panel: component -> [columns]
PANEL = {
    "BMX": ["BMXBMI", "BMXWAIST"],
    "BPX": ["BPXSY1", "BPXDI1"],
    "GHB": ["LBXGH"],
    "GLU": ["LBXGLU"],
    "HDL": ["LBDHDD"],
    "TCHOL": ["LBXTC"],
    "TRIGLY": ["LBXTR"],
    "BIOPRO": ["LBXSCR", "LBXSAL", "LBXSATSI"],
}
FEATURES = [c for cols in PANEL.values() for c in cols]


def _load(comp: str, cyc: str, cols) -> pd.DataFrame:
    df = pd.read_sas(os.path.join(NHANES, f"{comp}_{cyc}.xpt"))
    return df[["SEQN"] + [c for c in cols if c in df.columns]]


@dataclass
class Cohort:
    X_raw: np.ndarray          # (n, 12) with NaN for missing
    missing_frac: np.ndarray   # (n,) fraction of the 12 features missing
    age: np.ndarray
    sex: np.ndarray
    strata: np.ndarray         # age-band x sex label
    feature_names: list
    seqn: np.ndarray = None    # NHANES respondent id, for endpoint linkage

    @property
    def complete_mask(self):
        return self.missing_frac == 0.0

    def collapse_positive(self, threshold=0.5):
        return self.missing_frac >= threshold


def load_cycle(cyc: str = "G", min_age: int = 18) -> Cohort:
    demo = _load("DEMO", cyc, ["RIDAGEYR", "RIAGENDR"])
    panel = demo.copy()
    for comp, cols in PANEL.items():
        panel = panel.merge(_load(comp, cyc, cols), on="SEQN", how="left")
    panel = panel[panel["RIDAGEYR"] >= min_age].reset_index(drop=True)
    X = panel[FEATURES].to_numpy(dtype=float)
    miss = np.isnan(X).mean(axis=1)
    age = panel["RIDAGEYR"].to_numpy(dtype=float)
    sex = panel["RIAGENDR"].to_numpy(dtype=float)
    band = np.digitize(age, [40, 60])       # 0:18-39, 1:40-59, 2:60+
    strata = np.array([f"{b}_{int(s)}" for b, s in zip(band, sex)])
    seqn = panel["SEQN"].to_numpy(dtype=np.int64)
    return Cohort(X, miss, age, sex, strata, list(FEATURES), seqn=seqn)


__all__ = ["Cohort", "load_cycle", "PANEL", "FEATURES", "NHANES"]

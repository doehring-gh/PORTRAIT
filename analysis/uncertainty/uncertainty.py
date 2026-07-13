"""Calibrated intervals on a position centile: marginal and group-conditional split-conformal coverage.

Two distinct guarantees:
  MARGINAL (guaranteed): split conformal on the centile delivers finite-sample distribution-free
    marginal coverage >= 1 - alpha. A DKW band (Massart tight constant) bounds the whole CDF at
    the same confidence. This is the strong, honest guarantee.
  GROUP-CONDITIONAL (approximate): Mondrian / group-conditional conformal over pre-registered
    strata (e.g. age band x sex) applies split conformal WITHIN each stratum. Coverage is REPORTED
    per stratum, NOT gated—distribution-free CONDITIONAL coverage is impossible (Barber 2021b);
    what a finite stratum gives is marginal-within-stratum, which degrades as the stratum shrinks.

The conformity score for a centile target c in [0,1] with a point prediction c_hat is the absolute
residual |c - c_hat|; the split-conformal radius is the ceil((n+1)(1-alpha))/n empirical quantile
of calibration residuals. Interval = c_hat +/- q, clipped to [0,1].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


def dkw_band(n: int, alpha: float) -> float:
    """DKW-Massart half-width: sup_x |F_n(x) - F(x)| <= eps with prob >= 1-alpha, where
    eps = sqrt( ln(2/alpha) / (2n) ). Bounds the WHOLE empirical CDF simultaneously."""
    return float(np.sqrt(np.log(2.0 / alpha) / (2.0 * n)))


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal radius: the ceil((n+1)(1-alpha))/n empirical quantile of |residual|
    (finite-sample valid). Returns +inf if the level exceeds what n can certify."""
    r = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = len(r)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(r[k - 1])


@dataclass
class CoverageResult:
    alpha: float
    marginal_coverage: float
    marginal_radius: float
    dkw_halfwidth: float
    group_coverage: Dict[str, float] = field(default_factory=dict)
    group_n: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "marginal_coverage": self.marginal_coverage,
                "marginal_radius": self.marginal_radius, "dkw_halfwidth": self.dkw_halfwidth,
                "group_coverage": self.group_coverage, "group_n": self.group_n}


def split_conformal_coverage(c_true_cal: np.ndarray, c_hat_cal: np.ndarray,
                             c_true_test: np.ndarray, c_hat_test: np.ndarray,
                             alpha: float) -> tuple:
    """Fit the conformal radius on calibration residuals, measure coverage on test.
    Returns (empirical_coverage, radius)."""
    radius = conformal_quantile(c_true_cal - c_hat_cal, alpha)
    lo = np.clip(c_hat_test - radius, 0.0, 1.0)
    hi = np.clip(c_hat_test + radius, 0.0, 1.0)
    covered = (c_true_test >= lo) & (c_true_test <= hi)
    return float(np.mean(covered)), radius


def evaluate_coverage(c_true_cal: np.ndarray, c_hat_cal: np.ndarray,
                      c_true_test: np.ndarray, c_hat_test: np.ndarray, alpha: float,
                      strata_test: Optional[np.ndarray] = None,
                      strata_cal: Optional[np.ndarray] = None) -> CoverageResult:
    """Marginal split-conformal coverage (guaranteed) + optional group-conditional coverage
    (Mondrian: a separate radius fit per stratum on that stratum's calibration residuals)."""
    marg_cov, radius = split_conformal_coverage(c_true_cal, c_hat_cal, c_true_test, c_hat_test, alpha)
    dkw = dkw_band(len(c_true_cal), alpha)

    group_cov, group_n = {}, {}
    if strata_test is not None and strata_cal is not None:
        for s in np.unique(strata_test):
            cal_m = strata_cal == s
            test_m = strata_test == s
            group_n[str(s)] = int(test_m.sum())
            if cal_m.sum() < 5 or test_m.sum() < 1:
                group_cov[str(s)] = float("nan")   # too few to fit or measure—reported as nan
                continue
            gc, _ = split_conformal_coverage(c_true_cal[cal_m], c_hat_cal[cal_m],
                                             c_true_test[test_m], c_hat_test[test_m], alpha)
            group_cov[str(s)] = gc
    return CoverageResult(alpha, marg_cov, radius, dkw, group_cov, group_n)


__all__ = ["dkw_band", "conformal_quantile", "split_conformal_coverage",
           "evaluate_coverage", "CoverageResult"]

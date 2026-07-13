"""Conditional per-feature surprise: F(x_j | x_{-j}) via quantile regression + split-conformal.

For each feature j:
  - fit conditional quantile predictors q_tau(x_{-j}) of x_j given the other features
    (PRIMARY estimator = GradientBoostingRegressor pinball loss at a tau grid; a quantile forest
    surrogate, since sklearn has no native QRF; ROBUSTNESS estimator = linear quantile regression),
  - place the observed x_j at its conditional centile s_j = F_hat(x_j | x_{-j}) (fraction of tau grid
    below x_j, monotone-repaired),
  - split-conformal calibrate the (1-alpha) conditional interval using held-out conformity scores.

Validated by calibration (marginal coverage within a DKW band), not by comparison to a detector.
The surprise s_j in [0,1] is the per-feature conditional centile; |s_j - 0.5|*2 is a two-sided
surprise magnitude; the conformal interval gives the calibrated (1-alpha) acceptance region.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


TAU_GRID = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


@dataclass
class ConditionalSurprise:
    tau_grid: np.ndarray = field(default_factory=lambda: TAU_GRID.copy())
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    random_state: int = 0
    _models: dict = field(default_factory=dict, repr=False)   # feature j -> {tau: model}
    _conformal: dict = field(default_factory=dict, repr=False) # feature j -> conformal quantile

    def _fit_feature(self, X, j):
        others = [c for c in range(X.shape[1]) if c != j]
        y = X[:, j]; Xo = X[:, others]
        models = {}
        for tau in self.tau_grid:
            m = GradientBoostingRegressor(loss="quantile", alpha=float(tau),
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    learning_rate=self.learning_rate, random_state=self.random_state)
            m.fit(Xo, y)
            models[float(tau)] = m
        return others, models

    def fit(self, X_fit: np.ndarray):
        """Fit conditional quantile models per feature on the training half."""
        self._models = {}
        for j in range(X_fit.shape[1]):
            others, models = self._fit_feature(X_fit, j)
            self._models[j] = {"others": others, "models": models}
        return self

    def _cond_quantiles(self, X, j):
        info = self._models[j]; Xo = X[:, info["others"]]
        Q = np.column_stack([info["models"][float(t)].predict(Xo) for t in self.tau_grid])
        Q = np.sort(Q, axis=1)   # monotone repair (quantile crossing)
        return Q

    def centile(self, X: np.ndarray, j: int) -> np.ndarray:
        """Conditional centile s_j = F_hat(x_j | x_{-j}) in [0,1] via interpolation on the tau grid."""
        Q = self._cond_quantiles(X, j); x = X[:, j]
        s = np.empty(len(x))
        for i in range(len(x)):
            s[i] = np.interp(x[i], Q[i], self.tau_grid, left=0.0, right=1.0)
        return s

    def calibrate(self, X_cal: np.ndarray, alpha: float = 0.10):
        """Split-conformal: conformity score = distance of x_j from the conditional median, scaled by
        the conditional IQR; the (1-alpha) conformal quantile becomes the calibrated acceptance
        half-width (in scaled units). Stored per feature."""
        self._conformal = {}
        for j in range(X_cal.shape[1]):
            Q = self._cond_quantiles(X_cal, j)
            med = Q[:, np.argmin(np.abs(self.tau_grid - 0.5))]
            iqr = (Q[:, np.argmin(np.abs(self.tau_grid - 0.75))] -
                   Q[:, np.argmin(np.abs(self.tau_grid - 0.25))])
            iqr = np.maximum(iqr, 1e-9)
            score = np.abs(X_cal[:, j] - med) / iqr
            qlevel = min(1.0, np.ceil((len(score) + 1) * (1 - alpha)) / len(score))
            self._conformal[j] = {"q": float(np.quantile(score, qlevel)), "alpha": alpha}
        return self

    def interval(self, X: np.ndarray, j: int):
        """Calibrated (1-alpha) conditional interval [lo, hi] for feature j (requires calibrate())."""
        Q = self._cond_quantiles(X, j)
        med = Q[:, np.argmin(np.abs(self.tau_grid - 0.5))]
        iqr = np.maximum(Q[:, np.argmin(np.abs(self.tau_grid - 0.75))] -
                         Q[:, np.argmin(np.abs(self.tau_grid - 0.25))], 1e-9)
        c = self._conformal[j]["q"]
        return med - c * iqr, med + c * iqr

    def covered(self, X: np.ndarray, j: int) -> np.ndarray:
        lo, hi = self.interval(X, j)
        return (X[:, j] >= lo) & (X[:, j] <= hi)

    def surprise(self, X: np.ndarray) -> np.ndarray:
        """(n, d) matrix of conditional centiles s_j in [0,1]."""
        return np.column_stack([self.centile(X, j) for j in range(X.shape[1])])


def marginal_surprise(X_ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Baseline: marginal centile of each x_j against the reference marginal (a comparator that must
    fail on conditional structure)."""
    out = np.empty_like(X, dtype=float)
    for j in range(X.shape[1]):
        out[:, j] = np.searchsorted(np.sort(X_ref[:, j]), X[:, j]) / len(X_ref)
    return out


class ConditionalSurpriseLinearQR(ConditionalSurprise):
    """Robustness estimator: linear quantile regression per (feature, tau)
    instead of gradient-boosted trees. Much faster; a different functional form. If results
    differ between this and the GBR primary, the conclusion is estimator-dependent and must
    be reported explicitly."""
    def _fit_feature(self, X, j):
        from sklearn.linear_model import QuantileRegressor
        others = [c for c in range(X.shape[1]) if c != j]
        y = X[:, j]; Xo = X[:, others]
        models = {}
        for tau in self.tau_grid:
            m = QuantileRegressor(quantile=float(tau), alpha=0.0, solver="highs")
            m.fit(Xo, y)
            models[float(tau)] = m
        return others, models

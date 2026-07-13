"""Describability gate: group-conditional split-conformal over a fixed finite slice
partition, with per-record abstain decision, multiplicity-corrected simultaneous coverage test, and
orthogonality anti-cheat.

Prior art (cited, NOT claimed as novel): Mondrian conformal (Vovk 2005); CQR (Romano 2019); selective
/ reject-option conformal. Core contributions: the anti-cheat guard and the describability frontier
instrument.

Design choices:
  - slice partition is a fixed finite map x -> slice_id (e.g., age-band x sex on NHANES); on synthetic
    oracle the fixture's own slice_label is used.
  - per-slice split-conformal: within each slice, calibrate a nonconformity threshold at level
    (1-alpha) with the finite-sample correction; a record is DESCRIBABLE if its nonconformity is
    within its slice's calibrated threshold, else ABSTAIN. A whole slice is UNCERTIFIED (all its
    records abstain) if its calibration set is too small (n_cal < min_cal).
  - simultaneous pass rule: Bonferroni over the K fixed slices - every slice's empirical coverage on
    a held-out test fold must lie in the adjusted band [1-alpha-eps_K, 1-alpha+eps_K].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.stats import norm


def _finite_sample_qlevel(n, alpha):
    return min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)


@dataclass
class DescribabilityGate:
    surprise_model: object                  # a fitted ConditionalSurprise (provides nonconformity)
    alpha: float = 0.10
    min_cal: int = 50                        # a slice with fewer calibration rows is UNCERTIFIED
    _thresh: dict = field(default_factory=dict, repr=False)   # slice_id -> conformal threshold
    _uncertified: set = field(default_factory=set, repr=False)

    def _nonconformity(self, X):
        """Record-level nonconformity = max over features of the conditional conformal score
        |x_j - median_j| / IQR_j (the same scaled residual the surprise model calibrates per feature).
        A record inside every per-feature conditional band has low nonconformity."""
        m = self.surprise_model
        scores = np.zeros((len(X), X.shape[1]))
        for j in range(X.shape[1]):
            Q = m._cond_quantiles(X, j)
            med = Q[:, np.argmin(np.abs(m.tau_grid - 0.5))]
            iqr = np.maximum(Q[:, np.argmin(np.abs(m.tau_grid - 0.75))] -
                             Q[:, np.argmin(np.abs(m.tau_grid - 0.25))], 1e-9)
            scores[:, j] = np.abs(X[:, j] - med) / iqr
        return scores.max(1)

    def calibrate(self, X_cal, slices_cal):
        """Per-slice Mondrian calibration of the nonconformity threshold."""
        s = self._nonconformity(X_cal)
        self._thresh = {}; self._uncertified = set()
        for sl in np.unique(slices_cal):
            m = slices_cal == sl
            if m.sum() < self.min_cal:
                self._uncertified.add(int(sl)); continue
            self._thresh[int(sl)] = float(np.quantile(s[m], _finite_sample_qlevel(m.sum(), self.alpha)))
        return self

    def describe(self, X, slices):
        """Per-record decision: True = DESCRIBABLE (within slice threshold), False = ABSTAIN."""
        s = self._nonconformity(X); out = np.zeros(len(X), bool)
        for i in range(len(X)):
            sl = int(slices[i])
            if sl in self._uncertified or sl not in self._thresh:
                out[i] = False
            else:
                out[i] = s[i] <= self._thresh[sl]
        return out

    def abstain_rate(self, X, slices):
        return 1.0 - self.describe(X, slices).mean()


def slice_coverage_test(gate, X_te, slices_te, K, alpha=0.10, family_alpha=0.05):
    """Bonferroni-simultaneous per-slice coverage test on a held-out fold. Coverage in a slice =
    fraction of that slice's records that are DESCRIBABLE (accepted) among those whose TRUE value lies
    inside the calibrated conditional region. For a well-calibrated gate on exchangeable data, the
    acceptance rate among genuine in-distribution records approx (1-alpha); we test empirical coverage
    = P(accept | in slice) against the adjusted band.

    Returns per-slice coverage, the Bonferroni half-width, and pass/fail per slice + simultaneous."""
    describ = gate.describe(X_te, slices_te)
    z = norm.ppf(1 - family_alpha / (2 * K))               # Bonferroni two-sided per-slice
    rows = {}
    for sl in np.unique(slices_te):
        m = slices_te == sl
        cov = describ[m].mean()
        n = int(m.sum())
        hw = z * np.sqrt(max(cov * (1 - cov), 1e-9) / max(n, 1))    # Wald, Bonferroni-adjusted
        band = (1 - alpha - hw, 1 - alpha + hw)
        rows[int(sl)] = {"coverage": float(cov), "n": n, "band": [float(band[0]), float(band[1])],
                         "in_band": bool(band[0] <= cov <= band[1]), "halfwidth": float(hw)}
    simultaneous = all(r["in_band"] for r in rows.values())
    return {"per_slice": rows, "simultaneous_pass": bool(simultaneous),
            "K": int(K), "family_alpha": family_alpha, "alpha": alpha}


def orthogonality_anticheat(abstain_indicator, X, mu, missing_count=None):
    """The abstain decision must NOT be centroid distance in disguise. Regress abstain-indicator on
    ||x-mu|| (+ missing_count if given) with logistic regression; report McFadden pseudo-R^2.
    pseudo-R^2 > 0.5 => FAIL (the gate is a distance ruler)."""
    from sklearn.linear_model import LogisticRegression
    d = np.linalg.norm(X - mu, axis=1).reshape(-1, 1)
    feats = d if missing_count is None else np.column_stack([d, missing_count])
    y = abstain_indicator.astype(int)
    if y.mean() in (0.0, 1.0):
        return {"pseudo_r2": 0.0, "pass": True, "note": "degenerate abstain (all/none)"}
    lr = LogisticRegression(max_iter=1000).fit(feats, y)
    p = np.clip(lr.predict_proba(feats)[:, 1], 1e-9, 1 - 1e-9)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    pbar = y.mean(); ll0 = np.sum(y * np.log(pbar) + (1 - y) * np.log(1 - pbar))
    pr2 = 1 - ll / ll0
    return {"pseudo_r2": float(pr2), "pass": bool(pr2 <= 0.5),
            "note": "abstain vs ||x-mu||" + ("+missing_count" if missing_count is not None else "")}

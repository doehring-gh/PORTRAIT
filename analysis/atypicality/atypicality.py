"""Atypicality detection: off-manifold flag with finite-sample validity.

The core distinction: CENTRALITY (joint-density void: a point where the reference manifold
is hollow) is distinct from EXTREMITY (marginal tail: a point far out on one axis). A record
that is heavily imputed pulls toward the joint centre, reading as normal to a marginal-tail
detector but abnormal to a density-void score. This module scores density voids.

Calibration (Bates et al. 2023, conformal p-value): score every calibration (reference) point
and every test point with the SAME statistic; the conformal p-value of a test point is
  p = (1 + #{cal_score >= test_score}) / (n_cal + 1)
which is a valid marginal p-value under exchangeability (finite-sample). Multiplicity across
test points is controlled by Benjamini-Hochberg FDR at level q. This replaces an uncalibrated
threshold.

Score: negative local reference density = distance to the k-th nearest reference neighbour
(kNN distance). A density-void point has a LARGE kNN distance to the reference. (A far-tail
extremity point also has large kNN distance, so this raw score is an off-manifold score; the
centrality-vs-extremity contrast is exercised by the specificity controls, where on-shell
points must NOT fire and planted voids MUST.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def knn_void_score(reference: np.ndarray, query: np.ndarray, k: int = 10) -> np.ndarray:
    """Off-manifold score = distance to the k-th nearest REFERENCE neighbour. Large => in a void."""
    reference = np.atleast_2d(np.asarray(reference, dtype=float))
    query = np.atleast_2d(np.asarray(query, dtype=float))
    tree = cKDTree(reference)
    d, _ = tree.query(query, k=k)
    return d[:, -1] if d.ndim == 2 else d


def conformal_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Conformal p-value per test point (Bates 2023): right-tailed, larger score = more atypical.
    p_i = (1 + #{cal >= test_i}) / (n_cal + 1). Valid marginal p-value under exchangeability."""
    cal = np.sort(np.asarray(cal_scores, dtype=float))
    test = np.atleast_1d(np.asarray(test_scores, dtype=float))
    n = len(cal)
    ge = n - np.searchsorted(cal, test, side="left")   # #{cal >= test_i}
    return (1.0 + ge) / (n + 1.0)


def bh_reject(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg: boolean reject mask controlling FDR at level q."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    kmax = np.max(np.where(passed)[0])
    cutoff = p[order][kmax]
    return p <= cutoff


@dataclass
class AtypicalityResult:
    p_values: np.ndarray
    flagged: np.ndarray         # bool mask after FDR
    q: float
    k: int

    def as_dict(self) -> dict:
        return {"q": self.q, "k": self.k, "n_flagged": int(self.flagged.sum()),
                "n_test": int(len(self.p_values))}


def flag_atypical(reference: np.ndarray, cal: np.ndarray, test: np.ndarray,
                  k: int = 10, q: float = 0.1) -> AtypicalityResult:
    """Score cal + test against the reference; conformal p-value per test point; BH-FDR at q.
    `cal` and `reference` are both drawn from the on-manifold population; `cal` calibrates the
    p-value, `reference` defines the density."""
    cal_s = knn_void_score(reference, cal, k=k)
    test_s = knn_void_score(reference, test, k=k)
    p = conformal_pvalues(cal_s, test_s)
    flagged = bh_reject(p, q)
    return AtypicalityResult(p, flagged, q, k)


__all__ = ["knn_void_score", "conformal_pvalues", "bh_reject", "flag_atypical",
           "AtypicalityResult"]

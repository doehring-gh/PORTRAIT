"""Kozachenko-Leonenko kNN self-information / entropy estimators.

Computes typicality deviation D = Ihat - hhat, its sign branch (degenerate/over-dense D<0 vs 
genuinely-rare D>0), and collapse detection score. Uses numpy and scipy only. Densities are 
estimated using reference rows only, then queried for any row (so query points never define 
their own density).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln


def _log_unit_ball(d: int) -> float:
    # log volume of the d-dim unit Euclidean ball: (d/2)log(pi) - lgamma(d/2+1)
    return 0.5 * d * np.log(np.pi) - gammaln(d / 2.0 + 1.0)


def _standardise(Xq: np.ndarray, Xref: np.ndarray):
    mu = Xref.mean(0); sd = Xref.std(0); sd[sd == 0] = 1.0
    return (Xq - mu) / sd, (Xref - mu) / sd


def kl_self_information(Xq: np.ndarray, Xref: np.ndarray, k: int = 10) -> np.ndarray:
    """Ihat(x) = -log phat(x) via Kozachenko-Leonenko, densities fit on Xref, queried at Xq.
       Ihat = psi(n) - psi(k) + log c_d + d * log r_k(x)   (r_k = dist to k-th ref neighbour)."""
    Xq_s, Xref_s = _standardise(Xq, Xref)
    n, d = Xref_s.shape
    tree = cKDTree(Xref_s)
    # if a query row IS a reference row, its 0-distance self must not count: query k+1, drop the 0.
    dists, _ = tree.query(Xq_s, k=k + 1)
    # k-th strictly-positive neighbour distance (drops a 0-distance self-match when a query row is a
    # reference row); falls back to the largest positive distance if fewer than k positives exist.
    rk = np.array([row[row > 1e-12][k - 1] if (row > 1e-12).sum() >= k else row[row > 1e-12].max()
                   for row in dists])
    rk = np.maximum(rk, 1e-12)
    return digamma(n) - digamma(k) + _log_unit_ball(d) + d * np.log(rk)


def kl_entropy(Xref: np.ndarray, k: int = 10) -> float:
    """hhat = mean over reference rows of Ihat (leave-self-out via the k+1 query)."""
    return float(np.mean(kl_self_information(Xref, Xref, k=k)))


@dataclass
class D1Result:
    I: np.ndarray            # self-information / information content per query row
    D: np.ndarray            # typicality deviation Ihat - hhat
    h: float                 # reference entropy
    collapse: np.ndarray     # detection score = max(0, -D): larger => more degenerate/over-dense
    sign: np.ndarray         # 'neg' (over-dense) / 'pos' (rare) per row
    k: int


def typicality_deviation(Xq, Xref, k: int = 10) -> D1Result:
    h = kl_entropy(Xref, k=k)
    I = kl_self_information(Xq, Xref, k=k)
    D = I - h
    collapse = np.maximum(0.0, -D)               # over-dense (D<0) => positive detection score
    sign = np.where(D < 0, "neg", "pos")
    return D1Result(I=I, D=D, h=h, collapse=collapse, sign=sign, k=k)


def collapse_score(Xq, Xref, k: int = 10) -> np.ndarray:
    return typicality_deviation(Xq, Xref, k=k).collapse


def sign_branch(Xq, Xref, k: int = 10) -> np.ndarray:
    return typicality_deviation(Xq, Xref, k=k).sign


def per_feature_surprise(Xq, Xref, k: int = 10, order: Optional[list] = None) -> np.ndarray:
    """Per-feature surprise via a 1D conditional-kNN chain: surprise_j = -log phat(x_j | x_{<j}).
       Approximated by the 1D KL self-information of x_j within the neighbourhood defined by x_{<j}
       (frozen feature order = given order, else natural column order). Returns (n_q, d)."""
    Xq_s, Xref_s = _standardise(Xq, Xref)
    n_q, d = Xq_s.shape
    order = list(range(d)) if order is None else order
    out = np.zeros((n_q, d))
    for pos, j in enumerate(order):
        if pos == 0:
            out[:, j] = kl_self_information(Xq[:, [j]], Xref[:, [j]], k=k)
        else:
            cond = order[:pos]
            # condition: weight reference rows by proximity in x_{<j}; use the k nearest in cond-space
            tree = cKDTree(Xref_s[:, cond])
            idx = tree.query(Xq_s[:, cond], k=min(50, len(Xref_s)))[1]
            for i in range(n_q):
                local = Xref[idx[i]][:, [j]]
                out[i, j] = kl_self_information(Xq[i:i+1, [j]], local, k=min(k, len(local) - 1))
    return out

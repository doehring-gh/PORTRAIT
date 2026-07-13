"""Position (per-axis empirical-CDF centile).

A per-axis centile is an empirical CDF computed on each axis's marginal distribution independently,
not a multivariate statistical depth. For each axis, the centile is the fraction of a reference
sample at or below the query value. Two variants are provided: a standard empirical CDF and a
mid-rank variant that is unbiased for the query's own rank when the query is in the reference.

Centile is in [0,1], monotone non-decreasing in the axis value by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ecdf_centile(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Empirical-CDF centile of each query value against a 1-D reference sample.
    centile_i = mean( reference <= query_i ). In [0,1], monotone non-decreasing in query."""
    reference = np.asarray(reference, dtype=float)
    query = np.atleast_1d(np.asarray(query, dtype=float))
    # searchsorted on the sorted reference: count of ref <= q is 'right' insertion index
    order = np.sort(reference)
    counts = np.searchsorted(order, query, side="right")
    return counts / len(order)


def midrank_centile(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Mid-rank (Hazen-style) centile: (#ref < q + 0.5*#ref == q) / n. Unbiased for a query drawn
    from the reference distribution; avoids the 0/1 boundary pile-up of the plain ECDF."""
    reference = np.asarray(reference, dtype=float)
    query = np.atleast_1d(np.asarray(query, dtype=float))
    order = np.sort(reference)
    lt = np.searchsorted(order, query, side="left")
    le = np.searchsorted(order, query, side="right")
    eq = le - lt
    return (lt + 0.5 * eq) / len(order)


@dataclass
class PositionResult:
    centiles: np.ndarray        # (n_query, n_axes) in [0,1]
    method: str                 # "ecdf" | "midrank"

    def as_dict(self) -> dict:
        return {"method": self.method, "shape": list(self.centiles.shape),
                "min": float(self.centiles.min()), "max": float(self.centiles.max())}


def position_centiles(reference_scores: np.ndarray, query_scores: np.ndarray,
                      method: str = "midrank") -> PositionResult:
    """Per-axis centile for every query row against the reference, axis by axis.

    reference_scores : (n_ref, n_axes)   the axis scores of the reference population
    query_scores     : (n_query, n_axes) the axis scores of the samples to place
    Returns centiles (n_query, n_axes), each column an independent per-axis empirical CDF.
    """
    reference_scores = np.atleast_2d(np.asarray(reference_scores, dtype=float))
    query_scores = np.atleast_2d(np.asarray(query_scores, dtype=float))
    n_axes = reference_scores.shape[1]
    if query_scores.shape[1] != n_axes:
        raise ValueError(f"axis mismatch: reference {n_axes}, query {query_scores.shape[1]}")
    fn = {"ecdf": ecdf_centile, "midrank": midrank_centile}[method]
    cols = [fn(reference_scores[:, a], query_scores[:, a]) for a in range(n_axes)]
    return PositionResult(np.column_stack(cols), method)


__all__ = ["ecdf_centile", "midrank_centile", "position_centiles", "PositionResult"]

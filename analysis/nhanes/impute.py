"""Imputation using mean, median, or k-nearest neighbors fitted on a reference subset.
Missing values are replaced with the chosen statistic (mean/median fill to the global centroid;
kNN preserves local structure by using neighbor values). Returns a fully-observed matrix."""
from __future__ import annotations

import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def _reference_stats(X_ref: np.ndarray):
    return np.nanmean(X_ref, axis=0), np.nanmedian(X_ref, axis=0)


def impute(X_raw: np.ndarray, reference_mask: np.ndarray, method: str = "mean",
           k: int = 5) -> np.ndarray:
    """Fill NaNs in X_raw using statistics from the reference rows only.
    method: 'mean' | 'median' | 'knn'. Returns a fully-observed matrix."""
    X = X_raw.copy()
    X_ref = X_raw[reference_mask]
    if method in ("mean", "median"):
        mu, med = _reference_stats(X_ref)
        fill = mu if method == "mean" else med
        idx = np.where(np.isnan(X))
        X[idx] = np.take(fill, idx[1])
        return X
    if method == "knn":
        # standardise on reference, kNN-impute, invert - neighbours define the fill (least collapse)
        sc = StandardScaler().fit(X_ref)
        Xs = sc.transform(X)
        imp = KNNImputer(n_neighbors=k)
        imp.fit(sc.transform(X_ref))
        Xs_filled = imp.transform(Xs)
        return sc.inverse_transform(Xs_filled)
    raise ValueError(f"unknown impute method: {method}")


__all__ = ["impute"]

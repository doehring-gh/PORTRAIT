"""Round-trip recovery: does the generated data actually contain the planted latent?

This is a generator self-consistency check. Each recovery uses a family-appropriate recoverer 
and returns an error in [0, ~1] where 0 = perfect recovery, up to the family's natural 
invariance (label permutation; monotone/sign for a 1D coordinate; column permutation for 
archetype weights). Families with no latent return None.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.manifold import Isomap
from sklearn.metrics import adjusted_rand_score

from .scenarios import Scenario


def _corr_coord(true_t: np.ndarray, rec: np.ndarray) -> float:
    """Monotone-invariant 1D recovery error = 1 - |Spearman| (sign/direction free)."""
    rho, _ = spearmanr(true_t, rec)
    if not np.isfinite(rho):
        return 1.0
    return float(1.0 - abs(rho))


def _labels_error(true_labels: np.ndarray, X: np.ndarray, k: int, seed: int) -> float:
    """1 - ARI from KMeans with known k (permutation-invariant by construction)."""
    km = KMeans(n_clusters=k, n_init=10, random_state=int(seed))
    pred = km.fit_predict(X)
    ari = adjusted_rand_score(true_labels, pred)
    return float(1.0 - max(ari, 0.0))


def _weights_error(true_W: np.ndarray, X: np.ndarray, k: int, seed: int) -> float:
    """Recover simplex weights via NMF (k comps), align columns to truth by best assignment,
    error = 1 - mean aligned column Pearson correlation."""
    Xn = X - X.min(axis=0)                          # NMF needs non-negative input
    nmf = NMF(n_components=k, init="nndsvda", random_state=int(seed), max_iter=1000)
    Wh = nmf.fit_transform(Xn)
    # correlation matrix between true and recovered columns
    C = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            a, b = true_W[:, i], Wh[:, j]
            if a.std() == 0 or b.std() == 0:
                C[i, j] = 0.0
            else:
                C[i, j] = abs(np.corrcoef(a, b)[0, 1])
    row, col = linear_sum_assignment(-C)
    return float(1.0 - C[row, col].mean())


def _factor1d_error(true_s: np.ndarray, X: np.ndarray) -> float:
    """Recover a single latent factor as the top left-singular vector's score."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    rec = U[:, 0] * S[0]
    return _corr_coord(true_s, rec)


def _coord1d_error(sc: Scenario) -> float:
    """1D continuous latent. Use Isomap for a curved manifold (geodesic-aware); PC1 otherwise."""
    X = sc.X
    true_t = np.asarray(sc.ground_truth["latent_coord"])
    if sc.family == "curved_manifold":
        n_neighbors = min(15, max(5, X.shape[0] // 40))
        rec = Isomap(n_neighbors=n_neighbors, n_components=1).fit_transform(X)[:, 0]
    else:
        Xc = X - X.mean(axis=0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        rec = U[:, 0] * S[0]
    return _corr_coord(true_t, rec)


def roundtrip_error(sc: Scenario, seed: int = 0) -> Optional[float]:
    """Dispatch by recovery_kind. Returns recovery error (0 = perfect) or None when no latent."""
    kind = sc.recovery_kind
    if kind == "none":
        return None
    if kind == "labels":
        return _labels_error(np.asarray(sc.ground_truth["latent_labels"]),
                             sc.X, int(sc.ground_truth["k"]), seed)
    if kind == "coord1d":
        return _coord1d_error(sc)
    if kind == "weights":
        return _weights_error(np.asarray(sc.ground_truth["latent_weights"]),
                             sc.X, int(sc.ground_truth["k"]), seed)
    if kind == "factor1d":
        return _factor1d_error(np.asarray(sc.ground_truth["latent_factor"]), sc.X)
    raise ValueError(f"unknown recovery_kind: {kind}")

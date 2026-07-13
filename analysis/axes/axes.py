"""Sparse-NMF axes with resample stability and null-model comparison.

Validity of the axes rests on THREE properties, not on human naming:
  1. resample stability: the sparse support (top-loading features per axis) is stable across
     subsamples - feature-selection Jaccard >= 0.7 (win condition);
  2. beats-null: reconstruction of the data beats a covariance-matched null at matched rank;
  3. downstream sufficiency (tested on real data): axes stratify an eligible non-leaky
     outcome about as well as the full feature set.

Comparator: PCA at matched rank (Occam). NMF requires non-negative input, so features are shifted
to non-negative per column (min-shift) before factorization; PCA runs on standardized data.
No clinical name is emitted for any loading here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from sklearn.decomposition import NMF, PCA


def _nonneg(X: np.ndarray) -> np.ndarray:
    return X - X.min(axis=0, keepdims=True)


def fit_nmf(X: np.ndarray, rank: int, seed: int = 0, max_iter: int = 1000):
    """Sparse NMF (L1 on components). Returns (W scores, H loadings)."""
    nmf = NMF(n_components=rank, init="nndsvda", random_state=int(seed),
              max_iter=max_iter, l1_ratio=1.0, alpha_W=0.0, alpha_H=1e-3)
    W = nmf.fit_transform(_nonneg(X))
    return W, nmf.components_


def top_support(H: np.ndarray, top_k: int = 3) -> List[frozenset]:
    """Per-axis sparse support = indices of the top_k highest-loading features."""
    return [frozenset(np.argsort(h)[::-1][:top_k].tolist()) for h in H]


def _match_axes(Ha: np.ndarray, Hb: np.ndarray) -> np.ndarray:
    """Greedy match rows of Hb to rows of Ha by absolute cosine; return permutation of Hb rows."""
    from scipy.optimize import linear_sum_assignment
    A = Ha / (np.linalg.norm(Ha, axis=1, keepdims=True) + 1e-12)
    B = Hb / (np.linalg.norm(Hb, axis=1, keepdims=True) + 1e-12)
    C = np.abs(A @ B.T)
    row, col = linear_sum_assignment(-C)
    return col


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


@dataclass
class StabilityResult:
    rank: int
    top_k: int
    mean_jaccard: float
    sd_jaccard: float
    n_subsamples: int
    per_axis_jaccard: List[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"rank": self.rank, "top_k": self.top_k, "mean_jaccard": self.mean_jaccard,
                "sd_jaccard": self.sd_jaccard, "n_subsamples": self.n_subsamples,
                "per_axis_jaccard": self.per_axis_jaccard}


def stability(X: np.ndarray, rank: int, n_subsamples: int = 50, frac: float = 0.8,
              top_k: int = 3, seed: int = 0) -> StabilityResult:
    """Resample stability of the sparse NMF support (feature-selection Jaccard).

    Fit NMF on the full data (reference support), then on n_subsamples bootstrap-fraction
    subsamples; align axes to the reference by loading cosine; Jaccard of each axis's top-k
    support against the reference. Report mean/sd over axes x subsamples.
    """
    rng = np.random.default_rng(int(seed))
    n = len(X)
    _, H_ref = fit_nmf(X, rank, seed=seed)
    ref_sup = top_support(H_ref, top_k)
    jac = []
    per_axis = [[] for _ in range(rank)]
    for b in range(n_subsamples):
        idx = rng.choice(n, int(frac * n), replace=False)
        _, H_b = fit_nmf(X[idx], rank, seed=seed + 1 + b)
        perm = _match_axes(H_ref, H_b)
        sup_b = top_support(H_b, top_k)
        for ax in range(rank):
            jv = jaccard(ref_sup[ax], sup_b[perm[ax]])
            jac.append(jv)
            per_axis[ax].append(jv)
    return StabilityResult(rank, top_k, float(np.mean(jac)), float(np.std(jac)),
                           n_subsamples, [float(np.mean(a)) for a in per_axis])


def reconstruction_error(X: np.ndarray, rank: int, method: str = "nmf", seed: int = 0) -> float:
    """Relative Frobenius reconstruction error at the given rank."""
    if method == "nmf":
        W, H = fit_nmf(X, rank, seed=seed)
        Xhat = W @ H
        Xn = _nonneg(X)
        return float(np.linalg.norm(Xn - Xhat) / (np.linalg.norm(Xn) + 1e-12))
    elif method == "pca":
        p = PCA(n_components=rank, random_state=int(seed))
        Xc = X - X.mean(axis=0)
        Z = p.fit_transform(Xc)
        Xhat = Z @ p.components_
        return float(np.linalg.norm(Xc - Xhat) / (np.linalg.norm(Xc) + 1e-12))
    raise ValueError(method)


def reconstruction_gap_vs_null(X: np.ndarray, rank: int, n_null: int = 20, seed: int = 0,
                               method: str = "nmf"):
    """Reconstruction error at `rank` vs a covariance-matched null at the same rank.

    Note: low-rank reconstruction error does NOT separate structured data from a covariance-matched
    null—a Gaussian with the same covariance compresses about as well at matched rank. This returns
    the gap as a diagnostic, not a gate. The beyond-covariance decision belongs to the describability
    gate; axis validity here rests on (1) describability confirmed, (2) resample stability
    (this module), (3) downstream sufficiency (validation on real data).
    
    Returns (observed_err, null_err_mean, null_err_sd, gap_z).
    """
    rng = np.random.default_rng(int(seed))
    obs = reconstruction_error(X, rank, method=method, seed=seed)
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh((cov + cov.T) / 2.0)
    w = np.clip(w, 1e-12, None)
    root = V @ np.diag(np.sqrt(w)) @ V.T
    errs = []
    for i in range(n_null):
        Xn = mu + rng.normal(size=X.shape) @ root.T
        errs.append(reconstruction_error(Xn, rank, method=method, seed=seed + 1 + i))
    nm, nsd = float(np.mean(errs)), float(np.std(errs))
    gap_z = float((obs - nm) / (nsd + 1e-12))
    return obs, nm, nsd, gap_z


def axes_valid(X: np.ndarray, rank: int, s2_decision: str, jaccard_threshold: float = 0.7,
               n_subsamples: int = 50, top_k: int = 3, seed: int = 0):
    """Axis-validity verdict. Axes are valid only if the describability gate confirms structure
    beyond its covariance AND the sparse support is resample-stable. Downstream sufficiency is a
    separate validation step on real data.

    Returns a dict: {s2_gate, stability_jaccard, stable, emit_axes, reason}.
    emit_axes is False when the describability gate does not certify the data—abstention is the
    correct output in that case.
    """
    st = stability(X, rank, n_subsamples=n_subsamples, top_k=top_k, seed=seed)
    stable = st.mean_jaccard >= jaccard_threshold
    describable = (s2_decision == "DESCRIBE")
    emit = bool(describable and stable)
    if not describable:
        reason = "describability gate did not certify structure beyond the covariance -> abstain, no axes"
    elif not stable:
        reason = f"axis support unstable (Jaccard {st.mean_jaccard:.3f} < {jaccard_threshold})"
    else:
        reason = f"describable and stable (Jaccard {st.mean_jaccard:.3f}); downstream sufficiency not yet confirmed"
    return {"s2_gate": s2_decision, "stability_jaccard": st.mean_jaccard,
            "stable": stable, "emit_axes": emit, "reason": reason,
            "per_axis_jaccard": st.per_axis_jaccard}


__all__ = ["fit_nmf", "top_support", "jaccard", "stability", "StabilityResult",
           "reconstruction_error", "reconstruction_gap_vs_null", "axes_valid"]

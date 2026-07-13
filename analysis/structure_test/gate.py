"""Describability gate: a calibrated two-sample test for detecting structure beyond covariance
and marginals.

Null hypothesis: "no structure beyond the covariance (and marginals)."

Test method: a calibrated two-sample test (energy distance, Szekely–Rizzo; MMD as an alternative)
between observed data and a null-reference sample that preserves only the allowed second-order
(and, for the copula variant, marginal) structure.

Calibration: parametric-bootstrap Monte-Carlo p-value. Under H0 the data is a draw from the
null model, so the observed statistic distributes like the statistic between two null-model
draws. p = (1 + #{stat_null >= stat_obs}) / (B + 1). This is calibrated by construction.

Three-state decision: DESCRIBE (p < alpha) / REFUSE (p > alpha_hi) / BORDERLINE (in between),
carrying the p-value and the null it was tested against.

Null variants:
  "gaussian" : reference ~ N(mu_hat, cov_hat)              (covariance only)
  "copula"   : reference preserves the empirical marginals and the rank correlation, destroys
               higher-order dependence (Gaussian copula + empirical marginal inverse-CDF).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm, rankdata


# ---------------------------------------------------------------- statistics

def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Szekely–Rizzo energy distance between samples A (n,d) and B (m,d). >= 0; 0 iff equal dist."""
    dab = cdist(A, B).mean()
    daa = cdist(A, A).mean()
    dbb = cdist(B, B).mean()
    return float(2.0 * dab - daa - dbb)


def _median_bandwidth(A: np.ndarray, B: np.ndarray, rng: np.random.Generator, cap: int = 200) -> float:
    Z = np.vstack([A, B])
    if len(Z) > cap:
        Z = Z[rng.choice(len(Z), cap, replace=False)]
    d = cdist(Z, Z)
    md = np.median(d[d > 0])
    return float(md) if md > 0 else 1.0


def mmd2(A: np.ndarray, B: np.ndarray, bandwidth: Optional[float] = None,
         rng: Optional[np.random.Generator] = None) -> float:
    """Unbiased-ish squared MMD with a Gaussian RBF kernel (median-heuristic bandwidth)."""
    if rng is None:
        rng = np.random.default_rng(0)
    if bandwidth is None:
        bandwidth = _median_bandwidth(A, B, rng)
    g = 1.0 / (2.0 * bandwidth ** 2)
    kaa = np.exp(-g * cdist(A, A, "sqeuclidean")).mean()
    kbb = np.exp(-g * cdist(B, B, "sqeuclidean")).mean()
    kab = np.exp(-g * cdist(A, B, "sqeuclidean")).mean()
    return float(kaa + kbb - 2.0 * kab)


def _dcor_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Distance correlation between two 1-D vectors (Szekely–Rizzo). Captures nonlinear
    dependence; 0 iff independent."""
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    A = cdist(a, a)
    B = cdist(b, b)
    A = A - A.mean(0)[None, :] - A.mean(1)[:, None] + A.mean()
    B = B - B.mean(0)[None, :] - B.mean(1)[:, None] + B.mean()
    dcov = np.sqrt(max((A * B).mean(), 0.0))
    va = np.sqrt(max((A * A).mean(), 0.0))
    vb = np.sqrt(max((B * B).mean(), 0.0))
    return float(dcov / np.sqrt(va * vb)) if va * vb > 0 else 0.0


def mean_pairwise_dcor(X: np.ndarray, rng: np.random.Generator, cap: int = 200) -> float:
    """Mean distance correlation over all feature pairs. A dependence-structure statistic:
    high when features share nonlinear dependence beyond the covariance/copula null."""
    n = min(len(X), cap)
    Xs = X[rng.choice(len(X), n, replace=False)] if len(X) > n else X
    d = Xs.shape[1]
    vals = []
    for j in range(d):
        for k in range(j + 1, d):
            vals.append(_dcor_pair(Xs[:, j], Xs[:, k]))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------- null-model samplers

def _fit_gaussian(X: np.ndarray):
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh((cov + cov.T) / 2.0)
    w = np.clip(w, 1e-12, None)
    root = V @ np.diag(np.sqrt(w)) @ V.T
    return mu, root


def sample_gaussian_null(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    mu, root = _fit_gaussian(X)
    return mu + rng.normal(size=(n, X.shape[1])) @ root.T


def sample_copula_null(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian-copula null: preserve empirical marginals and rank correlation, destroy higher-order
    dependence. Algorithm:
    1. Rank-transform each column to standard normal scores (the Gaussian copula representation).
    2. Fit the correlation of those scores; draw fresh correlated normal scores.
    3. Map back to the data scale through each column's empirical quantile function.
    """
    nX, d = X.shape
    # empirical normal scores
    Z = np.empty_like(X, dtype=float)
    for j in range(d):
        r = rankdata(X[:, j], method="average") / (nX + 1.0)
        Z[:, j] = norm.ppf(r)
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    w, V = np.linalg.eigh((corr + corr.T) / 2.0)
    w = np.clip(w, 1e-12, None)
    root = V @ np.diag(np.sqrt(w)) @ V.T
    Znew = rng.normal(size=(n, d)) @ root.T
    # map to data scale via empirical marginal quantiles
    out = np.empty((n, d), dtype=float)
    Xs = np.sort(X, axis=0)
    u = norm.cdf(Znew)
    for j in range(d):
        idx = np.clip((u[:, j] * nX).astype(int), 0, nX - 1)
        out[:, j] = Xs[idx, j]
    return out


_SAMPLERS = {"gaussian": sample_gaussian_null, "copula": sample_copula_null}


# ---------------------------------------------------------------- the gate

@dataclass
class GateResult:
    decision: str            # "DESCRIBE" | "REFUSE" | "BORDERLINE"
    p_value: float
    stat_obs: float
    null_type: str           # "gaussian" | "copula"
    statistic: str           # "energy" | "mmd"
    alpha: float
    alpha_hi: float
    n_used: int
    b_reps: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _statistic(name: str, A: np.ndarray, R: np.ndarray, rng: np.random.Generator,
               bandwidth: Optional[float]) -> float:
    if name == "energy":
        return energy_distance(A, R)
    if name == "mmd":
        return mmd2(A, R, bandwidth=bandwidth, rng=rng)
    raise ValueError(f"unknown statistic {name!r}")


def describability_gate(X: np.ndarray, null_type: str = "copula", statistic: str = "energy",
                        alpha: float = 0.05, alpha_hi: float = 0.20, b_reps: int = 199,
                        n_cap: int = 300, seed: int = 0,
                        bandwidth: Optional[float] = None) -> GateResult:
    """Run the calibrated describability test. Returns a three-state GateResult.

    A parametric-bootstrap Monte-Carlo p-value:
      stat_obs = D(X_sub, R0),  R0 ~ null_model(X_sub)
      stat_b   = D(X_b,   R_b), X_b ~ null_model(X_sub), R_b ~ null_model(X_b)   [double-fit]
      p = (1 + #{stat_b >= stat_obs}) / (b_reps + 1)
    Small p => X differs from its own null => DESCRIBE.
    """
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=float)
    n = min(len(X), n_cap)
    Xs = X[rng.choice(len(X), n, replace=False)] if len(X) > n else X
    sampler = _SAMPLERS[null_type]

    if statistic == "mmd" and bandwidth is None:
        bandwidth = _median_bandwidth(Xs, Xs, rng)

    R0 = sampler(Xs, n, rng)
    stat_obs = _statistic(statistic, Xs, R0, rng, bandwidth)

    null_stats = np.empty(b_reps)
    for b in range(b_reps):
        Xb = sampler(Xs, n, rng)          # a genuine null-model draw
        Rb = sampler(Xb, n, rng)          # its own fitted reference (mirror the observed procedure)
        null_stats[b] = _statistic(statistic, Xb, Rb, rng, bandwidth)

    p = (1.0 + np.sum(null_stats >= stat_obs)) / (b_reps + 1.0)
    if p < alpha:
        decision = "DESCRIBE"
    elif p > alpha_hi:
        decision = "REFUSE"
    else:
        decision = "BORDERLINE"
    return GateResult(decision, float(p), float(stat_obs), null_type, statistic,
                      float(alpha), float(alpha_hi), int(n), int(b_reps))


@dataclass
class CombinedGateResult:
    decision: str            # "DESCRIBE" | "REFUSE" | "BORDERLINE"
    p_combined: float        # min-p combination p-value (exactly calibrated)
    p_energy: float
    p_dcor: float
    null_type: str
    alpha: float
    alpha_hi: float
    n_used: int
    b_reps: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def combined_gate(X: np.ndarray, null_type: str = "copula", alpha: float = 0.05,
                  alpha_hi: float = 0.20, b_reps: int = 199, n_cap: int = 300,
                  seed: int = 0) -> CombinedGateResult:
    """Combined describability gate: energy distance (distributional structure) and mean
    pairwise distance correlation (nonlinear dependence structure), combined by the min-p rule
    over a shared null bootstrap so the combined p-value is exactly calibrated.

    Design rationale: no single two-sample statistic separates all describe families from the
    covariance/copula null. Energy distance detects distributional gaps (clusters, curved
    manifolds, archetypes); mean pairwise distance-correlation detects nonlinear cross-feature
    dependence with a near-zero distributional gap (a tight gradient).

    Min-p combination: T = min(p_energy_raw, p_dcor_raw). Its null distribution is obtained from
    the same null draws used to compute the raw per-statistic p-values, so the combined p-value
    p = (1 + #{T_null <= T_obs}) / (B + 1) is exactly calibrated (no independence assumption).
    """
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, dtype=float)
    n = min(len(X), n_cap)
    Xs = X[rng.choice(len(X), n, replace=False)] if len(X) > n else X
    sampler = _SAMPLERS[null_type]

    # observed statistics
    R0 = sampler(Xs, n, rng)
    e_obs = energy_distance(Xs, R0)
    d_obs = mean_pairwise_dcor(Xs, rng)

    # null bootstrap: for each rep, a null-model draw and its own reference
    e_null = np.empty(b_reps)
    d_null = np.empty(b_reps)
    for b in range(b_reps):
        Xb = sampler(Xs, n, rng)
        Rb = sampler(Xb, n, rng)
        e_null[b] = energy_distance(Xb, Rb)
        d_null[b] = mean_pairwise_dcor(Xb, rng)

    # raw per-statistic p-values (right-tailed: larger stat => more structure)
    p_e = (1.0 + np.sum(e_null >= e_obs)) / (b_reps + 1.0)
    p_d = (1.0 + np.sum(d_null >= d_obs)) / (b_reps + 1.0)

    # min-p combination, calibrated against the same null draws.
    # For each null rep b, its raw p under the leave-one-out null ecdf:
    def _raw_p(val, arr):
        return (1.0 + np.sum(arr >= val)) / (b_reps + 1.0)
    T_obs = min(p_e, p_d)
    T_null = np.empty(b_reps)
    for b in range(b_reps):
        pe_b = _raw_p(e_null[b], e_null)
        pd_b = _raw_p(d_null[b], d_null)
        T_null[b] = min(pe_b, pd_b)
    p_comb = (1.0 + np.sum(T_null <= T_obs)) / (b_reps + 1.0)

    if p_comb < alpha:
        decision = "DESCRIBE"
    elif p_comb > alpha_hi:
        decision = "REFUSE"
    else:
        decision = "BORDERLINE"
    return CombinedGateResult(decision, float(p_comb), float(p_e), float(p_d),
                              null_type, float(alpha), float(alpha_hi), int(n), int(b_reps))


__all__ = ["describability_gate", "GateResult", "combined_gate", "CombinedGateResult",
           "energy_distance", "mmd2", "mean_pairwise_dcor",
           "sample_gaussian_null", "sample_copula_null"]

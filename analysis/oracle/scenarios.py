"""Synthetic truth oracle - scenario generators.

Each generator is deterministic and seeded, returns a Scenario carrying:
  X                       (n, d) float array
  ground_truth            dict: latent coordinates + planted deviations (or None)
  expected_describe_label "DESCRIBE" or "REFUSE"
  recovery_kind           how roundtrip.py should try to recover the planted latent
  spec                    the full param dict (with family + seed) used for data_hash
  difficulty              float in [0,1] mapping monotonically to the family's property

DESCRIBE families: clusters {clean, overlap, imbalanced}; non-Gaussian gradient; curved
manifold; partial-archetypal; heavy-tailed.
REFUSE families: isotropic Gaussian; diagonal Gaussian; covariance-preserving Gaussian-blob
control; curved-but-unstructured null.

Design note: the "round-trip" is a generator self-consistency check (does the data actually
contain the latent I planted, recoverable to near-zero error), NOT a method claim.
REFUSE families have latent = None and are not round-tripped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..manifest import hash_array, hash_spec_seed


@dataclass
class Scenario:
    family: str
    X: np.ndarray
    ground_truth: dict
    expected_describe_label: str          # "DESCRIBE" | "REFUSE"
    recovery_kind: str                    # "labels" | "coord1d" | "weights" | "factor1d" | "none"
    difficulty: float
    spec: dict
    data_hash: str = field(default="")

    def __post_init__(self) -> None:
        seed = int(self.spec.get("seed", 0))
        self.data_hash = hash_spec_seed(self.spec, seed)

    @property
    def X_hash(self) -> str:
        return hash_array(self.X)


# ------------------------------------------------------------------ helpers

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


# ------------------------------------------------------------------ DESCRIBE families

def clusters(seed: int = 0, n: int = 600, d: int = 8, k: int = 3,
             difficulty: float = 0.0, imbalance: float = 0.0) -> Scenario:
    """K Gaussian clusters. difficulty in [0,1] shrinks centroid separation (overlap up).
    imbalance in [0,1] skews cluster sizes. Latent = cluster labels + within-cluster coords."""
    rng = _rng(seed)
    sep = float(np.interp(difficulty, [0.0, 1.0], [6.0, 0.6]))   # separation shrinks with difficulty
    centroids = rng.normal(size=(k, d))
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True) * sep
    # cluster sizes
    base = np.ones(k)
    if imbalance > 0:
        base = np.linspace(1.0, 1.0 + 6.0 * imbalance, k)
    probs = base / base.sum()
    counts = rng.multinomial(n, probs)
    labels = np.repeat(np.arange(k), counts)
    within = rng.normal(size=(n, d))
    X = centroids[labels] + within
    gt = {"latent_labels": labels, "centroids": centroids, "k": k,
          "planted_deviations": None}
    spec = {"family": "clusters", "seed": int(seed), "n": n, "d": d, "k": k,
            "difficulty": float(difficulty), "imbalance": float(imbalance), "sep": sep}
    return Scenario("clusters", _standardize(X), gt, "DESCRIBE", "labels",
                    float(difficulty), spec)


def gradient(seed: int = 0, n: int = 600, d: int = 8, difficulty: float = 0.0) -> Scenario:
    """Non-Gaussian 1D gradient: latent t ~ Beta(2,5) (skewed), features are monotone
    nonlinear maps of t plus noise. difficulty raises noise. Latent = t."""
    rng = _rng(seed)
    noise = float(np.interp(difficulty, [0.0, 1.0], [0.15, 1.2]))
    t = rng.beta(2.0, 5.0, size=n)
    tc = t - t.mean()
    cols = []
    for j in range(d):
        # distinct monotone nonlinear channels
        if j % 3 == 0:
            f = np.tanh(3.0 * tc)
        elif j % 3 == 1:
            f = np.sign(tc) * np.abs(tc) ** 1.5
        else:
            f = tc + 0.5 * tc ** 2
        gain = 1.0 + 0.5 * ((j % 5) / 5.0)
        cols.append(gain * f + noise * rng.normal(size=n))
    X = np.column_stack(cols)
    gt = {"latent_coord": t, "planted_deviations": None}
    spec = {"family": "gradient", "seed": int(seed), "n": n, "d": d,
            "difficulty": float(difficulty), "noise": noise}
    return Scenario("gradient", _standardize(X), gt, "DESCRIBE", "coord1d",
                    float(difficulty), spec)


def curved_manifold(seed: int = 0, n: int = 600, d: int = 8, difficulty: float = 0.0) -> Scenario:
    """S-curve: a 1D intrinsic coordinate embedded nonlinearly in d dims. difficulty raises
    off-manifold noise. Latent = intrinsic coordinate t."""
    rng = _rng(seed)
    noise = float(np.interp(difficulty, [0.0, 1.0], [0.02, 0.5]))
    t = np.sort(rng.uniform(-1.5 * np.pi, 1.5 * np.pi, size=n))
    x0 = np.sin(t)
    x1 = np.sign(t) * (np.cos(t) - 1.0)
    x2 = t / 3.0
    base = np.column_stack([x0, x1, x2])
    # embed into d dims via a fixed random rotation, add off-manifold noise
    Q, _ = np.linalg.qr(rng.normal(size=(d, 3)))
    X = base @ Q.T + noise * rng.normal(size=(n, d))
    gt = {"latent_coord": t, "planted_deviations": None}
    spec = {"family": "curved_manifold", "seed": int(seed), "n": n, "d": d,
            "difficulty": float(difficulty), "noise": noise}
    return Scenario("curved_manifold", _standardize(X), gt, "DESCRIBE", "coord1d",
                    float(difficulty), spec)


def partial_archetypal(seed: int = 0, n: int = 600, d: int = 8, k: int = 3,
                       difficulty: float = 0.0) -> Scenario:
    """Convex mixtures of K archetypes; a fraction of points sit near vertices (archetypal),
    the rest in the interior. difficulty raises interior fraction + noise. Latent = weights W."""
    rng = _rng(seed)
    noise = float(np.interp(difficulty, [0.0, 1.0], [0.05, 0.6]))
    interior_frac = float(np.interp(difficulty, [0.0, 1.0], [0.3, 0.9]))
    archetypes = rng.uniform(0.0, 5.0, size=(k, d))   # non-negative vertices
    n_arch = int(round(n * (1.0 - interior_frac)))
    W = np.zeros((n, k))
    # archetypal points: near a single vertex
    vids = rng.integers(0, k, size=n_arch)
    W[np.arange(n_arch), vids] = 1.0
    W[:n_arch] += rng.dirichlet(np.ones(k) * 20.0, size=n_arch) * 0.05
    # interior points: broad Dirichlet
    W[n_arch:] = rng.dirichlet(np.ones(k) * 1.5, size=n - n_arch)
    W = W / W.sum(axis=1, keepdims=True)
    X = W @ archetypes + noise * rng.normal(size=(n, d))
    gt = {"latent_weights": W, "archetypes": archetypes, "k": k,
          "planted_deviations": None}
    spec = {"family": "partial_archetypal", "seed": int(seed), "n": n, "d": d, "k": k,
            "difficulty": float(difficulty), "noise": noise, "interior_frac": interior_frac}
    return Scenario("partial_archetypal", _standardize(X), gt, "DESCRIBE", "weights",
                    float(difficulty), spec)


def heavy_tailed(seed: int = 0, n: int = 600, d: int = 8, difficulty: float = 0.0) -> Scenario:
    """A single latent factor with heavy-tailed (Student-t) scores drives a low-rank signal;
    difficulty lowers the t degrees-of-freedom (heavier tails). Latent = factor scores s."""
    rng = _rng(seed)
    df = float(np.interp(difficulty, [0.0, 1.0], [8.0, 2.0]))    # heavier tails at high difficulty
    load = rng.normal(size=d)
    load = load / np.linalg.norm(load)
    s = rng.standard_t(df, size=n)
    s = (s - s.mean()) / s.std()
    X = np.outer(s, load) * 3.0 + 0.3 * rng.normal(size=(n, d))
    gt = {"latent_factor": s, "loading": load, "planted_deviations": None}
    spec = {"family": "heavy_tailed", "seed": int(seed), "n": n, "d": d,
            "difficulty": float(difficulty), "df": df}
    return Scenario("heavy_tailed", _standardize(X), gt, "DESCRIBE", "factor1d",
                    float(difficulty), spec)


# ------------------------------------------------------------------ REFUSE families

def isotropic_gaussian(seed: int = 0, n: int = 600, d: int = 8, difficulty: float = 0.0) -> Scenario:
    """X ~ N(0, sigma^2 I). No structure beyond covariance. difficulty scales sigma (inert)."""
    rng = _rng(seed)
    sigma = float(np.interp(difficulty, [0.0, 1.0], [1.0, 2.0]))
    X = rng.normal(scale=sigma, size=(n, d))
    gt = {"latent": None, "planted_deviations": None}
    spec = {"family": "isotropic_gaussian", "seed": int(seed), "n": n, "d": d,
            "difficulty": float(difficulty), "sigma": sigma}
    return Scenario("isotropic_gaussian", _standardize(X), gt, "REFUSE", "none",
                    float(difficulty), spec)


def diagonal_gaussian(seed: int = 0, n: int = 600, d: int = 8, difficulty: float = 0.0) -> Scenario:
    """X ~ N(0, diag(variances)). Anisotropic marginals, still no joint structure.
    difficulty raises the spread of the per-feature variances (anisotropy)."""
    rng = _rng(seed)
    spread = float(np.interp(difficulty, [0.0, 1.0], [0.2, 3.0]))
    log_var = rng.uniform(-spread, spread, size=d)
    scales = np.exp(0.5 * log_var)
    X = rng.normal(size=(n, d)) * scales
    gt = {"latent": None, "planted_deviations": None}
    spec = {"family": "diagonal_gaussian", "seed": int(seed), "n": n, "d": d,
            "difficulty": float(difficulty), "spread": spread}
    return Scenario("diagonal_gaussian", _standardize(X), gt, "REFUSE", "none",
                    float(difficulty), spec)


def covariance_matched_null(seed: int = 0, n: int = 600, d: int = 8, k: int = 3,
                            difficulty: float = 0.0, source: str = "clusters") -> Scenario:
    """Critical control: take a DESCRIBE family, measure its empirical covariance, then
    draw a Gaussian with the SAME mean+covariance. Second-order structure is preserved;
    the higher-order (cluster/gradient) structure is destroyed. A covariance-only method
    cannot tell this from the source; a real describability gate must refuse it.
    difficulty here is inherited from the source family (its structure strength)."""
    rng = _rng(seed)
    # generate the source deterministically from an embedded sub-seed
    src_seed = int(seed) * 1000 + 7
    if source == "clusters":
        src = clusters(seed=src_seed, n=n, d=d, k=k, difficulty=difficulty)
    elif source == "gradient":
        src = gradient(seed=src_seed, n=n, d=d, difficulty=difficulty)
    else:
        raise ValueError(f"unknown covariance source: {source}")
    S = src.X
    mu = S.mean(axis=0)
    cov = np.cov(S, rowvar=False)
    # symmetric PSD square-root via eigen-decomposition
    w, V = np.linalg.eigh((cov + cov.T) / 2.0)
    w = np.clip(w, 0.0, None)
    root = V @ np.diag(np.sqrt(w)) @ V.T
    Z = rng.normal(size=(n, d))
    X = mu + Z @ root.T
    gt = {"latent": None, "planted_deviations": None,
          "source_family": source, "matched_cov": True}
    spec = {"family": "covariance_matched_null", "seed": int(seed), "n": n, "d": d, "k": k,
            "difficulty": float(difficulty), "source": source, "src_seed": src_seed}
    return Scenario("covariance_matched_null", _standardize(X), gt, "REFUSE", "none",
                    float(difficulty), spec)


def curved_unstructured_null(seed: int = 0, n: int = 600, d: int = 8, difficulty: float = 0.0) -> Scenario:
    """Isotropic Gaussian passed through a fixed elementwise nonlinear warp: the point cloud
    acquires curvature/skew in its marginals but carries NO low-dimensional latent to recover
    (it stays full-dimensional noise). difficulty raises the warp strength."""
    rng = _rng(seed)
    strength = float(np.interp(difficulty, [0.0, 1.0], [0.3, 2.0]))
    Z = rng.normal(size=(n, d))
    X = Z + strength * np.sin(1.5 * Z) + 0.3 * strength * (Z ** 2 - 1.0)
    gt = {"latent": None, "planted_deviations": None}
    spec = {"family": "curved_unstructured_null", "seed": int(seed), "n": n, "d": d,
            "difficulty": float(difficulty), "strength": strength}
    return Scenario("curved_unstructured_null", _standardize(X), gt, "REFUSE", "none",
                    float(difficulty), spec)

"""Synthetic truth oracle - registry of deterministic, seeded scenario families.

REGISTRY maps family name -> generator callable. Each generator returns a Scenario
(X, ground_truth, expected_describe_label, recovery_kind, difficulty, spec, data_hash).
Use get(family, **kwargs) to build one; iter_families() to enumerate.
"""
from __future__ import annotations

from typing import Callable, Dict

from . import scenarios as _s
from .scenarios import Scenario
from .roundtrip import roundtrip_error

DESCRIBE_FAMILIES = (
    "clusters_clean", "clusters_overlap", "clusters_imbalanced",
    "gradient", "curved_manifold", "partial_archetypal", "heavy_tailed",
)
REFUSE_FAMILIES = (
    "isotropic_gaussian", "diagonal_gaussian",
    "covariance_matched_null", "curved_unstructured_null",
)
# Baseline families: what must round-trip / build for the baseline.
MVP_FAMILIES = ("clusters_clean", "clusters_overlap", "gradient", "covariance_matched_null")


def _clusters_clean(seed=0, **kw) -> Scenario:
    return _s.clusters(seed=seed, difficulty=kw.pop("difficulty", 0.0), imbalance=0.0, **kw)


def _clusters_overlap(seed=0, **kw) -> Scenario:
    return _s.clusters(seed=seed, difficulty=kw.pop("difficulty", 0.55), imbalance=0.0, **kw)


def _clusters_imbalanced(seed=0, **kw) -> Scenario:
    return _s.clusters(seed=seed, difficulty=kw.pop("difficulty", 0.1), imbalance=0.8, **kw)


REGISTRY: Dict[str, Callable[..., Scenario]] = {
    "clusters_clean": _clusters_clean,
    "clusters_overlap": _clusters_overlap,
    "clusters_imbalanced": _clusters_imbalanced,
    "gradient": _s.gradient,
    "curved_manifold": _s.curved_manifold,
    "partial_archetypal": _s.partial_archetypal,
    "heavy_tailed": _s.heavy_tailed,
    "isotropic_gaussian": _s.isotropic_gaussian,
    "diagonal_gaussian": _s.diagonal_gaussian,
    "covariance_matched_null": _s.covariance_matched_null,
    "curved_unstructured_null": _s.curved_unstructured_null,
}

ALL_FAMILIES = tuple(REGISTRY.keys())


def get(family: str, **kwargs) -> Scenario:
    if family not in REGISTRY:
        raise KeyError(f"unknown family {family!r}; known: {ALL_FAMILIES}")
    return REGISTRY[family](**kwargs)


def iter_families():
    for name in ALL_FAMILIES:
        yield name, REGISTRY[name]


__all__ = ["REGISTRY", "ALL_FAMILIES", "DESCRIBE_FAMILIES", "REFUSE_FAMILIES",
           "MVP_FAMILIES", "Scenario", "get", "iter_families", "roundtrip_error"]

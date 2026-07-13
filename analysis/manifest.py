"""Content hashing utilities.

Deterministic bytes for hashing arrays, configs, code, and scenario specs, so
results are reproducible and content-addressable.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import inspect
import json
from typing import Any, Callable

import numpy as np


def _canonical(obj: Any) -> bytes:
    """Deterministic bytes for hashing: sorted-key JSON, numpy arrays handled."""
    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=default).encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_config(config: dict) -> str:
    """SHA256 of the full resolved config dict."""
    return sha256_bytes(_canonical(config))


def hash_array(x: np.ndarray) -> str:
    """SHA256 of an array's exact bytes (shape + dtype + C-order content)."""
    x = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(str(x.dtype).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def hash_code(func: Callable) -> str:
    """SHA256 of a function's exact source (content hash)."""
    src = inspect.getsource(func)
    return sha256_bytes(src.encode("utf-8"))


def hash_spec_seed(spec: dict, seed: int) -> str:
    """Data hash for a scenario = SHA256 of scenario spec and seed."""
    return sha256_bytes(_canonical({"spec": spec, "seed": int(seed)}))


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

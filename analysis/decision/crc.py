"""Conformal Risk Control decision layer (Angelopoulos et al. 2024).

The serve-or-abstain decision: a system serves a rich description when a patient's typicality is
high enough, abstains otherwise. CRC picks the typicality threshold lambda so the expected loss on
served patients is bounded: E[loss] <= alpha. Loss here = "served but off-manifold" (we served a
rich description for a patient we should have abstained on).

Claim discipline: E[loss] <= alpha is a population bound, never a per-patient guarantee.

Mechanism: as lambda rises (serve only the most typical), fewer risky patients are served, so risk
falls monotonically. CRC calibrates lambda_hat on a calibration set as the smallest threshold whose
calibration risk (with the finite-sample (n+1)/n correction) is <= alpha, then that lambda is
applied to test. The empirical test risk is reported against alpha across an alpha grid.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _risk_at(threshold: float, typicality: np.ndarray, loss: np.ndarray, n_cal: int) -> float:
    """Finite-sample-corrected mean loss over served points (typicality >= threshold).
    Angelopoulos CRC correction: (n*R_hat + B) / (n+1) with B = loss upper bound = 1. If nothing
    is served, risk is defined as 0 (no served-but-off-manifold errors)."""
    served = typicality >= threshold
    if served.sum() == 0:
        return 0.0
    r_hat = loss[served].mean()
    return (n_cal * r_hat + 1.0) / (n_cal + 1.0)


def calibrate_lambda(typicality_cal: np.ndarray, loss_cal: np.ndarray, alpha: float,
                     grid: int = 200) -> float:
    """CRC threshold for a serve-if-typicality>=lambda rule whose risk is monotone non-increasing
    in lambda (raising the bar serves only more-typical patients, so served-but-off-manifold risk
    falls). Angelopoulos et al. 2024: scan lambda from high (serve fewest, safest) to low and take
    the smallest lambda for which the corrected risk stays <= alpha for that lambda and all higher
    lambdas. Scanning high to low and stopping at the first violation respects the monotone structure
    and avoids locking onto a low threshold whose large served set dipped below alpha by chance.

    Returns the chosen threshold; if even the highest threshold cannot meet alpha, returns max
    (serve almost nobody)."""
    typ = np.asarray(typicality_cal, dtype=float)
    loss = np.asarray(loss_cal, dtype=float)
    n = len(typ)
    lambdas = np.linspace(typ.min(), typ.max(), grid)[::-1]   # high -> low
    chosen = float(lambdas[0])
    for lam in lambdas:
        if _risk_at(lam, typ, loss, n) <= alpha:
            chosen = float(lam)      # still safe at this (lower) threshold; keep descending
        else:
            break                    # first violation: stop, keep the last safe threshold
    return chosen


@dataclass
class CRCResult:
    alpha: float
    lambda_hat: float
    test_risk: float
    served_fraction: float

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "lambda_hat": self.lambda_hat,
                "test_risk": self.test_risk, "served_fraction": self.served_fraction}


def crc_serve_abstain(typicality_cal: np.ndarray, loss_cal: np.ndarray,
                      typicality_test: np.ndarray, loss_test: np.ndarray,
                      alpha: float) -> CRCResult:
    """Calibrate lambda on (cal), apply to (test), report empirical test risk over served points."""
    lam = calibrate_lambda(typicality_cal, loss_cal, alpha)
    served = np.asarray(typicality_test, dtype=float) >= lam
    lt = np.asarray(loss_test, dtype=float)
    risk = float(lt[served].mean()) if served.sum() > 0 else 0.0
    return CRCResult(alpha, lam, risk, float(served.mean()))


__all__ = ["calibrate_lambda", "crc_serve_abstain", "CRCResult"]

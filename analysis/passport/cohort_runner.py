"""Cohort-agnostic PORTRAIT runner: generalises orchestrate.run() to any tabular cohort.

Imports the live analysis classes (no reimplementation). Produces a bundle whose shape is IDENTICAL
to cohort_bundle_full.json so the built app renders it with zero changes, and a validate() that scores
the gate coverage + coherence residual + abstention against pre-registered thresholds. No figures.

Governance: this runner is generic. Whether a given cohort's outputs may be committed is a per-cohort
decision (e.g. MIMIC-IV outputs are LOCAL-ONLY under its DUA and must never enter submission/).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple, Dict, List
import hashlib, json, datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from analysis.surprise.conditional import ConditionalSurprise
from analysis.describability_gate.describability import DescribabilityGate
from analysis.coherence.profile import ProfileCoherence
from analysis.passport.passport import PassportConfig, build_passport


@dataclass
class CohortSpec:
    name: str
    feature_ids: List[str]
    axis_names: Dict[str, str]
    units: Dict[str, str]
    reference_intervals: Dict[str, Tuple[float, float]]
    domain_label: str
    id_col: Optional[str] = None
    age_col: Optional[str] = None
    sex_col: Optional[str] = None
    slice_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None   # None => single slice 0
    alpha: float = 0.10
    min_cal: int = 50
    seed: int = 0


def _natfreq(centile: float) -> str:
    c = max(0.0, min(1.0, centile))
    if c <= 0.5:
        return f"about {max(1, round(c*100))} in 100 are this low or lower"
    return f"about {max(1, round((1-c)*100))} in 100 are this high or higher"


def _prov(spec: CohortSpec, df_hash: str) -> dict:
    return {"cohort": spec.name, "method_tags": [spec.name],
            "config_sha256": hashlib.sha256(json.dumps(
                {"feat": list(spec.feature_ids), "alpha": spec.alpha,
                 "domain": spec.domain_label}, sort_keys=True).encode()).hexdigest(),
            "data_hash": df_hash, "seed": spec.seed,
            "generated_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}


def build(df: pd.DataFrame, spec: CohortSpec) -> dict:
    # Generalised bundle builder: accepts any tabular cohort conforming to CohortSpec.
    feats = list(spec.feature_ids)
    df = df.dropna(subset=feats).reset_index(drop=True)
    n, d = len(df), len(feats)
    X_raw = df[feats].to_numpy(float)
    df_hash = hashlib.sha256(X_raw.tobytes()).hexdigest()[:16]
    sl = (np.zeros(n, int) if spec.slice_fn is None
          else np.asarray(spec.slice_fn(df)).astype(int))

    tr, te = train_test_split(np.arange(n), test_size=0.5, random_state=spec.seed)
    ref_raw = X_raw[tr]
    scaler = StandardScaler().fit(ref_raw)
    Z = scaler.transform(X_raw)
    cs = ConditionalSurprise(random_state=spec.seed).fit(Z[tr]).calibrate(Z[tr], alpha=spec.alpha)
    gate = DescribabilityGate(cs, alpha=spec.alpha, min_cal=spec.min_cal).calibrate(Z[tr], sl[tr])
    pc = ProfileCoherence().fit(ref_raw)
    prov = _prov(spec, df_hash)

    cfg = PassportConfig(feature_ids=feats, axis_names=dict(spec.axis_names),
                         units=dict(spec.units), reference_intervals=dict(spec.reference_intervals),
                         alpha=spec.alpha, domain_label=spec.domain_label)
    ids = df[spec.id_col].astype(str).tolist() if spec.id_col else [str(i) for i in range(n)]
    ages = df[spec.age_col].to_numpy() if spec.age_col else [None] * n
    sexes = df[spec.sex_col].to_numpy() if spec.sex_col else [None] * n

    patients = []
    for i in range(n):
        ps = build_passport(ref_raw, X_raw[i], int(sl[i]), cfg, scaler=scaler, gate=gate,
                            surprise=cs, coherence=pc, reference_n=len(tr), provenance=prov)
        rec = ps.to_dict()
        rec["id"] = f"P-{ids[i]}"
        rec["age"] = None if ages[i] is None else float(ages[i])
        rec["sex"] = None if sexes[i] is None else str(int(sexes[i]))
        rec["slice"] = int(sl[i])
        axes = rec.get("axes", [])
        nsurp = sum(1 for a in axes if a.get("surprising"))
        cohd = rec.get("coherence", {})
        rec["n_surprising"] = nsurp
        rec["coherence_fired"] = bool(cohd.get("assessed") and (cohd.get("centile") or 0) >= 90)
        rec["coherence_centile"] = cohd.get("centile") if cohd.get("assessed") else None
        for a in axes:
            a["natfreq"] = _natfreq(a.get("centile", 0.5))
        if rec["state"] == "DESCRIBE" and axes:
            so = [{"feature": a["feature"], "axis": a["axis"], "value": a["value"], "unit": a["unit"],
                   "centile": a["centile"], "direction": "high" if a["centile"] > 0.5 else "low",
                   "natfreq": _natfreq(a["centile"])} for a in axes if a.get("surprising")]
            so.sort(key=lambda s: abs(s["centile"] - 0.5), reverse=True)
            rec["standout"] = so
        patients.append(rec)

    from collections import Counter
    states = Counter(p["state"] for p in patients)
    describ = np.array([p["state"] == "DESCRIBE" for p in patients])
    overview = {"n_total": n, "n_real": n, "states": dict(states),
                "describable_rate": round(float(describ.mean()), 3),
                "abstain_rate": round(float(1 - describ.mean()), 3),
                "coherence_fired": int(sum(p["coherence_fired"] for p in patients)),
                "n_surprising_hist": {str(k): int(v) for k, v in
                                      sorted(Counter(p["n_surprising"] for p in patients).items())},
                "n_slices": int(len(np.unique(sl))), "n_features": d,
                "reference_n": int(len(tr))}
    bundle = {"axis_names": dict(spec.axis_names), "units": dict(spec.units),
              "reference_intervals": {k: list(v) for k, v in spec.reference_intervals.items()},
              "provenance": prov, "reference_n": int(len(tr)), "n_patients": n,
              "domain_label": spec.domain_label, "overview": overview, "patients": patients,
              "_internal": {"Z": Z, "sl": sl, "te": te, "tr": tr, "describe": describ,
                            "surprise": cs, "coherence": pc, "X_raw": X_raw, "d": d}}
    return bundle


def validate(bundle: dict, spec: CohortSpec, *, cov_lo=0.85, cov_hi=0.95,
             abstain_budget=0.20, ks_p_min=0.05, ortho_max=0.20) -> dict:
    """Score the built bundle against pre-registered WIN/KILL thresholds.
    
    Computes gate coverage (acceptance rate on held-out test data), coherence clean-residual
    normality, abstention rate, and orthogonality of coherence centile to surprising-feature count.
    Returns a dict with per-check results and overall verdict.
    """
    it = bundle["_internal"]; Z, sl, te, tr, d = it["Z"], it["sl"], it["te"], it["tr"], it["d"]
    cs, pc, X_raw = it["surprise"], it["coherence"], it["X_raw"]
    # Per-slice gate coverage = acceptance rate on the held-out TEST half (targets ~1-alpha).
    # This is the registered quantity: for a calibrated gate on exchangeable data, the fraction of
    # test records in a slice that are describable approximates (1-alpha).
    gate = DescribabilityGate(cs, alpha=spec.alpha, min_cal=spec.min_cal).calibrate(Z[tr], sl[tr])
    cov, fails = {}, False
    for s in np.unique(sl[te]):
        rows = te[sl[te] == s]
        if len(rows) < spec.min_cal:
            continue
        c = float(gate.describe(Z[rows], sl[rows]).mean())
        cov[int(s)] = round(c, 3)
        fails |= not (cov_lo <= c <= cov_hi)
    # Coherence clean-residual: per-feature standardized residual r_j ~ N(0,1) on reference
    R = pc.residuals(X_raw[it["tr"]]).ravel()
    ks_p = float(stats.kstest((R - R.mean()) / R.std(), "norm").pvalue)
    # Orthogonality of coherence centile to extremity (n_surprising) on describable rows
    mask = it["describe"]
    coh = np.array([p["coherence_centile"] for p in bundle["patients"]], float)
    nsurp = np.array([p["n_surprising"] for p in bundle["patients"]], float)
    good = mask & ~np.isnan(coh)
    rho = float(stats.spearmanr(coh[good], nsurp[good]).correlation) if good.sum() > 5 else 0.0
    abstain = bundle["overview"]["abstain_rate"]
    checks = {
        "coverage": {"per_slice": cov, "PASS": bool(cov) and not fails},
        "abstention": {"rate": abstain, "PASS": 0.02 <= abstain <= abstain_budget},
        "coherence_ks": {"p": round(ks_p, 4), "PASS": ks_p > ks_p_min},
        "orthogonality": {"rho": round(rho, 3), "PASS": abs(rho) < ortho_max},
    }
    return {"cohort": spec.name,
            "thresholds": {"cov": [cov_lo, cov_hi], "abstain_budget": abstain_budget,
                           "ks_p_min": ks_p_min, "ortho_max": ortho_max},
            "checks": checks, "verdict": "PASS" if all(c["PASS"] for c in checks.values()) else "FAIL"}


def strip_internal(bundle: dict) -> dict:
    """Return a JSON-serialisable copy without the _internal numpy payload (for writing the app bundle)."""
    return {k: v for k, v in bundle.items() if k != "_internal"}

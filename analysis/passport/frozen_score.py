"""Frozen-background scoring for PORTRAIT (deployment-correct design).

Fit the reference ONCE on a large pooled population, freeze it, then score any new
individual or cohort against that fixed background. Both the score-a-new-individual
tool and the cohort browser consume the SAME exported model, so they are provably
scoring against an identical background.

Primary result POSITIVE: gate coverage all 6 age-sex slices in [0.85,0.95] on held-out
test cycle scored against the frozen n=9421 reference (results/frozen_bg/).

Purity contract: this module does ALL statistics in Python; the renderer only displays
precomputed PatientPassport objects. No statistics ever run in the browser.
"""
from __future__ import annotations
import hashlib, pickle
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.preprocessing import StandardScaler

from analysis.surprise.conditional import ConditionalSurprise
from analysis.describability_gate.describability import DescribabilityGate
from analysis.coherence.profile import ProfileCoherence
from analysis.passport.passport import PassportConfig, build_passport, render_passport

# --- NHANES cardiometabolic domain config (same frozen 12-feature panel) ---
FEATURES = ["BMXBMI", "BMXWAIST", "BPXSY1", "BPXDI1", "LBXGH", "LBXGLU",
            "LBDHDD", "LBXTC", "LBXTR", "LBXSCR", "LBXSAL", "LBXSATSI"]
AXIS_NAMES = {"BMXBMI": "Body-mass index", "BMXWAIST": "Waist circumference",
              "BPXSY1": "Systolic BP", "BPXDI1": "Diastolic BP", "LBXGH": "HbA1c",
              "LBXGLU": "Fasting glucose", "LBDHDD": "HDL cholesterol", "LBXTC": "Total cholesterol",
              "LBXTR": "Triglycerides", "LBXSCR": "Creatinine", "LBXSAL": "Albumin", "LBXSATSI": "ALT"}
UNITS = {"BMXBMI": "kg/m2", "BMXWAIST": "cm", "BPXSY1": "mmHg", "BPXDI1": "mmHg", "LBXGH": "%",
         "LBXGLU": "mg/dL", "LBDHDD": "mg/dL", "LBXTC": "mg/dL", "LBXTR": "mg/dL",
         "LBXSCR": "mg/dL", "LBXSAL": "g/dL", "LBXSATSI": "U/L"}
REF_INTERVALS = {"LBXGH": (4.0, 5.6), "LBXGLU": (70, 99), "BPXSY1": (90, 120), "LBDHDD": (40, 100),
                 "LBXTC": (0, 200), "BMXBMI": (18.5, 25), "LBXSAL": (3.5, 5.0)}


def slice_of(age, sex):
    """NHANES age x sex slice id = digitize(age,[40,60])*10 + sex."""
    age = np.asarray(age); sex = np.asarray(sex).astype(int)
    return (np.digitize(age, [40, 60]) * 10 + sex).astype(int)


def _natfreq(centile: float) -> str:
    """Plain-language natural-frequency gloss for a 0..1 centile."""
    c = max(0.0, min(1.0, float(centile)))
    if c <= 0.5:
        n = max(1, round(c * 100)); return f"about {n} in 100 adults are this low or lower"
    n = max(1, round((1 - c) * 100)); return f"about {n} in 100 adults are this high or higher"


@dataclass
class FrozenModel:
    """A frozen reference background: raw reference matrix + fitted slots + config."""
    ref_raw: np.ndarray
    scaler: object
    surprise: object
    gate: object
    coherence: object
    cfg: PassportConfig
    reference_n: int
    reference_hash: str
    provenance: dict = field(default_factory=dict)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


def build_frozen_model(ref_raw, slices_ref, *, cfg=None, seed=0, provenance=None) -> FrozenModel:
    """Fit scaler + surprise + describability gate (in scaled space) + coherence (in raw space)
    ONCE on the pooled reference, then freeze. slices_ref = per-row slice ids for the reference."""
    ref_raw = np.asarray(ref_raw, float)
    if cfg is None:
        cfg = PassportConfig(feature_ids=list(FEATURES), axis_names=dict(AXIS_NAMES),
                             units=dict(UNITS), reference_intervals=dict(REF_INTERVALS),
                             alpha=0.10, domain_label="clinical (NHANES cardiometabolic)")
    sc = StandardScaler().fit(ref_raw)
    Z = sc.transform(ref_raw)
    cs = ConditionalSurprise(random_state=seed).fit(Z).calibrate(Z, alpha=cfg.alpha)
    gate = DescribabilityGate(cs, alpha=cfg.alpha, min_cal=50).calibrate(Z, np.asarray(slices_ref).astype(int))
    pc = ProfileCoherence().fit(ref_raw)
    rhash = hashlib.sha256(ref_raw.tobytes()).hexdigest()[:16]
    prov = dict(provenance or {}); prov.setdefault("reference_hash", rhash)
    prov.setdefault("reference_n", int(len(ref_raw)))
    prov.setdefault("design", "frozen-background (fit once, score many)")
    prov.setdefault("reference_design", "frozen-background")
    return FrozenModel(ref_raw=ref_raw, scaler=sc, surprise=cs, gate=gate, coherence=pc,
                       cfg=cfg, reference_n=int(len(ref_raw)), reference_hash=rhash, provenance=prov)


def _enrich(rec: dict) -> dict:
    """Add n_surprising, coherence flags, natfreq glosses, and standout list.
    Mirrors cohort_runner.build() per-record enrichment so the app bundle is identical."""
    axes = rec.get("axes", [])
    rec["n_surprising"] = sum(1 for a in axes if a.get("surprising"))
    cohd = rec.get("coherence", {}) or {}
    rec["coherence_fired"] = bool(cohd.get("assessed") and (cohd.get("centile") or 0) >= 90)
    rec["coherence_centile"] = cohd.get("centile") if cohd.get("assessed") else None
    for a in axes:
        a["natfreq"] = _natfreq(a.get("centile", 0.5))
    if rec.get("state") == "DESCRIBE" and axes:
        so = [{"feature": a["feature"], "axis": a["axis"], "value": a["value"], "unit": a["unit"],
               "centile": a["centile"], "direction": "high" if a["centile"] > 0.5 else "low",
               "natfreq": _natfreq(a["centile"])} for a in axes if a.get("surprising")]
        so.sort(key=lambda s: abs(s["centile"] - 0.5), reverse=True)
        rec["standout"] = so
    return rec


def score_record(model: FrozenModel, x_row, slice_id, *, force_refuse=False) -> dict:
    """Score ONE individual against the frozen background -> enriched passport dict."""
    ps = build_passport(model.ref_raw, np.asarray(x_row, float), int(slice_id), model.cfg,
                        scaler=model.scaler, gate=model.gate, surprise=model.surprise,
                        coherence=model.coherence, reference_n=model.reference_n,
                        provenance=model.provenance, force_refuse=force_refuse)
    return _enrich(ps.to_dict())


def render_record_html(model: FrozenModel, x_row, slice_id, *, force_refuse=False) -> str:
    """Score ONE individual and return the WCAG-2.2-AA Passport HTML."""
    ps = build_passport(model.ref_raw, np.asarray(x_row, float), int(slice_id), model.cfg,
                        scaler=model.scaler, gate=model.gate, surprise=model.surprise,
                        coherence=model.coherence, reference_n=model.reference_n,
                        provenance=model.provenance, force_refuse=force_refuse)
    return render_passport(ps, model.cfg)


def score_cohort(model: FrozenModel, X_raw, ids, ages, sexes, slices) -> dict:
    """Score a whole cohort against the frozen background -> bundle in the app's shape.
    None of these individuals need to be in the reference (that is the point)."""
    X_raw = np.asarray(X_raw, float); n = len(X_raw)
    slices = np.asarray(slices).astype(int)
    patients = []
    for i in range(n):
        rec = score_record(model, X_raw[i], int(slices[i]))
        rec["id"] = f"P-{ids[i]}"
        rec["age"] = None if ages[i] is None else float(ages[i])
        rec["sex"] = None if sexes[i] is None else str(int(sexes[i]))
        rec["slice"] = int(slices[i])
        patients.append(rec)
    describ = np.array([p["state"] == "DESCRIBE" for p in patients])
    strata = {}
    for p in patients:
        key = f"{p['slice']}"
        strata.setdefault(key, {"n": 0, "describe": 0})
        strata[key]["n"] += 1
        strata[key]["describe"] += int(p["state"] == "DESCRIBE")
    for k in strata:
        strata[k]["rate"] = round(strata[k]["describe"] / max(strata[k]["n"], 1), 3)
    overview = {"n_total": n, "n_real": n, "states": dict(Counter(p["state"] for p in patients)),
                "describable_rate": round(float(describ.mean()), 3),
                "abstain_rate": round(float(1 - describ.mean()), 3),
                "coherence_fired": int(sum(p["coherence_fired"] for p in patients)),
                "n_surprising_hist": {str(k): int(v) for k, v in
                                      sorted(Counter(p["n_surprising"] for p in patients).items())},
                "strata": strata, "n_slices": int(len(np.unique(slices))),
                "n_features": int(X_raw.shape[1]), "reference_n": model.reference_n}
    return {"axis_names": dict(model.cfg.axis_names), "units": dict(model.cfg.units),
            "reference_intervals": {k: list(v) for k, v in model.cfg.reference_intervals.items()},
            "provenance": model.provenance, "reference_n": model.reference_n, "n_patients": n,
            "domain_label": model.cfg.domain_label, "overview": overview, "patients": patients}

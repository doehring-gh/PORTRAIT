"""Orchestrator: chain the clinical assessment gates into per-patient NHANES
Patient Passports and render all three honest states.

Wired onto the post-validation slots:
  - Describability gate (per-patient DESCRIBE/ABSTAIN) - gates narrative summary
  - Profile coherence (+ attribution) on DESCRIBE
  - Conditional per-feature surprise on DESCRIBE
  - Midrank position + DKW uncertainty band (always shown)
Deterministic (seeded). Writes results/passports/.
"""
from __future__ import annotations

import hashlib
import json
import os
import datetime
import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from analysis.nhanes.cohort import load_cycle, FEATURES
from analysis.describability_gate.describability import DescribabilityGate
from analysis.surprise.conditional import ConditionalSurprise
from analysis.coherence.profile import ProfileCoherence
from analysis.passport.passport import PassportConfig, build_passport, render_passport

RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "passports"))

AXIS_NAMES = {
    "BMXBMI": "Body-mass index", "BMXWAIST": "Waist circumference",
    "BPXSY1": "Systolic BP", "BPXDI1": "Diastolic BP", "LBXGH": "HbA1c",
    "LBXGLU": "Fasting glucose", "LBDHDD": "HDL cholesterol", "LBXTC": "Total cholesterol",
    "LBXTR": "Triglycerides", "LBXSCR": "Creatinine", "LBXSAL": "Albumin", "LBXSATSI": "ALT",
}
UNITS = {"BMXBMI": "kg/m2", "BMXWAIST": "cm", "BPXSY1": "mmHg", "BPXDI1": "mmHg", "LBXGH": "%",
         "LBXGLU": "mg/dL", "LBDHDD": "mg/dL", "LBXTC": "mg/dL", "LBXTR": "mg/dL", "LBXSCR": "mg/dL",
         "LBXSAL": "g/dL", "LBXSATSI": "U/L"}
# display-only clinical reference intervals (adult, generic; not diagnostic thresholds)
REF_INTERVALS = {"LBXGH": (4.0, 5.6), "LBXGLU": (70, 99), "BPXSY1": (90, 120), "LBDHDD": (40, 100),
                 "BMXBMI": (18.5, 24.9)}

METHOD_TAGS = ["describability-gate", "profile-coherence", "passport-render"]


def _clinical_config():
    return PassportConfig(feature_ids=list(FEATURES), axis_names=dict(AXIS_NAMES), units=dict(UNITS),
                          reference_intervals=dict(REF_INTERVALS), alpha=0.10, domain_label="clinical (NHANES cardiometabolic)")


def _slice_of(age, sex):
    band = int(np.digitize(age, [40, 60]))
    return band * 10 + int(sex)


def _fit_slots(ref_raw, slices_tr, seed=0):
    """Fit scaler + conditional surprise + describability gate (in scaled space) + profile coherence (in raw space)."""
    sc = StandardScaler().fit(ref_raw)
    Z = sc.transform(ref_raw)
    cs = ConditionalSurprise(random_state=seed).fit(Z).calibrate(Z, alpha=0.10)
    gate = DescribabilityGate(cs, alpha=0.10, min_cal=50).calibrate(Z, slices_tr)
    pc = ProfileCoherence().fit(ref_raw)
    return sc, cs, gate, pc


def _prov(config, seed):
    code = hashlib.sha256(open(os.path.join(os.path.dirname(__file__), "passport.py")).read().encode()).hexdigest()
    cfg = hashlib.sha256(json.dumps({"feat": config.feature_ids, "alpha": config.alpha,
                                     "domain": config.domain_label}, sort_keys=True).encode()).hexdigest()
    return {"method_tags": METHOD_TAGS, "code_sha256": code, "config_sha256": cfg,
            "data_hash": "NHANES cycle-G complete reference (public)", "seed": seed,
            "generated_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}


def _cov_matched_null(X, rng):
    mu = X.mean(axis=0); cov = np.cov(X, rowvar=False)
    w, V = np.linalg.eigh(cov); w = np.clip(w, 1e-9, None)
    root = V @ np.diag(np.sqrt(w)) @ V.T
    return mu + rng.standard_normal(size=X.shape) @ root.T


def run(seed: int = 0, write: bool = True):
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(seed)
    coh = load_cycle("G")
    ref_raw = coh.X_raw[coh.complete_mask]
    age = coh.age[coh.complete_mask]; sex = coh.sex[coh.complete_mask]
    slices = np.array([_slice_of(a, s) for a, s in zip(age, sex)])
    tr, te = train_test_split(np.arange(len(ref_raw)), test_size=0.5, random_state=seed)
    sc, cs, gate, pc = _fit_slots(ref_raw[tr], slices[tr], seed=seed)
    cfg = _clinical_config()
    prov = _prov(cfg, seed)

    # exemplar selection on the held-out half (deterministic)
    Zte = sc.transform(ref_raw[te]); desc = gate.describe(Zte, slices[te])
    C = pc.centile(ref_raw[te]); cen = np.linalg.norm((ref_raw[te] - ref_raw[tr].mean(0)) / ref_raw[tr].std(0), axis=1)
    cen_pct = np.array([(cen < c).mean() for c in cen])
    _, R = pc.attribution(ref_raw[te], top=2); maxr = np.abs(R).max(1)
    gluc = FEATURES.index("LBXGLU")

    # DESCRIBE = the incoherent exemplar: describable, non-extreme, big single-feature residual, glucose driver
    inc = np.where(desc & (cen_pct < 0.60) & (maxr > 3))[0]
    inc = inc[np.argsort(-maxr[inc])]
    i_desc = next((i for i in inc if abs(R[i, gluc]) > 3), inc[0])
    # ABSTAIN = a real undescribable patient
    i_abst = np.where(~desc)[0][0]
    # REFUSE = covariance-matched null
    null_raw = _cov_matched_null(ref_raw[tr], rng)

    out = {}
    specs = [("describe", te[i_desc], False), ("abstain", te[i_abst], False)]
    passports = {}
    for name, gi, _fr in specs:
        ps = build_passport(ref_raw[tr], ref_raw[gi], int(slices[gi]), cfg, scaler=sc, gate=gate,
                            surprise=cs, coherence=pc, reference_n=len(tr), provenance=prov)
        passports[name] = ps
    # REFUSE on the null reference
    ps_ref = build_passport(null_raw, null_raw[0], int(slices[tr][0]), cfg, scaler=sc, gate=gate,
                            surprise=cs, coherence=pc, reference_n=len(null_raw), provenance=prov,
                            force_refuse=True)
    passports["refuse"] = ps_ref

    titles = {"describe": "PORTRAIT Patient Passport - NHANES (describable)",
              "abstain": "PORTRAIT Patient Passport - NHANES (abstain)",
              "refuse": "PORTRAIT Patient Passport - covariance-matched null (refuse)"}
    for name, ps in passports.items():
        html_doc = render_passport(ps, cfg, title=titles[name])
        if write:
            with open(os.path.join(RESULTS, f"passport_{name}.html"), "w") as fh:
                fh.write(html_doc)
            with open(os.path.join(RESULTS, f"passport_{name}.json"), "w") as fh:
                json.dump(ps.to_dict(), fh, indent=2, default=float)
        out[name] = {"state": ps.state, "describable": ps.describable,
                     "coherence_assessed": ps.coherence.get("assessed"),
                     "file": f"passport_{name}.html"}
    if write:
        with open(os.path.join(RESULTS, "s7_summary.json"), "w") as fh:
            json.dump({"seed": seed, "reference_n": len(tr), "passports": out,
                       "exemplar_glucose_residual": float(R[i_desc, gluc])}, fh, indent=2, default=float)
    return passports, out


if __name__ == "__main__":
    _, summary = run()
    print(json.dumps(summary, indent=2, default=float))

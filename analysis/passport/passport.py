"""PORTRAIT Patient Passport (data contract + pure build + pure render).

Introduces ONE per-patient document composed of:
  - describability gate (analysis.describability_gate.describability.DescribabilityGate) -> per-patient state
  - conditional surprise (analysis.surprise.conditional.ConditionalSurprise) -> per-feature flag
  - profile coherence (analysis.coherence.profile.ProfileCoherence) -> coherence + drivers
  - position centile (analysis.position.position.midrank_centile) -> marginal position
  - DKW band (analysis.uncertainty.uncertainty.dkw_band) -> guaranteed halfwidth

No atypicality detector. build_passport() computes; render_passport() is
PURE (no statistics) so the determinism test can pin the HTML byte-for-byte.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field, asdict

import numpy as np

from analysis.position.position import midrank_centile
from analysis.uncertainty.uncertainty import dkw_band


# ---------------------------------------------------------------- data contract

@dataclass
class PassportConfig:
    """Domain-agnostic configuration. A new domain changes ONLY this object, never the code."""
    feature_ids: list                       # ordered feature ids (columns of the reference matrix)
    axis_names: dict = field(default_factory=dict)          # id -> human label
    units: dict = field(default_factory=dict)               # id -> unit string ("" allowed)
    reference_intervals: dict = field(default_factory=dict)  # id -> (lo, hi), display-only, optional
    alpha: float = 0.10
    domain_label: str = "generic"

    def name(self, fid):
        return self.axis_names.get(fid, fid)

    def unit(self, fid):
        return self.units.get(fid, "")


@dataclass
class AxisCell:
    feature: str
    axis: str
    value: float
    unit: str
    centile: float                 # marginal population centile
    band_lo: float
    band_hi: float
    band_halfwidth: float          # DKW guaranteed halfwidth (marginal)
    ref_interval: tuple = None     # (lo, hi) display-only clinical reference range, optional
    surprise_centile: float = None  # conditional centile given the rest
    surprising: bool = False


@dataclass
class PatientPassport:
    state: str                                  # DESCRIBE | ABSTAIN | REFUSE
    describable: bool
    axes: list = field(default_factory=list)    # list[AxisCell]
    coherence: dict = field(default_factory=dict)
    abstention: dict = field(default_factory=dict)
    reference_n: int = 0
    provenance: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self):
        d = asdict(self)
        return d


# ---------------------------------------------------------------- pure builder

BANNED = ("diagnos", "treat", "prescrib", "you have", "risk of", "cure", "should see a")
DISCLAIMER = ("Research characterisation aid. Not a diagnostic or predictive device. Population "
              "position with calibrated marginal intervals; colour is never the sole channel.")
COHERENCE_HONESTY = ("Descriptive axis (Gaussian-copula person-fit); not a detector, not a diagnosis. "
                     "Reported only where the describability gate certifies coverage.")


def build_passport(reference, x_row, slice_id, config, *, scaler, gate, surprise=None, coherence=None,
                   reference_n=None, provenance=None, force_refuse=False, seed=0):
    """Compute the PatientPassport for ONE patient. Pure of I/O; all statistics happen here.

    reference : (n_ref, d) raw complete-row reference cohort (feature order = config.feature_ids)
    x_row     : (d,) the patient's raw feature vector
    slice_id  : the describability slice the patient maps to (from the caller's slice_fn)
    scaler    : the fitted transform the gate/surprise were calibrated in (travels WITH the gate)
    gate      : fitted analysis.describability_gate.describability.DescribabilityGate
    surprise  : fitted+calibrated analysis.surprise.conditional.ConditionalSurprise (DESCRIBE panels)
    coherence : fitted analysis.coherence.profile.ProfileCoherence (DESCRIBE panels)
    force_refuse : render the REFUSE state (reference is a covariance-matched null probe)
    """
    reference = np.asarray(reference, float)
    x = np.asarray(x_row, float).ravel()
    d = len(config.feature_ids)
    n_ref = int(reference_n if reference_n is not None else len(reference))
    prov = dict(provenance or {})
    prov.setdefault("seed", seed)

    # ---- STEP A: REFUSE (cohort-level demonstration/guard) ----
    if force_refuse:
        return PatientPassport(state="REFUSE", describable=False, reference_n=n_ref, provenance=prov,
                               note=("Input reference is a covariance-matched null: no structure beyond "
                                     "second order. PORTRAIT refuses to describe."))

    # ---- Panel 2 (ALWAYS): marginal position + DKW band ----
    axes = []
    for j, fid in enumerate(config.feature_ids):
        ref_j = reference[:, j]
        c = float(midrank_centile(ref_j, np.array([x[j]]))[0])
        hw = float(dkw_band(len(ref_j), config.alpha))
        axes.append(AxisCell(feature=fid, axis=config.name(fid), value=float(x[j]), unit=config.unit(fid),
                             centile=c, band_lo=max(0.0, c - hw), band_hi=min(1.0, c + hw),
                             band_halfwidth=hw,
                             ref_interval=tuple(config.reference_intervals[fid])
                             if fid in config.reference_intervals else None))

    # ---- STEP B: per-patient describability in the SCALED space (scaler travels with the gate) ----
    Xg = scaler.transform(x.reshape(1, -1))
    describable = bool(gate.describe(Xg, np.array([slice_id]))[0])

    if not describable:
        # ---- STEP C: ABSTAIN ----
        drivers = []
        if surprise is not None:
            mags = []
            for j in range(d):
                sc = float(surprise.centile(Xg, j)[0])
                mags.append((config.feature_ids[j], abs(sc - 0.5) * 2.0))
            tot = sum(m for _, m in mags) or 1.0
            drivers = [{"feature": f, "share": round(m / tot, 3)}
                       for f, m in sorted(mags, key=lambda t: -t[1])[:3]]
        return PatientPassport(
            state="ABSTAIN", describable=False, axes=axes,
            coherence={"assessed": False,
                       "note": "Coherence not assessed - patient outside the describable region."},
            abstention={"reason": ("The describability gate abstains for this patient: their profile "
                                   "falls outside the region where PORTRAIT can certify coverage."),
                        "drivers": drivers},
            reference_n=n_ref, provenance=prov,
            note=("PORTRAIT reports marginal position only and abstains from a rich, gated description "
                  "for this patient."))

    # ---- STEP D: DESCRIBE (rich) ----
    if surprise is not None:
        lo, hi = config.alpha / 2.0, 1.0 - config.alpha / 2.0
        for j, cell in enumerate(axes):
            sc = float(surprise.centile(Xg, j)[0])
            cell.surprise_centile = sc
            cell.surprising = bool(sc <= lo or sc >= hi)
    coh = {"assessed": False}
    if coherence is not None:
        cent = float(coherence.centile(x.reshape(1, -1))[0])
        (idx2,), R = coherence.attribution(x.reshape(1, -1), top=2)
        r = R[0]
        absr = np.abs(r)
        concentration = float(absr.max() / (np.sqrt((r ** 2).sum()) + 1e-9))
        coh = {"assessed": True, "centile": cent, "concentration": concentration,
               "drivers": [{"feature": config.feature_ids[j], "value": float(x[j]),
                            "residual": float(r[j]), "impossible": bool(abs(r[j]) > 2)} for j in idx2],
               "note": COHERENCE_HONESTY}
    return PatientPassport(state="DESCRIBE", describable=True, axes=axes, coherence=coh,
                           reference_n=n_ref, provenance=prov,
                           note=("This patient falls inside the describable region: PORTRAIT gives a "
                                 "gated description with per-feature surprise and a coherence read."))


# ---------------------------------------------------------------- pure renderer

# state -> (colour meeting >=4.5:1 on white, glyph, word). Colour NEVER the sole channel.
_STATE = {
    "DESCRIBE": ("#1b7837", "\u25CF", "DESCRIBE"),   # filled circle, dark green
    "ABSTAIN":  ("#8a5a00", "\u25D1", "ABSTAIN"),    # half circle, dark amber
    "REFUSE":   ("#6a1b9a", "\u25CB", "REFUSE"),     # open circle, dark purple
}


def _bar(c, lo, hi, label):
    """Deterministic inline SVG centile bar with a band, 0..1 -> 0..300px; aria-label for parity."""
    x = int(round(c * 300)); a = int(round(lo * 300)); b = int(round(hi * 300))
    return ('<svg width="320" height="18" role="img" aria-label="' + html.escape(label) + '">'
            '<rect x="0" y="7" width="300" height="4" fill="#c8c8c8"/>'
            '<rect x="' + str(a) + '" y="5" width="' + str(max(1, b - a)) + '" height="8" fill="#7a94b8"/>'
            '<circle cx="' + str(x) + '" cy="9" r="5" fill="#1b1b1b"/></svg>')


def render_passport(ps: PatientPassport, config: PassportConfig,
                    title: str = "PORTRAIT Patient Passport") -> str:
    """PURE render: PatientPassport -> self-contained WCAG-2.2-AA HTML. No statistics computed here."""
    colour, glyph, word = _STATE[ps.state]
    esc = html.escape
    UP = "&#9650;"      # up-triangle glyph (avoids a literal > ( sequence in source)

    axes_block = ""
    if ps.axes:
        rows = ""
        for a in ps.axes:
            cp = a.centile * 100.0
            surp = (' <span class="flag">' + UP + ' surprising given the rest</span>') if a.surprising else ""
            ref = ""
            if a.ref_interval is not None:
                ref = ('<div class="ref">ref ' + ("%g" % a.ref_interval[0]) + "\u2013"
                       + ("%g" % a.ref_interval[1]) + " " + esc(a.unit) + "</div>")
            bar_lbl = (a.axis + ": centile " + ("%.0f" % cp) + " of 100, marginal band plus or minus "
                       + ("%.0f" % (a.band_halfwidth * 100)))
            rows += ('<tr><th scope="row">' + esc(a.axis) + '<div class="val">' + ("%.1f" % a.value)
                     + " " + esc(a.unit) + "</div>" + ref + "</th>"
                     + "<td>" + _bar(a.centile, a.band_lo, a.band_hi, bar_lbl) + "</td>"
                     + '<td class="num">' + ("%.0f" % cp) + '<span class="pm"> &plusmn;'
                     + ("%.0f" % (a.band_halfwidth * 100)) + "</span>" + surp + "</td></tr>")
        axes_block = ('<h2>Position on named axes</h2>'
                      '<table><thead><tr><th scope="col">Axis (value)</th>'
                      '<th scope="col">Population centile (marginal band)</th>'
                      '<th scope="col">%</th></tr></thead><tbody>' + rows + "</tbody></table>"
                      '<p class="cap">Bands are population-<b>marginal</b> (DKW); they do not assert a '
                      'per-patient guarantee (conditional coverage is provably impossible).</p>')

    coh_block = ""
    if ps.state == "DESCRIBE" and ps.coherence.get("assessed"):
        cc = ps.coherence
        dl = ""
        for drv in cc["drivers"]:
            imp = " <b>(impossible given the rest)</b>" if drv["impossible"] else ""
            dl += ("<li>" + esc(config.name(drv["feature"])) + " = " + ("%.1f" % drv["value"]) + " "
                   + esc(config.unit(drv["feature"])) + " &mdash; conditional residual r="
                   + ("%+.2f" % drv["residual"]) + imp + "</li>")
        flagged = cc["centile"] >= 90
        cglyph = UP if flagged else "&#10003;"
        clabel = "unusual COMBINATION of features" if flagged else "internally coherent"
        coh_block = ('<h2>Profile coherence</h2>'
                     '<div class="coh ' + ("flag" if flagged else "ok") + '"><span class="glyph">'
                     + cglyph + "</span> <b>" + clabel + "</b> (coherence-deviation centile "
                     + ("%.0f" % cc["centile"]) + "/100; concentration " + ("%.2f" % cc["concentration"])
                     + ")." + '<div class="attr">Driven by:<ul>' + dl + "</ul></div>"
                     + '<div class="prov">' + esc(cc["note"]) + "</div></div>")

    ab_block = ""
    if ps.state == "ABSTAIN" and ps.abstention:
        ab = ps.abstention
        dl = "".join("<li>" + esc(config.name(x["feature"])) + " &mdash; share "
                     + ("%.0f%%" % (x["share"] * 100)) + "</li>" for x in ab.get("drivers", []))
        driver_ul = ('<div class="attr">Largest surprise contributors:<ul>' + dl + "</ul></div>") if dl else ""
        ab_block = ('<h2>Why PORTRAIT abstains here</h2>'
                    '<div class="coh abstain"><span class="glyph">&#9888;</span> ' + esc(ab["reason"])
                    + driver_ul + "</div>")

    p = ps.provenance
    prov_block = ('<div class="foot"><b>Provenance.</b> method: ' + esc(str(p.get("method_tags", "")))
                  + "; code " + esc(str(p.get("code_sha256", ""))[:12]) + "; config "
                  + esc(str(p.get("config_sha256", ""))[:12]) + "; data " + esc(str(p.get("data_hash", "")))
                  + "; seed " + esc(str(p.get("seed", ""))) + "; reference n=" + str(ps.reference_n) + ". "
                  + '<span class="gen">generated ' + esc(str(p.get("generated_utc", ""))) + "</span><br>"
                  + "Regenerate: <code>python -m analysis.passport.orchestrate</code>.</div>")

    css = ("body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:760px;margin:2rem auto;color:#1b1b1b;line-height:1.4}"
           "h1{font-size:1.35rem;margin-bottom:.3rem} h2{font-size:1.05rem;margin:1.4rem 0 .4rem}"
           ".state{display:inline-block;padding:.3rem .7rem;border:2px solid " + colour
           + ";border-radius:6px;color:" + colour + ";font-weight:700;letter-spacing:.03em}"
           ".sub{color:#555;font-size:.9rem}"
           "table{border-collapse:collapse;width:100%;margin-top:.4rem}"
           "td,th{padding:.35rem .5rem;border-bottom:1px solid #eee;font-size:.93rem;vertical-align:top}"
           "thead th{text-align:left;color:#555;font-weight:600}"
           "th[scope=row]{text-align:left;font-weight:600} .val{color:#444;font-weight:400;font-size:.85rem}"
           ".ref{color:#777;font-size:.78rem} .num{text-align:right;white-space:nowrap} .pm{color:#777}"
           ".flag{color:#8a5a00;font-weight:600} .cap{color:#777;font-size:.8rem;margin:.3rem 0 0}"
           ".coh{border-left:5px solid #8a8f98;padding:.7rem .9rem;margin:.4rem 0;background:#f7f8fa;border-radius:4px}"
           ".coh.ok{border-color:#1b7837} .coh.flag{border-color:#b00020} .coh.abstain{border-color:#8a5a00;background:#fdf7e8}"
           ".glyph{font-size:1.2em;margin-right:.3em} .attr ul{margin:.3rem 0 .3rem 1.2rem}"
           ".prov{font-size:.8rem;color:#666;margin-top:.5rem}"
           ".note{background:#f6f6f4;border-left:3px solid " + colour + ";padding:.6rem .8rem;margin-top:1rem;font-size:.9rem}"
           ".foot{margin-top:1.6rem;color:#888;font-size:.78rem} code{background:#eee;padding:.05rem .3rem;border-radius:3px}")

    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            "<title>" + esc(title) + "</title><style>" + css + "</style></head><body><main>")
    header = ("<h1>" + esc(title) + "</h1>"
              '<div class="state" aria-label="describability state ' + esc(word) + '">' + glyph
              + "&nbsp;" + esc(word) + "</div>"
              '<span class="sub">&nbsp;domain: ' + esc(config.domain_label) + " &middot; reference n="
              + str(ps.reference_n) + "</span>"
              '<div class="note">' + esc(ps.note) + "</div>")
    return (head + header + axes_block + coh_block + ab_block
            + '<p class="cap">' + esc(DISCLAIMER) + "</p>" + prov_block + "</main></body></html>")

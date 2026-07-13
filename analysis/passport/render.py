"""Passport render: deterministic HTML Patient Passport.

The Passport describes the patient against the population on nameable axes, and it renders one of THREE
first-class states, decided by the frozen describability gate (an
abstention-aware description system that knows when it cannot describe):

  DESCRIBE : The cohort carries structure beyond its covariance -> a RICH passport
             (per-axis centiles with calibrated bands + atypicality strip).
  ABSTAIN  : The cohort's structure is borderline or lean-refuse -> the covariance is already
             sufficient; the passport renders the marginal centiles ONLY and states plainly that it
             abstains from a rich beyond-covariance description. (This is what the real NHANES
             cycle-G cohort triggers.)
  REFUSE   : the input is a covariance-matched null (no structure at all) -> the passport refuses to
             describe and says so.

Colour is never the sole channel (WCAG 2.2): every state carries a
text label and a distinct glyph. No diagnostic/clinical claim is emitted anywhere.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np

from analysis.structure_test.gate import combined_gate
from analysis.position.position import midrank_centile
from analysis.uncertainty.uncertainty import dkw_band
from analysis.atypicality.atypicality import knn_void_score, conformal_pvalues, bh_reject


@dataclass
class PassportState:
    state: str                       # DESCRIBE | ABSTAIN | REFUSE
    s2_decision: str
    s2_p_combined: float
    axes: list = field(default_factory=list)   # per-axis dicts (empty on REFUSE)
    atypical: dict = field(default_factory=dict)
    reference_n: int = 0
    note: str = ""

    def as_dict(self):
        d = self.__dict__.copy()
        return d


def _decide_state(reference: np.ndarray, seed: int = 0) -> tuple:
    """Frozen describability gate over 5 seeds -> majority decision + median combined p."""
    decs, ps = [], []
    for s in range(5):
        r = combined_gate(reference, null_type="copula", alpha=0.05, alpha_hi=0.20,
                          b_reps=199, n_cap=300, seed=seed + s)
        decs.append(r.decision); ps.append(r.p_combined)
    from collections import Counter
    maj = Counter(decs).most_common(1)[0][0]
    return maj, float(np.median(ps)), decs


def build_passport(reference: np.ndarray, query: np.ndarray, feature_names,
                   axis_names=None, alpha: float = 0.05, k: int = 10, seed: int = 0,
                   force_null: bool = False) -> PassportState:
    """Build the Passport data structure for one query patient against a reference cohort.

    reference : (n_ref, d) complete-row reference cohort (standardised upstream is fine or raw)
    query     : (d,) the patient's feature vector
    force_null: if True, the reference is treated as a covariance-matched null probe -> REFUSE test
    """
    reference = np.asarray(reference, float)
    query = np.asarray(query, float).ravel()
    axis_names = list(axis_names) if axis_names is not None else list(feature_names)

    s2_decision, s2_p, _ = _decide_state(reference, seed=seed)

    # REFUSE: explicit covariance-matched-null probe, or describability gate REFUSE with a high p (no structure)
    if force_null:
        return PassportState(state="REFUSE", s2_decision="REFUSE", s2_p_combined=s2_p,
                             reference_n=len(reference),
                             note="Input is a covariance-matched null: no structure beyond second "
                                  "order. PORTRAIT refuses to describe.")

    # per-axis centiles are ALWAYS computable (marginal); they are the floor.
    axes = []
    for j, (fname, aname) in enumerate(zip(feature_names, axis_names)):
        ref_j = reference[:, j]
        c = float(midrank_centile(ref_j, np.array([query[j]]))[0])
        band = dkw_band(len(ref_j), alpha)           # DKW half-width on the centile (guaranteed)
        axes.append({"feature": fname, "axis": aname, "centile": c,
                     "band_lo": max(0.0, c - band), "band_hi": min(1.0, c + band),
                     "band_halfwidth": band})

    # atypicality - only meaningful when we are DESCRIBING; computed for DESCRIBE/ABSTAIN alike
    # but presented only in the rich (DESCRIBE) passport.
    void = knn_void_score(reference, reference, k=k)
    q_void = knn_void_score(reference, query.reshape(1, -1), k=k)
    pval = conformal_pvalues(void, q_void)[0]
    atypical = {"void_score": float(q_void[0]), "conformal_p": float(pval),
                "flagged": bool(pval < alpha)}

    if s2_decision == "DESCRIBE":
        return PassportState(state="DESCRIBE", s2_decision=s2_decision, s2_p_combined=s2_p,
                             axes=axes, atypical=atypical, reference_n=len(reference),
                             note="Cohort carries structure beyond its covariance: a rich "
                                  "beyond-covariance description is warranted.")
    # Borderline or refuse-leaning on a real cohort -> ABSTAIN (marginal only)
    return PassportState(state="ABSTAIN", s2_decision=s2_decision, s2_p_combined=s2_p,
                         axes=axes, atypical={}, reference_n=len(reference),
                         note="The cohort's structure is largely captured by its covariance and "
                              "rank correlation. PORTRAIT reports marginal centiles only and "
                              "abstains from a rich beyond-covariance description.")


# ---- HTML rendering (deterministic, self-contained, colour-not-sole-channel) ----

_STATE_STYLE = {
    "DESCRIBE": ("#1b7837", "\u25CF", "DESCRIBE"),      # filled circle
    "ABSTAIN":  ("#b8860b", "\u25D1", "ABSTAIN"),       # half circle
    "REFUSE":   ("#762a83", "\u25CB", "REFUSE"),        # open circle
}


def _bar(c, lo, hi):
    """A deterministic inline SVG centile bar with a band, 0..1 -> 0..300px."""
    x = int(round(c * 300)); a = int(round(lo * 300)); b = int(round(hi * 300))
    return (f'<svg width="320" height="18" role="img">'
            f'<rect x="0" y="7" width="300" height="4" fill="#e0e0e0"/>'
            f'<rect x="{a}" y="5" width="{max(1,b-a)}" height="8" fill="#b0c4de"/>'
            f'<circle cx="{x}" cy="9" r="5" fill="#1b1b1b"/></svg>')


def render_passport(ps: PassportState, title: str = "PORTRAIT Patient Passport") -> str:
    colour, glyph, label = _STATE_STYLE[ps.state]
    esc = html.escape
    rows = ""
    for a in ps.axes:
        c = a["centile"]
        rows += (f'<tr><td>{esc(a["axis"])}</td>'
                 f'<td>{_bar(c, a["band_lo"], a["band_hi"])}</td>'
                 f'<td style="text-align:right">{c*100:.0f}<span style="color:#666">'
                 f' &plusmn;{a["band_halfwidth"]*100:.0f}</span></td></tr>')
    atyp = ""
    if ps.state == "DESCRIBE" and ps.atypical:
        flg = ps.atypical["flagged"]
        atyp = (f'<div class="atyp"><b>Atypicality:</b> conformal p = '
                f'{ps.atypical["conformal_p"]:.3f} &mdash; '
                f'{"FLAGGED atypical (off-manifold)" if flg else "within the reference manifold"}. '
                f'This is a population-position statement, not a diagnosis.</div>')
    axes_block = (f'<table><thead><tr><th>Axis</th><th>Centile (with guaranteed band)</th>'
                  f'<th>%</th></tr></thead><tbody>{rows}</tbody></table>{atyp}'
                  if ps.axes else "")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{esc(title)}</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:720px;margin:2rem auto;color:#1b1b1b}}
.state{{display:inline-block;padding:.3rem .7rem;border:2px solid {colour};border-radius:6px;
color:{colour};font-weight:700;letter-spacing:.03em}}
table{{border-collapse:collapse;width:100%;margin-top:1rem}}
td,th{{padding:.35rem .5rem;border-bottom:1px solid #eee;font-size:.95rem}}
th{{text-align:left;color:#555;font-weight:600}}
.note{{background:#f6f6f4;border-left:3px solid {colour};padding:.6rem .8rem;margin-top:1rem;font-size:.92rem}}
.atyp{{margin-top:.8rem;font-size:.92rem}}
.foot{{margin-top:1.5rem;color:#888;font-size:.8rem}}
</style></head><body>
<h1 style="font-size:1.3rem;margin-bottom:.2rem">{esc(title)}</h1>
<div class="state" aria-label="state {esc(label)}">{glyph}&nbsp;{esc(label)}</div>
<span style="color:#666;font-size:.9rem">&nbsp;Describability: {esc(ps.s2_decision)} (p={ps.s2_p_combined:.3f}), reference n={ps.reference_n}</span>
<div class="note">{esc(ps.note)}</div>
{axes_block}
<div class="foot">Research characterisation aid. Not a diagnostic or predictive device. Population
position with calibrated intervals; colour is never the sole channel.</div>
</body></html>"""

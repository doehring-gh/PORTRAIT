"""PORTRAIT profile-coherence coordinate (descriptive).

Gaussian-copula person-fit: rank-transform each feature to normal scores (removes marginal position),
then the closed-form leave-one-out conditional distribution z_j | z_{-j} gives a per-feature standardized
residual r_j (marginally ~ N(0,1) under the Gaussian-copula null). Coordinate C(x) = sum_j r_j^2. The
d LOO residuals are correlated, so C is a quadratic form in correlated residuals: it is NOT chi^2_d and
is NOT the squared Mahalanobis distance z' R^-1 z (that identity holds only for sequential-Cholesky
residuals). C correlates empirically ~0.88 with copula-Mahalanobis but is deliberately not identical -
which is why it names DIFFERENT patients than raw extremity (orthogonality |rho|~0.44). C is calibrated by
the empirical reference centile, never the chi^2 tail. Attribution = argmax|r_j| = the feature(s) that fail
to cohere (symmetric, order-invariant). Coherence centile = reference percentile of C (higher = LESS
coherent).

Provenance: psychometric person-fit / appropriateness measurement (lz*, Snijders 2001); copula PIT
(actuarial/quant-finance); the closed-form conditional residual is standard multivariate-normal algebra.
NOVELTY is only the synthesis into a calibrated, attributable clinical DESCRIPTION axis. This is NOT a
detector: a covariance detector (Mahalanobis) ties it on any labelled discrimination task. Its value
is descriptive, not discriminative - the per-feature DECOMPOSITION a single
distance number cannot give, and the fact that it names DIFFERENT patients than extremity.
"""
import numpy as np
from scipy.stats import norm


def normal_scores(Xref_raw, Xmat):
    """Rank each column of Xmat within Xref_raw's marginal, map to normal scores (copula PIT)."""
    d = Xref_raw.shape[1]; nref = len(Xref_raw)
    out = np.empty((len(Xmat), d), float)
    for j in range(d):
        r = np.searchsorted(np.sort(Xref_raw[:, j]), Xmat[:, j])
        out[:, j] = norm.ppf(np.clip((r + 0.5) / (nref + 1), 1e-4, 1 - 1e-4))
    return out


class ProfileCoherence:
    def __init__(self, mode="loo"):
        # 'loo' (default): leave-one-out full-complement residuals (each z_j conditioned on ALL others). Symmetric
        #   and order-invariant attribution. C = sum_j r_j^2 is a quadratic form in correlated residuals; it is
        #   NOT the squared Mahalanobis distance and NOT chi^2_d, and is calibrated by the empirical reference
        #   centile. It correlates ~0.88 with copula-Mahalanobis but is deliberately not identical - that is why
        #   it names DIFFERENT patients than raw extremity (orthogonality |rho|~0.44).
        # 'cholesky': sequential residuals in panel order, r = L^-1 z (R = L L'); then C = z' R^-1 z EXACTLY (the
        #   copula-Mahalanobis identity, chi^2_d under the Gaussian-copula null), but the coordinate becomes
        #   essentially Mahalanobis (orthogonality ~0.82) and attribution is order-dependent. Retained for comparison.
        self.mode = mode

    def fit(self, Xref_raw):
        self.Xref = np.asarray(Xref_raw, float)
        Z = normal_scores(self.Xref, self.Xref)
        self.R = np.corrcoef(Z, rowvar=False)
        self.d = Z.shape[1]
        # leave-one-out conditioning (symmetric)
        self.cond = []
        for j in range(self.d):
            oth = [k for k in range(self.d) if k != j]
            Roo_inv = np.linalg.pinv(self.R[np.ix_(oth, oth)])
            beta = self.R[j, oth] @ Roo_inv
            s2 = max(1.0 - beta @ self.R[oth, j], 1e-6)
            self.cond.append((oth, beta, np.sqrt(s2)))
        # sequential-Cholesky whitening matrix in panel order: r = Z @ Linv.T, R = L L'
        self._Linv = np.linalg.inv(np.linalg.cholesky(self.R))
        self._Cref = self.coordinate(self.Xref)  # reference distribution for the centile transform
        return self

    def _loo_residuals(self, Z):
        R = np.empty_like(Z)
        for j, (oth, beta, s) in enumerate(self.cond):
            R[:, j] = (Z[:, j] - Z[:, oth] @ beta) / s
        return R

    def residuals(self, Xmat):
        """Per-feature standardized residual r_j ~ N(0,1) under the Gaussian-copula null.

        Cholesky mode: sequential residual (z_j conditioned on its predecessors in panel order); the vector
        r = L^-1 z whitens the copula, so sum_j r_j^2 = z' R^-1 z exactly. LOO mode: full-complement residual.
        """
        Z = normal_scores(self.Xref, np.asarray(Xmat, float))
        if getattr(self, "mode", "cholesky") == "cholesky":
            return Z @ self._Linv.T
        return self._loo_residuals(Z)

    def coordinate(self, Xmat):
        """C(x) = sum_j r_j^2. Cholesky mode: C = z' R^-1 z (squared copula-Mahalanobis), chi^2_d under the
        Gaussian-copula null; calibrated in practice by the empirical reference centile. Higher = less coherent."""
        return (self.residuals(Xmat) ** 2).sum(1)

    def centile(self, Xmat):
        """0-100 coherence-deviation centile vs the reference (100 = most incoherent)."""
        C = self.coordinate(Xmat)
        return 100.0 * np.searchsorted(np.sort(self._Cref), C) / len(self._Cref)

    def attribution(self, Xmat, top=2):
        """Indices of the top-`top` features by |r_j| (the features that fail to cohere)."""
        R = self.residuals(Xmat)
        return np.argsort(np.abs(R), 1)[:, -top:][:, ::-1], R


def coherence_panel(pc, gate, x_row, slice_id, feature_names, units=None, gate_transform=None):
    """Render the coherence axis as an HTML Passport panel for ONE patient.

    Chained to the describability gate: if the gate ABSTAINS for this record, no coherence claim is
    made (the panel says so). Otherwise it shows the calibrated coherence centile + the per-feature
    attribution (which features fail to cohere). No diagnostic claim; WCAG 2.2 - state carries a text
    label + glyph, never colour alone. Domain-agnostic: axis names/units read from arguments.

    IMPORTANT: `pc` (ProfileCoherence) rank-transforms internally, so it takes the RAW feature row.
    The gate is calibrated in a SEPARATE space (e.g. StandardScaler); pass `gate_transform` (a
    callable row->gate-space row) so the gate sees the space it was calibrated on. If omitted, the raw
    row is passed to the gate (only correct when the gate was calibrated on raw features).
    """
    import html as _html
    import numpy as _np
    X = x_row.reshape(1, -1)
    Xg = gate_transform(X) if gate_transform is not None else X
    describable = bool(gate.describe(Xg, _np.array([slice_id]))[0])
    if not describable:
        return ('<div class="coh abstain"><span class="glyph">&#9888;</span> '
                '<b>Coherence: not assessed</b> &mdash; this patient falls outside the describable '
                'region (the describability gate abstains), so PORTRAIT makes no coherence claim here.</div>')
    centile = float(pc.centile(X)[0])
    (idx2,), R = pc.attribution(X, top=2)
    r = R[0]
    label = ("internally coherent" if centile < 90 else "shows an unusual COMBINATION of features")
    glyph = "&#10003;" if centile < 90 else "&#9650;"
    rows = ""
    for j in idx2:
        val = x_row[j]; u = (units or {}).get(feature_names[j], "")
        flag = " (impossible given the rest)" if abs(r[j]) > 2 else ""
        rows += (f'<li>{_html.escape(feature_names[j])} = {val:.1f} {u} '
                 f'&mdash; conditional residual r={r[j]:+.2f}{flag}</li>')
    return (f'<div class="coh {"ok" if centile<90 else "flag"}">'
            f'<span class="glyph">{glyph}</span> <b>Profile coherence: {label}</b> '
            f'(coherence-deviation centile {centile:.0f}/100).'
            f'<div class="attr">Driven by:<ul>{rows}</ul></div>'
            f'<div class="prov">Descriptive axis (Gaussian-copula person-fit); not a diagnosis, '
            f'not a detector. Reported only where the describability gate certifies coverage.</div></div>')

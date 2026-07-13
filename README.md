# PORTRAIT — a calibrated, abstention-aware Patient Passport

[![DOI](https://zenodo.org/badge/1297704517.svg)](https://doi.org/10.5281/zenodo.21343950)

PORTRAIT describes an individual against a reference population on nameable, clinician-readable
axes — with calibrated uncertainty, and an explicit refusal to describe when the data do not
support a description beyond what covariance alone explains.

It is a **research characterisation aid, not a diagnostic or predictive device.** Every statement
it makes is a *population-position* statement with a stated uncertainty, never a clinical prediction.

## What it does

Given a reference cohort (a table of features for a population) and one individual:

1. **Describability gate.** A calibrated two-sample test decides whether the reference carries
   structure *beyond its own covariance*. If it does not, PORTRAIT **abstains** — it will not
   manufacture a rich description from noise. This is the core guarantee.
2. **Position.** Each feature is placed as a population centile with a distribution-free
   (DKW) confidence band, so the position carries a guaranteed-coverage interval.
3. **Atypicality.** A conformal, multiplicity-controlled flag for individuals who sit in a
   sparsely populated region of feature space.
4. **Profile coherence.** A per-feature decomposition of how internally consistent an
   individual's profile is, naming *which* features fail to cohere — information a single
   distance number cannot give.
5. **Conditional surprise.** Which individual features are surprising given the others.
6. **Passport render.** A self-contained, accessible HTML page (WCAG 2.2: state is always
   carried by a text label and glyph, never colour alone) showing the above for one person,
   in one of three honest states: **DESCRIBE**, **ABSTAIN**, or **REFUSE**.

## Install

Requires Python 3.9+ (reference results were produced on Python 3.13; 3.9 imports and runs).

    pip install -e .

This installs the `analysis` package and its dependencies. (For the exact pinned versions used
to produce the reference results, use `pip install -r requirements.txt` instead.)

## Quick start — the built-in demo (no data to download)

    portrait-demo            # installed console script
    # or, equivalently:
    python -m analysis.run

This runs the method on synthetic generators and writes `portrait_demo_passport.html`:
it shows the describability gate DESCRIBING structured data and declining a covariance-matched
null, then renders one Passport.

## Run it on your own data

The method modules are plain NumPy/scikit-learn and take arrays directly:

```python
import numpy as np
from analysis.structure_test.gate import describability_gate
from analysis.passport.render import build_passport, render_passport

reference = ...        # (n_people, n_features) reference-population array
individual = ...       # (n_features,) the person to describe
names = [...]          # feature names

gate = describability_gate(reference)          # DESCRIBE / BORDERLINE / REFUSE
ps = build_passport(reference, individual, names, seed=0)
open("passport.html", "w").write(render_passport(ps))
```

The worked clinical example uses the public NHANES adult cardiometabolic panel. The loader in
`analysis/nhanes/cohort.py` reads the public NHANES `.xpt` files from `$NHANES_DIR`
(default `./data/NHANES`); download them from CDC NHANES and point `NHANES_DIR` at that folder.
**No patient data ships in this repository** — it is referenced by path only.

## Layout

- `analysis/` — the method. Independent components (`structure_test`, `axes`, `position`,
  `uncertainty`, `atypicality`, `coherence`, `surprise`, `decision`, `typicality`), the
  `passport` render/orchestration layer, the `app` HTML builders, the `nhanes` data loader,
  and a synthetic-data `oracle` so the method can be exercised with nothing to download.
- `results/app/` — a prebuilt interactive Passport explorer (self-contained HTML).
- `README`, `LICENSE`, `requirements.txt`, `reproduce.sh`.

## Honest limitations

- All statements are *within-cohort population positions*, not predictions and not diagnoses.
- The describability guarantee is a calibrated statistical test; like any test it has power
  limits at small sample sizes, and it will abstain rather than overclaim when data are thin.
- Profile coherence is **descriptive**, not a discriminator: on a labelled discrimination task a
  plain covariance distance ties it. Its value is the per-feature attribution and that it names
  different individuals than raw extremity.
- The interactive app depends on a browser; the numeric method depends only on the standard
  scientific-Python stack.

## Licence

MIT — see `LICENSE`.

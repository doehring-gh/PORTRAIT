# analysis/

The PORTRAIT method. Self-contained: standard public libraries only
(numpy, scipy, scikit-learn).

Entry point: `run.py` (also invoked by `../reproduce.sh`) runs a data-free demo on the
built-in synthetic generators and writes a Passport HTML.

Components (each usable on its own array inputs):

- `structure_test/` - the calibrated describability gate (DESCRIBE / BORDERLINE / REFUSE).
- `axes/` - sparse-NMF axes with resample-stability validity.
- `position/` - population centiles.
- `uncertainty/` - distribution-free (DKW) and conformal coverage.
- `atypicality/` - conformal, multiplicity-controlled outlier flag.
- `coherence/` - profile-coherence coordinate with per-feature attribution.
- `surprise/` - conditional per-feature surprise.
- `decision/` - conformal-risk serve/abstain decision.
- `typicality/` - typicality deviation and per-feature decomposition.
- `passport/` - compose the components into a per-person Passport (build + render + orchestrate).
- `app/` - inject a precomputed bundle into the interactive HTML explorer.
- `nhanes/` - loader for the public NHANES cardiometabolic panel (referenced by path; no data ships).
- `oracle/` - synthetic-data generators so the method can be exercised with nothing to download.

Everything is seeded and deterministic.

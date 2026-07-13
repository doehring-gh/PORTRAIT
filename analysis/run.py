"""PORTRAIT - self-contained demo (no external data required).

Runs the method end to end on the built-in synthetic generators, so you can see
what PORTRAIT does with nothing to download:

  1. Describability gate: on data with genuine structure the gate describes; on a
     covariance-matched null it abstains. This is the abstention guarantee.
  2. Patient Passport: render one individual against the synthetic reference
     population - per-axis centiles with a calibrated band, an atypicality flag,
     and the surprising features - as a self-contained HTML page.

Standard scientific-Python libraries only (numpy/scipy/scikit-learn). Writes the
demo Passport to ./portrait_demo_passport.html. To run PORTRAIT on your own data,
import the method modules directly (see README) or point the NHANES loader at a
local download via the NHANES_DIR environment variable.
"""
import sys
import numpy as np

from analysis.oracle import scenarios
from analysis.structure_test.gate import describability_gate
from analysis.passport.render import build_passport, render_passport


def main():
    print("== PORTRAIT demo (synthetic, no external data) ==")

    print("-- Describability gate on structured vs null data")
    structured = scenarios.clusters(seed=0)            # expected: DESCRIBE
    null = scenarios.covariance_matched_null(seed=0)   # expected: ABSTAIN/REFUSE
    g_struct = describability_gate(structured.X, null_type="copula", statistic="energy")
    g_null = describability_gate(null.X, null_type="copula", statistic="energy")
    print(f"   structured data : p={g_struct.p_value:.4f}  decision={g_struct.decision}")
    print(f"   covariance null : p={g_null.p_value:.4f}  decision={g_null.decision}")

    print("-- Render one Patient Passport against the synthetic reference")
    ref = structured.X
    query = ref[0]
    names = [f"feature_{i+1}" for i in range(ref.shape[1])]
    ps = build_passport(ref, query, names, seed=0)
    html = render_passport(ps)
    out = "portrait_demo_passport.html"
    with open(out, "w") as fh:
        fh.write(html)
    print(f"   wrote {out}  (state: {ps.state})")

    print("== done ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Clinical UI builder: inject a precomputed patient data bundle into the app template.

The app is a thin shell over a patient data contract: all statistics are computed upstream
and serialized into the bundle; the browser only presents them. No statistic runs
in JavaScript — the same purity guarantee as the upstream computation. Self-contained (inline CSS+JS, no external assets, no
network) so it is offline and air-gapped safe for hospital or community deployment.
"""
from __future__ import annotations
import json
import os

HERE = os.path.dirname(__file__)
TEMPLATE = os.path.join(HERE, "app_template.html")


def build_app(bundle: dict, out_path: str) -> str:
    tpl = open(TEMPLATE).read()
    payload = json.dumps(bundle, default=float, ensure_ascii=False)
    # single literal substitution; JSON is inert data, not code
    html_doc = tpl.replace("__BUNDLE__", payload)
    with open(out_path, "w") as fh:
        fh.write(html_doc)
    return out_path


if __name__ == "__main__":
    import sys
    b = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else None
    if b:
        print(build_app(b, sys.argv[2] if len(sys.argv) > 2 else "portrait_app.html"))

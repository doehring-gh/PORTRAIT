"""PORTRAIT - score a NEW individual against a frozen population reference.

This is the deployment tool: it scores a person who was NOT in the reference data, against
a reference model fit once from a large population sample (the pooled NHANES adult reference).

Usage:
    # build & export the frozen model once (writes frozen_model.pkl):
    python -m analysis.passport.score_individual --build-model frozen_model.pkl

    # score one individual from a JSON file of {feature: value, age:, sex:} -> Passport HTML:
    python -m analysis.passport.score_individual --model frozen_model.pkl \
        --person person.json --out passport.html

    # or score inline (12 values in FEATURES order, plus --age --sex):
    python -m analysis.passport.score_individual --model frozen_model.pkl \
        --values 24.5 88 118 76 5.2 92 52 185 110 0.95 4.3 22 --age 45 --sex 1 --out passport.html

sex: 1 = male, 2 = female (NHANES coding). All statistics run here in Python; the emitted
HTML only displays the precomputed Passport (purity contract preserved).
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np

from analysis.passport import frozen_score as FS


def build_and_export(out_path):
    from analysis.nhanes.cohort import load_cycle
    refX, refAge, refSex = [], [], []
    for c in ["D", "E", "F", "H"]:
        co = load_cycle(c); m = co.complete_mask
        refX.append(co.X_raw[m]); refAge.append(co.age[m]); refSex.append(co.sex[m])
    Xref = np.vstack(refX)
    age = np.concatenate(refAge); sex = np.concatenate(refSex)
    sl = FS.slice_of(age, sex)
    model = FS.build_frozen_model(Xref, sl, seed=0,
                provenance={"reference_cycles": "NHANES D+E+F+H", "source": "NHANES public"})
    model.save(out_path)
    print(f"frozen model exported -> {out_path} (ref_n={model.reference_n}, hash={model.reference_hash})")
    return model


def _person_to_vector(person: dict):
    missing = [f for f in FS.FEATURES if f not in person]
    if missing:
        sys.exit(f"ERROR: person is missing required features: {missing}")
    x = np.array([float(person[f]) for f in FS.FEATURES], float)
    age = person.get("age"); sex = person.get("sex")
    if age is None or sex is None:
        sys.exit("ERROR: person JSON must include 'age' and 'sex' (1=male, 2=female).")
    return x, float(age), int(sex)


def main(argv=None):
    ap = argparse.ArgumentParser(description="PORTRAIT: score a new individual against the frozen background.")
    ap.add_argument("--build-model", metavar="PATH", help="build the frozen model and write it to PATH, then exit")
    ap.add_argument("--model", metavar="PATH", help="path to an exported frozen_model.pkl")
    ap.add_argument("--person", metavar="JSON", help="JSON file: {feature: value, ..., age:, sex:}")
    ap.add_argument("--values", nargs=12, type=float, metavar="V",
                    help="12 feature values in FEATURES order: " + " ".join(FS.FEATURES))
    ap.add_argument("--age", type=float, help="age in years (with --values)")
    ap.add_argument("--sex", type=int, choices=[1, 2], help="1=male, 2=female (with --values)")
    ap.add_argument("--out", metavar="HTML", default="passport.html", help="output HTML path")
    ap.add_argument("--json-out", metavar="JSON", help="also write the passport dict as JSON")
    args = ap.parse_args(argv)

    if args.build_model:
        build_and_export(args.build_model); return

    if not args.model:
        sys.exit("ERROR: --model is required (or run --build-model first).")
    model = FS.FrozenModel.load(args.model)

    if args.person:
        person = json.load(open(args.person))
        x, age, sex = _person_to_vector(person)
    elif args.values is not None:
        if args.age is None or args.sex is None:
            sys.exit("ERROR: --values requires --age and --sex.")
        x, age, sex = np.array(args.values, float), args.age, args.sex
    else:
        sys.exit("ERROR: supply --person or --values.")

    slice_id = int(FS.slice_of([age], [sex])[0])
    rec = FS.score_record(model, x, slice_id)
    html = FS.render_record_html(model, x, slice_id)
    open(args.out, "w").write(html)
    if args.json_out:
        json.dump(rec, open(args.json_out, "w"), indent=2, default=float)

    print(f"state: {rec['state']}  |  n_surprising: {rec['n_surprising']}  "
          f"|  reference_n: {model.reference_n}  |  hash: {model.reference_hash}")
    if rec.get("standout"):
        s = rec["standout"][0]
        print(f"top surprise: {s['axis']} = {s['value']} {s['unit']} ({s['natfreq']})")
    print(f"Passport HTML -> {args.out}")


if __name__ == "__main__":
    main()

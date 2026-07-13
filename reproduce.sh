#!/usr/bin/env bash
# PORTRAIT - one-command check that the method runs end to end on synthetic data.
set -euo pipefail

# Run from anywhere: make the repo root importable without needing `pip install`.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
cd "$HERE"

echo "== PORTRAIT =="

echo "-- 1. Interpreter"
python3 --version
py_minor=$(python3 -c 'import sys; print(sys.version_info[1])')
if [ "$py_minor" -lt 11 ]; then
  echo "   WARN: interpreter < 3.11; reference results were produced on 3.13 and may drift."
fi

echo "-- 2. Dependencies"
if python3 -c "import numpy, scipy, sklearn" 2>/dev/null; then
  echo "   numpy/scipy/scikit-learn present"
else
  echo "   Missing deps. Run: python3 -m pip install -r requirements.txt"
  exit 1
fi

echo "-- 3. Package imports (self-contained, standard libs only)"
python3 -c "import analysis; import analysis.structure_test.gate, analysis.passport.render" \
  && echo "   analysis import OK"

echo "-- 4. Run the built-in demo (no external data needed)"
python3 -m analysis.run

echo "== done =="

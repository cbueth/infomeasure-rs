#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Run the cross-package collection. Intended to run inside the :bench image
# (Rust toolchain + JIDT + bench Python venv + Julia + R). Env:
#   BENCH_DATA_DIR (default target/bench-data)
#   BENCH_SHORT=1  small/fast slice for iteration (default: full, adaptive)
#   BENCH_PACKAGES comma-separated package ids to collect; default = all.
#                  When set, existing fragments in results/ are kept and only
#                  the listed collectors run (selective update).
#   BENCH_KEEP_RESULTS=1  keep existing fragments instead of clearing results/
#   BENCH_SIZES, BENCH_WARMUP_MAX, BENCH_MIN_ITERS, BENCH_MAX_ITERS, ...
#   BENCH_PYTHON (default /opt/bench-venv/bin/python)
#   JIDT_JAR (default /opt/jidt/infodynamics.jar)
set -euo pipefail

# Equalisation: single-threaded everywhere (JIDT NUM_THREADS=1 is set in the
# collector; these cover BLAS/OpenMP/rayon in the Python/R/Julia/native libs).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 RAYON_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$REPO/target/bench-data}"
PY="${BENCH_PYTHON:-/opt/bench-venv/bin/python}"
JIDT_JAR="${JIDT_JAR:-/opt/jidt/infodynamics.jar}"
SHORT="${BENCH_SHORT:-0}"
export BENCH_SHORT="$SHORT"
SHORT_FLAG=""
[ "$SHORT" = "1" ] && SHORT_FLAG="--short"

PACKAGES="${BENCH_PACKAGES:-}"
selected() {
  if [ -z "$PACKAGES" ] || [ "$PACKAGES" = "all" ]; then return 0; fi
  case ",$PACKAGES," in *",$1,"*) return 0 ;; esac
  return 1
}

# Fresh fragments only for a full run; selective runs keep the others.
if [ -n "$PACKAGES" ] && [ "$PACKAGES" != "all" ]; then
  export BENCH_KEEP_RESULTS="${BENCH_KEEP_RESULTS:-1}"
fi
# BENCH_RESUME=1 keeps existing fragments so collectors skip finished entries
# and a cancelled run can be resumed.
export BENCH_RESUME="${BENCH_RESUME:-0}"
if [ "${BENCH_KEEP_RESULTS:-0}" != "1" ] && [ "$BENCH_RESUME" != "1" ]; then
  rm -f "$BENCH_DATA_DIR"/results/*.json
fi
mkdir -p "$BENCH_DATA_DIR/results"

echo "=== datasets ==="
cargo bench --bench gen_datasets

SEEDS="$("$PY" -c "import json;print(','.join(map(str,json.load(open('$BENCH_DATA_DIR/manifest.json'))['seeds'])))")"
SIZES="$("$PY" -c "import json;print(','.join(map(str,json.load(open('$BENCH_DATA_DIR/manifest.json'))['sizes'])))")"

if selected infomeasure-rs; then
  echo "=== infomeasure-rs ==="
  cargo bench --bench collect_cross_package
fi
if selected infomeasure-python; then
  echo "=== infomeasure-python ==="
  "$PY" "$SCRIPT_DIR/collect_infomeasure_python.py"
fi
if selected pyinform; then
  echo "=== pyinform ==="; "$PY" "$SCRIPT_DIR/collect_pyinform.py"
fi
if selected pyitlib; then
  echo "=== pyitlib ==="; "$PY" "$SCRIPT_DIR/collect_pyitlib.py"
fi
if selected pyentrp; then
  echo "=== pyentrp ==="; "$PY" "$SCRIPT_DIR/collect_pyentrp.py"
fi
if selected syntropy; then
  echo "=== syntropy ==="; "$PY" "$SCRIPT_DIR/collect_syntropy.py"
fi
if selected dit; then
  echo "=== dit ==="; "$PY" "$SCRIPT_DIR/collect_dit.py"
fi
if selected discreteentropyjl; then
  echo "=== DiscreteEntropy.jl ==="
  julia "$SCRIPT_DIR/collect_discreteentropyjl.jl" \
    --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS"
fi
if selected jidt; then
  echo "=== JIDT ==="
  rm -rf /tmp/jidtcls && mkdir -p /tmp/jidtcls
  javac -cp "$JIDT_JAR" -d /tmp/jidtcls "$SCRIPT_DIR/JidtCollector.java"
  java -cp "$JIDT_JAR:/tmp/jidtcls" JidtCollector \
    --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS" $SHORT_FLAG
fi
if selected rtransferentropy; then
  echo "=== RTransferEntropy ==="
  Rscript "$SCRIPT_DIR/collect_rtransferentropy.R" \
    --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS"
fi

# --- Alphabet-scaling family (discrete MLE across state counts) -------------
AB_STATES="$("$PY" -c "import sys;sys.path.insert(0,'$SCRIPT_DIR');import _grid;print(','.join(map(str,_grid.alphabet()['states'])))")"
AB_SIZES="$("$PY" -c "import sys;sys.path.insert(0,'$SCRIPT_DIR');import _grid;print(','.join(map(str,_grid.alphabet_sizes())))")"
AB_BUDGET="$("$PY" -c "import sys;sys.path.insert(0,'$SCRIPT_DIR');import _grid;print(_grid.alphabet()['budget_s'])")"
AB_CAPS="$("$PY" -c "import sys;sys.path.insert(0,'$SCRIPT_DIR');import _grid;print(','.join(f'{k}:{v}' for k,v in _grid.alphabet()['caps'].items()))")"

if selected infomeasure-rs; then
  echo "=== infomeasure-rs (alphabet) ==="
  cargo bench --bench collect_alphabet
fi
# One generic Python collector covers all Python providers; restrict it to the
# selected package ids.
if selected infomeasure-python || selected pyinform || selected pyitlib || selected pyentrp || selected dit; then
  AB_PKGS=""
  for p in infomeasure-python pyinform pyitlib pyentrp dit; do
    if selected "$p"; then AB_PKGS="${AB_PKGS:+$AB_PKGS,}$p"; fi
  done
  echo "=== Python packages (alphabet): $AB_PKGS ==="
  BENCH_ALPHABET_PACKAGES="$AB_PKGS" "$PY" "$SCRIPT_DIR/collect_alphabet.py"
fi
if selected jidt; then
  echo "=== JIDT (alphabet) ==="
  javac -cp "$JIDT_JAR" -d /tmp/jidtcls "$SCRIPT_DIR/JidtAlphabetCollector.java"
  java -cp "$JIDT_JAR:/tmp/jidtcls" JidtAlphabetCollector \
    --data-dir "$BENCH_DATA_DIR" --states "$AB_STATES" --sizes "$AB_SIZES" \
    --caps "$AB_CAPS" --budget "$AB_BUDGET" --seeds "$SEEDS" $SHORT_FLAG
fi

echo "=== merge ==="
"$PY" "$SCRIPT_DIR/merge_results.py"

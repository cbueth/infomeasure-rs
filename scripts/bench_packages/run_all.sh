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
#   PAGES_DIR  optional checkout of the `pages` branch. When set, fragments are
#              published to it after every package; the first publish creates a
#              commit and later ones amend it, so a run leaves one commit.
#
# Every collector is non-fatal: a failure is recorded and the run continues, so
# completed work still publishes. The script exits non-zero at the end if any
# collector (or publish) failed.
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
PAGES_DIR="${PAGES_DIR:-}"
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

FAILED=""
record_failure() {
  echo "ERROR: $1 failed (continuing)" >&2
  FAILED="${FAILED}${FAILED:+, }$1"
}
# Run a collector, recording (not propagating) a failure.
step() {
  local name="$1"; shift
  "$@" || record_failure "$name"
}

# Publish the fragments so far. The first call commits; later calls amend that
# commit so the run leaves a single commit on `pages`. No-op without PAGES_DIR.
PUBLISHED=0
publish() {
  [ -n "$PAGES_DIR" ] || return 0
  local amend=0
  [ "$PUBLISHED" = "1" ] && amend=1
  if RESULTS_DIR="$BENCH_DATA_DIR/results" PAGES_DIR="$PAGES_DIR" AMEND="$amend" \
       bash "$SCRIPT_DIR/publish_fragments.sh"; then
    PUBLISHED=1
  else
    record_failure publish
  fi
}

echo "=== datasets ==="
step datasets cargo bench --bench gen_datasets

if ! SEEDS="$("$PY" -c "import json;print(','.join(map(str,json.load(open('$BENCH_DATA_DIR/manifest.json'))['seeds'])))")"; then
  record_failure seeds
  SEEDS=""
fi
if ! SIZES="$("$PY" -c "import json;print(','.join(map(str,json.load(open('$BENCH_DATA_DIR/manifest.json'))['sizes'])))")"; then
  record_failure sizes
  SIZES=""
fi

if selected infomeasure-rs; then
  echo "=== infomeasure-rs ==="
  step infomeasure-rs cargo bench --bench collect_cross_package
  publish
fi
if selected infomeasure-python; then
  echo "=== infomeasure-python ==="
  step infomeasure-python "$PY" "$SCRIPT_DIR/collect_infomeasure_python.py"
  publish
fi
if selected pyinform; then
  echo "=== pyinform ==="
  step pyinform "$PY" "$SCRIPT_DIR/collect_pyinform.py"
  publish
fi
if selected pyitlib; then
  echo "=== pyitlib ==="
  step pyitlib "$PY" "$SCRIPT_DIR/collect_pyitlib.py"
  publish
fi
if selected pyentrp; then
  echo "=== pyentrp ==="
  step pyentrp "$PY" "$SCRIPT_DIR/collect_pyentrp.py"
  publish
fi
if selected syntropy; then
  echo "=== syntropy ==="
  step syntropy "$PY" "$SCRIPT_DIR/collect_syntropy.py"
  publish
fi
if selected dit; then
  echo "=== dit ==="
  step dit "$PY" "$SCRIPT_DIR/collect_dit.py"
  publish
fi
if selected discreteentropyjl; then
  echo "=== DiscreteEntropy.jl ==="
  step discreteentropyjl julia "$SCRIPT_DIR/collect_discreteentropyjl.jl" \
    --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS"
  publish
fi
if selected jidt; then
  echo "=== JIDT ==="
  step jidt bash -c "rm -rf /tmp/jidtcls && mkdir -p /tmp/jidtcls && \
    javac -cp '$JIDT_JAR' -d /tmp/jidtcls '$SCRIPT_DIR/JidtCollector.java' && \
    java -cp '$JIDT_JAR:/tmp/jidtcls' JidtCollector --data-dir '$BENCH_DATA_DIR' --sizes '$SIZES' --seeds '$SEEDS' $SHORT_FLAG"
  publish
fi
if selected rtransferentropy; then
  echo "=== RTransferEntropy ==="
  step rtransferentropy Rscript "$SCRIPT_DIR/collect_rtransferentropy.R" \
    --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS"
  publish
fi

# --- Alphabet-scaling family (discrete MLE across state counts) -------------
ab_cfg() {
  "$PY" -c "import sys;sys.path.insert(0,'$SCRIPT_DIR');import _grid;print($1)"
}
if ! AB_STATES="$(ab_cfg "','.join(map(str,_grid.alphabet()['states']))")"; then
  record_failure alphabet-config
fi
if ! AB_SIZES="$(ab_cfg "','.join(map(str,_grid.alphabet_sizes()))")"; then
  record_failure alphabet-sizes
fi
if ! AB_BUDGET="$(ab_cfg "_grid.alphabet()['budget_s']")"; then
  record_failure alphabet-budget
fi
if ! AB_CAPS="$(ab_cfg "','.join(f'{k}:{v}' for k,v in _grid.alphabet()['caps'].items())")"; then
  record_failure alphabet-caps
fi

if selected infomeasure-rs; then
  echo "=== infomeasure-rs (alphabet) ==="
  step infomeasure-rs-alphabet cargo bench --bench collect_alphabet
  publish
fi
# One generic Python collector covers all Python providers; restrict it to the
# selected package ids.
if selected infomeasure-python || selected pyinform || selected pyitlib || selected pyentrp || selected dit; then
  AB_PKGS=""
  for p in infomeasure-python pyinform pyitlib pyentrp dit; do
    if selected "$p"; then AB_PKGS="${AB_PKGS:+$AB_PKGS,}$p"; fi
  done
  echo "=== Python packages (alphabet): $AB_PKGS ==="
  step python-alphabet env BENCH_ALPHABET_PACKAGES="$AB_PKGS" "$PY" "$SCRIPT_DIR/collect_alphabet.py"
  publish
fi
if selected jidt; then
  echo "=== JIDT (alphabet) ==="
  step jidt-alphabet bash -c "rm -rf /tmp/jidtcls && mkdir -p /tmp/jidtcls && \
    javac -cp '$JIDT_JAR' -d /tmp/jidtcls '$SCRIPT_DIR/JidtAlphabetCollector.java' && \
    java -cp '$JIDT_JAR:/tmp/jidtcls' JidtAlphabetCollector --data-dir '$BENCH_DATA_DIR' \
      --states '$AB_STATES' --sizes '$AB_SIZES' --caps '$AB_CAPS' --budget '$AB_BUDGET' \
      --seeds '$SEEDS' $SHORT_FLAG"
  publish
fi
if selected discreteentropyjl; then
  echo "=== DiscreteEntropy.jl (alphabet) ==="
  step discreteentropyjl-alphabet julia "$SCRIPT_DIR/collect_discreteentropyjl.jl" \
    --data-dir "$BENCH_DATA_DIR" --family alphabet \
    --states "$AB_STATES" --sizes "$AB_SIZES" --caps "$AB_CAPS" \
    --budget "$AB_BUDGET" --seeds "$SEEDS"
  publish
fi

echo "=== merge ==="
step merge "$PY" "$SCRIPT_DIR/merge_results.py"
publish

if [ -n "$FAILED" ]; then
  echo "=== FAILED collectors: $FAILED ===" >&2
  exit 1
fi
echo "=== all collectors succeeded ==="

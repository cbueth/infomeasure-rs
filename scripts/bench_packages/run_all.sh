#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Run the full cross-package collection. Intended to run inside the :bench
# image (Rust toolchain + JIDT + bench Python venv + Julia + R). Env:
#   BENCH_DATA_DIR (default target/bench-data)
#   BENCH_SHORT=1  small/fast slice for iteration (default: full, adaptive)
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

echo "=== infomeasure-rs (build + generate + collect) ==="
# Fresh fragments each full run: avoids stale collectors leaking into the merge.
rm -f "$BENCH_DATA_DIR"/results/*.json
cargo bench --bench gen_datasets
cargo bench --bench collect_cross_package

SEEDS="$("$PY" -c "import json;print(','.join(map(str,json.load(open('$BENCH_DATA_DIR/manifest.json'))['seeds'])))")"
SIZES="$("$PY" -c "import json;print(','.join(map(str,json.load(open('$BENCH_DATA_DIR/manifest.json'))['sizes'])))")"

echo "=== python collectors ==="
"$PY" "$SCRIPT_DIR/collect_infomeasure_python.py"
"$PY" "$SCRIPT_DIR/collect_pyinform.py"
"$PY" "$SCRIPT_DIR/collect_pyitlib.py"
"$PY" "$SCRIPT_DIR/collect_pyentrp.py"
"$PY" "$SCRIPT_DIR/collect_syntropy.py"
"$PY" "$SCRIPT_DIR/collect_dit.py"

echo "=== DiscreteEntropy.jl (julia) ==="
julia "$SCRIPT_DIR/collect_discreteentropyjl.jl" \
  --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS"

echo "=== JIDT (native java) ==="
rm -rf /tmp/jidtcls && mkdir -p /tmp/jidtcls
javac -cp "$JIDT_JAR" -d /tmp/jidtcls "$SCRIPT_DIR/JidtCollector.java"
java -cp "$JIDT_JAR:/tmp/jidtcls" JidtCollector \
  --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS" $SHORT_FLAG

echo "=== RTransferEntropy (R) ==="
Rscript "$SCRIPT_DIR/collect_rtransferentropy.R" \
  --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS"

echo "=== merge ==="
"$PY" "$SCRIPT_DIR/merge_results.py"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Run the full cross-package collection. Intended to run inside the :bench
# image (Rust toolchain + JIDT + bench Python venv + Julia + R). Env:
#   BENCH_DATA_DIR (default target/bench-data)
#   BENCH_SHORT=1  (default)  small/fast slice for iteration
#   BENCH_SIZES, BENCH_WARMUP, BENCH_ITERATIONS
#   BENCH_PYTHON (default /opt/bench-venv/bin/python)
#   JIDT_JAR (default /opt/jidt/infodynamics.jar)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
export BENCH_DATA_DIR="${BENCH_DATA_DIR:-$REPO/target/bench-data}"
PY="${BENCH_PYTHON:-/opt/bench-venv/bin/python}"
JIDT_JAR="${JIDT_JAR:-/opt/jidt/infodynamics.jar}"
SHORT="${BENCH_SHORT:-1}"

if [ "$SHORT" = "1" ]; then
  WARMUP="${BENCH_WARMUP:-1}"
  ITERS="${BENCH_ITERATIONS:-3}"
  SHORT_FLAG="--short"
else
  WARMUP="${BENCH_WARMUP:-3}"
  ITERS="${BENCH_ITERATIONS:-10}"
  SHORT_FLAG=""
fi
export BENCH_WARMUP="$WARMUP" BENCH_ITERATIONS="$ITERS"

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
  --data-dir "$BENCH_DATA_DIR" --sizes "${SIZES:-}" --seeds "${SEEDS:-}" \
  --warmup "$WARMUP" --iterations "$ITERS"

echo "=== JIDT (native java) ==="
rm -rf /tmp/jidtcls && mkdir -p /tmp/jidtcls
javac -cp "$JIDT_JAR" -d /tmp/jidtcls "$SCRIPT_DIR/JidtCollector.java"
java -cp "$JIDT_JAR:/tmp/jidtcls" JidtCollector \
  --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS" $SHORT_FLAG

echo "=== RTransferEntropy (R) ==="
Rscript "$SCRIPT_DIR/collect_rtransferentropy.R" \
  --data-dir "$BENCH_DATA_DIR" --sizes "$SIZES" --seeds "$SEEDS" \
  --warmup "$WARMUP" --iterations "$ITERS"

echo "=== merge ==="
"$PY" "$SCRIPT_DIR/merge_results.py"

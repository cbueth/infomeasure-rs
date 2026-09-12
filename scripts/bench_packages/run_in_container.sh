#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Run the full cross-package collection inside the :bench image on the CI
# runner (host-side launcher; the collection itself is run_all.sh in-container).
#
#   REPO=~/code/infomeasure-rs bash scripts/bench_packages/run_in_container.sh
#
# Optional env forwarded: BENCH_SIZES, BENCH_SHORT, BENCH_WARMUP,
# BENCH_ITERATIONS, BENCH_DATA_DIR.
set -euo pipefail

REPO="${REPO:-$HOME/code/infomeasure-rs}"

args=(
  --rm --user 1000:100
  -v "$REPO:/work" -w /work
  -e CARGO_HOME=/work/target/cargo-home
  -e CARGO_TARGET_DIR=/work/target/cargo
  -e BENCH_DATA_DIR="${BENCH_DATA_DIR:-/work/target/bench-data}"
)
for var in BENCH_SIZES BENCH_SHORT BENCH_WARMUP_MAX BENCH_WARMUP_BUDGET_S \
           BENCH_MIN_ITERS BENCH_MAX_ITERS BENCH_ITER_BUDGET_S; do
  if [ -n "${!var:-}" ]; then args+=(-e "$var=${!var}"); fi
done

exec docker run "${args[@]}" --entrypoint bash im-bench:dev -c \
  'export PATH=/usr/local/cargo/bin:$PATH; bash scripts/bench_packages/run_all.sh'

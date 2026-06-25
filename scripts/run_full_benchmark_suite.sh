#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Full benchmark suite orchestration script.
#
# Usage:
#   # Quick development smoke test (minimal params)
#   bash scripts/run_full_benchmark_suite.sh --quick
#
#   # Full production run
#   bash scripts/run_full_benchmark_suite.sh
#
#   # Custom sizes
#   bash scripts/run_full_benchmark_suite.sh --sizes "100,500,1000,5000,10000"
#
# shellcheck disable=SC2317

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARKS_DATA_DIR="$SCRIPT_DIR/benchmarks_data"
INTERNAL_DIR="$REPO_DIR/internal"
RESULTS_DIR="$INTERNAL_DIR/benchmark_results"
VENV_PYTHON="$REPO_DIR/tests/validation_crate/.venv/bin/python"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Default: full production params
SIZES="100,500,1000,5000,10000"
HISTORY_LENS="2,3,4,5,6"
K_VALUES="2,3,4,5"
BANDWIDTHS="0.1,0.3,0.5,1.0,1.5"
ORDERS="2,3,4"
DELAYS="0,1,2"
ITERATIONS=10
WARMUP=3

# Parse args
QUICK_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --sizes)
            SIZES="$2"
            shift 2
            ;;
        --history-lens)
            HISTORY_LENS="$2"
            shift 2
            ;;
        --k-values)
            K_VALUES="$2"
            shift 2
            ;;
        --bandwidths)
            BANDWIDTHS="$2"
            shift 2
            ;;
        --orders)
            ORDERS="$2"
            shift 2
            ;;
        --delays)
            DELAYS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if $QUICK_MODE; then
    echo "=== QUICK MODE (development smoke test) ==="
    SIZES="100,500"
    HISTORY_LENS="2,3"
    K_VALUES="3"
    BANDWIDTHS="0.5"
    ORDERS="2,3"
    DELAYS="1"
    ITERATIONS=2
    WARMUP=1
fi

echo "=== infomeasure-rs Benchmark Suite ==="
echo "Timestamp: $TIMESTAMP"
echo "Sizes: $SIZES"
echo "History lengths: $HISTORY_LENS"
echo "K values: $K_VALUES"
echo "Bandwidths: $BANDWIDTHS"
echo "Orders: $ORDERS"
echo "Delays: $DELAYS"
echo "Iterations: $ITERATIONS"
echo "Warmup: $WARMUP"
echo ""

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Step 1: Capture hardware specs
# ---------------------------------------------------------------------------
echo "=== Step 1: Capture hardware specs ==="
if command -v system_profiler &>/dev/null; then
    system_profiler SPHardwareDataType > "$INTERNAL_DIR/hardware_specs.txt" 2>/dev/null
    echo "Hardware specs saved to $INTERNAL_DIR/hardware_specs.txt"
else
    echo "system_profiler not available, recording basic info"
    uname -a > "$INTERNAL_DIR/hardware_specs.txt"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2: Run Rust benchmarks (criterion)
# ---------------------------------------------------------------------------
echo "=== Step 2: Run Rust benchmarks ==="

# Full production list
RUST_BENCHES=(
    "entropy_discrete"
    "entropy_discrete_bias_corrected"
    "entropy_ordinal"
    "entropy_expfam"
    "mi"
    "mi_discrete_bias_corrected"
    "mi_expfam"
    "te"
    "cmi"
    "cte"
)

if $QUICK_MODE; then
    # Quick: only run the key representative benches
    RUST_BENCHES=(
        "entropy_discrete"
        "mi"
        "te"
        "cmi"
        "cte"
    )
fi

for bench in "${RUST_BENCHES[@]}"; do
    echo "  Running: cargo bench --bench $bench"
    cargo bench --bench "$bench"
    echo ""
done

# Optionally run GPU bench
if [[ "${INCLUDE_GPU:-false}" == "true" ]] || $QUICK_MODE; then
    echo "  Skipping kernel_gpu (use INCLUDE_GPU=true to enable)"
fi

# ---------------------------------------------------------------------------
# Step 3: Run Python benchmarks
# ---------------------------------------------------------------------------
echo "=== Step 3: Run Python benchmarks ==="

"$VENV_PYTHON" "$SCRIPT_DIR/run_all_benchmarks.py" \
    --sizes "$SIZES" \
    --history-lens "$HISTORY_LENS" \
    --k-values "$K_VALUES" \
    --bandwidths "$BANDWIDTHS" \
    --orders "$ORDERS" \
    --delays "$DELAYS" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP"

echo ""

# ---------------------------------------------------------------------------
# Step 4: Compare Rust vs Python
# ---------------------------------------------------------------------------
echo "=== Step 4: Compare Rust vs Python ==="

# For now, placeholder — compare_benchmarks.py needs criterion JSON paths
# which are auto-generated. We'll wire this properly once collect_rust_results.py
# is fully stable.
echo "  (compare_benchmarks.py integration will be wired after collect step)"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Collect & aggregate results
# ---------------------------------------------------------------------------
echo "=== Step 5: Collect & aggregate results ==="

if [ -f "$SCRIPT_DIR/aggregate_results.py" ]; then
    "$VENV_PYTHON" "$SCRIPT_DIR/aggregate_results.py" \
        --results-dir "$RESULTS_DIR" \
        --timestamp "$TIMESTAMP" \
        --sizes "$SIZES"
fi

echo ""
echo "=== Done ==="
echo "Results directory: $RESULTS_DIR"
echo "Hardware specs: $INTERNAL_DIR/hardware_specs.txt"
echo "Python benchmark data: $BENCHMARKS_DATA_DIR/"

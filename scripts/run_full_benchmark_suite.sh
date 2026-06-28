#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Full benchmark suite orchestration script.
#
# Results are stored in persistent run directories under
# internal/benchmark_results/runs/run_<timestamp>_<features>/. Each run
# is self-contained with metadata, Rust results, and Python results.
# The aggregate step merges all runs into summary.csv with a `features`
# column distinguishing CPU vs GPU configurations.
#
# Skip logic: before running a bench binary, discovers its criterion groups
# via --list, then checks each group directly in target/criterion/ for
# existing new/estimates.json files. Groups that already have results are
# skipped; only missing groups are run. Results are collected incrementally
# after each group finishes, so partial runs are never lost.
#
# Usage:
#   # Quick development smoke test
#   bash scripts/run_full_benchmark_suite.sh --quick
#
#   # Full production run (GPU acceleration enabled by default)
#   bash scripts/run_full_benchmark_suite.sh
#
#   # CPU-only run (disable GPU)
#   INCLUDE_GPU=false bash scripts/run_full_benchmark_suite.sh
#
#   # Clear all previous outputs and start fresh
#   bash scripts/run_full_benchmark_suite.sh --clear
#
#   # Add more sizes to existing results
#   bash scripts/run_full_benchmark_suite.sh --sizes "20000,50000"
#
# shellcheck disable=SC2317

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INTERNAL_DIR="$REPO_DIR/internal"
RESULTS_DIR="$INTERNAL_DIR/benchmark_results"
RUNS_DIR="$RESULTS_DIR/runs"
VENV_PYTHON="$REPO_DIR/tests/validation_crate/.venv/bin/python"

TOTAL_STEPS=6

# Default: full production params
SIZES="100,200,400,800,1600,3200,6400,12500"
SIZES_EXTENDED="100,200,400,800,1600,3200,6400,12500" #,25000,50000,100000"
K_VALUES="3,5"
BANDWIDTHS="0.3,0.5,1.0,1.5"
KERNEL_TYPES="gaussian,box"
KERNEL_ITERATIONS=3
ORDERS="2,3,4"

DIMENSIONS="1"
ALPHAS="0.6,1.2,1.8"
Q_VALUES="0.6,1.2,1.8"
ITERATIONS=10
WARMUP=3

# Parse args
QUICK_MODE=false
CLEAR=false
COMPARE_ONLY=false
COMPARE_RUN_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)       QUICK_MODE=true; shift ;;
        --clear)       CLEAR=true; shift ;;
        --compare-only)
            COMPARE_ONLY=true
            if [ $# -ge 2 ] && [[ "${2:-}" != --* ]]; then
                COMPARE_RUN_DIR="$2"; shift 2
            else
                shift
            fi
            ;;
        --sizes)       SIZES="$2"; SIZES_EXTENDED="$2"; shift 2 ;;
        --k-values)    K_VALUES="$2"; shift 2 ;;
        --history-lens) echo "Warning: --history-lens is no longer used; ignoring"; shift 2 ;;
        --orders)      ORDERS="$2"; shift 2 ;;
        --delays)      echo "Warning: --delays is no longer used; ignoring"; shift 2 ;;
        --dimensions)  DIMENSIONS="$2"; shift 2 ;;
        --alphas)      ALPHAS="$2"; shift 2 ;;
        --q-values)    Q_VALUES="$2"; shift 2 ;;
        --iterations)  ITERATIONS="$2"; shift 2 ;;
        --warmup)      WARMUP="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $QUICK_MODE; then
    SIZES="100,500"
    SIZES_EXTENDED="100,500"
    K_VALUES="3"
    BANDWIDTHS="0.5"
    KERNEL_TYPES="gaussian,box"
    KERNEL_ITERATIONS=2
    ORDERS="2,3"

    DIMENSIONS="1"
    ALPHAS="0.5,1.0"
    Q_VALUES="0.5,1.5"
    ITERATIONS=2
    WARMUP=1
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INCLUDE_GPU="${INCLUDE_GPU:-false}"
FEATURES=""
SUFFIX=""
if $INCLUDE_GPU; then
    FEATURES="--features gpu"
    SUFFIX="gpu"
fi

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR_NAME="run_${TIMESTAMP}_${SUFFIX}"
RUN_DIR="$RUNS_DIR/$RUN_DIR_NAME"

# Rust bench binary list (all measures, all approaches)
RUST_BENCHES=(
    "entropy_discrete"
    "entropy_discrete_bias_corrected"
    "entropy_ordinal"
    "entropy_expfam"
    "mi"
    "mi_discrete_bias_corrected"
    "mi_expfam"
    "te"
    "te_discrete_bias_corrected"
    "cmi"
    "cmi_discrete_bias_corrected"
    "cte"
    "cte_discrete_bias_corrected"
)

if $QUICK_MODE; then
    RUST_BENCHES=(
        "entropy_discrete"
        "entropy_discrete_bias_corrected"
        "mi"
        "mi_discrete_bias_corrected"
        "te"
        "cmi"
        "cte"
    )
fi

ALL_RUST_BENCHES=("${RUST_BENCHES[@]}")

# Feature tag for display
FEATURE_LABEL="${SUFFIX:-baseline (CPU)}"

# ---------------------------------------------------------------------------
# Compare-only mode: re-generate comparisons without running benchmarks
# ---------------------------------------------------------------------------
if $COMPARE_ONLY; then
    if [ -z "$COMPARE_RUN_DIR" ]; then
        COMPARE_RUN_DIR=$(ls -dt "$RUNS_DIR"/run_*_ 2>/dev/null | head -1 || true)
        if [ -z "$COMPARE_RUN_DIR" ]; then
            echo "No run directories found in $RUNS_DIR"
            exit 1
        fi
    fi
    if [ ! -d "$COMPARE_RUN_DIR" ]; then
        echo "Run directory not found: $COMPARE_RUN_DIR"
        exit 1
    fi
    RUN_DIR="$COMPARE_RUN_DIR"
    echo "=== Compare-only mode ==="
    echo "  Run dir: $RUN_DIR"
    echo ""

    # Step 4: re-collect Rust results (safety net)
    echo "[1/2] Re-collect Rust results"
    "$VENV_PYTHON" "$SCRIPT_DIR/collect_rust_results.py" \
        --criterion-dir "$REPO_DIR/target/criterion" \
        --output "$RUN_DIR/rust_results.json" \
        --features "$SUFFIX" \
        --merge 2>&1 || true
    echo ""

    # Step 6: compare & aggregate
    echo "[2/2] Compare & aggregate"
    PY_COUNT=0
    for py_file in "$RUN_DIR"/python_*.json; do
        [ -f "$py_file" ] || continue
        base=$(basename "$py_file" .json)
        group="${base#python_}"
        output_report="$RUN_DIR/comparison_${group}.md"
        echo "  [$((PY_COUNT + 1))/??] Comparing $group..."
        "$VENV_PYTHON" "$SCRIPT_DIR/compare_benchmarks.py" \
            --python "$py_file" \
            --rust "$RUN_DIR/rust_results.json" \
            --output "$output_report" 2>&1 || true
        PY_COUNT=$((PY_COUNT + 1))
        echo "  -> $group done ($output_report)"
    done
    echo "  Merging comparison reports..."
    find "$RUN_DIR" -maxdepth 1 -name 'comparison_*.md' ! -name 'comparison_report.md' \
        -exec cat {} + > "$RUN_DIR/comparison_report.md" 2>/dev/null || true
    echo "  $PY_COUNT comparison report(s) in $RUN_DIR/comparison_*.md"
    echo ""
    echo "  Aggregating all runs into summary.csv..."
    "$VENV_PYTHON" "$SCRIPT_DIR/aggregate_results.py" \
        --runs-dir "$RUNS_DIR" \
        --output "$RESULTS_DIR/summary.csv" 2>&1

    echo "  Generating viewer data..."
    "$VENV_PYTHON" "$SCRIPT_DIR/generate_benchmark_json.py" \
        --run-dir "$RUN_DIR" \
        --hardware "$INTERNAL_DIR/hardware_specs.txt" \
        --output "$REPO_DIR/docs/benchmark_data.json" 2>&1

    echo ""
    echo "=== Done ==="
    echo "  Run dir:  $RUN_DIR"
    echo "  Summary:  $RESULTS_DIR/summary.csv"
    exit 0
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

status() {
    local step="$1"
    local msg="$2"
    printf "\n[%s/%s] %s\n" "$step" "$TOTAL_STEPS" "$msg"
}

# Check if a criterion group already has results in target/criterion/.
group_has_results() {
    local group="$1"
    [ -d "$REPO_DIR/target/criterion/$group" ] || return 1
    find "$REPO_DIR/target/criterion/$group" -maxdepth 5 -name estimates.json \
        -path "*/new/estimates.json" -print -quit 2>/dev/null | grep -q . || return 1
    return 0
}

# Discover criterion group names for a bench binary via --list.
discover_groups() {
    local bench="$1"
    cargo bench --bench "$bench" $FEATURES -- --list 2>/dev/null \
        | sed 's/ *:.*$//' \
        | sed 's|/.*||' \
        | sort -u \
        | grep . \
        | tr '\n' ',' \
        | sed 's/,$//' || true
}

# Collect results for specific groups and merge into run dir.
collect_groups() {
    local groups="$1"
    [ -z "$groups" ] && return 0
    "$VENV_PYTHON" "$SCRIPT_DIR/collect_rust_results.py" \
        --criterion-dir "$REPO_DIR/target/criterion" \
        --output "$RUN_DIR/rust_results.json" \
        --features "$SUFFIX" \
        --merge 2>&1 || true
}

# ---------------------------------------------------------------------------
# Step 0: Clear previous outputs (if requested)
# ---------------------------------------------------------------------------
if $CLEAR; then
    echo "=== Clearing all benchmark outputs ==="
    rm -rf "$RUNS_DIR"
    rm -f "$RESULTS_DIR/summary.csv"
    rm -rf "$REPO_DIR/target/criterion"
    echo "Cleared run dirs + criterion cache. Run without --clear to start fresh."
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 1: Capture hardware specs + version info
# ---------------------------------------------------------------------------
status 1 "Hardware specs"
mkdir -p "$INTERNAL_DIR"
if command -v system_profiler &>/dev/null; then
    system_profiler SPHardwareDataType > "$INTERNAL_DIR/hardware_specs.txt" 2>/dev/null
    echo "  Saved to $INTERNAL_DIR/hardware_specs.txt"
else
    uname -a > "$INTERNAL_DIR/hardware_specs.txt"
    echo "  Saved basic info to $INTERNAL_DIR/hardware_specs.txt"
fi

# Capture toolchain and library versions
PYTHON_VERSION=$("$VENV_PYTHON" --version 2>&1 | head -1)
RUSTC_VERSION=$(rustc --version 2>&1 | head -1)
IM_PYTHON_VERSION=$("$VENV_PYTHON" -c "import infomeasure; print(getattr(infomeasure, '__version__', 'unknown'))" 2>&1)
IM_RUST_VERSION=$(grep -m1 '^version' "$REPO_DIR/Cargo.toml" 2>/dev/null | sed 's/.*"\(.*\)".*/\1/' || echo "unknown")
echo "  Python: $PYTHON_VERSION"
echo "  Rustc: $RUSTC_VERSION"
echo "  infomeasure (Python): $IM_PYTHON_VERSION"
echo "  infomeasure (Rust): $IM_RUST_VERSION"

# ---------------------------------------------------------------------------
# Step 2: Create run directory with metadata
# ---------------------------------------------------------------------------
status 2 "Run directory"
mkdir -p "$RUN_DIR"

"$VENV_PYTHON" -c "
import json
metadata = {
    'timestamp': '$TIMESTAMP',
    'features': '$SUFFIX',
    'versions': {
        'python': '$PYTHON_VERSION',
        'rustc': '$RUSTC_VERSION',
        'infomeasure_python': '$IM_PYTHON_VERSION',
        'infomeasure_rust': '$IM_RUST_VERSION',
    },
    'config': {
        'sizes': [$SIZES],
        'dimensions': [$DIMENSIONS],
        'k_values': [$K_VALUES],
        'bandwidths': [$BANDWIDTHS],
    },
}
with open('$RUN_DIR/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
"

echo "  Created: $RUN_DIR"
echo "  Features: $FEATURE_LABEL"
echo "  Sizes: $SIZES"
echo "  Extended sizes: $SIZES_EXTENDED"

# ---------------------------------------------------------------------------
# Step 3: Run Rust benchmarks (criterion), incrementally collect per group
# ---------------------------------------------------------------------------
status 3 "Rust benchmarks"

# Initialize rust_results.json
if [ ! -f "$RUN_DIR/rust_results.json" ]; then
    echo '{"benches":{},"metadata":{"source":"criterion","path":"target/criterion","total_benchmarks":0}}' > "$RUN_DIR/rust_results.json"
fi

# Recover any existing results from target/criterion/
echo "  Recovering existing results from target/criterion/..."
"$VENV_PYTHON" "$SCRIPT_DIR/collect_rust_results.py" \
    --criterion-dir "$REPO_DIR/target/criterion" \
    --output "$RUN_DIR/rust_results.json" \
    --features "$SUFFIX" \
    --merge 2>&1 || true

echo "  Features: ${FEATURES:-none}"

# Export benchmark parameters for Rust bench binaries to discover via env var
export BENCH_SIZES="$SIZES"
export BENCH_SIZES_EXTENDED="$SIZES_EXTENDED"
export BENCH_K_VALUES="$K_VALUES"
export BENCH_BANDWIDTHS="$BANDWIDTHS"
export BENCH_ORDERS="$ORDERS"
export BENCH_ALPHAS="$ALPHAS"
export BENCH_Q_VALUES="$Q_VALUES"
echo ""

BENCH_COUNT="${#ALL_RUST_BENCHES[@]}"
BENCH_NUM=0
for bench in "${ALL_RUST_BENCHES[@]}"; do
    BENCH_NUM=$((BENCH_NUM + 1))

    # Discover which criterion groups this bench binary produces
    groups=$(discover_groups "$bench")

    if [ -z "$groups" ]; then
        echo "  [$BENCH_NUM/$BENCH_COUNT] $bench → discovery failed, running full bench"
        cargo bench --bench "$bench" $FEATURES 2>&1 \
            || echo "  Warning: $bench failed, continuing"
        groups=$(discover_groups "$bench")
        if [ -n "$groups" ]; then
            echo "    Collecting results..."
            collect_groups "$groups"
        fi
        echo ""
        continue
    fi

    # Check which groups have results in target/criterion/
    missing=""
    IFS=',' read -ra group_list <<< "$groups"
    for g in "${group_list[@]}"; do
        if ! group_has_results "$g"; then
            [ -n "$missing" ] && missing="${missing},"
            missing="${missing}${g}"
        fi
    done

    if [ -z "$missing" ]; then
        echo "  [$BENCH_NUM/$BENCH_COUNT] $bench → all groups have results, skipping"
        echo "    Groups: $(echo "$groups" | tr ',' ', ')"
    else
        echo "  [$BENCH_NUM/$BENCH_COUNT] $bench → running missing groups:"
        IFS=',' read -ra missing_list <<< "$missing"
        for mg in "${missing_list[@]}"; do
            echo "    → cargo bench --bench $bench $FEATURES -- $mg"
            cargo bench --bench "$bench" $FEATURES -- "$mg" 2>&1 \
                || echo "    Warning: $mg failed, continuing"
        done
        echo "    Collecting results..."
        collect_groups "$groups"
    fi
    echo ""
done

# Also run scaling_nd if not in quick mode
if ! $QUICK_MODE; then
    bench="scaling_nd"
    groups=$(discover_groups "$bench")

    if [ -z "$groups" ]; then
        echo "  [$BENCH_NUM/$BENCH_COUNT] $bench → discovery failed, running full bench"
        cargo bench --bench "$bench" $FEATURES 2>&1 \
            || echo "  Warning: $bench failed, continuing"
    else
        missing=""
        IFS=',' read -ra group_list <<< "$groups"
        for g in "${group_list[@]}"; do
            if ! group_has_results "$g"; then
                [ -n "$missing" ] && missing="${missing},"
                missing="${missing}${g}"
            fi
        done

        if [ -z "$missing" ]; then
            echo "  [$BENCH_NUM/$BENCH_COUNT] $bench → all groups have results, skipping"
        else
            echo "  [$BENCH_NUM/$BENCH_COUNT] $bench → running missing groups:"
            IFS=',' read -ra missing_list <<< "$missing"
            for mg in "${missing_list[@]}"; do
                echo "    → cargo bench --bench $bench $FEATURES -- $mg"
                cargo bench --bench "$bench" $FEATURES -- "$mg" 2>&1 \
                    || echo "    Warning: $mg failed, continuing"
            done
            echo "    Collecting results..."
            collect_groups "$groups"
        fi
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Step 4: Re-collect all Rust results (safety net for missed incrementals)
# ---------------------------------------------------------------------------
status 4 "Re-collect all Rust results (safety net)"

"$VENV_PYTHON" "$SCRIPT_DIR/collect_rust_results.py" \
    --criterion-dir "$REPO_DIR/target/criterion" \
    --output "$RUN_DIR/rust_results.json" \
    --features "$SUFFIX" \
    --merge 2>&1

echo ""

# ---------------------------------------------------------------------------
# Step 5: Run Python benchmarks
# ---------------------------------------------------------------------------
status 5 "Python benchmarks"

"$VENV_PYTHON" "$SCRIPT_DIR/run_all_benchmarks.py" \
    --sizes "$SIZES" \
    --k-values "$K_VALUES" \
    --bandwidths "$BANDWIDTHS" \
    --kernel-types "$KERNEL_TYPES" \
    --kernel-iterations "$KERNEL_ITERATIONS" \
    --orders "$ORDERS" \
    --dimensions "$DIMENSIONS" \
    --alphas "$ALPHAS" \
    --q-values "$Q_VALUES" \
    --iterations "$ITERATIONS" \
    --warmup "$WARMUP" \
    --output-dir "$RUN_DIR" 2>&1

echo ""

# ---------------------------------------------------------------------------
# Step 6: Compare Rust vs Python + aggregate results
# ---------------------------------------------------------------------------
status 6 "Compare & aggregate"

# Run comparison for each Python group in this run dir
PY_COUNT=0
for py_file in "$RUN_DIR"/python_*.json; do
    [ -f "$py_file" ] || continue
    base=$(basename "$py_file" .json)
    group="${base#python_}"
    output_report="$RUN_DIR/comparison_${group}.md"
    echo "  [$((PY_COUNT + 1))/??] Comparing $group..."
    "$VENV_PYTHON" "$SCRIPT_DIR/compare_benchmarks.py" \
        --python "$py_file" \
        --rust "$RUN_DIR/rust_results.json" \
        --output "$output_report" 2>&1 || true
    PY_COUNT=$((PY_COUNT + 1))
    echo "  -> $group done ($output_report)"
done

echo "  Merging comparison reports..."
find "$RUN_DIR" -maxdepth 1 -name 'comparison_*.md' ! -name 'comparison_report.md' \
    -exec cat {} + > "$RUN_DIR/comparison_report.md" 2>/dev/null || true

echo "  $PY_COUNT comparison report(s) in $RUN_DIR/comparison_*.md"

# Aggregate all runs into summary CSV
echo ""
echo "  Aggregating all runs into summary.csv..."
"$VENV_PYTHON" "$SCRIPT_DIR/aggregate_results.py" \
    --runs-dir "$RUNS_DIR" \
    --output "$RESULTS_DIR/summary.csv" 2>&1

echo "  Generating viewer data..."
"$VENV_PYTHON" "$SCRIPT_DIR/generate_benchmark_json.py" \
    --run-dir "$RUN_DIR" \
    --hardware "$INTERNAL_DIR/hardware_specs.txt" \
    --output "$REPO_DIR/docs/benchmark_data.json" 2>&1

echo ""

# Print summary
total_benches=$("$VENV_PYTHON" -c "
import json
d = json.load(open('$RUN_DIR/rust_results.json'))
print(d['metadata']['total_benchmarks'])
" 2>/dev/null || echo "?")
echo "=== Done ==="
echo "  Run dir:      $RUN_DIR"
echo "  Features:     $FEATURE_LABEL"
echo "  Benchmarks:   $total_benches collected"
echo "  Results dir:  $RESULTS_DIR"
echo "  Summary:      $RESULTS_DIR/summary.csv"
echo ""
echo "To clear all results and start fresh:"
echo "  bash $0 --clear"

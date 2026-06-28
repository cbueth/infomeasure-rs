#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Benchmark comparison script for comparing Rust vs Python infomeasure performance.

Maps Python benchmark names (e.g. "kernel/mi/100/bw0.5/d1/hist2") to Rust
criterion keys (e.g. "mi_kernel/bw_0_5/100") using a known mapping table.

Usage:
    # Compare Python results from a run dir against Rust results
    python scripts/compare_benchmarks.py \
        --python runs/run_xxx/python_kernel.json \
        --rust runs/run_xxx/rust_results.json \
        --output comparison_kernel.md

    # Just show unmatched Python entries (debugging)
    python scripts/compare_benchmarks.py \
        --python runs/run_xxx/python_discrete.json \
        --rust runs/run_xxx/rust_results.json \
        --show-unmatched
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Python → Rust naming mapping
# =============================================================================

# Maps (python_group, python_measure) → rust_criterion_group_name.
# None means no Rust benchmark equivalent exists yet.
PYTHON_TO_RUST_GROUP: Dict[Tuple[str, str], Optional[str]] = {
    ("discrete", "entropy"): "entropy_discrete_small",
    ("discrete", "mi"): "mi_discrete",
    ("discrete", "cmi"): "cmi_discrete",
    ("discrete", "te"): "te_discrete",
    ("discrete", "cte"): "cte_discrete",
    ("kernel", "entropy"): None,  # no kernel entropy Rust bench yet
    ("kernel", "mi"): "mi_kernel",
    ("kernel", "cmi"): "cmi_kernel",
    ("kernel", "te"): "te_kernel",
    ("kernel", "cte"): "cte_kernel",
    ("kl", "entropy"): "entropy_kl",
    ("kl", "mi"): "mi_ksg",  # Python uses approach="ksg" for MI
    ("kl", "cmi"): "cmi_ksg",  # Python uses approach="ksg" for CMI
    ("kl", "te"): "te_ksg",  # Python uses approach="ksg" for TE
    ("kl", "cte"): "cte_ksg",  # Python uses approach="ksg" for CTE
    ("ordinal", "entropy"): "entropy_ordinal",
    ("ordinal", "mi"): "mi_ordinal",
    ("ordinal", "cmi"): "cmi_ordinal",
    ("ordinal", "te"): "te_ordinal",
    ("ordinal", "cte"): "cte_ordinal",
    ("renyi", "entropy"): "entropy_renyi",
    ("renyi", "mi"): "mi_renyi",
    ("renyi", "cmi"): "cmi_renyi",
    ("renyi", "te"): "te_renyi",
    ("renyi", "cte"): "cte_renyi",
    ("tsallis", "entropy"): "entropy_tsallis",
    ("tsallis", "mi"): "mi_tsallis",
    ("tsallis", "cmi"): "cmi_tsallis",
    ("tsallis", "te"): "te_tsallis",
    ("tsallis", "cte"): "cte_tsallis",
}


def _normalize_float(val: str) -> str:
    """Normalize Python float string to Rust f64::to_string() convention.

    Python str(1.0) = '1.0', but Rust 1.0f64.to_string() = '1'.
    Python str(0.5) = '0.5', Rust 0.5f64.to_string() = '0.5'.
    After normalization: strip trailing zeros + dot, then replace remaining '.' with '_'.
    """
    v = val.rstrip("0").rstrip(".")
    if "." in v:
        v = v.replace(".", "_")
    return v


def py_param_to_rust(
    py_group: str, py_measure: str, extra: Dict[str, str]
) -> Optional[str]:
    """Convert Python benchmark extra params to Rust criterion key param."""
    if py_group == "discrete":
        return "mle" if py_measure != "entropy" else "discrete"
    if py_group == "kernel":
        # Rust kernel benches use box kernel by default. Gaussian has no Rust match.
        if extra.get("kernel_type") == "gaussian":
            return None
        bw = extra.get("bw", "")
        bw_rust = bw.replace(".", "_")
        if bw_rust.endswith("_0") and bw_rust != "_0":
            bw_rust = bw_rust[:-2]
        return f"bw_{bw_rust}"
    if py_group == "kl":
        k = extra.get("k", "")
        return f"k{k}" if k else None
    if py_group == "ordinal":
        order = extra.get("order", "")
        return f"order_{order}" if order else None
    if py_group == "renyi":
        k = extra.get("k", "")
        alpha = extra.get("alpha", "")
        if k and alpha:
            alpha_rust = _normalize_float(alpha)
            return f"k{k}_alpha{alpha_rust}"
        return None
    if py_group == "tsallis":
        k = extra.get("k", "")
        q = extra.get("q", "")
        if k and q:
            q_rust = _normalize_float(q)
            return f"k{k}_q{q_rust}"
        return None
    return None


def parse_py_name(name: str) -> Optional[dict]:
    """Parse a Python benchmark name into components.

    Python format:  {group}/{measure}/{size}/{param1}/{param2}/...
    Known params:   bwX, kX, orderX, delayX, dX, histX
    """
    parts = name.split("/")
    if len(parts) < 3:
        return None
    group, measure, size = parts[0], parts[1], parts[2]
    extra = {}
    for p in parts[3:]:
        if p.startswith("bw"):
            extra["bw"] = p[2:]
        elif p.startswith("k") and not p.startswith("ksg"):
            extra["k"] = p[1:]
        elif p.startswith("order"):
            extra["order"] = p[5:]
        elif p.startswith("delay"):
            extra["delay"] = p[5:]
        elif p.startswith("d"):
            extra["dims"] = p[1:]
        elif p.startswith("hist"):
            extra["hist_len"] = p[4:]
        elif p.startswith("alpha"):
            extra["alpha"] = p[5:]
        elif p.startswith("q"):
            extra["q"] = p[1:]
        elif p in ("gaussian", "box"):
            extra["kernel_type"] = p
    return {"group": group, "measure": measure, "size": size, "extra": extra}


def py_name_to_rust_key(name: str) -> Optional[str]:
    """Translate a Python benchmark name into a Rust criterion key."""
    parsed = parse_py_name(name)
    if parsed is None:
        return None

    rust_group = PYTHON_TO_RUST_GROUP.get((parsed["group"], parsed["measure"]))
    if rust_group is None:
        return None

    rust_param = py_param_to_rust(parsed["group"], parsed["measure"], parsed["extra"])
    if rust_param is None:
        return None

    return f"{rust_group}/{rust_param}/{parsed['size']}"


# =============================================================================
# Data structures
# =============================================================================


@dataclass
class BenchmarkEntry:
    name: str
    mean: float
    stddev: float
    value: Optional[float] = None


@dataclass
class ComparisonResult:
    name: str
    python_mean: float
    rust_mean: float
    speedup: float
    python_value: Optional[float]
    rust_value: Optional[float]
    value_diff: Optional[float] = None


# =============================================================================
# Parsers
# =============================================================================


def parse_rust_criterion_json(json_path: str) -> Dict[str, BenchmarkEntry]:
    """Parse Rust criterion JSON output (collected format)."""
    with open(json_path) as f:
        data = json.load(f)

    benchmarks = {}
    if "benches" in data:
        for group_name, benches in data["benches"].items():
            for bench_key, stats in benches.items():
                benchmarks[bench_key] = BenchmarkEntry(
                    name=bench_key,
                    mean=stats.get("mean", 0),
                    stddev=stats.get("stddev", 0),
                    value=stats.get("value"),
                )
    return benchmarks


def parse_python_benchmark_json(json_path: str) -> Dict[str, BenchmarkEntry]:
    """Parse Python benchmark JSON output."""
    with open(json_path) as f:
        data = json.load(f)

    benchmarks = {}
    for bench in data.get("benchmarks", []):
        name = bench.get("name", "")
        if not name:
            continue
        stats = bench.get("statistics", {})
        benchmarks[name] = BenchmarkEntry(
            name=name,
            mean=stats.get("mean", 0),
            stddev=stats.get("stddev", 0),
            value=bench.get("value"),
        )
    return benchmarks


# =============================================================================
# Comparison
# =============================================================================


def compare_benchmarks(
    python_benchmarks: Dict[str, BenchmarkEntry],
    rust_benchmarks: Dict[str, BenchmarkEntry],
    show_unmatched: bool = False,
) -> List[ComparisonResult]:
    """Compare Python and Rust benchmarks using the name mapping."""
    results = []
    unmatched = []

    for py_name, py_entry in python_benchmarks.items():
        rust_key = py_name_to_rust_key(py_name)
        if rust_key is None:
            unmatched.append((py_name, "no Rust group mapping"))
            continue

        rust_entry = rust_benchmarks.get(rust_key)
        if rust_entry is None:
            unmatched.append((py_name, f"no Rust data for key {rust_key}"))
            continue

        speedup = py_entry.mean / rust_entry.mean if rust_entry.mean > 0 else 0
        value_diff = None
        if py_entry.value is not None and rust_entry.value is not None:
            value_diff = abs(py_entry.value - rust_entry.value)

        results.append(
            ComparisonResult(
                name=py_name,
                python_mean=py_entry.mean,
                rust_mean=rust_entry.mean,
                speedup=speedup,
                python_value=py_entry.value,
                rust_value=rust_entry.value,
                value_diff=value_diff,
            )
        )

    if show_unmatched:
        for name, reason in unmatched:
            print(f"  UNMATCHED: {name}  ({reason})")

    return results


# =============================================================================
# Report generation
# =============================================================================


def generate_report(comparisons: List[ComparisonResult], output_path: str) -> str:
    """Generate markdown comparison report."""
    lines = [
        "# Rust vs Python Benchmark Comparison",
        "",
        "| Benchmark | Python (s) | Rust (s) | Speedup |",
        "|-----------|-------------|-----------|---------|",
    ]

    for c in comparisons:
        lines.append(
            f"| {c.name} | {c.python_mean:.6f} | {c.rust_mean:.6f} | {c.speedup:.2f}x |"
        )

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    return report


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compare Rust vs Python benchmark results"
    )
    parser.add_argument(
        "--python", "-p", required=True, help="Python benchmark JSON file"
    )
    parser.add_argument("--rust", "-r", required=True, help="Rust benchmark JSON file")
    parser.add_argument(
        "--output",
        "-o",
        default="comparison_report.md",
        help="Output report file",
    )
    parser.add_argument(
        "--show-unmatched",
        action="store_true",
        help="Print Python entries that could not be matched to Rust",
    )
    args = parser.parse_args()

    # Load
    print(f"Loading Python benchmarks from: {args.python}")
    python_benchmarks = parse_python_benchmark_json(args.python)
    print(f"Found {len(python_benchmarks)} Python benchmarks")

    print(f"Loading Rust benchmarks from: {args.rust}")
    rust_benchmarks = parse_rust_criterion_json(args.rust)
    print(f"Found {len(rust_benchmarks)} Rust benchmarks")

    # Compare
    comparisons = compare_benchmarks(
        python_benchmarks,
        rust_benchmarks,
        show_unmatched=args.show_unmatched,
    )
    print(f"Matched {len(comparisons)} benchmarks")

    # Generate report
    if comparisons:
        report = generate_report(comparisons, args.output)
        print(f"\nReport saved to: {args.output}")
        print("\n" + report)
    else:
        print("\nNo benchmarks matched. Use --show-unmatched to debug.")


if __name__ == "__main__":
    main()

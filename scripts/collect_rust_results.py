#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Extract Rust criterion benchmark results from target/criterion/ into structured JSON.

Usage:
    python scripts/collect_rust_results.py --output internal/rust_results.json

    # Only specific benches
    python scripts/collect_rust_results.py --benches entropy_discrete,mi,cmi --output results.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def find_estimate_files(criterion_dir: Path) -> list[Path]:
    """Find all 'new/estimates.json' files in criterion benchmark directories."""
    return list(criterion_dir.rglob("new/estimates.json"))


def parse_bench_name(estimate_path: Path, criterion_dir: Path) -> Optional[str]:
    """Extract the benchmark name from criterion directory structure.

    Criterion structure: target/criterion/<group>/<param>/<size>/new/estimates.json
    """
    try:
        relative = estimate_path.relative_to(criterion_dir)
        parts = relative.parts
        # parts = (<group>, <param>, <size>, 'new', 'estimates.json')
        if len(parts) >= 4:
            group = parts[0]
            param = parts[1]
            size_val = parts[2]
            return f"{group}/{param}/{size_val}"
    except ValueError:
        pass
    return None


def parse_estimates(path: Path) -> Dict[str, Any]:
    """Parse criterion estimates.json to extract timing data."""
    with open(path) as f:
        data = json.load(f)

    mean_entry = data.get("mean", {})
    std_dev_entry = data.get("std_dev", {})

    # All times are in nanoseconds — convert to seconds
    mean_ns = mean_entry.get("point_estimate", 0)
    stddev_ns = std_dev_entry.get("point_estimate", 0)

    result = {
        "mean": mean_ns / 1e9,
        "stddev": stddev_ns / 1e9,
        "mean_ns": mean_ns,
        "stddev_ns": stddev_ns,
    }

    # Extract confidence intervals
    mean_ci = mean_entry.get("confidence_interval", {})
    if mean_ci:
        result["ci_lower"] = mean_ci.get("lower_bound", 0) / 1e9
        result["ci_upper"] = mean_ci.get("upper_bound", 0) / 1e9

    # Extract median
    median_entry = data.get("median", {})
    if median_entry:
        result["median"] = median_entry.get("point_estimate", 0) / 1e9

    # Extract slope (might differ from mean for some estimators)
    slope_entry = data.get("slope", {})
    if slope_entry:
        result["slope"] = slope_entry.get("point_estimate", 0) / 1e9

    return result


def collect_results(
    criterion_dir: Path, bench_filter: set[str] | None = None
) -> Dict[str, Any]:
    """Collect all criterion benchmark results."""
    results = {
        "benches": {},
        "metadata": {"source": "criterion", "path": str(criterion_dir)},
    }

    if not criterion_dir.exists():
        print(
            f"Warning: criterion directory not found: {criterion_dir}", file=sys.stderr
        )
        return results

    estimate_files = find_estimate_files(criterion_dir)
    if not estimate_files:
        print(f"Warning: no estimate files found in {criterion_dir}", file=sys.stderr)
        return results

    for est_path in sorted(estimate_files):
        bench_name = parse_bench_name(est_path, criterion_dir)
        if bench_name is None:
            continue

        group_name = bench_name.split("/")[0]
        if bench_filter and group_name not in bench_filter:
            continue

        stats = parse_estimates(est_path)

        if group_name not in results["benches"]:
            results["benches"][group_name] = {}
        results["benches"][group_name][bench_name] = stats

    # Count total
    total = sum(len(v) for v in results["benches"].values())
    results["metadata"]["total_benchmarks"] = total

    return results


def main():
    parser = argparse.ArgumentParser(description="Collect Rust criterion results")
    parser.add_argument(
        "--criterion-dir",
        default="target/criterion",
        help="Path to criterion output directory",
    )
    parser.add_argument(
        "--benches",
        help="Comma-separated list of bench groups to include (default: all)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="internal/rust_results.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    bench_filter = None
    if args.benches:
        bench_filter = set(b.strip() for b in args.benches.split(","))

    criterion_dir = Path(args.criterion_dir)
    results = collect_results(criterion_dir, bench_filter)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    total = results["metadata"]["total_benchmarks"]
    print(f"Collected {total} benchmarks from {len(results['benches'])} groups")
    print(f"Written to: {args.output}")

    if total == 0:
        print(f"\nHint: criterion directory '{criterion_dir}' may be empty or missing.")
        print("Run some Rust benchmarks first:")
        print("  cargo bench --bench entropy_discrete")


if __name__ == "__main__":
    main()

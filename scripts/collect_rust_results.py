#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Extract Rust criterion benchmark results from target/criterion/ into structured JSON.

Collects results from criterion's new/estimates.json, tags them with the
feature set used, and writes to a run directory for persistent storage.

Usage:
    python scripts/collect_rust_results.py --run-dir runs/my_run --features gpu
    python scripts/collect_rust_results.py --run-dir runs/my_run --output rust.json
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


def find_sample_file(estimates_path: Path) -> Optional[Path]:
    """Find corresponding sample.json in the same directory as estimates.json."""
    sample = estimates_path.parent / "sample.json"
    return sample if sample.exists() else None


def parse_bench_name(estimate_path: Path, criterion_dir: Path) -> Optional[str]:
    """Extract the benchmark name from criterion directory structure.

    Criterion structure: target/criterion/<group>/[...id_parts]/<size>/new/estimates.json
    where id_parts may span multiple segments when BenchmarkId contains '/' (e.g. "box/bw0_1").
    """
    try:
        relative = estimate_path.relative_to(criterion_dir)
        parts = relative.parts
        if len(parts) >= 4:
            size_val = parts[
                -3
            ]  # size is always 3rd from end (before /new/estimates.json)
            group = parts[0]
            id_parts = parts[1:-3]  # everything between group and size
            return f"{group}/{'/'.join(id_parts)}/{size_val}"
    except ValueError:
        pass
    return None


def parse_estimates(estimates_path: Path) -> Dict[str, Any]:
    """Parse criterion estimates.json + sample.json to extract timing data."""
    with open(estimates_path) as f:
        data = json.load(f)

    mean_entry = data.get("mean", {})
    std_dev_entry = data.get("std_dev", {})

    mean_ns = mean_entry.get("point_estimate", 0)
    stddev_ns = std_dev_entry.get("point_estimate", 0)
    std_err_ns = mean_entry.get("standard_error", 0)

    result = {
        "mean": mean_ns / 1e9,
        "stddev": stddev_ns / 1e9,
        "std_err": std_err_ns / 1e9,
        "mean_ns": mean_ns,
        "stddev_ns": stddev_ns,
        "std_err_ns": std_err_ns,
    }

    # Read sample.json for sample count (N)
    sample_path = find_sample_file(estimates_path)
    if sample_path:
        try:
            with open(sample_path) as f:
                sample_data = json.load(f)
            sample_times = sample_data.get("times", [])
            result["n_samples"] = len(sample_times)
        except (json.JSONDecodeError, KeyError):
            pass

    mean_ci = mean_entry.get("confidence_interval", {})
    if mean_ci:
        result["ci_lower"] = mean_ci.get("lower_bound", 0) / 1e9
        result["ci_upper"] = mean_ci.get("upper_bound", 0) / 1e9

    median_entry = data.get("median", {})
    if median_entry:
        result["median"] = median_entry.get("point_estimate", 0) / 1e9

    slope_entry = data.get("slope", {})
    if slope_entry:
        result["slope"] = slope_entry.get("point_estimate", 0) / 1e9

    return result


def collect_results(
    criterion_dir: Path, bench_filter: Optional[set[str]] = None
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

    total = sum(len(v) for v in results["benches"].values())
    results["metadata"]["total_benchmarks"] = total

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect Rust criterion results with feature tagging"
    )
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
        "--output", "-o", default="rust_results.json", help="Output JSON filename"
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Run directory (e.g. runs/run_2025-01-01_cpu). "
        "Output is written to <run-dir>/<output>.",
    )
    parser.add_argument(
        "--features",
        default="",
        help="Feature set used for this run (e.g. 'gpu', '' for baseline)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing output file instead of overwriting",
    )
    args = parser.parse_args()

    bench_filter = None
    if args.benches:
        bench_filter = set(b.strip() for b in args.benches.split(","))

    criterion_dir = Path(args.criterion_dir)
    results = collect_results(criterion_dir, bench_filter)

    # Add feature tag to each entry
    features_tag = [f.strip() for f in args.features.split(",") if f.strip()]
    for group_name, benches in results["benches"].items():
        for bench_key, stats in benches.items():
            stats["features"] = features_tag

    results["metadata"]["features"] = features_tag
    results["metadata"]["collection_timestamp"] = (
        __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat()
    )

    # Determine output path
    output_dir = Path(args.run_dir) if args.run_dir else Path.cwd()
    output_path = output_dir / args.output
    os.makedirs(output_path.parent, exist_ok=True)

    # Merge with existing file if requested
    if args.merge and output_path.exists():
        try:
            with open(output_path) as f:
                existing = json.load(f)
            for g, benches in results["benches"].items():
                existing.setdefault("benches", {})[g] = benches
            total = sum(len(v) for v in existing["benches"].values())
            existing["metadata"]["total_benchmarks"] = total
            if features_tag:
                existing["metadata"]["features"] = features_tag
            existing["metadata"]["collection_timestamp"] = results["metadata"][
                "collection_timestamp"
            ]
            results = existing
        except (json.JSONDecodeError, KeyError):
            pass

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total = results["metadata"]["total_benchmarks"]
    print(f"Collected {total} benchmarks from {len(results['benches'])} groups")
    print(f"Features: {features_tag}")
    print(f"Written to: {output_path}")

    if total == 0:
        print(f"\nHint: criterion directory '{criterion_dir}' may be empty or missing.")
        print("Run some Rust benchmarks first:")
        print("  cargo bench --bench entropy_discrete")


if __name__ == "__main__":
    main()

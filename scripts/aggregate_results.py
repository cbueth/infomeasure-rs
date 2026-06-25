#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Aggregate Rust + Python benchmark results into summary CSVs.

Usage:
    # Collect Rust results first, then aggregate
    python scripts/collect_rust_results.py --output internal/rust_results.json
    python scripts/aggregate_results.py

    # Full workflow with custom parameters
    python scripts/aggregate_results.py \
        --rust internal/rust_results.json \
        --python-dir scripts/benchmarks_data \
        --results-dir internal/benchmark_results \
        --timestamp "2025-01-01"
"""

import argparse
import json
import os
import sys
import csv
from pathlib import Path
from typing import Dict, Any, Optional


def load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_csv(rows: list[Dict[str, Any]], path: str):
    if not rows:
        print(f"  Warning: no data for {path}, skipping")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Use union of all keys across all rows (handles mixed fields)
    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def aggregate_summary(
    rust_path: Optional[str],
    python_dir: str,
    results_dir: str,
    timestamp: str,
):
    """Generate summary CSV with all benchmark data."""
    summary_rows = []

    # Load Rust data if available
    rust_data = {}
    if rust_path and Path(rust_path).exists():
        rust_data = load_json(rust_path)
        for group_name, benches in rust_data.get("benches", {}).items():
            for bench_key, stats in benches.items():
                summary_rows.append(
                    {
                        "source": "rust",
                        "group": group_name,
                        "benchmark": bench_key,
                        "mean_s": stats.get("mean", 0),
                        "stddev_s": stats.get("stddev", 0),
                        "min_s": stats.get("min", 0),
                        "max_s": stats.get("max", 0),
                        "median_s": stats.get("median", 0),
                        "samples": stats.get("count", 0),
                        "timestamp": timestamp,
                    }
                )

    # Load Python data if available
    python_dir_path = Path(python_dir)
    if python_dir_path.exists():
        for json_file in sorted(python_dir_path.glob("*.json")):
            group_name = json_file.stem
            py_data = load_json(str(json_file))
            for bench in py_data.get("benchmarks", []):
                stats = bench.get("statistics", {})
                summary_rows.append(
                    {
                        "source": "python",
                        "group": bench.get("group", group_name),
                        "benchmark": bench.get("name", ""),
                        "mean_s": stats.get("mean", 0),
                        "stddev_s": stats.get("stddev", 0),
                        "min_s": stats.get("min", 0),
                        "max_s": stats.get("max", 0),
                        "median_s": stats.get("median", 0),
                        "value": bench.get("value"),
                        "timestamp": timestamp,
                    }
                )

    write_csv(summary_rows, os.path.join(results_dir, "summary.csv"))


def aggregate_python_parity(
    python_dir: str,
    results_dir: str,
    timestamp: str,
):
    """Generate Python-parity CSV (Rust-only until comparison data is available)."""
    # Phase 2: merge with Rust comparison data from compare_benchmarks.py
    # For now, just note that data exists
    python_dir_path = Path(python_dir)
    total = 0
    measures = set()
    groups = set()
    if python_dir_path.exists():
        for json_file in sorted(python_dir_path.glob("*.json")):
            py_data = load_json(str(json_file))
            total += len(py_data.get("benchmarks", []))
            for b in py_data.get("benchmarks", []):
                measures.add(b.get("measure"))
                groups.add(b.get("group"))

    # Write a metadata file about what's available
    info = {
        "timestamp": timestamp,
        "python_benchmark_total": total,
        "groups": sorted(groups),
        "measures": sorted(measures),
    }
    info_path = os.path.join(results_dir, "python_data_available.json")
    os.makedirs(os.path.dirname(info_path) or ".", exist_ok=True)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Python benchmark info written to {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results")
    parser.add_argument(
        "--rust",
        default="internal/rust_results.json",
        help="Path to Rust results JSON (from collect_rust_results.py)",
    )
    parser.add_argument(
        "--python-dir",
        default="scripts/benchmarks_data",
        help="Directory containing Python benchmark JSON files",
    )
    parser.add_argument(
        "--results-dir",
        default="internal/benchmark_results",
        help="Directory to write aggregated results",
    )
    parser.add_argument(
        "--timestamp",
        default="",
        help="Timestamp for this run (default: auto)",
    )
    parser.add_argument(
        "--sizes",
        default="",
        help="Sizes used (for metadata)",
    )
    args = parser.parse_args()

    timestamp = args.timestamp or "unknown"

    print("Aggregating benchmark results...")
    aggregate_summary(args.rust, args.python_dir, args.results_dir, timestamp)
    aggregate_python_parity(args.python_dir, args.results_dir, timestamp)
    print("Done.")


if __name__ == "__main__":
    main()

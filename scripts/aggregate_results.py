#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""
Aggregate Rust + Python benchmark results from multiple run directories.

Scans <runs-dir>/run_*/ for rust_results.json and python_*.json files,
merges them into a summary CSV with a `features` column distinguishing
different configurations (CPU, GPU, etc.). Deduplicates by
(source, group, benchmark, features), keeping the latest run.

Usage:
    # Aggregate all runs in default directory
    python scripts/aggregate_results.py

    # Custom paths
    python scripts/aggregate_results.py \
        --runs-dir internal/benchmark_results/runs \
        --output internal/benchmark_results/summary.csv
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: skipping {path}: {e}", file=sys.stderr)
        return None


def extract_rust_rows(data: Dict, run_id: str) -> list[Dict]:
    """Extract rows from a collected Rust results JSON."""
    rows = []
    features = data.get("metadata", {}).get("features", [])
    features_str = ",".join(sorted(features)) if features else ""
    timestamp = data.get("metadata", {}).get("collection_timestamp", "")

    for group_name, benches in data.get("benches", {}).items():
        for bench_key, stats in benches.items():
            rows.append(
                {
                    "source": "rust",
                    "group": group_name,
                    "benchmark": bench_key,
                    "features": features_str,
                    "mean_s": stats.get("mean", 0),
                    "stddev_s": stats.get("stddev", 0),
                    "min_s": stats.get("min", 0),
                    "max_s": stats.get("max", 0),
                    "median_s": stats.get("median", 0),
                    "ci_lower_s": stats.get("ci_lower"),
                    "ci_upper_s": stats.get("ci_upper"),
                    "samples": stats.get("count", 0),
                    "run_id": run_id,
                    "timestamp": timestamp,
                }
            )
    return rows


def extract_python_rows(data: Dict, run_id: str) -> list[Dict]:
    """Extract rows from a Python benchmark results JSON."""
    rows = []
    features = data.get("metadata", {}).get("extra_params", {}).get("features", "")
    features_str = (
        features
        if isinstance(features, str)
        else ",".join(sorted(features))
        if features
        else ""
    )
    timestamp = data.get("metadata", {}).get("timestamp", "")

    for bench in data.get("benchmarks", []):
        stats = bench.get("statistics", {})
        rows.append(
            {
                "source": "python",
                "group": bench.get("group", ""),
                "measure": bench.get("measure", ""),
                "benchmark": bench.get("name", ""),
                "features": features_str,
                "mean_s": stats.get("mean", 0),
                "stddev_s": stats.get("stddev", 0),
                "min_s": stats.get("min", 0),
                "max_s": stats.get("max", 0),
                "median_s": stats.get("median", 0),
                "value": bench.get("value"),
                "params": json.dumps(bench.get("params", {})),
                "run_id": run_id,
                "timestamp": timestamp,
            }
        )
    return rows


def scan_run_directory(runs_dir: Path) -> list[Dict]:
    """Scan all run subdirectories and extract benchmark data."""
    all_rows = []

    if not runs_dir.exists():
        print(f"  Run directory not found: {runs_dir}")
        return all_rows

    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    )
    if not run_dirs:
        print(f"  No run directories found in {runs_dir}")
        return all_rows

    print(f"  Found {len(run_dirs)} run(s)")

    for run_dir in run_dirs:
        run_id = run_dir.name

        # Rust results
        rust_file = run_dir / "rust_results.json"
        if rust_file.exists():
            data = load_json(rust_file)
            if data:
                rows = extract_rust_rows(data, run_id)
                all_rows.extend(rows)
                print(f"    {run_id}: {len(rows)} Rust entries")

        # Python results
        for py_file in sorted(run_dir.glob("python_*.json")):
            data = load_json(py_file)
            if data:
                rows = extract_python_rows(data, run_id)
                all_rows.extend(rows)
                print(f"    {run_id}: {len(rows)} Python entries ({py_file.name})")

    return all_rows


def deduplicate_rows(rows: list[Dict]) -> list[Dict]:
    """Deduplicate by (source, group, benchmark, features), keeping last occurrence.

    Input rows should be ordered by run timestamp ascending.
    The last row for each unique key wins.
    """
    seen = {}
    for row in rows:
        key = (
            row.get("source", ""),
            row.get("group", ""),
            row.get("benchmark", ""),
            row.get("features", ""),
        )
        seen[key] = row  # last one wins
    return list(seen.values())


def write_csv(rows: list[Dict], path: str):
    if not rows:
        print(f"  No data to write for {path}")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Wrote {len(rows)} aggregated rows to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Rust + Python benchmark results from run directories"
    )
    parser.add_argument(
        "--runs-dir",
        default="internal/benchmark_results/runs",
        help="Directory containing run_* subdirectories",
    )
    parser.add_argument(
        "--output",
        default="internal/benchmark_results/summary.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    print("Aggregating benchmark results...")
    all_rows = scan_run_directory(runs_dir)

    if not all_rows:
        print("No data found. Run some benchmarks first:")
        print(f"  bash scripts/run_full_benchmark_suite.sh --quick")
        sys.exit(0)

    # Sort by timestamp then deduplicate (keep latest)
    all_rows.sort(key=lambda r: r.get("timestamp", ""))
    deduped = deduplicate_rows(all_rows)
    duplicates = len(all_rows) - len(deduped)
    if duplicates > 0:
        print(f"\n  Removed {duplicates} duplicate(s) (keeping latest per config)")

    write_csv(deduped, args.output)

    # Print summary by features
    from collections import Counter

    features_counts = Counter(r.get("features", "") for r in deduped)
    print("\n  Entries by feature set:")
    for feat, count in sorted(features_counts.items()):
        label = feat if feat else "(baseline / CPU)"
        print(f"    [{label}] {count} entries")


if __name__ == "__main__":
    main()

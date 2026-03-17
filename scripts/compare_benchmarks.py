#!/usr/bin/env python3
"""
Benchmark comparison script for comparing Rust vs Python infomeasure performance.

This script:
1. Runs Python benchmarks with matching parameters
2. Parses Rust criterion JSON output
3. Produces comparison reports with speedup calculations

Usage:
    # Run Python benchmarks
    python scripts/python_benchmark.py --output internal/python_bench.json --sizes 100,1000,10000

    # Run Rust benchmarks (in another terminal)
    cargo bench --bench entropy_discrete -- --output-format=json > rust_bench.json

    # Compare
    python scripts/compare_benchmarks.py --python internal/python_bench.json --rust-json target/criterion/... --output internal/comparison.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class BenchmarkEntry:
    """Single benchmark entry."""

    name: str
    mean: float
    stddev: float
    value: Optional[float] = None
    times: List[float] = None


@dataclass
class ComparisonResult:
    """Comparison between Python and Rust."""

    name: str
    python_mean: float
    rust_mean: float
    speedup: float
    python_value: Optional[float]
    rust_value: Optional[float]
    value_diff: Optional[float] = None


def parse_rust_criterion_json(json_path: str) -> Dict[str, BenchmarkEntry]:
    """Parse Rust criterion JSON output."""
    with open(json_path, "r") as f:
        data = json.load(f)

    benchmarks = {}
    for bench in data.get("benchmarks", []):
        name = bench.get("id", bench.get("name", ""))
        if not name:
            continue

        # Extract parameter info from name
        # Format: "group/param/size"

        stats = bench.get("statistics", bench.get("mean", {}))
        if isinstance(stats, dict):
            mean = stats.get("mean", 0)
            stddev = stats.get("stddev", 0)
        else:
            mean = stats
            stddev = 0

        # Get the value (entropy/mi/te estimate)
        value_field = bench.get("value", bench.get("result", {}).get("value"))

        benchmarks[name] = BenchmarkEntry(
            name=name,
            mean=mean,
            stddev=stddev,
            value=value_field,
            times=bench.get("times", []),
        )

    return benchmarks


def parse_python_benchmark_json(json_path: str) -> Dict[str, BenchmarkEntry]:
    """Parse Python benchmark JSON output."""
    with open(json_path, "r") as f:
        data = json.load(f)

    benchmarks = {}
    for bench in data.get("benchmarks", []):
        name = bench.get("name", "")
        if not name:
            continue

        stats = bench.get("statistics", {})
        mean = stats.get("mean", 0)
        stddev = stats.get("stddev", 0)

        benchmarks[name] = BenchmarkEntry(
            name=name,
            mean=mean,
            stddev=stddev,
            value=bench.get("value"),
            times=bench.get("times", []),
        )

    return benchmarks


def normalize_name(name: str) -> str:
    """Normalize benchmark name for comparison."""
    # Remove prefixes and normalize
    name = name.replace("entropy_discrete/entropy_discrete/", "entropy_discrete/")
    name = name.replace("mi_discrete/mi_discrete/", "mi_discrete/")
    name = name.replace("te_discrete/te_discrete/", "te_discrete/")
    return name


def compare_benchmarks(
    python_benchmarks: Dict[str, BenchmarkEntry],
    rust_benchmarks: Dict[str, BenchmarkEntry],
) -> List[ComparisonResult]:
    """Compare Python and Rust benchmarks."""
    results = []

    # Create normalized lookup for Rust
    rust_normalized = {normalize_name(k): v for k, v in rust_benchmarks.items()}

    for py_name, py_entry in python_benchmarks.items():
        py_normalized = normalize_name(py_name)

        rust_entry = rust_normalized.get(py_normalized)
        if rust_entry is None:
            # Try to find a close match
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

    return results


def generate_report(comparisons: List[ComparisonResult], output_path: str):
    """Generate comparison report."""
    lines = [
        "# Rust vs Python Benchmark Comparison",
        "",
        "| Benchmark | Python (s) | Rust (s) | Speedup | Value Diff |",
        "|-----------|-------------|-----------|---------|------------|",
    ]

    for c in comparisons:
        value_diff_str = f"{c.value_diff:.4f}" if c.value_diff is not None else "N/A"
        lines.append(
            f"| {c.name} | {c.python_mean:.6f} | {c.rust_mean:.6f} | {c.speedup:.2f}x | {value_diff_str} |"
        )

    # Add summary
    speedups = [c.speedup for c in comparisons if c.speedup > 0]
    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        lines.extend(
            [
                "",
                f"**Average Speedup: {avg_speedup:.2f}x**",
                "",
                f"**Total Benchmarks: {len(comparisons)}**",
            ]
        )

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare Rust vs Python benchmark results"
    )
    parser.add_argument(
        "--python", "-p", required=True, help="Python benchmark JSON file"
    )
    parser.add_argument("--rust", "-r", help="Rust benchmark JSON file (or directory)")
    parser.add_argument(
        "--output", "-o", default="comparison_report.md", help="Output report file"
    )
    parser.add_argument("--json", "-j", help="Output JSON comparison file")
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format",
    )

    args = parser.parse_args()

    # Load Python benchmarks
    print(f"Loading Python benchmarks from: {args.python}")
    python_benchmarks = parse_python_benchmark_json(args.python)
    print(f"Found {len(python_benchmarks)} Python benchmarks")

    rust_benchmarks = {}
    if args.rust:
        # Handle directory or file
        rust_path = Path(args.rust)
        if rust_path.is_dir():
            # Look for criterion JSON files
            json_files = list(rust_path.glob("**/*.json"))
            print(f"Found {len(json_files)} JSON files in {rust_path}")
        else:
            print(f"Loading Rust benchmarks from: {args.rust}")
            rust_benchmarks = parse_rust_criterion_json(args.rust)
            print(f"Found {len(rust_benchmarks)} Rust benchmarks")

    # Compare if Rust data available
    if rust_benchmarks:
        comparisons = compare_benchmarks(python_benchmarks, rust_benchmarks)
        print(f"Matched {len(comparisons)} benchmarks")

        if args.json:
            # Save JSON comparison
            json_output = {
                "comparisons": [
                    {
                        "name": c.name,
                        "python_mean": c.python_mean,
                        "rust_mean": c.rust_mean,
                        "speedup": c.speedup,
                        "python_value": c.python_value,
                        "rust_value": c.rust_value,
                        "value_diff": c.value_diff,
                    }
                    for c in comparisons
                ]
            }
            with open(args.json, "w") as f:
                json.dump(json_output, f, indent=2)
            print(f"JSON comparison saved to: {args.json}")

        # Generate report
        if args.format == "markdown" or args.output.endswith(".md"):
            report = generate_report(comparisons, args.output)
            print(f"\nReport saved to: {args.output}")
            print("\n" + report)
    else:
        # Just show Python results
        print("\nPython Benchmark Results:")
        print("-" * 60)
        for name, entry in sorted(python_benchmarks.items()):
            print(f"{name}: {entry.mean:.6f}s (value={entry.value})")


if __name__ == "__main__":
    main()

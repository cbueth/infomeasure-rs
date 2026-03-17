#!/usr/bin/env python3
"""
Run all benchmark groups with standard parameters and generate outputs.

Usage:
    python scripts/run_all_benchmarks.py

    # Or with custom parameters
    python scripts/run_all_benchmarks.py --sizes 100,1000,10000 --iterations 10
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd):
    """Run a command and return its result."""
    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(
        cmd, cwd=Path(__file__).parent.parent / "tests/validation_crate"
    )
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run all benchmark groups")
    parser.add_argument(
        "--sizes", default="100,200,500,1000,2000,5000,10000", help="Data sizes to benchmark"
    )
    parser.add_argument(
        "--history-lens", default="2,3,4,5,6", help="History lengths for MI/TE"
    )
    parser.add_argument("--k-values", default="2,3,4,5", help="K values for KL/KSG")
    parser.add_argument(
        "--bandwidths", default="0.1,0.3,0.5,1.0,1.5", help="Bandwidths for kernel"
    )
    parser.add_argument("--orders", default="2,3,4", help="Orders for ordinal")
    parser.add_argument("--delays", default="0,1,2", help="Delays for ordinal")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of iterations"
    )
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument(
        "--measures", default="entropy,mi,cmi,te,cte", help="Measures to run"
    )
    parser.add_argument(
        "--skip-markdown", action="store_true", help="Skip markdown generation"
    )

    args = parser.parse_args()

    # Find Python interpreter in venv
    venv_python = (
        Path(__file__).parent.parent / "tests/validation_crate/.venv/bin/python"
    )
    if not venv_python.exists():
        print("Error: Could not find Python venv")
        sys.exit(1)

    script_dir = Path(__file__).parent
    benchmark_script = script_dir / "python_benchmark.py"
    data_dir = script_dir / "benchmarks_data"
    data_dir.mkdir(exist_ok=True)

    common_args = [
        str(venv_python),
        str(benchmark_script),
        f"--sizes={args.sizes}",
        f"--history-lens={args.history_lens}",
        f"--iterations={args.iterations}",
        f"--warmup={args.warmup}",
        f"--measures={args.measures}",
    ]

    if not args.skip_markdown:
        common_args.append("--markdown")

    groups = [
        {
            "name": "discrete",
            "args": [
                f"--output={data_dir}/discrete.json",
            ],
        },
        {
            "name": "kernel",
            "args": [
                f"--bandwidths={args.bandwidths}",
                f"--output={data_dir}/kernel.json",
            ],
        },
        {
            "name": "kl",
            "args": [
                f"--k-values={args.k_values}",
                f"--output={data_dir}/kl.json",
            ],
        },
        {
            "name": "ordinal",
            "args": [
                f"--orders={args.orders}",
                f"--delays={args.delays}",
                f"--output={data_dir}/ordinal.json",
            ],
        },
    ]

    print(f"\nRunning benchmarks for {len(groups)} groups")
    print(f"Sizes: {args.sizes}")
    print(f"History lengths: {args.history_lens}")
    print(f"Measures: {args.measures}")

    all_success = True
    for group in groups:
        cmd = common_args + [f"--group={group['name']}"] + group["args"]
        if not run_command(cmd):
            all_success = False

    if all_success:
        print("\n" + "=" * 60)
        print("All benchmarks completed successfully!")
        print("=" * 60)
        print(f"\nOutput files:")
        for group in groups:
            print(f"  - {data_dir}/{group['name']}.json")
    else:
        print("\n" + "=" * 60)
        print("Some benchmarks failed!")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()

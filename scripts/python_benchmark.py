#!/usr/bin/env python3
"""
Python benchmarking script for comparing infomeasure Python vs Rust performance.

Structured by estimator type (group) with relevant parameters for each measure.

Usage:
    # Run by estimator group
    python scripts/python_benchmark.py --group kl --measures mi,te --sizes 100,1000

    # Run all groups
    python scripts/python_benchmark.py --all --measures entropy,mi,cmi,te,cte

Groups (estimator types):
    discrete  - No special params (just data size)
    kernel   - bandwidth, kernel type
    kl       - k (neighbors)
    ksg      - k, metric
    renyi    - k, alpha
    tsallis  - k, q
    ordinal  - order, delay

Measures:
    entropy  - Just the estimator
    mi       - Mutual Information
    cmi      - Conditional Mutual Information
    te       - Transfer Entropy
    cte      - Conditional Transfer Entropy
"""

import argparse
import json
import os
import platform
import sys
import time
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict, field
from statistics import mean, stdev, median

try:
    import infomeasure as im
except ImportError:
    print("Error: infomeasure package not found.")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    group: str
    measure: str
    name: str
    params: Dict[str, Any]
    mean: float
    stddev: float
    min: float
    max: float
    median: float
    times: List[float]
    value: Optional[float] = None


@dataclass
class BenchmarkMetadata:
    timestamp: str
    python_version: str
    infomeasure_version: str
    platform: str
    processor: str
    group: str
    measures: List[str]
    sizes: List[int]
    iterations: int
    warmup: int
    extra_params: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Data Generation Functions (matching Rust benchmarks)
# ============================================================================


def generate_discrete_data(
    size: int, num_states: int = 10, seed: int = 42
) -> np.ndarray:
    """Generate random discrete data."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_states, size=size, dtype=np.int32)


def generate_gaussian_data(size: int, dims: int = 1, seed: int = 42) -> np.ndarray:
    """Generate random Gaussian data."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(size, dims)).astype(np.float64)


def generate_correlated_pair(size: int, correlation: float = 0.5, seed: int = 42):
    """Generate correlated Gaussian pair."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size).astype(np.float64)
    w = rng.standard_normal(size).astype(np.float64)
    x = z
    y = (correlation * z + np.sqrt(1 - correlation**2) * w).astype(np.float64)
    return x, y


def generate_time_series(
    size: int, coupling: float = 0.5, lag: int = 1, seed: int = 42
):
    """Generate coupled time series."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(size + lag).astype(np.float64)
    target = np.zeros(size, dtype=np.float64)
    for i in range(size):
        noise = rng.standard_normal()
        target[i] = coupling * source[i + lag] + np.sqrt(1 - coupling**2) * noise
    return source[:size], target


# ============================================================================
# Benchmark Runner
# ============================================================================


def run_benchmark(func, *args, warmup: int = 3, iterations: int = 10, **kwargs) -> Dict:
    """Run a benchmark with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Timed runs
    times = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = func(*args, **kwargs)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)

    return {
        "times": times,
        "mean": mean(times),
        "stddev": stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "median": median(times),
        "value": float(result) if result is not None else None,
    }


# ============================================================================
# Estimator-Specific Benchmarks
# ============================================================================


def benchmark_discrete(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    history_lens: List[int],
) -> List[BenchmarkResult]:
    """Benchmark discrete estimator."""
    results = []
    num_states = 10
    seed = 42

    for measure in measures:
        for size in sizes:
            for hist_len in history_lens if measure != "entropy" else [None]:
                # Generate data
                x = generate_discrete_data(size, num_states, seed)

                # Build params dict
                params = {"num_states": num_states}
                if hist_len:
                    params["history_len"] = hist_len

                name_base = f"discrete/{measure}/{size}"
                if hist_len:
                    name_base += f"/hist{hist_len}"

                # Run appropriate measure
                try:
                    if measure == "entropy":
                        stats = run_benchmark(
                            im.entropy,
                            x,
                            approach="discrete",
                            warmup=warmup,
                            iterations=iterations,
                        )
                    elif measure == "mi":
                        y = generate_discrete_data(size, num_states, seed + 1)
                        stats = run_benchmark(
                            im.mutual_information,
                            x,
                            y,
                            approach="discrete",
                            warmup=warmup,
                            iterations=iterations,
                        )
                    elif measure == "cmi":
                        y = generate_discrete_data(size, num_states, seed + 1)
                        z = generate_discrete_data(size, num_states, seed + 2)
                        stats = run_benchmark(
                            im.mutual_information,
                            x,
                            y,
                            cond=z,
                            approach="discrete",
                            warmup=warmup,
                            iterations=iterations,
                        )
                    elif measure == "te":
                        y = np.roll(x, hist_len or 1)
                        y[0] = x[0]
                        stats = run_benchmark(
                            im.transfer_entropy,
                            x,
                            y,
                            approach="discrete",
                            warmup=warmup,
                            iterations=iterations // 2,
                        )
                    elif measure == "cte":
                        y = np.roll(x, hist_len or 1)
                        y[0] = x[0]
                        z = generate_discrete_data(size, num_states, seed + 3)
                        stats = run_benchmark(
                            im.transfer_entropy,
                            x,
                            y,
                            cond=z,
                            approach="discrete",
                            warmup=warmup,
                            iterations=iterations // 2,
                        )
                    else:
                        continue

                    results.append(
                        BenchmarkResult(
                            group="discrete",
                            measure=measure,
                            name=name_base,
                            params=params,
                            mean=stats["mean"],
                            stddev=stats["stddev"],
                            min=stats["min"],
                            max=stats["max"],
                            median=stats["median"],
                            times=stats["times"],
                            value=stats["value"],
                        )
                    )
                except Exception as e:
                    print(f"  Warning: {measure} failed: {e}")

    return results


def benchmark_kernel(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    history_lens: List[int],
    bandwidths: List[float],
) -> List[BenchmarkResult]:
    """Benchmark kernel estimator."""
    results = []
    seed = 42

    for measure in measures:
        for size in sizes:
            for bw in bandwidths:
                for hist_len in history_lens if measure != "entropy" else [None]:
                    # Generate data
                    x = generate_gaussian_data(size, 1, seed).reshape(-1)

                    params = {"bandwidth": bw, "kernel": "gaussian"}
                    if hist_len:
                        params["history_len"] = hist_len

                    name_base = f"kernel/{measure}/{size}/bw{bw}"
                    if hist_len:
                        name_base += f"/hist{hist_len}"

                    try:
                        if measure == "entropy":
                            stats = run_benchmark(
                                im.entropy,
                                x,
                                approach="kernel",
                                kernel="gaussian",
                                bandwidth=bw,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "mi":
                            x, y = generate_correlated_pair(size, 0.5, seed)
                            stats = run_benchmark(
                                im.mutual_information,
                                x,
                                y,
                                approach="kernel",
                                kernel="gaussian",
                                bandwidth=bw,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "cmi":
                            x, y = generate_correlated_pair(size, 0.5, seed)
                            z = generate_gaussian_data(size, 1, seed + 2).reshape(-1)
                            stats = run_benchmark(
                                im.mutual_information,
                                x,
                                y,
                                cond=z,
                                approach="kernel",
                                kernel="gaussian",
                                bandwidth=bw,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "te":
                            src, tgt = generate_time_series(
                                size, 0.5, hist_len or 1, seed
                            )
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                approach="kernel",
                                kernel="gaussian",
                                bandwidth=bw,
                                warmup=warmup,
                                iterations=iterations // 2,
                            )
                        elif measure == "cte":
                            src, tgt = generate_time_series(
                                size, 0.5, hist_len or 1, seed
                            )
                            cond = generate_gaussian_data(size, 1, seed + 3).reshape(-1)
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                cond=cond,
                                approach="kernel",
                                kernel="gaussian",
                                bandwidth=bw,
                                warmup=warmup,
                                iterations=iterations // 2,
                            )
                        else:
                            continue

                        results.append(
                            BenchmarkResult(
                                group="kernel",
                                measure=measure,
                                name=name_base,
                                params=params,
                                mean=stats["mean"],
                                stddev=stats["stddev"],
                                min=stats["min"],
                                max=stats["max"],
                                median=stats["median"],
                                times=stats["times"],
                                value=stats["value"],
                            )
                        )
                    except Exception as e:
                        print(f"  Warning: {measure} failed: {e}")

    return results


def benchmark_kl(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    history_lens: List[int],
    k_values: List[int],
) -> List[BenchmarkResult]:
    """Benchmark KL (Kozachenko-Leonenko) estimator."""
    results = []
    seed = 42
    noise = 1e-10

    for measure in measures:
        for size in sizes:
            for k in k_values:
                for hist_len in history_lens if measure != "entropy" else [None]:
                    x = generate_gaussian_data(size, 1, seed).reshape(-1)

                    params = {"k": k}
                    if hist_len:
                        params["history_len"] = hist_len

                    name_base = f"kl/{measure}/{size}/k{k}"
                    if hist_len:
                        name_base += f"/hist{hist_len}"

                    try:
                        if measure == "entropy":
                            stats = run_benchmark(
                                im.entropy,
                                x.reshape(-1, 1),
                                approach="kl",
                                k=k,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "mi":
                            x, y = generate_correlated_pair(size, 0.5, seed)
                            stats = run_benchmark(
                                im.mutual_information,
                                x,
                                y,
                                approach="ksg",
                                k=k,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "cmi":
                            x, y = generate_correlated_pair(size, 0.5, seed)
                            z = generate_gaussian_data(size, 1, seed + 2).reshape(-1)
                            stats = run_benchmark(
                                im.mutual_information,
                                x,
                                y,
                                cond=z,
                                approach="ksg",
                                k=k,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "te":
                            src, tgt = generate_time_series(
                                size, 0.5, hist_len or 1, seed
                            )
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                approach="ksg",
                                k=k,
                                warmup=warmup,
                                iterations=iterations // 2,
                            )
                        elif measure == "cte":
                            src, tgt = generate_time_series(
                                size, 0.5, hist_len or 1, seed
                            )
                            cond = generate_gaussian_data(size, 1, seed + 3).reshape(-1)
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                cond=cond,
                                approach="ksg",
                                k=k,
                                warmup=warmup,
                                iterations=iterations // 2,
                            )
                        else:
                            continue

                        results.append(
                            BenchmarkResult(
                                group="kl",
                                measure=measure,
                                name=name_base,
                                params=params,
                                mean=stats["mean"],
                                stddev=stats["stddev"],
                                min=stats["min"],
                                max=stats["max"],
                                median=stats["median"],
                                times=stats["times"],
                                value=stats["value"],
                            )
                        )
                    except Exception as e:
                        print(f"  Warning: {measure} failed: {e}")

    return results


def benchmark_ordinal(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    history_lens: List[int],
    orders: List[int],
    delays: List[int],
) -> List[BenchmarkResult]:
    """Benchmark ordinal estimator."""
    results = []
    seed = 42

    for measure in measures:
        for size in sizes:
            for order in orders:
                for delay in delays:
                    for hist_len in history_lens if measure != "entropy" else [None]:
                        x = generate_gaussian_data(size, 1, seed).reshape(-1)

                        params = {"order": order, "delay": delay}
                        if hist_len:
                            params["history_len"] = hist_len

                        name_base = (
                            f"ordinal/{measure}/{size}/order{order}/delay{delay}"
                        )
                        if hist_len:
                            name_base += f"/hist{hist_len}"

                        try:
                            if measure == "entropy":
                                stats = run_benchmark(
                                    im.entropy,
                                    x,
                                    approach="ordinal",
                                    embedding_dim=order,
                                    warmup=warmup,
                                    iterations=iterations,
                                )
                            elif measure == "mi":
                                x, y = generate_correlated_pair(size, 0.5, seed)
                                stats = run_benchmark(
                                    im.mutual_information,
                                    x,
                                    y,
                                    approach="ordinal",
                                    embedding_dim=order,
                                    warmup=warmup,
                                    iterations=iterations,
                                )
                            elif measure == "cmi":
                                x, y = generate_correlated_pair(size, 0.5, seed)
                                z = generate_gaussian_data(size, 1, seed + 2).reshape(
                                    -1
                                )
                                stats = run_benchmark(
                                    im.mutual_information,
                                    x,
                                    y,
                                    cond=z,
                                    approach="ordinal",
                                    embedding_dim=order,
                                    warmup=warmup,
                                    iterations=iterations,
                                )
                            elif measure == "te":
                                src, tgt = generate_time_series(
                                    size, 0.5, hist_len or 1, seed
                                )
                                stats = run_benchmark(
                                    im.transfer_entropy,
                                    src,
                                    tgt,
                                    approach="ordinal",
                                    embedding_dim=order,
                                    warmup=warmup,
                                    iterations=iterations // 2,
                                )
                            elif measure == "cte":
                                src, tgt = generate_time_series(
                                    size, 0.5, hist_len or 1, seed
                                )
                                cond = generate_gaussian_data(
                                    size, 1, seed + 3
                                ).reshape(-1)
                                stats = run_benchmark(
                                    im.transfer_entropy,
                                    src,
                                    tgt,
                                    cond=cond,
                                    approach="ordinal",
                                    embedding_dim=order,
                                    warmup=warmup,
                                    iterations=iterations // 2,
                                )
                            else:
                                continue

                            results.append(
                                BenchmarkResult(
                                    group="ordinal",
                                    measure=measure,
                                    name=name_base,
                                    params=params,
                                    mean=stats["mean"],
                                    stddev=stats["stddev"],
                                    min=stats["min"],
                                    max=stats["max"],
                                    median=stats["median"],
                                    times=stats["times"],
                                    value=stats["value"],
                                )
                            )
                        except Exception as e:
                            print(f"  Warning: {measure} failed: {e}")

    return results


def benchmark_renyi(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    history_lens: List[int],
    k_values: List[int],
    alphas: List[float],
) -> List[BenchmarkResult]:
    """Benchmark Rényi entropy estimator."""
    results = []
    seed = 42
    noise = 1e-10

    for measure in measures:
        if measure == "entropy":
            for size in sizes:
                for k in k_values:
                    for alpha in alphas:
                        x = generate_gaussian_data(size, 1, seed).reshape(-1)
                        params = {"k": k, "alpha": alpha}
                        name_base = f"renyi/{measure}/{size}/k{k}/alpha{alpha}"

                        try:
                            stats = run_benchmark(
                                im.entropy,
                                x.reshape(-1, 1),
                                approach="renyi",
                                k=k,
                                alpha=noise,
                                warmup=warmup,
                                iterations=iterations,
                            )

                            results.append(
                                BenchmarkResult(
                                    group="renyi",
                                    measure=measure,
                                    name=name_base,
                                    params=params,
                                    mean=stats["mean"],
                                    stddev=stats["stddev"],
                                    min=stats["min"],
                                    max=stats["max"],
                                    median=stats["median"],
                                    times=stats["times"],
                                    value=stats["value"],
                                )
                            )
                        except Exception as e:
                            print(f"  Warning: {measure} failed: {e}")
        else:
            print(f"  Note: renyi not fully implemented for {measure}, skipping")

    return results


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark infomeasure by estimator group"
    )
    parser.add_argument(
        "--group",
        "-g",
        choices=["discrete", "kernel", "kl", "ordinal", "renyi", "all"],
        help="Estimator group to benchmark",
    )
    parser.add_argument(
        "--measures",
        "-m",
        default="entropy,mi,te",
        help="Comma-separated measures (entropy,mi,cmi,te,cte)",
    )
    parser.add_argument(
        "--sizes", "-s", default="100,1000,10000", help="Comma-separated data sizes"
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=10, help="Number of iterations"
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=3, help="Number of warmup runs"
    )
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Also generate markdown table for docs",
    )

    # Extra parameters
    parser.add_argument("--k-values", default="1,3,5", help="K values for KL/KSG")
    parser.add_argument(
        "--bandwidths", default="0.1,0.5,1.0", help="Bandwidths for kernel"
    )
    parser.add_argument("--orders", default="2,3,4", help="Orders for ordinal")
    parser.add_argument("--delays", default="1", help="Delays for ordinal")
    parser.add_argument(
        "--alphas", default="0.5,1.0,2.0", help="Alpha values for Rényi"
    )
    parser.add_argument(
        "--history-lens", default="1,2,3", help="History lengths for MI/TE"
    )

    args = parser.parse_args()

    # Parse parameters
    sizes = [int(x) for x in args.sizes.split(",")]
    measures = [x.strip() for x in args.measures.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]
    bandwidths = [float(x) for x in args.bandwidths.split(",")]
    orders = [int(x) for x in args.orders.split(",")]
    delays = [int(x) for x in args.delays.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    history_lens = [int(x) for x in args.history_lens.split(",")]

    # Determine groups to run
    if args.group == "all":
        groups = ["discrete", "kernel", "kl", "ordinal", "renyi"]
    else:
        groups = [args.group] if args.group else ["discrete"]

    # Get versions
    try:
        version = getattr(im, "__version__", "unknown")
    except:
        version = "unknown"

    all_results = []

    for group in groups:
        print(f"\n=== Benchmarking {group} ===")

        if group == "discrete":
            all_results.extend(
                benchmark_discrete(
                    sizes, measures, args.warmup, args.iterations, history_lens
                )
            )
        elif group == "kernel":
            all_results.extend(
                benchmark_kernel(
                    sizes,
                    measures,
                    args.warmup,
                    args.iterations,
                    history_lens,
                    bandwidths,
                )
            )
        elif group == "kl":
            all_results.extend(
                benchmark_kl(
                    sizes,
                    measures,
                    args.warmup,
                    args.iterations,
                    history_lens,
                    k_values,
                )
            )
        elif group == "ordinal":
            all_results.extend(
                benchmark_ordinal(
                    sizes,
                    measures,
                    args.warmup,
                    args.iterations,
                    history_lens,
                    orders,
                    delays,
                )
            )
        elif group == "renyi":
            all_results.extend(
                benchmark_renyi(
                    sizes,
                    measures,
                    args.warmup,
                    args.iterations,
                    history_lens,
                    k_values,
                    alphas,
                )
            )

    # Build metadata
    metadata = BenchmarkMetadata(
        timestamp=datetime.now(timezone.utc).isoformat(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        infomeasure_version=version,
        platform=platform.platform(),
        processor=platform.processor(),
        group=args.group or "mixed",
        measures=measures,
        sizes=sizes,
        iterations=args.iterations,
        warmup=args.warmup,
        extra_params={
            "k_values": k_values,
            "bandwidths": bandwidths,
            "orders": orders,
            "delays": delays,
            "alphas": alphas,
            "history_lens": history_lens,
        },
    )

    # Output
    output = {
        "metadata": asdict(metadata),
        "benchmarks": [
            {
                "group": r.group,
                "measure": r.measure,
                "name": r.name,
                "params": r.params,
                "statistics": {
                    "mean": r.mean,
                    "stddev": r.stddev,
                    "min": r.min,
                    "max": r.max,
                    "median": r.median,
                },
                "times": r.times,
                "value": r.value,
            }
            for r in all_results
        ],
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults: {len(all_results)} benchmarks")
    print(f"Written to: {args.output}")

    # Generate markdown table if requested
    if args.markdown:
        md_output = args.output.replace(".json", ".md")
        generate_markdown_table(output, md_output)
        print(f"Markdown table written to: {md_output}")


def generate_markdown_table(data: dict, output_path: str):
    """Generate a markdown table for docs.rs showing time vs data size."""
    benchmarks = data.get("benchmarks", [])
    metadata = data.get("metadata", {})

    # Group by estimator group and measure
    by_group = {}
    for b in benchmarks:
        key = (b.get("group"), b.get("measure"))
        if key not in by_group:
            by_group[key] = []
        by_group[key].append(b)

    lines = [
        "<!-- Auto-generated from benchmark data -->",
        "",
        "## Performance Benchmarks",
        "",
        f"*Measured on: {metadata.get('platform', 'unknown')} ({metadata.get('processor', 'unknown')})*",
        f"*Python version: {metadata.get('python_version', 'unknown')}*",
        "",
    ]

    for (group, measure), items in sorted(by_group.items()):
        lines.append(f"### {group.title()} {measure.upper()}")
        lines.append("")

        # Create table - use the name directly for clarity
        lines.append("| Benchmark | Mean Time (s) | Std Dev |")
        lines.append("|-----------|---------------|---------|")

        # Sort by name for consistent ordering
        for b in sorted(items, key=lambda x: x["name"]):
            name = b["name"]
            mean_time = b["statistics"]["mean"]
            std = b["statistics"]["stddev"]

            lines.append(f"| {name} | {mean_time:.6f} | {std:.6f} |")

        lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

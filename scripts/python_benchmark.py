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
    size: int, coupling: float = 0.5, lag: int = 1, seed: int = 42, dims: int = 1
):
    """Generate coupled time series. When dims > 1, returns (N, D) arrays."""
    rng = np.random.default_rng(seed)
    source = rng.standard_normal(size=(size + lag, dims)).astype(np.float64)
    noise = rng.standard_normal(size=(size, dims)).astype(np.float64)
    target = coupling * source[lag:] + np.sqrt(1 - coupling**2) * noise
    if dims == 1:
        return source[:size, 0], target[:, 0]
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


BIAS_METHODS = [
    "mle",
    "grassberger",
    "nsb",
    "miller_madow",
    "ansb",
    "chao_shen",
    "chao_wang_jost",
    "shrink",
    "bayes",
    "bonachela",
    "zhang",
]


def benchmark_discrete(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
) -> List[BenchmarkResult]:
    """Benchmark discrete estimator."""
    results = []
    num_states = 10
    seed = 42

    for measure in measures:
        for size in sizes:
            x = generate_discrete_data(size, num_states, seed)

            methods = BIAS_METHODS

            for method in methods:
                params = {"num_states": num_states, "method": method}
                name_base = f"discrete/{measure}/{size}/{method}"

                approach = method if method != "mle" else "discrete"
                extra = {"alpha": 1.0} if method == "bayes" else {}

                try:
                    if measure == "entropy":
                        stats = run_benchmark(
                            im.entropy,
                            x,
                            approach=approach,
                            warmup=warmup,
                            iterations=iterations,
                            **extra,
                        )
                    elif measure == "mi":
                        y = generate_discrete_data(size, num_states, seed + 1)
                        stats = run_benchmark(
                            im.mutual_information,
                            x,
                            y,
                            approach=approach,
                            warmup=warmup,
                            iterations=iterations,
                            **extra,
                        )
                    elif measure == "cmi":
                        y = generate_discrete_data(size, num_states, seed + 1)
                        z = generate_discrete_data(size, num_states, seed + 2)
                        stats = run_benchmark(
                            im.mutual_information,
                            x,
                            y,
                            cond=z,
                            approach=approach,
                            warmup=warmup,
                            iterations=iterations,
                            **extra,
                        )
                    elif measure == "te":
                        y = np.roll(x, 1)
                        y[0] = x[0]
                        stats = run_benchmark(
                            im.transfer_entropy,
                            x,
                            y,
                            approach=approach,
                            src_hist_len=1,
                            dest_hist_len=1,
                            warmup=warmup,
                            iterations=iterations // 2,
                            **extra,
                        )
                    elif measure == "cte":
                        y = np.roll(x, 1)
                        y[0] = x[0]
                        z = generate_discrete_data(size, num_states, seed + 3)
                        stats = run_benchmark(
                            im.transfer_entropy,
                            x,
                            y,
                            cond=z,
                            approach=approach,
                            src_hist_len=1,
                            dest_hist_len=1,
                            cond_hist_len=1,
                            warmup=warmup,
                            iterations=iterations // 2,
                            **extra,
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
                    print(f"  Warning: {measure}/{method} failed: {e}")

    return results


def benchmark_kernel(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    bandwidths: List[float],
    kernel_types: Optional[List[str]] = None,
    dimensions: Optional[List[int]] = None,
    kernel_iterations: Optional[int] = None,
) -> List[BenchmarkResult]:
    """Benchmark kernel estimator."""
    results = []
    seed = 42
    kiter = kernel_iterations if kernel_iterations is not None else iterations
    kernel_types = kernel_types or ["gaussian", "box"]
    dimensions = dimensions or [1]

    total = (
        len(measures)
        * len(dimensions)
        * len(kernel_types)
        * len(sizes)
        * len(bandwidths)
    )
    done = 0

    for dims in dimensions:
        for ktype in kernel_types:
            for measure in measures:
                for size in sizes:
                    for bw in bandwidths:
                        params = {"bandwidth": bw, "kernel": ktype, "dims": dims}

                        name_base = f"kernel/{measure}/{size}/bw{bw}/{ktype}/d{dims}"

                        done += 1
                        print(
                            f"  [{done}/{total}] {name_base}...",
                            end=" ",
                            flush=True,
                        )

                        try:
                            if measure == "entropy":
                                x = generate_gaussian_data(size, dims, seed)
                                if dims == 1:
                                    x = x.reshape(-1)
                                stats = run_benchmark(
                                    im.entropy,
                                    x,
                                    approach="kernel",
                                    kernel=ktype,
                                    bandwidth=bw,
                                    warmup=warmup,
                                    iterations=kiter,
                                )
                            elif measure == "mi":
                                x = generate_gaussian_data(size, dims, seed)
                                y = generate_gaussian_data(size, dims, seed + 1)
                                if dims == 1:
                                    x, y = x.reshape(-1), y.reshape(-1)
                                stats = run_benchmark(
                                    im.mutual_information,
                                    x,
                                    y,
                                    approach="kernel",
                                    kernel=ktype,
                                    bandwidth=bw,
                                    warmup=warmup,
                                    iterations=kiter,
                                )
                            elif measure == "cmi":
                                x = generate_gaussian_data(size, dims, seed)
                                y = generate_gaussian_data(size, dims, seed + 1)
                                z = generate_gaussian_data(size, dims, seed + 2)
                                if dims == 1:
                                    x, y, z = (
                                        x.reshape(-1),
                                        y.reshape(-1),
                                        z.reshape(-1),
                                    )
                                stats = run_benchmark(
                                    im.mutual_information,
                                    x,
                                    y,
                                    cond=z,
                                    approach="kernel",
                                    kernel=ktype,
                                    bandwidth=bw,
                                    warmup=warmup,
                                    iterations=kiter,
                                )
                            elif measure == "te":
                                src, tgt = generate_time_series(
                                    size, 0.5, 1, seed, dims
                                )
                                stats = run_benchmark(
                                    im.transfer_entropy,
                                    src,
                                    tgt,
                                    approach="kernel",
                                    kernel=ktype,
                                    bandwidth=bw,
                                    src_hist_len=1,
                                    dest_hist_len=1,
                                    step_size=1,
                                    warmup=warmup,
                                    iterations=kiter // 2 + 1,
                                )
                            elif measure == "cte":
                                src, tgt = generate_time_series(
                                    size, 0.5, 1, seed, dims
                                )
                                cond = generate_gaussian_data(size, dims, seed + 3)
                                if dims == 1:
                                    cond = cond.reshape(-1)
                                stats = run_benchmark(
                                    im.transfer_entropy,
                                    src,
                                    tgt,
                                    cond=cond,
                                    approach="kernel",
                                    kernel=ktype,
                                    bandwidth=bw,
                                    src_hist_len=1,
                                    dest_hist_len=1,
                                    cond_hist_len=1,
                                    step_size=1,
                                    warmup=warmup,
                                    iterations=kiter // 2 + 1,
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
                            print(f"{stats['mean'] * 1000:.1f}ms")
                        except Exception as e:
                            print(f"FAILED: {e}")

    return results


def benchmark_kl(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    k_values: List[int],
    dimensions: Optional[List[int]] = None,
) -> List[BenchmarkResult]:
    """Benchmark KL (Kozachenko-Leonenko) estimator."""
    results = []
    seed = 42
    dimensions = dimensions or [1]

    total = len(dimensions) * len(measures) * len(sizes) * len(k_values)
    done = 0

    for dims in dimensions:
        for measure in measures:
            for size in sizes:
                for k in k_values:
                    done += 1
                    params = {"k": k, "dims": dims}

                    name_base = f"kl/{measure}/{size}/k{k}/d{dims}"

                    print(
                        f"  [{done}/{total}] {name_base}...",
                        end=" ",
                        flush=True,
                    )

                    try:
                        if measure == "entropy":
                            x = generate_gaussian_data(size, dims, seed)
                            stats = run_benchmark(
                                im.entropy,
                                x,
                                approach="kl",
                                k=k,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "mi":
                            x = generate_gaussian_data(size, dims, seed)
                            y = generate_gaussian_data(size, dims, seed + 1)
                            if dims == 1:
                                x, y = x.reshape(-1), y.reshape(-1)
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
                            x = generate_gaussian_data(size, dims, seed)
                            y = generate_gaussian_data(size, dims, seed + 1)
                            z = generate_gaussian_data(size, dims, seed + 2)
                            if dims == 1:
                                x, y, z = (
                                    x.reshape(-1),
                                    y.reshape(-1),
                                    z.reshape(-1),
                                )
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
                            src, tgt = generate_time_series(size, 0.5, 1, seed, dims)
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                approach="ksg",
                                k=k,
                                src_hist_len=1,
                                dest_hist_len=1,
                                warmup=warmup,
                                iterations=iterations // 2,
                            )
                        elif measure == "cte":
                            src, tgt = generate_time_series(size, 0.5, 1, seed, dims)
                            cond = generate_gaussian_data(size, dims, seed + 3)
                            if dims == 1:
                                cond = cond.reshape(-1)
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                cond=cond,
                                approach="ksg",
                                k=k,
                                src_hist_len=1,
                                dest_hist_len=1,
                                cond_hist_len=1,
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
                        print(f"{stats['mean'] * 1000:.1f}ms")
                    except Exception as e:
                        print(f"FAILED: {e}")

    return results


def benchmark_ordinal(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    orders: List[int],
) -> List[BenchmarkResult]:
    """Benchmark ordinal estimator."""
    results = []
    seed = 42

    for measure in measures:
        for size in sizes:
            for order in orders:
                x = generate_gaussian_data(size, 1, seed).reshape(-1)

                params = {"order": order}

                name_base = f"ordinal/{measure}/{size}/order{order}"

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
                        z = generate_gaussian_data(size, 1, seed + 2).reshape(-1)
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
                        src, tgt = generate_time_series(size, 0.5, 1, seed)
                        stats = run_benchmark(
                            im.transfer_entropy,
                            src,
                            tgt,
                            approach="ordinal",
                            embedding_dim=order,
                            src_hist_len=1,
                            dest_hist_len=1,
                            step_size=1,
                            warmup=warmup,
                            iterations=iterations // 2,
                        )
                    elif measure == "cte":
                        src, tgt = generate_time_series(size, 0.5, 1, seed)
                        cond = generate_gaussian_data(size, 1, seed + 3).reshape(-1)
                        stats = run_benchmark(
                            im.transfer_entropy,
                            src,
                            tgt,
                            cond=cond,
                            approach="ordinal",
                            embedding_dim=order,
                            src_hist_len=1,
                            dest_hist_len=1,
                            cond_hist_len=1,
                            step_size=1,
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
                    print(f"{stats['mean'] * 1000:.1f}ms")
                except Exception as e:
                    print(f"  Warning: {measure} failed: {e}")

    return results


def benchmark_renyi(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    k_values: List[int],
    alphas: List[float],
    dimensions: Optional[List[int]] = None,
) -> List[BenchmarkResult]:
    """Benchmark Rényi estimator (entropy, mi, cmi, te, cte)."""
    results = []
    seed = 42
    noise = 1e-10
    dimensions = dimensions or [1]

    total = len(dimensions) * len(measures) * len(sizes) * len(k_values) * len(alphas)

    if total == 0:
        return results

    done = 0
    for dims in dimensions:
        for measure in measures:
            for size in sizes:
                for k in k_values:
                    for alpha in alphas:
                        done += 1
                    params = {"k": k, "alpha": alpha, "dims": dims}
                    name_base = f"renyi/{measure}/{size}/k{k}/alpha{alpha}/d{dims}"

                    print(
                        f"  [{done}/{total}] {name_base}...",
                        end=" ",
                        flush=True,
                    )

                    try:
                        if measure == "entropy":
                            x = generate_gaussian_data(size, dims, seed)
                            if dims == 1:
                                x = x.reshape(-1)
                            stats = run_benchmark(
                                im.entropy,
                                x,
                                approach="renyi",
                                k=k,
                                alpha=alpha,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "mi":
                            x = generate_gaussian_data(size, dims, seed)
                            y = generate_gaussian_data(size, dims, seed + 1)
                            if dims == 1:
                                x, y = x.reshape(-1), y.reshape(-1)
                            stats = run_benchmark(
                                im.mutual_information,
                                x,
                                y,
                                approach="renyi",
                                k=k,
                                alpha=alpha,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "cmi":
                            x = generate_gaussian_data(size, dims, seed)
                            y = generate_gaussian_data(size, dims, seed + 1)
                            z = generate_gaussian_data(size, dims, seed + 2)
                            if dims == 1:
                                x, y, z = (
                                    x.reshape(-1),
                                    y.reshape(-1),
                                    z.reshape(-1),
                                )
                            stats = run_benchmark(
                                im.mutual_information,
                                x,
                                y,
                                cond=z,
                                approach="renyi",
                                k=k,
                                alpha=alpha,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "te":
                            src, tgt = generate_time_series(size, 0.5, 1, seed, dims)
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                approach="renyi",
                                k=k,
                                alpha=alpha,
                                src_hist_len=1,
                                dest_hist_len=1,
                                warmup=warmup,
                                iterations=iterations // 2,
                            )
                        elif measure == "cte":
                            src, tgt = generate_time_series(size, 0.5, 1, seed, dims)
                            cond = generate_gaussian_data(size, dims, seed + 3)
                            if dims == 1:
                                cond = cond.reshape(-1)
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                cond=cond,
                                approach="renyi",
                                k=k,
                                alpha=alpha,
                                src_hist_len=1,
                                dest_hist_len=1,
                                cond_hist_len=1,
                                warmup=warmup,
                                iterations=iterations // 2,
                            )
                        else:
                            continue

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
                        print(f"{stats['mean'] * 1000:.1f}ms")
                    except Exception as e:
                        print(f"FAILED: {e}")

    return results


def benchmark_tsallis(
    sizes: List[int],
    measures: List[str],
    warmup: int,
    iterations: int,
    k_values: List[int],
    q_values: List[float],
    dimensions: Optional[List[int]] = None,
) -> List[BenchmarkResult]:
    """Benchmark Tsallis estimator (entropy, mi, cmi, te, cte)."""
    results = []
    seed = 42
    dimensions = dimensions or [1]

    total = len(dimensions) * len(measures) * len(sizes) * len(k_values) * len(q_values)

    if total == 0:
        return results

    done = 0
    for dims in dimensions:
        for measure in measures:
            for size in sizes:
                for k in k_values:
                    for q in q_values:
                        done += 1
                    params = {"k": k, "q": q, "dims": dims}
                    name_base = f"tsallis/{measure}/{size}/k{k}/q{q}/d{dims}"

                    print(
                        f"  [{done}/{total}] {name_base}...",
                        end=" ",
                        flush=True,
                    )

                    try:
                        if measure == "entropy":
                            x = generate_gaussian_data(size, dims, seed)
                            if dims == 1:
                                x = x.reshape(-1)
                            stats = run_benchmark(
                                im.entropy,
                                x,
                                approach="tsallis",
                                k=k,
                                q=q,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "mi":
                            x = generate_gaussian_data(size, dims, seed)
                            y = generate_gaussian_data(size, dims, seed + 1)
                            if dims == 1:
                                x, y = x.reshape(-1), y.reshape(-1)
                            stats = run_benchmark(
                                im.mutual_information,
                                x,
                                y,
                                approach="tsallis",
                                k=k,
                                q=q,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "cmi":
                            x = generate_gaussian_data(size, dims, seed)
                            y = generate_gaussian_data(size, dims, seed + 1)
                            z = generate_gaussian_data(size, dims, seed + 2)
                            if dims == 1:
                                x, y, z = (
                                    x.reshape(-1),
                                    y.reshape(-1),
                                    z.reshape(-1),
                                )
                            stats = run_benchmark(
                                im.mutual_information,
                                x,
                                y,
                                cond=z,
                                approach="tsallis",
                                k=k,
                                q=q,
                                warmup=warmup,
                                iterations=iterations,
                            )
                        elif measure == "te":
                            src, tgt = generate_time_series(size, 0.5, 1, seed, dims)
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                approach="tsallis",
                                k=k,
                                q=q,
                                src_hist_len=1,
                                dest_hist_len=1,
                                warmup=warmup,
                                iterations=iterations // 2,
                            )
                        elif measure == "cte":
                            src, tgt = generate_time_series(size, 0.5, 1, seed, dims)
                            cond = generate_gaussian_data(size, dims, seed + 3)
                            if dims == 1:
                                cond = cond.reshape(-1)
                            stats = run_benchmark(
                                im.transfer_entropy,
                                src,
                                tgt,
                                cond=cond,
                                approach="tsallis",
                                k=k,
                                q=q,
                                src_hist_len=1,
                                dest_hist_len=1,
                                cond_hist_len=1,
                                warmup=warmup,
                                iterations=iterations // 2,
                            )
                        else:
                            continue

                        results.append(
                            BenchmarkResult(
                                group="tsallis",
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
                        print(f"{stats['mean'] * 1000:.1f}ms")
                    except Exception as e:
                        print(f"FAILED: {e}")

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
        choices=["discrete", "kernel", "kl", "ordinal", "renyi", "tsallis", "all"],
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
    parser.add_argument(
        "--bandwidths", default="0.1,0.5,1.0", help="Bandwidths for kernel"
    )
    parser.add_argument("--k-values", default="1,3,5", help="K values for KL/KSG")
    parser.add_argument(
        "--kernel-types",
        default="gaussian,box",
        help="Kernel types for kernel estimator",
    )
    parser.add_argument(
        "--kernel-iterations",
        type=int,
        default=0,
        help="Iterations for kernel benchmarks (0 = use --iterations value)",
    )
    parser.add_argument("--orders", default="2,3,4", help="Orders for ordinal")
    parser.add_argument(
        "--alphas", default="0.5,1.0,2.0", help="Alpha values for Rényi"
    )
    parser.add_argument(
        "--q-values", default="0.5,1.5,2.0", help="Q values for Tsallis"
    )
    parser.add_argument(
        "--dimensions",
        default="1",
        help="Comma-separated RV dimensions for scaling analysis (used by kernel/kl/renyi)",
    )

    args = parser.parse_args()

    # Parse parameters
    sizes = [int(x) for x in args.sizes.split(",")]
    measures = [x.strip() for x in args.measures.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]
    bandwidths = [float(x) for x in args.bandwidths.split(",")]
    orders = [int(x) for x in args.orders.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    q_values = [float(x) for x in args.q_values.split(",")]
    # Parse dimensions
    dimensions = [int(x) for x in args.dimensions.split(",")]

    # Determine groups to run
    if args.group == "all":
        groups = ["discrete", "kernel", "kl", "ordinal", "renyi", "tsallis"]
    else:
        groups = [args.group] if args.group else ["discrete"]

    # Get versions
    try:
        version = getattr(im, "__version__", "unknown")
    except:
        version = "unknown"

    # Resume support: load existing results and collect completed benchmark names
    completed_names = set()
    all_results = []
    if args.output and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                existing = json.load(f)
            for b in existing.get("benchmarks", []):
                completed_names.add(b["name"])
            # Reuse existing results (don't re-benchmark)
            for b in existing.get("benchmarks", []):
                all_results.append(b)
            print(
                f"Resuming: {len(completed_names)} benchmarks already completed in {args.output}"
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: could not load existing output ({e}), starting fresh")

    # Wrap benchmark functions to skip completed names
    def skipped_benchmark_discrete(*a, **kw):
        new_results = benchmark_discrete(*a, **kw)
        fresh = [r for r in new_results if r.name not in completed_names]
        skipped = len(new_results) - len(fresh)
        if skipped:
            print(f"  (skipped {skipped} already-completed discrete benchmarks)")
        return fresh

    def skipped_benchmark_kernel(*a, **kw):
        new_results = benchmark_kernel(*a, **kw)
        fresh = [r for r in new_results if r.name not in completed_names]
        skipped = len(new_results) - len(fresh)
        if skipped:
            print(f"  (skipped {skipped} already-completed kernel benchmarks)")
        return fresh

    def skipped_benchmark_kl(*a, **kw):
        new_results = benchmark_kl(*a, **kw)
        fresh = [r for r in new_results if r.name not in completed_names]
        skipped = len(new_results) - len(fresh)
        if skipped:
            print(f"  (skipped {skipped} already-completed kl benchmarks)")
        return fresh

    def skipped_benchmark_ordinal(*a, **kw):
        new_results = benchmark_ordinal(*a, **kw)
        fresh = [r for r in new_results if r.name not in completed_names]
        skipped = len(new_results) - len(fresh)
        if skipped:
            print(f"  (skipped {skipped} already-completed ordinal benchmarks)")
        return fresh

    def skipped_benchmark_renyi(*a, **kw):
        new_results = benchmark_renyi(*a, **kw)
        fresh = [r for r in new_results if r.name not in completed_names]
        skipped = len(new_results) - len(fresh)
        if skipped:
            print(f"  (skipped {skipped} already-completed renyi benchmarks)")
        return fresh

    def skipped_benchmark_tsallis(*a, **kw):
        new_results = benchmark_tsallis(*a, **kw)
        fresh = [r for r in new_results if r.name not in completed_names]
        skipped = len(new_results) - len(fresh)
        if skipped:
            print(f"  (skipped {skipped} already-completed tsallis benchmarks)")
        return fresh

    for group in groups:
        print(f"\n=== Benchmarking {group} ===")

        if group == "discrete":
            all_results.extend(
                skipped_benchmark_discrete(
                    sizes, measures, args.warmup, args.iterations
                )
            )
        elif group == "kernel":
            kernel_types = [x.strip() for x in args.kernel_types.split(",")]
            kernel_iterations = args.kernel_iterations or args.iterations
            all_results.extend(
                skipped_benchmark_kernel(
                    sizes,
                    measures,
                    args.warmup,
                    args.iterations,
                    bandwidths,
                    kernel_types,
                    dimensions,
                    kernel_iterations,
                )
            )
        elif group == "kl":
            all_results.extend(
                skipped_benchmark_kl(
                    sizes,
                    measures,
                    args.warmup,
                    args.iterations,
                    k_values,
                    dimensions,
                )
            )
        elif group == "ordinal":
            all_results.extend(
                skipped_benchmark_ordinal(
                    sizes,
                    measures,
                    args.warmup,
                    args.iterations,
                    orders,
                )
            )
        elif group == "renyi":
            all_results.extend(
                skipped_benchmark_renyi(
                    sizes,
                    measures,
                    args.warmup,
                    args.iterations,
                    k_values,
                    alphas,
                    dimensions,
                )
            )
        elif group == "tsallis":
            all_results.extend(
                skipped_benchmark_tsallis(
                    sizes,
                    measures,
                    args.warmup,
                    args.iterations,
                    k_values,
                    q_values,
                    dimensions,
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
            "alphas": alphas,
            "q_values": q_values,
            "dimensions": dimensions,
        },
    )

    # Output
    def _serialize_benchmark(r):
        if isinstance(r, dict):
            return r
        return {
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

    output = {
        "metadata": asdict(metadata),
        "benchmarks": [_serialize_benchmark(r) for r in all_results],
    }

    # Ensure directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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

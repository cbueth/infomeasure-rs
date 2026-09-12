# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Shared helpers for the cross-package benchmark collectors.

Every collector reads the canonical datasets written by
``benches/gen_datasets.rs`` and writes a schema-v2 *fragment*
(``<data>/results/<package>.json``) that ``merge_results.py`` unions into the
final ``cross_package.json``. Timing contract: warm-up calls are discarded,
then a fixed number of timed iterations; statistics are pooled across seeds.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

COLS = {"entropy": 1, "mi": 2, "te": 2, "cmi": 3, "cte": 3}
MEASURES = ["entropy", "mi", "cmi", "te", "cte"]


def data_dir() -> Path:
    return Path(os.environ.get("BENCH_DATA_DIR", "target/bench-data"))


def results_dir() -> Path:
    d = data_dir() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_manifest() -> dict:
    import json

    return json.loads((data_dir() / "manifest.json").read_text())


def sizes_and_seeds() -> tuple[list[int], list[int]]:
    m = read_manifest()
    sizes = m["sizes"]
    override = os.environ.get("BENCH_SIZES")
    if override:
        sizes = [int(s) for s in override.split(",") if s.strip()]
    return sizes, m["seeds"]


def load(measure: str, kind: str, seed: int, n: int) -> np.ndarray:
    """Load one dataset as an (n, cols) array."""
    dtype = "<i4" if kind == "discrete" else "<f8"
    path = data_dir() / f"{measure}_{kind}_s{seed}_n{n}.bin"
    arr = np.fromfile(path, dtype=dtype)
    return arr.reshape(-1, COLS[measure])


def timing_config() -> tuple[bool, int, int]:
    short = os.environ.get("BENCH_SHORT", "") in ("1", "true", "True")
    warmup = int(os.environ.get("BENCH_WARMUP", 1 if short else 3))
    iterations = int(os.environ.get("BENCH_ITERATIONS", 3 if short else 10))
    return short, warmup, iterations


def time_call(fn, warmup: int, iterations: int) -> tuple[list[float], float | None]:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    value = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - t0)
    return times, value


def stats(times: list[float]) -> dict:
    t = np.asarray(times, dtype=float)
    n = t.size
    mean = float(t.mean())
    stddev = float(t.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * stddev / np.sqrt(n) if n > 0 else 0.0
    return {
        "mean": mean,
        "stddev": stddev,
        "min": float(t.min()),
        "max": float(t.max()),
        "median": float(np.median(t)),
        "samples": n,
        "ci_lower": mean - half,
        "ci_upper": mean + half,
    }


def entry(
    package: str,
    language: str,
    measure: str,
    approach: str,
    function: str,
    n: int,
    params: dict,
    st: dict,
    value: float | None = None,
    notes: str | None = None,
) -> dict:
    return {
        "id": f"{measure}/{approach}/n{n}/{package}",
        "package": package,
        "language": language,
        "measure": measure,
        "approach": approach,
        "function": function,
        "params": params,
        "statistics": st,
        "value": value,
        "notes": notes,
    }


def write_fragment(
    package: str,
    language: str,
    version: str,
    benchmarks: list[dict],
    seeds: list[int],
    warmup: int,
    iterations: int,
    short: bool,
    extra: dict | None = None,
    limitations: str | None = None,
) -> Path:
    import json

    pkg = {"id": package, "language": language, "version": version}
    if extra:
        pkg.update(extra)
    # Human-readable coverage limits, rendered as an on-page note next to the
    # "N/A" cells for this package (e.g. "discrete only", "no continuous CTE").
    pkg["limitations"] = limitations
    meta = {
        "schema": 2,
        "run_id": f"fragment_{int(time.time())}",
        "hardware": None,
        "runtime": {
            "threads": 1,
            "warmup": warmup,
            "iterations": iterations,
            "short": short,
        },
        "seeds": seeds,
        "packages": [pkg],
        "coverage": sorted({(b["measure"], b["approach"]) for b in benchmarks}),
    }
    out = results_dir() / f"{package}.json"
    out.write_text(json.dumps({"meta": meta, "benchmarks": benchmarks}, indent=2))
    print(f"wrote {len(benchmarks)} entries to {out}")
    return out


def default_params(measure: str, approach: str, n: int) -> dict:
    return {
        "n": n,
        "k": 4 if approach in ("ksg",) else None,
        "bandwidth": 0.5 if approach.startswith("kernel") else None,
        "order": None,
        "delay": 1,
        "alpha": None,
        "q": None,
        "dims": 1,
        "method": "mle" if approach == "discrete" else None,
        "kernel_type": approach.split("_", 1)[1]
        if approach.startswith("kernel_")
        else None,
    }

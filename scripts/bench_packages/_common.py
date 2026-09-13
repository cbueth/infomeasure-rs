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


def silence_logging() -> None:
    """Silence library logging so the timed region measures compute only.

    infomeasure-python logs ANSB / Chao-Wang-Jost warnings through the stdlib
    ``logging`` module on every call; the adaptive loop calls the estimator many
    times, so those messages are formatted *inside* the timed call. They are
    informational (the estimators still run) and dataset-independent, so
    suppressing them is the fair fix rather than picking data that avoids the
    warning. ``loguru`` is also muted defensively for any dependency that uses
    it.
    """
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    try:
        from loguru import logger as _loguru

        _loguru.remove()
    except Exception:  # noqa: BLE001 - loguru is optional
        pass


silence_logging()


def data_dir() -> Path:
    return Path(os.environ.get("BENCH_DATA_DIR", "target/bench-data"))


def results_dir() -> Path:
    d = data_dir() / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def resuming() -> bool:
    """Whether this run should keep and skip an existing fragment."""
    return os.environ.get("BENCH_RESUME", "") in ("1", "true", "True")


def existing_benchmarks(package: str, fingerprint: str | None = None) -> list[dict]:
    """Benchmarks from an existing ``results/<package>.json`` fragment.

    Used by resumable collectors: load the partial fragment, skip the entry ids
    it already contains, and rewrite it as new measures complete. When
    ``fingerprint`` is given, the fragment is only reused if its stored
    fingerprint matches — so a package-version, grid or config change starts a
    fresh collection instead of silently keeping stale entries.
    """
    import json

    path = results_dir() / f"{package}.json"
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if fingerprint is not None:
        got = obj.get("meta", {}).get("fingerprint")
        if got != fingerprint:
            print(
                f"resume: {package} fingerprint changed ({got!r} != {fingerprint!r}); "
                "starting fresh"
            )
            return []
    benchmarks = obj.get("benchmarks", [])
    return benchmarks if isinstance(benchmarks, list) else []


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


def timing_config(detail_only: bool = False) -> dict:
    """Timing/budget config.

    Short mode is fixed (1 warm-up + up to 3 iters) for fast local iteration.
    Full mode is *adaptive* to bound CI wall time: warm up to `warmup_max` times
    (stop early once `warmup_budget` seconds elapsed), then run at least
    `min_iters` and up to `max_iters` timed iterations, stopping once
    `iter_budget` seconds elapsed (always at least `min_iters`).

    `detail_only=True` selects a tighter budget for the detailed grid variants
    (single seed) so the full infomeasure grid stays inside its time target.
    """
    short = os.environ.get("BENCH_SHORT", "") in ("1", "true", "True")
    if short:
        return {
            "short": True,
            "warmup_max": 1,
            "warmup_budget": 0.0,
            "min_iters": 1,
            "max_iters": 2,
            "iter_budget": 0.0,
        }
    if detail_only:
        # Profile A: guarantees >=5 samples (10 for fast calls) so the
        # reported stddev/CI are meaningful. Set BENCH_DETAIL_MIN_ITERS=3 for
        # profile B if the wall-time budget is tight.
        return {
            "short": False,
            "warmup_max": int(os.environ.get("BENCH_DETAIL_WARMUP_MAX", 1)),
            "warmup_budget": float(os.environ.get("BENCH_DETAIL_WARMUP_BUDGET_S", 0.1)),
            "min_iters": int(os.environ.get("BENCH_DETAIL_MIN_ITERS", 5)),
            "max_iters": int(os.environ.get("BENCH_DETAIL_MAX_ITERS", 10)),
            "iter_budget": float(os.environ.get("BENCH_DETAIL_ITER_BUDGET_S", 0.3)),
        }
    return {
        "short": False,
        "warmup_max": int(os.environ.get("BENCH_WARMUP_MAX", 3)),
        "warmup_budget": float(os.environ.get("BENCH_WARMUP_BUDGET_S", 0.4)),
        "min_iters": int(os.environ.get("BENCH_MIN_ITERS", 3)),
        "max_iters": int(os.environ.get("BENCH_MAX_ITERS", 10)),
        "iter_budget": float(os.environ.get("BENCH_ITER_BUDGET_S", 1.5)),
    }


def time_call(fn, cfg: dict) -> tuple[list[float], float | None]:
    t0 = time.perf_counter()
    warm = 0
    while True:
        fn()
        warm += 1
        if warm >= cfg["warmup_max"]:
            break
        if cfg["warmup_budget"] > 0 and time.perf_counter() - t0 >= cfg["warmup_budget"]:
            break
    times: list[float] = []
    value = None
    t0 = time.perf_counter()
    while True:
        start = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - start)
        n = len(times)
        if n >= cfg["max_iters"]:
            break
        if n >= cfg["min_iters"] and (
            cfg["iter_budget"] <= 0 or time.perf_counter() - t0 >= cfg["iter_budget"]
        ):
            break
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
    slug: str | None = None,
    representative: bool = True,
) -> dict:
    id_suffix = f"/{slug}" if slug else ""
    return {
        "id": f"{measure}/{approach}{id_suffix}/n{n}/{package}",
        "package": package,
        "language": language,
        "measure": measure,
        "approach": approach,
        "function": function,
        "representative": representative,
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
    cfg: dict,
    extra: dict | None = None,
    limitations: str | None = None,
    fingerprint: str | None = None,
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
        "fingerprint": fingerprint,
        "hardware": None,
        "runtime": {
            "threads": 1,
            "adaptive": not cfg["short"],
            "warmup_max": cfg["warmup_max"],
            "warmup_budget_s": cfg["warmup_budget"],
            "min_iters": cfg["min_iters"],
            "max_iters": cfg["max_iters"],
            "iter_budget_s": cfg["iter_budget"],
        },
        "seeds": seeds,
        "packages": [pkg],
        "coverage": sorted({(b["measure"], b["approach"]) for b in benchmarks}),
    }
    out = results_dir() / f"{package}.json"
    out.write_text(json.dumps(_clean_nonfinite({"meta": meta, "benchmarks": benchmarks}), indent=2))
    print(f"wrote {len(benchmarks)} entries to {out}")
    return out


def _clean_nonfinite(obj):
    """Replace non-finite floats with None so the fragment is valid JSON.

    ``json.dumps`` emits ``NaN``/``Infinity`` by default, which is not valid
    JSON and makes the browser's ``JSON.parse`` throw.
    """
    import math

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nonfinite(v) for v in obj]
    return obj


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

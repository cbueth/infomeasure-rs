#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Collector: the Python infomeasure package.

Times only the estimator call on the shared canonical datasets and writes
``results/infomeasure-python.json`` (schema v2 fragment). The full detailed
grid (``benches/detailed_grid.json``) is collected: representative variants
(flagged ``representative: true``) use all seeds and the cross-package size
set, detailed-only variants use the first seed and a tighter budget. See
``scripts/bench_packages/_common.py`` for the timing contract.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _grid  # noqa: E402
from _common import (  # noqa: E402
    entry,
    load,
    read_manifest,
    stats,
    time_call,
    timing_config,
    write_fragment,
)

import infomeasure as im  # noqa: E402

APPROACHES = ("discrete", "ksg", "kernel", "ordinal", "renyi", "tsallis")

HIST = {"src_hist_len": 1, "dest_hist_len": 1}
HIST_COND = {"src_hist_len": 1, "dest_hist_len": 1, "cond_hist_len": 1}


def build_fn(measure: str, v: dict, cols):
    """Return ``(callable, function_name)`` for one grid variant."""
    x = cols[:, 0]
    approach = v["approach"]

    if approach == "discrete":
        method = v["method"] or "mle"
        kw: dict = {"approach": "discrete" if method == "mle" else method}
        if method == "bayes":
            kw["alpha"] = 1.0
    elif approach == "ksg":
        # infomeasure names the kNN entropy estimator "kl", MI/CMI/TE "ksg".
        kw = {"approach": "kl" if measure == "entropy" else "ksg", "k": v["k"]}
    elif approach == "kernel":
        kw = {
            "approach": "kernel",
            "kernel": v["kernel"],
            "bandwidth": v["bandwidth"],
        }
    elif approach == "ordinal":
        kw = {"approach": "ordinal", "embedding_dim": v["order"]}
    elif approach == "renyi":
        kw = {"approach": "renyi", "k": v["k"], "alpha": v["alpha"]}
    elif approach == "tsallis":
        kw = {"approach": "tsallis", "k": v["k"], "q": v["q"]}
    else:  # pragma: no cover - rust-only variants are filtered by the grid
        raise ValueError(f"unsupported approach for python: {approach}")

    if measure == "entropy":
        return lambda: float(im.entropy(x, **kw)), "im.entropy"
    y = cols[:, 1]
    if measure == "mi":
        return lambda: float(im.mutual_information(x, y, **kw)), "im.mutual_information"
    if measure == "cmi":
        z = cols[:, 2]
        return (
            lambda: float(im.mutual_information(x, y, cond=z, **kw)),
            "im.mutual_information",
        )
    if measure == "te":
        return (
            lambda: float(im.transfer_entropy(x, y, **HIST, **kw)),
            "im.transfer_entropy",
        )
    if measure == "cte":
        z = cols[:, 2]
        return (
            lambda: float(im.transfer_entropy(x, y, cond=z, **HIST_COND, **kw)),
            "im.transfer_entropy",
        )
    raise ValueError(measure)


def main() -> int:
    cross_cfg = timing_config()
    detail_cfg = timing_config(detail_only=True)
    cross_sizes, detailed_sizes = _grid.sizes()
    override = os.environ.get("BENCH_SIZES")
    if override:
        parsed = [int(s) for s in override.split(",") if s.strip()]
        if parsed:
            cross_sizes = detailed_sizes = parsed
    seeds_all = read_manifest()["seeds"]
    benchmarks: list[dict] = []

    cross_set = set(cross_sizes)
    for v in _grid.variants("python"):
        measure = v["measure"]
        approach = v["approach"]
        kind = "discrete" if approach == "discrete" else "continuous"

        for n in detailed_sizes:
            # Representative entries: a representative variant at a cross size.
            # All seeds + standard adaptive budget; everything else one seed +
            # the tight detailed budget.
            is_rep = v["cross"] and n in cross_set
            seeds = seeds_all if is_rep else seeds_all[:1]
            cfg = cross_cfg if is_rep else detail_cfg

            times: list[float] = []
            value = None
            for seed in seeds:
                # File I/O is deliberately outside the timed region.
                cols = load(measure, kind, seed, n)
                fn, fname = build_fn(measure, v, cols)
                t, value = time_call(fn, cfg)
                times.extend(t)
            st = stats(times)
            print(
                f"  {measure:>7} {approach:<16} {v['slug']:<22} n={n:<6} "
                f"{st['mean'] * 1e3:>9.3f} ms{'  [cross]' if is_rep else ''}"
            )
            benchmarks.append(
                entry(
                    "infomeasure-python",
                    "python",
                    measure,
                    approach,
                    fname,
                    n,
                    _grid.params(v, n),
                    st,
                    value,
                    slug=v["slug"],
                    representative=is_rep,
                )
            )

    write_fragment(
        "infomeasure-python",
        "python",
        getattr(im, "__version__", "unknown"),
        benchmarks,
        seeds_all,
        cross_cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

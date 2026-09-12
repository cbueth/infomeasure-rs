#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cross-package collector: the Python infomeasure package.

Times only the estimator call on the shared canonical datasets and writes
``results/infomeasure-python.json`` (schema v2 fragment). See
``scripts/bench_packages/_common.py`` for the timing contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    MEASURES,
    default_params,
    entry,
    load,
    sizes_and_seeds,
    stats,
    time_call,
    timing_config,
    write_fragment,
)

import infomeasure as im  # noqa: E402

APPROACHES = ["discrete", "ksg", "kernel_box", "kernel_gaussian"]


def build_fn(measure: str, approach: str, cols):
    x = cols[:, 0]
    if approach == "discrete":
        kw = {"approach": "discrete"}
    elif approach == "ksg":
        # infomeasure names the kNN entropy estimator "kl", MI/CMI/TE "ksg".
        kw = {"approach": "kl" if measure == "entropy" else "ksg", "k": 4}
    else:
        kw = {
            "approach": "kernel",
            "kernel": approach.split("_", 1)[1],
            "bandwidth": 0.5,
        }

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
            lambda: float(
                im.transfer_entropy(
                    x, y, src_hist_len=1, dest_hist_len=1, **kw
                )
            ),
            "im.transfer_entropy",
        )
    if measure == "cte":
        z = cols[:, 2]
        return (
            lambda: float(
                im.transfer_entropy(
                    x,
                    y,
                    cond=z,
                    src_hist_len=1,
                    dest_hist_len=1,
                    cond_hist_len=1,
                    **kw,
                )
            ),
            "im.transfer_entropy",
        )
    raise ValueError(measure)


def main() -> int:
    cfg = timing_config()
    sizes, seeds = sizes_and_seeds()
    benchmarks: list[dict] = []

    for measure in MEASURES:
        for approach in APPROACHES:
            kind = "discrete" if approach == "discrete" else "continuous"
            for n in sizes:
                times: list[float] = []
                value = None
                for seed in seeds:
                    cols = load(measure, kind, seed, n)
                    fn, fname = build_fn(measure, approach, cols)
                    t, value = time_call(fn, cfg)
                    times.extend(t)
                st = stats(times)
                print(
                    f"  {measure:>7} {approach:<16} n={n:<6} "
                    f"{st['mean'] * 1e3:>9.3f} ms"
                )
                benchmarks.append(
                    entry(
                        "infomeasure-python",
                        "python",
                        measure,
                        approach,
                        fname,
                        n,
                        default_params(measure, approach, n),
                        st,
                        value,
                    )
                )

    write_fragment(
        "infomeasure-python",
        "python",
        getattr(im, "__version__", "unknown"),
        benchmarks,
        seeds,
        cfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

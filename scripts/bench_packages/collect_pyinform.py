#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cross-package collector: PyInform (discrete, C ``inform`` library).

Times only the estimator call on the shared canonical datasets and writes
``results/pyinform.json``. PyInform's time-series API covers entropy, MI, TE
and CTE; it has no first-class conditional MI, so that row is omitted.
Results are in bits (base 2); history k=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    default_params,
    entry,
    load,
    sizes_and_seeds,
    stats,
    time_call,
    timing_config,
    write_fragment,
)

import pyinform  # noqa: E402
from pyinform import Dist, shannon, transfer_entropy, mutual_info  # noqa: E402

APPROACH = "discrete"
K_HISTORY = 1


def entropy_fn(x):
    xi = np.asarray(x, dtype=int)
    mn = int(xi.max()) + 1

    def call():
        counts = np.bincount(xi, minlength=mn)
        return float(shannon.entropy(Dist(counts.tolist())))

    return call


def mi_fn(x, y):
    xs, ys = x.tolist(), y.tolist()
    return lambda: float(mutual_info(xs, ys))


def te_fn(x, y):
    xs, ys = x.tolist(), y.tolist()
    return lambda: float(transfer_entropy(xs, ys, K_HISTORY))


def cte_fn(x, y, z):
    xs, ys, zs = x.tolist(), y.tolist(), z.tolist()
    return lambda: float(transfer_entropy(xs, ys, K_HISTORY, condition=zs))


def build_fn(measure, cols):
    x = cols[:, 0]
    if measure == "entropy":
        return entropy_fn(x), "pyinform.shannon.entropy"
    y = cols[:, 1]
    if measure == "mi":
        return mi_fn(x, y), "pyinform.mutual_info"
    if measure == "te":
        return te_fn(x, y), "pyinform.transfer_entropy"
    if measure == "cte":
        return cte_fn(x, y, cols[:, 2]), "pyinform.transfer_entropy"
    raise ValueError(measure)


def main() -> int:
    # Measures PyInform supports as time-series calls.
    measures = ["entropy", "mi", "te", "cte"]
    short, warmup, iterations = timing_config()
    sizes, seeds = sizes_and_seeds()
    benchmarks: list[dict] = []

    for measure in measures:
        for n in sizes:
            times: list[float] = []
            value = None
            for seed in seeds:
                cols = load(measure, "discrete", seed, n)
                fn, fname = build_fn(measure, cols)
                t, value = time_call(fn, warmup, iterations)
                times.extend(t)
            st = stats(times)
            print(
                f"  {measure:>7} {APPROACH:<14} n={n:<6} "
                f"{st['mean'] * 1e3:>9.3f} ms"
            )
            params = default_params(measure, APPROACH, n)
            params["k"] = K_HISTORY
            benchmarks.append(
                entry(
                    "pyinform",
                    "python",
                    measure,
                    APPROACH,
                    fname,
                    n,
                    params,
                    st,
                    value,
                )
            )

    write_fragment(
        "pyinform",
        "python",
        "0.2.0",
        benchmarks,
        seeds,
        warmup,
        iterations,
        short,
        extra={"base": 2, "library": "inform (C)"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cross-package collector: Syntropy (syntropyx), KNN family only.

Syntropy's KNN (Kraskov) estimators are sample-based and directly comparable
to our KSG rows. Its discrete (distribution-input), Gaussian (covariance-input),
neural/mixed families and its PID/higher-order measures are out of scope, as is
transfer entropy (Syntropy has none). Writes ``results/syntropy.json``.
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

from syntropy.knn import (  # noqa: E402
    conditional_mutual_information,
    differential_entropy,
    mutual_information,
)

APPROACH = "ksg"
MEASURES = ["entropy", "mi", "cmi"]
K = 4


def build_fn(measure, cols):
    # Syntropy KNN expects (features, samples).
    data = np.ascontiguousarray(cols.T.astype(float))
    if measure == "entropy":
        return (
            lambda: float(differential_entropy(data, k=K, idxs=(0,))[1]),
            "syntropy.knn.differential_entropy",
        )
    if measure == "mi":
        return (
            lambda: float(
                mutual_information(idxs_x=(0,), idxs_y=(1,), k=K, data=data)[1]
            ),
            "syntropy.knn.mutual_information",
        )
    if measure == "cmi":
        return (
            lambda: float(
                conditional_mutual_information(
                    idxs_x=(0,), idxs_y=(1,), idxs_z=(2,), k=K, data=data
                )[1]
            ),
            "syntropy.knn.conditional_mutual_information",
        )
    raise ValueError(measure)


def main() -> int:
    short, warmup, iterations = timing_config()
    sizes, seeds = sizes_and_seeds()
    benchmarks: list[dict] = []

    for measure in MEASURES:
        for n in sizes:
            times: list[float] = []
            value = None
            for seed in seeds:
                cols = load(measure, "continuous", seed, n)
                fn, fname = build_fn(measure, cols)
                t, value = time_call(fn, warmup, iterations)
                times.extend(t)
            st = stats(times)
            print(f"  {measure:>7} {APPROACH:<14} n={n:<6} {st['mean'] * 1e3:>9.3f} ms")
            benchmarks.append(
                entry(
                    "syntropy",
                    "python",
                    measure,
                    APPROACH,
                    fname,
                    n,
                    default_params(measure, APPROACH, n),
                    st,
                    value,
                )
            )

    write_fragment(
        "syntropy",
        "python",
        "0.0.2",
        benchmarks,
        seeds,
        warmup,
        iterations,
        short,
        extra={"base": "nats"},
        limitations=(
            "Only the sample-based KNN (Kraskov) family is compared: "
            "differential entropy/MI/CMI, k=4, nats. No transfer entropy. "
            "Its discrete (distribution-input), Gaussian (covariance-input), "
            "neural and mixed families and its PID/higher-order/temporal "
            "measures are out of scope."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cross-package collector: pyitlib (discrete, MLE/MAP/… estimators).

Covers discrete entropy, MI and conditional MI; no transfer entropy. Writes
``results/pyitlib.json``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

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

warnings.filterwarnings("ignore")
from pyitlib import discrete_random_variable as drv  # noqa: E402

APPROACH = "discrete"
MEASURES = ["entropy", "mi", "cmi"]


def build_fn(measure, cols):
    x = cols[:, 0].tolist()
    if measure == "entropy":
        return lambda: float(drv.entropy(x, base=2)), "pyitlib.entropy"
    y = cols[:, 1].tolist()
    if measure == "mi":
        return (
            lambda: float(drv.information_mutual(x, y, base=2)),
            "pyitlib.information_mutual",
        )
    if measure == "cmi":
        z = cols[:, 2].tolist()
        return (
            lambda: float(drv.information_mutual_conditional(x, y, z, base=2)),
            "pyitlib.information_mutual_conditional",
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
                cols = load(measure, "discrete", seed, n)
                fn, fname = build_fn(measure, cols)
                t, value = time_call(fn, warmup, iterations)
                times.extend(t)
            st = stats(times)
            print(f"  {measure:>7} {APPROACH:<14} n={n:<6} {st['mean'] * 1e3:>9.3f} ms")
            benchmarks.append(
                entry(
                    "pyitlib",
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
        "pyitlib",
        "python",
        "0.3.1",
        benchmarks,
        seeds,
        warmup,
        iterations,
        short,
        extra={"base": 2},
        limitations="Discrete only (entropy/MI/CMI); no transfer entropy. Base 2.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

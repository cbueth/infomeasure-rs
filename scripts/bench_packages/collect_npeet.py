#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cross-package collector: NPEET (non-parametric kNN / KSG estimators).

Covers continuous kNN (KSG) entropy, MI and conditional MI; no transfer
entropy. Writes ``results/npeet.json``.
"""

from __future__ import annotations

import sys
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

from npeet import entropy_estimators as ee  # noqa: E402

APPROACH = "ksg"
MEASURES = ["entropy", "mi", "cmi"]
K = 4


def build_fn(measure, cols):
    x = cols[:, 0].reshape(-1, 1)
    if measure == "entropy":
        return lambda: float(ee.entropy(x, k=K, base=2)), "npeet.entropy"
    y = cols[:, 1].reshape(-1, 1)
    if measure == "mi":
        return lambda: float(ee.mi(x, y, k=K, base=2)), "npeet.mi"
    if measure == "cmi":
        z = cols[:, 2].reshape(-1, 1)
        return lambda: float(ee.cmi(x, y, z, k=K, base=2)), "npeet.cmi"
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
                    "npeet",
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
        "npeet",
        "python",
        "1.0.1",
        benchmarks,
        seeds,
        warmup,
        iterations,
        short,
        extra={"base": 2},
        limitations=(
            "Continuous kNN (KSG) only: entropy/MI/CMI, k=4, base 2. "
            "No transfer entropy."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cross-package collector: dit (discrete information theory).

This dit version exposes Shannon entropy and MI (no conditional MI, no
transfer entropy). The empirical distribution is built inside the timed call.
Writes ``results/dit.json``.
"""

from __future__ import annotations

import sys
from collections import Counter
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

import dit  # noqa: E402

APPROACH = "discrete"
MEASURES = ["entropy", "mi"]


def distribution(rows):
    counts = Counter(rows)
    n = len(rows)
    return dit.Distribution(list(counts.keys()), [v / n for v in counts.values()])


def build_fn(measure, cols):
    if measure == "entropy":
        x = cols[:, 0].tolist()

        def call_entropy():
            return float(dit.shannon.entropy(distribution(x)))

        return call_entropy, "dit.shannon.entropy"

    if measure == "mi":
        pairs = list(zip(cols[:, 0].tolist(), cols[:, 1].tolist()))

        def call_mi():
            d = distribution(pairs)
            return float(dit.shannon.mutual_information(d, [0], [1]))

        return call_mi, "dit.shannon.mutual_information"

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
                    "dit",
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
        "dit",
        "python",
        "1.5",
        benchmarks,
        seeds,
        warmup,
        iterations,
        short,
        extra={"base": 2},
        limitations=(
            "Discrete only; this version exposes entropy and MI (no conditional "
            "MI, no transfer entropy). Base 2."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

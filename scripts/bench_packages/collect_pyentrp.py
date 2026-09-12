#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Cross-package collector: pyEntrp (time-series entropy).

Only ``shannon_entropy`` maps to our axis: it is a plug-in MLE estimate over
the unique-value frequencies of a discrete series (base 2). pyEntrp's other
measures (sample/multiscale/permutation entropy) are different notions and are
out of scope. Writes ``results/pyentrp.json``.
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

from pyentrp import entropy as ent  # noqa: E402

APPROACH = "discrete"


def main() -> int:
    cfg = timing_config()
    sizes, seeds = sizes_and_seeds()
    benchmarks: list[dict] = []

    for n in sizes:
        times: list[float] = []
        value = None
        for seed in seeds:
            x = load("entropy", "discrete", seed, n)[:, 0]
            fn = lambda: float(ent.shannon_entropy(x))  # noqa: E731
            t, value = time_call(fn, cfg)
            times.extend(t)
        st = stats(times)
        print(f"  {'entropy':>7} {APPROACH:<14} n={n:<6} {st['mean'] * 1e3:>9.3f} ms")
        benchmarks.append(
            entry(
                "pyentrp",
                "python",
                "entropy",
                APPROACH,
                "pyentrp.shannon_entropy",
                n,
                default_params("entropy", APPROACH, n),
                st,
                value,
            )
        )

    write_fragment(
        "pyentrp",
        "python",
        "2.1.0",
        benchmarks,
        seeds,
        cfg,
        extra={"base": 2},
        limitations=(
            "Entropy only: plug-in MLE over unique-value frequencies, base 2. "
            "No MI/CMI/TE; sample/permutation/multiscale entropy are out of "
            "scope."
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

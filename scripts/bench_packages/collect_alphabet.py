#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Alphabet-scaling collector for the Python packages.

Times each package's discrete MLE estimator across state counts at the sizes in
the grid's ``alphabet`` block, skipping larger N once a cell's mean exceeds the
budget. Writes one fragment per package (``results/<package>_alphabet.json``).

Env:
  BENCH_DATA_DIR            dataset dir (default ``target/bench-data``)
  BENCH_SHORT=1             warm-up 1 + up to 2 iterations
  BENCH_ALPHABET_SIZES      override the size list (comma-separated)
  BENCH_ALPHABET_PACKAGES   comma-separated subset of package ids
  BENCH_RESUME=1            keep an existing fragment when the fingerprint matches
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _grid  # noqa: E402
from _common import (  # noqa: E402
    existing_benchmarks,
    load_alphabet,
    read_manifest,
    resuming,
    stats,
    time_call,
    timing_config,
    write_fragment,
)

FAMILY = "alphabet"

# (package id, dist name, collector module, supported measures, limitations)
PROVIDERS = [
    (
        "infomeasure-python",
        "infomeasure",
        "collect_infomeasure_python",
        ["entropy", "mi", "cmi", "te", "cte"],
        "Discrete MLE; alphabet inferred; nats.",
    ),
    (
        "pyitlib",
        "pyitlib",
        "collect_pyitlib",
        ["entropy", "mi", "cmi"],
        "Discrete (base 2); no transfer entropy.",
    ),
    (
        "dit",
        "dit",
        "collect_dit",
        ["entropy", "mi"],
        "Discrete (base 2); entropy and MI only.",
    ),
    (
        "pyinform",
        "pyinform",
        "collect_pyinform",
        ["entropy", "mi", "cmi", "te", "cte"],
        "Discrete (bits); TE/CTE history k=1.",
    ),
    (
        "pyentrp",
        "pyentrp",
        "collect_pyentrp",
        ["entropy"],
        "Shannon entropy (base 2) only.",
    ),
]


def dist_version(dist: str, fallback: str = "unknown") -> str:
    try:
        import importlib.metadata as md

        return md.version(dist)
    except Exception:  # noqa: BLE001
        return fallback


def fingerprint(version: str, cfg: dict) -> str:
    h = hashlib.sha256()
    h.update(str(version).encode())
    h.update(os.environ.get("BENCH_COMMIT", "").encode())
    h.update(_grid.source_hash().encode())
    h.update(str(sorted(_grid.alphabet().get("caps", {}).items())).encode())
    h.update(str(_grid.alphabet_sizes()).encode())
    for key in ("warmup_max", "warmup_budget", "min_iters", "max_iters", "iter_budget"):
        h.update(repr(cfg.get(key)).encode())
    return h.hexdigest()[:16]


def provider_build(name: str, mod, measure: str, cols):
    if name == "infomeasure-python":
        return mod.build_fn(measure, {"approach": "discrete", "method": "mle"}, cols)
    if name == "pyentrp":
        # pyentrp has no shared build_fn; only shannon_entropy maps here.
        from pyentrp import entropy as ent

        x = cols[:, 0]
        return (lambda: float(ent.shannon_entropy(x))), "pyentrp.shannon_entropy"
    return mod.build_fn(measure, cols)


def params(measure: str, states: int, n: int) -> dict:
    return {
        "n": n,
        "states": states,
        "method": "mle",
        "delay": 1 if measure in ("te", "cte") else None,
        "k": None,
        "bandwidth": None,
        "order": None,
        "alpha": None,
        "q": None,
        "dims": 1,
        "kernel_type": None,
    }


def collect(name: str, mod, version: str, measures: list[str], limitations: str,
            cfg: dict, seeds: list[int], ab: dict) -> None:
    fp = fingerprint(version, cfg)
    benchmarks = existing_benchmarks(name, fp, family=FAMILY) if resuming() else []
    done = {b.get("id") for b in benchmarks}
    sizes = _grid.alphabet_sizes(ab)

    for measure in measures:
        states_list = _grid.alphabet_states(measure, ab)
        if not states_list:
            continue
        for states in states_list:
            stopped = False
            for n in sizes:
                if stopped:
                    break
                eid = f"{measure}/discrete/mle/b{states}/n{n}/{name}"
                if eid in done:
                    continue
                times: list[float] = []
                value = None
                fname = "?"
                for seed in seeds:
                    cols = load_alphabet(measure, states, seed, n)
                    fn, fname = provider_build(name, mod, measure, cols)
                    t, v = time_call(fn, cfg)
                    times.extend(t)
                    value = v
                st = stats(times)
                benchmarks.append(
                    {
                        "id": eid,
                        "package": name,
                        "language": "python",
                        "measure": measure,
                        "approach": "discrete",
                        "function": fname,
                        "representative": True,
                        "params": params(measure, states, n),
                        "statistics": st,
                        "value": value,
                        "notes": None,
                    }
                )
                done.add(eid)
                print(
                    f"  {name:20} {measure:8} b{states:<4} n={n:<6} {st['mean'] * 1e3:>9.3f} ms"
                )
                if st["mean"] > ab["budget_s"]:
                    print(f"  -> {name} b{states} n={n} exceeded {ab['budget_s']}s; skipping larger N")
                    stopped = True
        # Flush per measure (not per entry) so a cancelled run resumes while
        # keeping fragment writes infrequent.
        write_fragment(
            name, "python", version, benchmarks, seeds, cfg,
            fingerprint=fp, family=FAMILY, limitations=limitations,
        )
    write_fragment(
        name, "python", version, benchmarks, seeds, cfg,
        fingerprint=fp, family=FAMILY, limitations=limitations,
    )


def main() -> int:
    cfg = timing_config()
    ab = _grid.alphabet()
    seeds = read_manifest()["seeds"]
    selected = os.environ.get("BENCH_ALPHABET_PACKAGES")
    want = {s.strip() for s in selected.split(",")} if selected else None

    for name, dist, module_name, measures, limitations in PROVIDERS:
        if want is not None and name not in want:
            continue
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {e}")
            continue
        version = dist_version(dist, getattr(mod, "__version__", "unknown"))
        try:
            collect(name, mod, version, measures, limitations, cfg, seeds, ab)
        except Exception as e:  # noqa: BLE001
            print(f"error collecting {name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Merge per-package schema-v2 fragments into the final cross_package.json.

Reads every ``results/*.json`` written by the collectors, unions the package
list and benchmark arrays, and writes ``cross_package.json`` next to them.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import data_dir  # noqa: E402

# Packages deliberately not benchmarked, with the reason (rendered on the page).
EXCLUDED = [
    {
        "id": "idtxl",
        "language": "python",
        "category": "framework",
        "repo": "https://github.com/pwollstadt/IDTxl",
        "docs": "https://pwollstadt.github.io/IDTxl/html/index.html",
        "reason": (
            "A network-inference / effective-connectivity framework rather than "
            "a single-estimator library. Its CPU estimators delegate to JIDT, "
            "and its only independent implementation is a GPU-only OpenCL KSG "
            "estimator."
        ),
    },
    {
        "id": "npeet",
        "language": "python",
        "repo": "https://github.com/gregversteeg/NPEET",
        "reason": (
            "Unmaintained since 2022. Its KSG estimator is already covered by "
            "JIDT, infomeasure and Syntropy."
        ),
    },
    {
        "id": "tet",
        "language": "matlab",
        "reason": "Matlab-only, 2013, binary time series; out of scope.",
    },
    {
        "id": "trentool",
        "language": "matlab",
        "reason": "Matlab-only, 2017, TE only; out of scope.",
    },
    {
        "id": "infotheory",
        "language": "python/c++",
        "reason": "Defunct (last release 2020); discrete/continuous limited.",
    },
    {
        "id": "infotheoryjl",
        "language": "julia",
        "reason": "Unmaintained since 2016; entropy only.",
    },
    {
        "id": "pyentropy",
        "language": "python",
        "reason": "Google Code archive; superseded by other packages.",
    },
]


def main() -> int:
    d = data_dir()
    rdir = d / "results"
    fragments = sorted(rdir.glob("*.json"))
    if not fragments:
        print(f"no fragments in {rdir}", file=sys.stderr)
        return 1

    packages: dict[str, dict] = {}
    benchmarks: list[dict] = []
    coverage: set[tuple[str, str]] = set()
    meta: dict = {}
    runtime: dict | None = None
    loaded: list[tuple[Path, dict]] = []
    for f in fragments:
        obj = json.loads(f.read_text())
        loaded.append((f, obj))
        m = obj.get("meta", {})
        meta = meta or {
            k: v
            for k, v in m.items()
            if k not in ("packages", "run_id", "coverage")
        }
        if m.get("hardware") and not meta.get("hardware"):
            meta["hardware"] = m["hardware"]
        # Use the reference harness (infomeasure-rs) for the run-level runtime.
        if f.name == "infomeasure-rs.json" and m.get("runtime"):
            runtime = m["runtime"]
        for p in m.get("packages", []):
            packages[p["id"]] = p
        for b in obj.get("benchmarks", []):
            benchmarks.append(b)
            coverage.add((b["measure"], b["approach"]))
        print(f"  {f.name}: {len(obj.get('benchmarks', []))} entries")

    # Stamp the run's hardware into every fragment that lacks it, so all
    # fragments from one collection agree on the machine (pages read fragments,
    # not this merged file).
    hw = meta.get("hardware")
    if hw:
        for f, obj in loaded:
            if not obj.get("meta", {}).get("hardware"):
                obj.setdefault("meta", {})["hardware"] = hw
                f.write_text(json.dumps(obj, indent=2))

    benchmarks.sort(key=lambda b: (b["measure"], b["approach"], b["params"]["n"], b["package"]))
    if runtime:
        meta = {**meta, "runtime": runtime}
    out = {
        "meta": {
            **meta,
            "schema": 2,
            "generated": datetime.now(timezone.utc).isoformat(),
            "run_id": f"cross_package_{int(time.time())}",
            "packages": list(packages.values()),
            "coverage": sorted([list(c) for c in coverage]),
            "excluded": EXCLUDED,
        },
        "benchmarks": benchmarks,
    }
    path = d / "cross_package.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"merged {len(benchmarks)} entries from {len(packages)} packages -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

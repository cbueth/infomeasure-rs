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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import data_dir  # noqa: E402


def main() -> int:
    d = data_dir()
    rdir = d / "results"
    fragments = sorted(rdir.glob("*.json"))
    if not fragments:
        print(f"no fragments in {rdir}", file=sys.stderr)
        return 1

    packages: dict[str, dict] = {}
    benchmarks: list[dict] = []
    meta: dict = {}
    for f in fragments:
        obj = json.loads(f.read_text())
        m = obj.get("meta", {})
        meta = meta or {
            k: v
            for k, v in m.items()
            if k not in ("packages", "run_id")
        }
        if m.get("hardware") and not meta.get("hardware"):
            meta["hardware"] = m["hardware"]
        for p in m.get("packages", []):
            packages[p["id"]] = p
        benchmarks.extend(obj.get("benchmarks", []))
        print(f"  {f.name}: {len(obj.get('benchmarks', []))} entries")

    benchmarks.sort(key=lambda b: (b["measure"], b["approach"], b["params"]["n"], b["package"]))
    out = {
        "meta": {**meta, "schema": 2, "packages": list(packages.values())},
        "benchmarks": benchmarks,
    }
    path = d / "cross_package.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"merged {len(benchmarks)} entries from {len(packages)} packages -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

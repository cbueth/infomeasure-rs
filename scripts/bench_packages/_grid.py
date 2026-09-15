# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Expander for the shared detailed benchmark grid.

Mirrors ``benches/utils/grid.rs`` so both infomeasure packages measure an
identical set of variants. The single source is ``benches/detailed_grid.json``.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

_GRID_PATH = Path(__file__).resolve().parents[2] / "benches" / "detailed_grid.json"


def _load() -> dict:
    return json.loads(_GRID_PATH.read_text())


def source_hash() -> str:
    """Stable hash of the grid definition, for resumable-run invalidation."""
    import hashlib

    return hashlib.sha256(_GRID_PATH.read_bytes()).hexdigest()[:16]


def sizes() -> tuple[list[int], list[int]]:
    """Return ``(cross_sizes, detailed_sizes)``."""
    g = _load()
    return g["sizes"]["cross"], g["sizes"]["detailed"]


def alphabet() -> dict:
    """The alphabet-scaling family configuration."""
    return _load()["alphabet"]


def alphabet_states(measure: str, cfg: dict | None = None) -> list[int]:
    """State counts to collect for a measure, capped by the memory guard."""
    cfg = cfg or alphabet()
    cap = cfg.get("caps", {}).get(measure, 0)
    return [s for s in cfg["states"] if s <= cap]


def alphabet_sizes(cfg: dict | None = None) -> list[int]:
    """Sizes for the alphabet sweep (``BENCH_ALPHABET_SIZES``/``BENCH_SIZES``)."""
    import os

    cfg = cfg or alphabet()
    raw = os.environ.get("BENCH_ALPHABET_SIZES") or os.environ.get("BENCH_SIZES")
    if raw:
        sizes = [int(s) for s in raw.split(",") if s.strip()]
        if sizes:
            return sizes
    return list(cfg["sizes"])


def _num_slug(v) -> str:
    return str(v).replace(".", "_")


def _slug(v: dict) -> str:
    approach = v["approach"]
    if approach == "discrete":
        return v["method"] or "mle"
    if approach == "kernel":
        return f"{v['kernel'] or 'box'}_bw{_num_slug(v['bandwidth'])}"
    if approach == "ordinal":
        return f"order{v['order']}"
    if approach == "renyi":
        return f"alpha{_num_slug(v['alpha'])}_k{v['k']}"
    if approach == "tsallis":
        return f"q{_num_slug(v['q'])}_k{v['k']}"
    return f"k{v['k']}"


def variants(lang: str) -> list[dict]:
    """Expand the grid for one language (``"rust"`` or ``"python"``)."""
    g = _load()
    values = g["values"]
    cross = g["cross"]
    out: list[dict] = []

    for measure, entries in g["approaches"].items():
        for entry in entries:
            if "langs" in entry and lang not in entry["langs"]:
                continue
            axes = entry["axes"]
            # An entry may override a global axis' values (e.g. entropy's KL).
            per_entry = entry.get("values", {})
            for combo_vals in product(*(per_entry.get(a, values[a]) for a in axes)):
                combo = dict(zip(axes, combo_vals))
                v = {
                    "measure": measure,
                    "approach": entry["approach"],
                    "method": None,
                    "k": None,
                    "bandwidth": None,
                    "kernel": None,
                    "order": None,
                    "alpha": None,
                    "q": None,
                }
                for axis, val in combo.items():
                    if axis == "method":
                        v["method"] = val
                    elif axis in ("k", "kl_k"):
                        v["k"] = val
                    elif axis == "bandwidth":
                        v["bandwidth"] = val
                    elif axis == "kernel":
                        v["kernel"] = val
                    elif axis == "order":
                        v["order"] = val
                    elif axis == "alpha":
                        v["alpha"] = val
                    elif axis == "q":
                        v["q"] = val
                spec = cross.get(entry["approach"])
                v["cross"] = bool(spec) and all(
                    combo.get(k) == want for k, want in spec.items()
                )
                v["detail_only"] = not v["cross"]
                v["slug"] = _slug(v)
                out.append(v)
    return out


def params(v: dict, n: int) -> dict:
    return {
        "n": n,
        "k": v["k"],
        "bandwidth": v["bandwidth"],
        "order": v["order"],
        "delay": 1 if v["measure"] in ("te", "cte") else None,
        "alpha": v["alpha"],
        "q": v["q"],
        "dims": 1,
        "method": v["method"],
        "kernel_type": v["kernel"],
    }

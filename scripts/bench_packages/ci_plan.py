#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Decide what a benchmark CI run should do, and write ``plan.env``.

Inputs (environment, set by Woodpecker): ``CI_PIPELINE_EVENT``,
``CI_COMMIT_BRANCH``, ``CI_COMMIT_TAG``. Files: ``pages/registry.json``,
``versions.json`` (from check_versions.py, cron only) and ``pages/data/``.

Outputs ``plan.env`` with:
    ACTION     collect | open-pr | none
    PACKAGES   all | comma-separated ids
    MODE       full | selective | registry-pr | none

Policy (single workflow on ``main``):
    manual / tag        -> collect everything
    cron                -> open a registry PR if upstream moved, otherwise
                           collect the packages whose published fragment is
                           missing or pinned to a different version
    push (main)         -> nothing (the image step handles ``.docker/**``)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PAGES = Path("pages")
REGISTRY = PAGES / "registry.json"
DATA = PAGES / "data"


def published_versions() -> dict[str, str | None]:
    """Map package id -> version recorded in its published fragment.

    Fragments live on ``pages`` as ``data/<id>.json`` and are the *result* of
    the last collection, so a mismatch against ``registry.json`` means the pin
    moved (a merged registry PR) and the package needs re-collecting.
    """
    versions: dict[str, str | None] = {}
    if not DATA.is_dir():
        return versions
    for frag in DATA.glob("*.json"):
        try:
            obj = json.loads(frag.read_text())
        except json.JSONDecodeError:
            continue
        for pkg in obj.get("meta", {}).get("packages", []):
            pid = pkg.get("id")
            if pid:
                versions[pid] = pkg.get("version")
    return versions


def stale_packages() -> list[str]:
    """Registry packages whose published fragment is missing or out of date."""
    try:
        registry = json.loads(REGISTRY.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    published = published_versions()
    return sorted(
        pkg["id"]
        for pkg in registry.get("packages", [])
        if published.get(pkg["id"]) != pkg.get("version")
    )


def _write(action: str, packages: str, mode: str) -> int:
    lines = [f"ACTION={action}", f"PACKAGES={packages}", f"MODE={mode}"]
    Path("plan.env").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    event = os.environ.get("CI_PIPELINE_EVENT", "")
    tag = os.environ.get("CI_COMMIT_TAG", "")

    action, packages, mode = "none", "", "none"

    if event == "cron":
        # Bi-weekly gate: the Woodpecker cron fires weekly, but the registry
        # check only acts on even ISO weeks when BENCH_CRON_BIWEEKLY=1.
        if os.environ.get("BENCH_CRON_BIWEEKLY") == "1":
            week = int(datetime.now(timezone.utc).strftime("%V"))
            if week % 2 != 0:
                print("bi-weekly: off week", file=sys.stderr)
                return _write("none", "", "none")

        changes: list[str] = []
        report_path = Path("versions.json")
        if report_path.exists():
            try:
                changes = json.loads(report_path.read_text()).get("collectible_ids", [])
            except json.JSONDecodeError:
                changes = []
        if changes:
            action, packages, mode = "open-pr", ",".join(changes), "registry-pr"
        else:
            stale = stale_packages()
            if stale:
                have_data = DATA.is_dir() and any(DATA.glob("*.json"))
                action = "collect"
                packages = ",".join(stale)
                mode = "selective" if have_data else "full"
    elif event in ("manual", "tag") or tag:
        action, packages, mode = "collect", "all", "full"

    return _write(action, packages, mode)


if __name__ == "__main__":
    raise SystemExit(main())

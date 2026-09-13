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
    MODE       full | selective | none

Policy:
    manual / tag        -> collect everything
    push to pages       -> collect the packages whose pins changed (or all if
                           no fragments exist yet)
    cron                -> open a registry-update PR if upstream moved
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PAGES = Path("pages")
REGISTRY = PAGES / "registry.json"


def _ids(registry: dict) -> dict[str, str]:
    return {p["id"]: p.get("version") for p in registry.get("packages", [])}


def changed_via_git() -> list[str]:
    """Packages whose pinned version changed in the pages push."""
    try:
        prev = subprocess.check_output(
            ["git", "-C", str(PAGES), "show", "HEAD~1:registry.json"],
            text=True,
        )
        old = _ids(json.loads(prev))
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return []
    new = _ids(json.loads(REGISTRY.read_text()))
    return sorted(pid for pid in set(old) | set(new) if old.get(pid) != new.get(pid))


def main() -> int:
    event = os.environ.get("CI_PIPELINE_EVENT", "")
    branch = os.environ.get("CI_COMMIT_BRANCH", "")
    tag = os.environ.get("CI_COMMIT_TAG", "")

    action, packages, mode = "none", "", "none"

    if event == "cron":
        # Optional bi-weekly gate: the Woodpecker cron fires weekly, but the
        # registry check only runs on even ISO weeks when BENCH_CRON_BIWEEKLY=1.
        if os.environ.get("BENCH_CRON_BIWEEKLY") == "1":
            week = int(datetime.now(timezone.utc).strftime("%V"))
            if week % 2 != 0:
                Path("plan.env").write_text("ACTION=none\nPACKAGES=\nMODE=none\n")
                print("ACTION=none (bi-weekly: off week)")
                return 0
        report_path = Path("versions.json")
        changes = []
        if report_path.exists():
            try:
                changes = json.loads(report_path.read_text()).get("collectible_ids", [])
            except json.JSONDecodeError:
                changes = []
        if changes:
            action, packages, mode = "open-pr", ",".join(changes), "registry-pr"
    elif event == "push" and branch == "pages":
        action = "collect"
        data = PAGES / "data"
        have_data = data.is_dir() and any(data.glob("*.json"))
        changed = changed_via_git()
        packages = ",".join(changed) if (have_data and changed) else "all"
        mode = "selective" if packages != "all" else "full"
    elif event in ("manual", "tag") or tag:
        action, packages, mode = "collect", "all", "full"

    lines = [f"ACTION={action}", f"PACKAGES={packages}", f"MODE={mode}"]
    Path("plan.env").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

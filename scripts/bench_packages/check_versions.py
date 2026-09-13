#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Check the benchmark registry against upstream releases.

Reads ``registry.json`` (the single source of truth, on the ``pages`` branch),
queries each package's ecosystem for its latest version and release date, and
reports which pins are out of date. With ``--write`` it updates the registry in
place and writes the list of changed package ids so CI can (after the update
PR is merged) collect only those packages.

Ecosystems:
    pypi      PyPI JSON API
    crates    crates.io API
    julia     Julia General registry (Versions.toml)
    cran      CRAN DESCRIPTION
    artifact  flagged for review (e.g. JIDT needs a new sha256 — not auto-bumped)

Usage:
    python scripts/bench_packages/check_versions.py \
        --registry pages/registry.json --write \
        --report /tmp/versions.json --changed-out /tmp/changed.txt
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

UA = "infomeasure-bench-ci/1.0 (+https://codeberg.org/cbueth/infomeasure-rs)"


def _get(url: str) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _get_json(url: str):
    text = _get(url)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _vkey(version: str) -> tuple:
    """Loose version sort key (handles 1.2.3, 1.2, 0.0.2-rc1)."""
    parts = re.split(r"[.\-+]", version)
    key = []
    for p in parts:
        key.append((0, int(p)) if p.isdigit() else (1, p))
    return tuple(key)


def latest_pypi(pkg: dict) -> tuple[str | None, str | None]:
    dist = pkg.get("distribution", pkg["id"])
    data = _get_json(f"https://pypi.org/pypi/{dist}/json")
    if not data:
        return None, None
    version = data.get("info", {}).get("version")
    released = None
    files = data.get("urls") or []
    if files:
        released = (files[0].get("upload_time_iso_8601") or files[0].get("upload_time") or "")[:10] or None
    return version, released


def latest_crates(pkg: dict) -> tuple[str | None, str | None]:
    data = _get_json(f"https://crates.io/api/v1/crates/{pkg['id']}")
    if not data:
        return None, None
    version = data.get("crate", {}).get("max_stable_version") or data.get("crate", {}).get("max_version")
    released = None
    versions = data.get("versions") or []
    for v in versions:
        if v.get("num") == version:
            released = (v.get("created_at") or "")[:10] or None
            break
    return version, released


def _julia_prefix(name: str) -> str:
    return name[0].upper()


def latest_julia(pkg: dict) -> tuple[str | None, str | None]:
    name = pkg.get("distribution", pkg["id"])
    base = f"https://raw.githubusercontent.com/JuliaRegistries/General/master/{_julia_prefix(name)}/{name}"
    text = _get(f"{base}/Versions.toml")
    if not text:
        return None, None
    versions = re.findall(r'^\["([^"]+)"\]', text, re.M)
    if not versions:
        return None, None
    version = max(versions, key=_vkey)
    released = None
    pkg_toml = _get(f"{base}/Package.toml") or ""
    m = re.search(r'repo\s*=\s*"([^"]+)"', pkg_toml)
    if m:
        repo = m.group(1).removesuffix(".git").replace("https://github.com/", "https://api.github.com/repos/")
        commit = _get_json(f"{repo}/commits/v{version}")
        if commit:
            released = (commit.get("commit", {}).get("committer", {}).get("date") or "")[:10] or None
    return version, released


def latest_cran(pkg: dict) -> tuple[str | None, str | None]:
    name = pkg.get("distribution", pkg["id"])
    text = _get(f"https://cran.r-project.org/web/packages/{name}/DESCRIPTION")
    if not text:
        return None, None
    m = re.search(r"^Version:\s*(\S+)", text, re.M)
    page = _get(f"https://cran.r-project.org/web/packages/{name}/index.html") or ""
    pub = re.search(r"Published:\s*</td>\s*<td>\s*([0-9]{4}-[0-9]{2}-[0-9]{2})", page)
    return (m.group(1) if m else None), (pub.group(1) if pub else None)


def latest_artifact(pkg: dict) -> tuple[str | None, str | None]:
    """Detect a newer GitHub release, but never auto-bump (sha256 is manual)."""
    repo = pkg.get("repo")
    if not repo:
        return None, None
    m = re.search(r"github\.com/([^/]+/[^/]+)", repo)
    if not m:
        return None, None
    data = _get_json(f"https://api.github.com/repos/{m.group(1)}/releases/latest")
    if not data:
        return None, None
    tag = data.get("tag_name", "").lstrip("v")
    return tag or None, (data.get("published_at") or "")[:10] or None


LOOKUPS = {
    "pypi": latest_pypi,
    "crates": latest_crates,
    "julia": latest_julia,
    "cran": latest_cran,
    "artifact": latest_artifact,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default="registry.json")
    ap.add_argument("--write", action="store_true", help="update the registry in place")
    ap.add_argument("--report", help="write a JSON report of detected changes")
    ap.add_argument("--changed-out", help="write a comma-separated list of changed ids")
    args = ap.parse_args()

    path = Path(args.registry)
    reg = json.loads(path.read_text())

    changes: dict[str, dict] = {}
    for pkg in reg.get("packages", []):
        lookup = LOOKUPS.get(pkg.get("ecosystem", ""))
        if not lookup:
            continue
        latest, released = lookup(pkg)
        if not latest or latest == pkg["version"]:
            continue
        changes[pkg["id"]] = {
            "from": pkg["version"],
            "to": latest,
            "released": released,
            "policy": pkg.get("policy", "auto"),
            "needs_manual_sha": pkg.get("ecosystem") == "artifact",
        }

    if args.write and changes:
        for pkg in reg.get("packages", []):
            c = changes.get(pkg["id"])
            if not c or c["needs_manual_sha"]:
                continue  # artifact bumps need a new sha256 and stay manual
            pkg["version"] = c["to"]
            if c["released"]:
                pkg["released"] = c["released"]
        reg["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path.write_text(json.dumps(reg, indent=2) + "\n")

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "changes": changes,
        "changed_ids": sorted(changes),
        "collectible_ids": sorted(k for k, v in changes.items() if not v["needs_manual_sha"]),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    if args.changed_out:
        Path(args.changed_out).write_text(",".join(report["collectible_ids"]) + "\n")

    for pid, c in changes.items():
        note = " (needs manual sha256)" if c["needs_manual_sha"] else ""
        print(f"  {pid}: {c['from']} -> {c['to']} [{c['policy']}]{note}")
    if not changes:
        print("  all pins up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

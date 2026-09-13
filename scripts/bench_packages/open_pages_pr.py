#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Commit a registry bump in the pages clone and open a PR against `pages`.

Run after ``check_versions.py --write`` has modified ``pages/registry.json``.
Uses the forge API with ``GITEA_TOKEN``; the git push uses the token embedded
in the remote URL by the CI checkout step.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PAGES = Path("pages")
BRANCH = "bench/registry"


def run(*cmd: str) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def main() -> int:
    if not run("git", "-C", str(PAGES), "status", "--porcelain", "registry.json"):
        print("registry unchanged; nothing to open")
        return 0

    token = os.environ.get("GITEA_TOKEN")
    repo_link = os.environ.get("CI_REPO_LINK", "https://codeberg.org/cbueth/infomeasure-rs")
    repo = os.environ.get("CI_REPO", "cbueth/infomeasure-rs")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    branch = f"{BRANCH}-{today}"

    subprocess.run(["git", "-C", str(PAGES), "checkout", "-B", branch], check=True)
    subprocess.run(["git", "-C", str(PAGES), "add", "registry.json"], check=True)
    subprocess.run(
        [
            "git", "-C", str(PAGES),
            "-c", "user.name=infomeasure-bench-ci",
            "-c", "user.email=ci@cbueth.de",
            "commit", "-m", f"pages: bump benchmark registry pins ({today})",
        ],
        check=True,
    )
    subprocess.run(["git", "-C", str(PAGES), "push", "-f", "origin", branch], check=True)

    if not token:
        print("no GITEA_TOKEN; pushed branch only")
        return 0

    report = {}
    if Path("versions.json").exists():
        report = json.loads(Path("versions.json").read_text())
    rows = "\n".join(
        f"- `{pid}`: {c['from']} -> {c['to']} [{c['policy']}]"
        + (" — needs a new sha256 (manual)" if c.get("needs_manual_sha") else "")
        for pid, c in sorted(report.get("changes", {}).items())
    )
    body = (
        "Automated benchmark-registry update.\n\n"
        f"{rows or '_(no version changes)_'}\n\n"
        "Merging this updates `registry.json` on `pages`; the pages push then "
        "triggers a collection for the changed packages.\n"
    )
    payload = json.dumps({"base": "pages", "head": branch, "title": f"Benchmark registry update ({today})", "body": body}).encode()
    req = urllib_request(repo_link, repo, token, payload)
    print(req)


def urllib_request(repo_link: str, repo: str, token: str, payload: bytes) -> str:
    import urllib.request

    url = f"{repo_link}/api/v1/repos/{repo}/pulls"
    r = urllib.request.Request(
        url,
        data=payload,
        headers={"Authorization": f"token {token}", "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return f"opened PR #{data.get('number')}: {data.get('html_url')}"
    except Exception as e:  # noqa: BLE001
        return f"PR creation failed: {e}"


if __name__ == "__main__":
    raise SystemExit(main())

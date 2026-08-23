#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Post a Bencher report summary as a create-or-update comment on a Codeberg PR.

Runs in `.woodpecker/bench.yml` after the `bench` step, on `pull_request`
events. It:
  1. Fetches the latest Bencher report for the PR source branch.
  2. Verifies it belongs to `$CI_COMMIT_SHA` (rejects stale reports on reruns).
  3. Fetches the target branch's latest report as a baseline.
  4. Renders a compact Markdown summary (alerts + per-benchmark deltas).
  5. Finds an existing comment by a trailing marker tag and PATCHes it,
     or POSTs a new one.

Uses only the Python standard library. Env vars:
  BENCHER_API_URL   (default https://api.bencher.dev)
  BENCHER_PROJECT   (default infomeasure-rs)
  BENCHER_DELTA_THRESHOLD (default 10; % |Δ| for improved/regressed classification)
  CODEGBERG_PR_COMMENT_TOKEN (Forgejo API token, write:issue scope)
  CI_REPO_OWNER, CI_REPO_NAME, CI_COMMIT_PULL_REQUEST,
  CI_COMMIT_SOURCE_BRANCH, CI_COMMIT_TARGET_BRANCH, CI_COMMIT_SHA
"""

import http.client
import json
import os
import sys
import time
import typing
import urllib.error
import urllib.parse
import urllib.request

BENCHER_API_URL = os.environ.get("BENCHER_API_URL", "https://api.bencher.dev")
BENCHER_PROJECT = os.environ.get("BENCHER_PROJECT", "infomeasure-rs")
CODEGBERG_TOKEN = os.environ.get("CODEGBERG_PR_COMMENT_TOKEN", "")
CODEGBERG_URL = "https://codeberg.org"
TOP_CHANGES = 10
DEFAULT_DELTA_THRESHOLD = 10.0


def delta_threshold() -> float:
    raw = os.environ.get("BENCHER_DELTA_THRESHOLD", "")
    try:
        return float(raw) if raw else DEFAULT_DELTA_THRESHOLD
    except ValueError:
        return DEFAULT_DELTA_THRESHOLD

# The marker tag is the last thing in the comment body. It lets us find an
# existing comment to update on a rerun instead of creating duplicates.
def marker_tag() -> str:
    return f'<div id="bencher.dev/projects/{BENCHER_PROJECT}/comment"></div>'


def required(name: str) -> str:
    val = os.environ.get(name, "")
    if not val:
        print(f"missing required env var: {name}")
        sys.exit(1)
    return val


def bencher_get(path: str) -> dict:
    url = f"{BENCHER_API_URL}/v0/projects/{BENCHER_PROJECT}{path}"
    last_err = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, http.client.HTTPException, ValueError) as e:
            last_err = e
            time.sleep(2)
    raise last_err if last_err else RuntimeError(f"GET {url} failed")


def bencher_list_reports(branch: str, per_page: int = 1, testbed: typing.Optional[str] = None) -> list:
    q = {"branch": branch, "per_page": per_page}
    if testbed:
        q["testbed"] = testbed
    return bencher_get(f"/reports?{urllib.parse.urlencode(q)}")


def bencher_get_report(
    branch: str,
    hash: typing.Optional[str] = None,
    testbed: typing.Optional[str] = None,
) -> typing.Optional[dict]:
    """Return a report for a branch. If `hash` is given, find the (oldest)
    report whose head commit matches it; otherwise return the latest report.
    `testbed` restricts the lookup so the baseline is only drawn from the same
    testbed (e.g. `self-hosted-gpu`) — otherwise old numbers from another
    machine (e.g. the VPS `self-hosted`) would be compared against it."""
    reports = bencher_list_reports(branch, per_page=100, testbed=testbed)
    if not reports:
        return None
    if hash is not None:
        for report in reports:
            if report["branch"]["head"]["version"]["hash"] == hash:
                return bencher_get(f"/reports/{report['uuid']}")
        return None
    return bencher_get(f"/reports/{reports[0]['uuid']}")


def format_ns(value: float) -> str:
    if value >= 1e9:
        return f"{value / 1e9:.3f} s"
    if value >= 1e6:
        return f"{value / 1e6:.3f} ms"
    if value >= 1e3:
        return f"{value / 1e3:.3f} µs"
    return f"{value:.0f} ns"


def percent_delta(pr_value: float, base_value: float) -> float:
    if base_value == 0 or not base_value or not pr_value:
        return 0.0
    return (pr_value - base_value) / base_value * 100.0


def measure_value(results: list, benchmark_name: str) -> typing.Optional[float]:
    """Find the first (iteration 0) latency value for a benchmark."""
    for iteration in results:
        for result in iteration:
            if result.get("benchmark", {}).get("name") == benchmark_name:
                for m in result.get("measures", []):
                    if m.get("measure", {}).get("slug") == "latency":
                        return m.get("metric", {}).get("value")
    return None


def _collect_rows(results: list, base_results: list) -> list:
    """Return (benchmark name, formatted value, delta %) per benchmark."""
    rows = []
    for iteration in results:
        for result in iteration:
            name = result["benchmark"]["name"]
            for m in result.get("measures", []):
                value = m.get("metric", {}).get("value")
                if value is None:
                    continue
                base_value = measure_value(base_results, name)
                delta = percent_delta(value, base_value) if base_value is not None else None
                rows.append((name, format_ns(value), delta))
    return rows


def _markdown_table(rows: list) -> str:
    lines = ["| Benchmark | Result | Δ vs target |", "| --- | ---: | ---: |"]
    for name, value_str, delta in rows:
        delta_str = f"{delta:+.1f}%" if delta is not None else "—"
        lines.append(f"| {name} | {value_str} | {delta_str} |")
    return "\n".join(lines)


def render_markdown(pr_report: dict, base_report: typing.Optional[dict]) -> str:
    project = pr_report["project"]["name"]
    branch = pr_report["branch"]["name"]
    testbed = pr_report["testbed"]["name"]
    sha = pr_report["branch"]["head"]["version"]["hash"]
    report_uuid = pr_report["uuid"]

    lines = []
    lines.append(f"## 🐰 Bencher Report — {branch}")
    lines.append("")
    lines.append(
        f"[View full report in Bencher](https://bencher.dev/perf/{BENCHER_PROJECT}/reports/{report_uuid})"
    )
    lines.append("")
    lines.append(f"- **Project:** {project}")
    lines.append(f"- **Branch:** {branch}")
    lines.append(f"- **Testbed:** {testbed}")
    lines.append(f"- **Commit:** `{sha[:10]}`")
    lines.append("")

    alerts = pr_report.get("alerts") or []
    if alerts:
        lines.append(f"### 🚨 {len(alerts)} Alert(s)")
        lines.append("")
        for alert in alerts:
            b = alert["benchmark"]["name"]
            m = alert["threshold"]["measure"]["name"]
            lines.append(f"- **{b}** ({m}) — see report")
        lines.append("")

    results = pr_report.get("results") or []
    base_results = (base_report.get("results") or []) if base_report else []
    benchmark_count = sum(len(it) for it in results)
    if benchmark_count == 0:
        lines.append("_No benchmarks found in this report._")
        lines.append("")
    else:
        rows = _collect_rows(results, base_results)
        # Benchmarks without a baseline sort last; the rest by |Δ| descending.
        rows.sort(
            key=lambda r: (
                r[2] is None,
                -(abs(r[2]) if r[2] is not None else 0.0),
            )
        )

        threshold = delta_threshold()
        improved = sum(1 for _, _, d in rows if d is not None and d <= -threshold)
        regressed = sum(1 for _, _, d in rows if d is not None and d >= threshold)
        unchanged = len(rows) - improved - regressed

        lines.append("### Summary")
        lines.append("")
        lines.append(
            f"✅ **Improved:** {improved} · "
            f"⚪ **Unchanged:** {unchanged} · "
            f"❌ **Regressed:** {regressed}"
        )
        lines.append(
            f"<sub>|Δ| ≥ {threshold:g}% counts as a change; "
            "Δ vs the start-point baseline</sub>"
        )
        lines.append("")

        shown = rows[:TOP_CHANGES]
        lines.append("### Top changes")
        lines.append("")
        lines.append(_markdown_table(shown))
        lines.append("")

        if len(rows) > TOP_CHANGES:
            hidden = rows[TOP_CHANGES:]
            lines.append("<details>")
            lines.append(f"<summary>All {len(rows)} benchmarks (show/hide)</summary>")
            lines.append("")
            lines.append(_markdown_table(hidden))
            lines.append("")
            lines.append("</details>")
            lines.append("")

    lines.append("---")
    lines.append("")
    # The marker tag MUST stay the last thing in the body.
    lines.append(marker_tag())
    return "\n".join(lines)


def codeberg_request(method: str, path: str, body: typing.Optional[dict] = None):
    url = f"{CODEGBERG_URL}/api/v1/repos/{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"token {CODEGBERG_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        print(f"Codeberg API {method} {url} failed: {e.code} {e.reason}")
        print(e.read().decode(errors="replace"))
        raise

def find_existing_comment(owner: str, repo: str, pr: int) -> typing.Optional[int]:
    page = 1
    tag = marker_tag()
    while True:
        comments = codeberg_request(
            "GET",
            f"{owner}/{repo}/issues/{pr}/comments?per_page=100&page={page}",
        )
        for comment in comments:
            body = comment.get("body") or ""
            if body.rstrip().endswith(tag):
                return comment["id"]
        if len(comments) < 100:
            return None
        page += 1


def main() -> int:
    if not CODEGBERG_TOKEN:
        print("CODEGBERG_PR_COMMENT_TOKEN not set; skipping comment")
        return 0

    owner = required("CI_REPO_OWNER")
    repo = required("CI_REPO_NAME")
    pr = required("CI_COMMIT_PULL_REQUEST")
    source = required("CI_COMMIT_SOURCE_BRANCH")
    target = required("CI_COMMIT_TARGET_BRANCH")
    sha = required("CI_COMMIT_SHA")

    pr_report = bencher_get_report(source)
    if pr_report is None:
        print(f"No Bencher report found for branch '{source}'; skipping comment")
        return 0

    report_hash = pr_report["branch"]["head"]["version"]["hash"]
    if report_hash != sha:
        print(
            f"Report hash {report_hash[:10]} != CI commit {sha[:10]}; "
            "report is stale (rerun in progress?), skipping"
        )
        return 0

    base_report = bencher_get_report(
        target,
        hash=pr_report["branch"]["head"]["start_point"]["version"]["hash"],
        testbed=pr_report["testbed"]["slug"],
    )
    if base_report is None:
        print(
            f"No baseline report found on '{target}' for start point "
            f"{pr_report['branch']['head']['start_point']['version']['hash'][:10]}; "
            "deltas will be omitted"
        )
    body = render_markdown(pr_report, base_report)

    comment_id = find_existing_comment(owner, repo, int(pr))
    if comment_id is not None:
        codeberg_request(
            "PATCH",
            f"{owner}/{repo}/issues/comments/{comment_id}",
            {"body": body},
        )
        print(f"Updated comment {comment_id} on PR #{pr}")
    else:
        codeberg_request(
            "POST",
            f"{owner}/{repo}/issues/{pr}/comments",
            {"body": body},
        )
        print(f"Created comment on PR #{pr}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

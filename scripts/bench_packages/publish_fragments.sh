#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Copy collected schema-v2 fragments into a `pages` checkout and push them.
# Env:
#   RESULTS_DIR  directory with <package>.json fragments
#   PAGES_DIR    checkout of the `pages` branch (remote already authenticated)
#   AMEND=1      amend the run's fragment commit instead of creating a new one
#                (used by run_all.sh so a whole run leaves a single commit)
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
PAGES_DIR="${PAGES_DIR:?PAGES_DIR is required}"
AMEND="${AMEND:-0}"
SUBJECT="pages: refresh benchmark fragments"

mkdir -p "$PAGES_DIR/data"
cp -f "$RESULTS_DIR"/*.json "$PAGES_DIR/data/"

cd "$PAGES_DIR"
git add data
if git diff --cached --quiet; then
  echo "no fragment changes to publish"
  exit 0
fi

if [ "$AMEND" = "1" ] && git log -1 --format=%s 2>/dev/null | grep -q "^${SUBJECT}"; then
  git -c user.name=infomeasure-bench-ci -c user.email=ci@cbueth.de commit --amend --no-edit
  git push --force-with-lease origin HEAD:pages
  echo "amended the fragment commit on pages"
else
  git -c user.name=infomeasure-bench-ci -c user.email=ci@cbueth.de \
    commit -m "${SUBJECT} ($(date -u +%Y-%m-%d))"
  git push origin HEAD:pages
  echo "published $(git diff --name-only HEAD~1 HEAD -- data | wc -l) fragment(s)"
fi
# Keep the remote-tracking ref in step with what we just pushed so the next
# amend can use --force-with-lease safely.
git update-ref "refs/remotes/origin/pages" HEAD

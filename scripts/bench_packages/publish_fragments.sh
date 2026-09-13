#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Copy collected schema-v2 fragments into a `pages` checkout and push them.
# Env:
#   RESULTS_DIR  directory with <package>.json fragments
#   PAGES_DIR    checkout of the `pages` branch (remote already authenticated)
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
PAGES_DIR="${PAGES_DIR:?PAGES_DIR is required}"

mkdir -p "$PAGES_DIR/data"
cp -f "$RESULTS_DIR"/*.json "$PAGES_DIR/data/"

cd "$PAGES_DIR"
if [ -z "$(git status --porcelain data)" ]; then
  echo "no fragment changes to publish"
  exit 0
fi

git add data
git -c user.name=infomeasure-bench-ci -c user.email=ci@cbueth.de \
  commit -m "pages: refresh benchmark fragments ($(date -u +%Y-%m-%d))"
git push origin HEAD:pages
echo "published $(git diff --name-only HEAD~1 HEAD -- data | wc -l) fragment(s)"

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Generate derived pin files from the benchmark registry.

The registry (``registry.json`` on the ``pages`` branch) is the **single source
of truth** for pinning package and runtime versions. This script expands it
into the files the container build consumes, so nothing is hand-maintained in
two places:

    requirements.txt   pinned Python deps + benchmarked PyPI packages
    pins.env           JULIA_VERSION / JULIA_SHA256 / JDK_PACKAGE /
                       JIDT_VERSION / JIDT_JAR_SHA256 + Julia package pins

CI checks out ``pages`` and runs this before building
``.docker/Dockerfile.bench``; locally, run it the same way (see run_in_container
docs).

Usage:
    python scripts/bench_packages/sync_pins.py \
        --registry /path/to/pages/registry.json --out .
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default="registry.json", help="Path to registry.json")
    ap.add_argument("--out", default=".pins", help="Output directory for derived files")
    args = ap.parse_args()

    reg = json.loads(Path(args.registry).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # requirements.txt — non-benchmarked deps first, then benchmarked PyPI packages.
    reqs: list[str] = []
    for dep in reg.get("dependencies", {}).get("python", []):
        reqs.append(f"{dep['name']}=={dep['version']}")
    for pkg in reg.get("packages", []):
        if pkg.get("ecosystem") == "pypi":
            name = pkg.get("distribution", pkg["id"])
            reqs.append(f"{name}=={pkg['version']}")
    (out / "requirements.txt").write_text("\n".join(sorted(reqs)) + "\n")

    # pins.env — runtime + artifact pins consumed by the Dockerfile.
    env: dict[str, str] = {}
    runtimes = reg.get("runtimes", {})
    if "julia" in runtimes:
        env["JULIA_VERSION"] = runtimes["julia"]["version"]
        env["JULIA_SHA256"] = runtimes["julia"]["sha256"]
    if "jdk" in runtimes:
        env["JDK_PACKAGE"] = runtimes["jdk"]["package"]
    if "r" in runtimes and "cran_snapshot" in runtimes["r"]:
        env["R_CRAN_SNAPSHOT"] = runtimes["r"]["cran_snapshot"]
    for pkg in reg.get("packages", []):
        if pkg["id"] == "jidt":
            env["JIDT_VERSION"] = pkg["version"]
            env["JIDT_JAR_SHA256"] = pkg["artifact_sha256"]
    for dep in reg.get("dependencies", {}).get("julia", []):
        env[f"JULIA_PKG_{dep['name'].upper()}"] = dep["version"]
    for pkg in reg.get("packages", []):
        if pkg.get("ecosystem") == "julia":
            name = pkg.get("distribution", pkg["id"]).upper()
            env[f"JULIA_PKG_{name}"] = pkg["version"]
    (out / "pins.env").write_text("\n".join(f"{k}={v}" for k, v in sorted(env.items())) + "\n")

    print(f"wrote {out/'requirements.txt'} and {out/'pins.env'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Smoke-test every benchmark dependency inside the :bench image by running a
# real (small) computation, not just an import.
# Run: docker run --rm im-bench:dev bench-smoke

set -uo pipefail

JIDT_JAR="${JIDT_JAR:-/opt/jidt/infodynamics.jar}"
BENCH_PY="${BENCH_VENV:-/opt/bench-venv}/bin/python"
status=0
ok()   { echo "  OK   $1"; }
fail() { echo "  FAIL $1"; status=1; }

echo "=== system ==="
. /etc/os-release 2>/dev/null && echo "os: ${PRETTY_NAME:-?}"
echo

echo "=== java / JIDT ==="
command -v java >/dev/null && java -version 2>&1 | head -1 || { fail "java missing"; }
if [ -f "$JIDT_JAR" ]; then
    echo "jar: $JIDT_JAR ($(stat -c%s "$JIDT_JAR") bytes)"
    if java -cp "$JIDT_JAR:/opt/jidt-smoke" JidtSmoke; then ok "JIDT KSG MI"; else fail "JIDT KSG MI"; fi
else
    fail "JIDT jar missing at $JIDT_JAR"
fi
echo

echo "=== python ($BENCH_PY) ==="
if [ -x "$BENCH_PY" ]; then
    "$BENCH_PY" - <<'PY'
import sys

bad = 0

def check(name, fn):
    global bad
    try:
        fn()
        print(f"  OK   {name}")
    except Exception as e:
        bad += 1
        print(f"  FAIL {name}: {type(e).__name__}: {e}")

import numpy as np
rng = np.random.default_rng(0)
x = rng.standard_normal(200)
y = 0.5 * x + np.sqrt(0.75) * rng.standard_normal(200)

def do_infomeasure():
    import infomeasure as im
    h = float(im.entropy(np.array([1, 2, 1, 2, 1, 2]), approach="discrete"))
    print(f"       infomeasure {getattr(im, '__version__', '?')} H={h:.6f}")

def do_pyinform():
    import pyinform
    te = float(pyinform.transfer_entropy((x > 0).astype(int).tolist(),
                                         (y > 0).astype(int).tolist(), k=1))
    print(f"       pyinform TE={te:.6f}")

def do_dit():
    import dit
    d = dit.Distribution(["0", "1"], [0.5, 0.5])
    print(f"       dit {getattr(dit, '__version__', '?')} H={float(dit.shannon.entropy(d)):.6f}")

def do_pyitlib():
    from pyitlib import discrete_random_variable as drv
    print(f"       pyitlib H={float(drv.entropy([1, 2, 1, 2])):.6f}")

def do_npeet():
    from npeet import entropy_estimators as ee
    mi = float(ee.mi(x.reshape(-1, 1).tolist(), y.reshape(-1, 1).tolist()))
    print(f"       npeet MI={mi:.6f}")

for name, fn in [("infomeasure", do_infomeasure), ("pyinform", do_pyinform),
                 ("dit", do_dit), ("pyitlib", do_pyitlib), ("npeet", do_npeet)]:
    check(name, fn)

sys.exit(1 if bad else 0)
PY
    [ $? -ne 0 ] && status=1
else
    fail "bench venv missing at $BENCH_PY"
fi
echo

echo "=== R ==="
if command -v Rscript >/dev/null; then
    if Rscript -e 'suppressMessages(library(RTransferEntropy)); set.seed(0); x <- rnorm(200); y <- 0.5*x + rnorm(200); cat(sprintf("       R %s, RTransferEntropy %s, TE=%.6f\n", as.character(getRversion()), as.character(packageVersion("RTransferEntropy")), calc_te(x, y)))' 2>&1 | tail -3; then
        ok "RTransferEntropy"
    else
        fail "RTransferEntropy"
    fi
else
    fail "Rscript missing"
fi
echo

if [ "$status" -eq 0 ]; then echo "SMOKE: ALL OK"; else echo "SMOKE: FAILURES PRESENT"; fi
exit "$status"

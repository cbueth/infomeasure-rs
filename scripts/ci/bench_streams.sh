#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
#
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Run the PR benchmark suite with 3 concurrent non-`*_parallel` streams, then
# the `*_parallel` groups alone, then the GPU groups alone. The concatenated
# Criterion output is printed on stdout for `bencher run` to parse.
#
# Why: the box has 6 cores / 12 threads and Criterion runs one group at a time.
# Cross-talk measured on this box (i7-8750H, no-turbo): co-running up to 3 light
# groups stays within run-to-run noise, 4 degrades, and the 4-thread
# `*_parallel` groups are the most contention-sensitive. The GPU groups are the
# noisiest benches, so they also run alone.
#
# Env:
#   BENCH_ARGS   the `--bench <name>` list (required)
#   CRIT_FLAGS   Criterion flags (required)
#   BENCH_FEATURES  cargo features for the bench run (default: gpu,parallel)
#   GPU_TARGETS  bench targets kept alone (default: kernel_gpu gpu_crossover)
#   STREAMS      number of concurrent non-parallel streams (default: 3)
set -uo pipefail

: "${BENCH_ARGS:?BENCH_ARGS not set}"
: "${CRIT_FLAGS:?CRIT_FLAGS not set}"
FEATURES="${BENCH_FEATURES:-gpu,parallel}"
GPU_TARGETS="${GPU_TARGETS:-kernel_gpu gpu_crossover}"
STREAMS="${STREAMS:-3}"
HEAVY_RE='kernel|ksg|kl|renyi|tsallis'

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Targets from "--bench a --bench b ...".
TARGETS=()
prev=""
for w in $BENCH_ARGS; do
  [ "$prev" = "--bench" ] && TARGETS+=("$w")
  prev="$w"
done

is_gpu() { case " $GPU_TARGETS " in *" $1 "*) return 0 ;; esac; return 1; }

groups_for() {
  cargo bench --features "$FEATURES" --bench "$1" -- --list 2>/dev/null \
    | sed -n 's#^\([^/]*\)/.*: benchmark$#\1#p'
}

STREAM_GROUPS=()
PAR_GROUPS=()
for t in "${TARGETS[@]}"; do
  is_gpu "$t" && continue
  while IFS= read -r g; do
    [ -n "$g" ] || continue
    case "$g" in
      *_parallel) PAR_GROUPS+=("$g") ;;
      *)          STREAM_GROUPS+=("$g") ;;
    esac
  done < <(groups_for "$t")
done

# Partition: heavy groups round-robin first (spread the expensive kernel/ksg/kl
# ones), then the light ones, so the streams finish together.
declare -a B0=() B1=() B2=()
h=0; l=0
for g in "${STREAM_GROUPS[@]}"; do
  [[ $g =~ $HEAVY_RE ]] || continue
  case $((h % STREAMS)) in 0) B0+=("$g");; 1) B1+=("$g");; *) B2+=("$g");; esac
  h=$((h + 1))
done
for g in "${STREAM_GROUPS[@]}"; do
  [[ $g =~ $HEAVY_RE ]] && continue
  case $((l % STREAMS)) in 0) B0+=("$g");; 1) B1+=("$g");; *) B2+=("$g");; esac
  l=$((l + 1))
done

regex_for() {
  local IFS='|'
  printf '^(%s)/' "$*"
}

echo "bench_streams: ${#TARGETS[@]} targets; ${#STREAM_GROUPS[@]} stream groups; ${#PAR_GROUPS[@]} parallel groups; gpu=[$GPU_TARGETS]" >&2

PIDS=()
run_bg() { # bucket_index regex
  local i="$1" re="$2"
  cargo bench --features "$FEATURES" $BENCH_ARGS -- "$re" $CRIT_FLAGS >"$TMP/s$i" 2>&1 &
  PIDS+=("$!")
}

[ ${#B0[@]} -gt 0 ] && run_bg 0 "$(regex_for "${B0[@]}")"
[ ${#B1[@]} -gt 0 ] && run_bg 1 "$(regex_for "${B1[@]}")"
[ ${#B2[@]} -gt 0 ] && run_bg 2 "$(regex_for "${B2[@]}")"

s_rc=0
if [ ${#PIDS[@]} -gt 0 ]; then
  for p in "${PIDS[@]}"; do wait "$p" || s_rc=1; done
fi

# `*_parallel` groups alone (they need 4 free cores each).
p_rc=0
if [ ${#PAR_GROUPS[@]} -gt 0 ]; then
  cargo bench --features "$FEATURES" $BENCH_ARGS -- '_parallel' $CRIT_FLAGS >"$TMP/par" 2>&1 || p_rc=1
fi

# GPU groups alone.
g_rc=0
gpu_args=""
for t in $GPU_TARGETS; do gpu_args="$gpu_args --bench $t"; done
if [ -n "$gpu_args" ]; then
  # shellcheck disable=SC2086
  cargo bench --features "$FEATURES" $gpu_args -- $CRIT_FLAGS >"$TMP/gpu" 2>&1 || g_rc=1
fi

# Emit all Criterion output (bencher parses stdout).
for f in "$TMP/s0" "$TMP/s1" "$TMP/s2" "$TMP/par" "$TMP/gpu"; do
  [ -f "$f" ] && cat "$f"
done

if [ "$s_rc" -ne 0 ] || [ "$p_rc" -ne 0 ] || [ "$g_rc" -ne 0 ]; then
  echo "bench_streams: a bench phase failed (streams=$s_rc parallel=$p_rc gpu=$g_rc)" >&2
  exit 1
fi

<!--
SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Codeberg CI Configuration

This repository uses Woodpecker CI for continuous integration with separate pipelines for linting and testing.

## Pipeline Structure

### Lint Pipeline (`.woodpecker/lint.yml`)
Fast code quality checks that run on every push and pull request to `main` and `develop` branches:

- **Formatting Check**: Uses `rustfmt` to ensure consistent code formatting
- **Clippy Linting**: Runs `clippy` with strict warnings as errors
- **Documentation Check**: Verifies documentation builds without warnings

### Runner rules

Two self-hosted runners, selected by label:

- **CPU runner** (`type: cpu`) — lint and the nightly (`test-nightly.yml`).
- **GPU runner** (`type: gpu`) — `bench.yml`, `test-gpu.yml`, `bench-packages.yml`.

The PR test jobs (`test.yml`) only request `provider: self-hosted`, so either
runner takes them: whichever is free grabs the next topic and the rest queue.
Benchmark jobs must run alone (their numbers are threshold-gated and sensitive to
contention); both agents run one workflow at a time, and `bench` /
`bench-packages` share the `gpu-exclusive` concurrency group as a safeguard if
that cap is ever raised. A floating test job can therefore delay a bench run but
never overlap it.

### Test Pipeline (`.woodpecker/test.yml`)
PR gate on either runner. The suite is split by topic (`expfam`, `kernel`,
`discrete`, `misc`, `ordinal`, `lib`, `noparallel`) as a matrix, so several
smaller jobs distribute across the two runners; each job builds for itself
(jobs may land on different runners and don't share a workspace). The heavier
topics finish sooner on the GPU box.

### Nightly Test Pipeline (`.woodpecker/test-nightly.yml`)
Triggered by a cron job named `test-nightly` on `main` (repo settings → Cron).
Runs `beta` on the CPU runner, plus beta `clippy`, so PRs can stay stable-only.

### GPU Test Pipeline (`.woodpecker/test-gpu.yml`)
Runs on the GPU runner with `--features gpu`: the GPU-accelerated correctness
tests plus the ignored discrete GPU smoke test. A guard fails the run if only a
software adapter is found.

### Benchmark Pipeline (`.woodpecker/bench.yml`)
Runs Criterion benchmarks on the GPU runner and reports to
[Bencher](https://bencher.dev/perf/infomeasure-rs): a PR comment on pull
requests, and the threshold baseline on `main`. The suite is collected by
`scripts/ci/bench_streams.sh`: **3 concurrent non-`*_parallel` streams**, then
the `*_parallel` groups and the GPU groups alone. Cross-talk on this box stays
within run-to-run noise at 3 concurrent groups but degrades at 4, and the
4-thread `*_parallel`/GPU benches are the most sensitive, so they never co-run.

- **Testbed = environment.** Bencher keys comparisons and thresholds on
  (branch, testbed, measure), so `--testbed` names the machine's power state (and
  collection mode). Changing it starts a fresh baseline instead of comparing
  across environments; the current testbed ends in `parallel3`.
- **Pinned CPU clock.** The GPU runner runs at a fixed, throttling-free clock
  (TLP, no turbo) for stable timings.
- **Alerts fail PRs** (`--error-on-alert`); if the clock or hardware changes,
  bump the testbed name so the old thresholds are not applied.
- **Dense/sparse guards.** The discrete-MLE benches (`mi/te/cmi/cte`) keep their
  default small base and add one large-alphabet point (`mle_b*` /
  `mle_alphabet_b*`), so Bencher tracks the dense joint, the sparse hash joint
  (`cmi`/`te` base 100, `cte` base 50) and MI's dense-table fallback (base 1500).
  `entropy_discrete` adds the known-alphabet histogram lever
  (`alphabet_b10`/`alphabet_b200`). Only the new `b*` ids start fresh Bencher
  baselines; the existing ids are unchanged.

## Python Integration

The test pipeline installs micromamba before running tests:
- Downloads micromamba binary from https://micro.mamba.pm
- Creates Python `infomeasure-rs-validation` environment from `tests/validation_crate/environment.yml`
- All 27 validation tests automatically use micromamba environment
- Single test step handles both Rust compilation and Python validation

## Docker Strategy

- **Lint pipeline**: Uses standard `rust:latest` (fast, no micromamba needed)
- **Test pipeline**: Uses `rust:${TOOLCHAIN}` + micromamba installation
- **Why this approach**: Simple, single test step, no over-engineering

## CI Triggers

Both pipelines trigger on:
- Push events to `main` or `develop` branches
- Pull requests targeting `main` or `develop` branches

## Resource Considerations

- GPU support testing runs on the dedicated self-hosted GPU runner (`.woodpecker/test-gpu.yml`)
- Micromamba is installed at runtime for efficiency
- Tests run only with relevant feature flags to optimize CI time

## Cross-package benchmark pipeline (`.woodpecker/bench-packages.yml`)

Collects the runtime benchmark data and publishes it to the `pages` branch. It
runs on the self-hosted GPU testbed and never touches `main`. The workflow and
all collector code live on `main` only; `pages` carries `registry.json`, the
viewer and `data/`.

Stages (sequential steps in one workflow):

1. **checkout** – clones `pages` (registry + existing fragments) and the source
   (`main`, or the triggering branch on a manual run).
2. **plan** – `check_versions.py` compares the pins to upstream; `ci_plan.py`
   decides the action (`collect` / `open-pr` / `none`) and writes `plan.env`.
3. **sync-pins** – expands `pages/registry.json` into `.pins/` for the image.
4. **image** – builds/publishes `codeberg.org/cbueth/infomeasure-rs:bench` with
   kaniko (registry-backed layer cache in `cbueth/infomeasure-rs-cache`; the
   plugin prepends the registry, so the setting omits `codeberg.org/`).
5. **collect-and-publish** – runs the collectors inside `:bench`, then pushes the
   changed fragments to `pages`.
6. **registry-pr** – writes and opens the registry bump PR against `pages`
   (cron only).

Policy by trigger:

| Trigger | Action |
| --- | --- |
| manual | collect **all** packages, publish |
| release tag | collect **all** packages, publish |
| cron `bench-*` | if upstream moved, open a `pages` registry PR; otherwise collect the packages whose published fragment is missing or pinned to a different version |
| push to `main` (`.docker/**`) | rebuild/publish `:bench` only |

There is **no** workflow copy on `pages` (single-workflow Option B): Woodpecker
reads `.woodpecker/` from the branch that triggers, so a manual/tag/`bench-*`
cron run on `main` uses this file. Consequently a `push to pages` triggers
nothing — a merged registry PR is picked up by the next `bench-*` cron via the
staleness check (`pages/data` fragment version vs `registry.json`), which delays
collection by up to one cron interval. Fragments are published directly to
`pages`; nothing is pushed to `main`.

The collector source is cloned from `main` for tag/cron events; a **manual** run
uses the selected branch instead, so the workflow can be tested on a PR branch
before it is merged.

### Long, resumable runs

The full detailed grid is expensive — the CPU Gaussian kernel is O(N²), so the
large sizes dominate — and the shared Codeberg CI can drop the agent connection
(woodpecker#6803; the runner already sets `WOODPECKER_KEEPALIVE_TIME=30s` and
`WOODPECKER_RETRY_TIMEOUT=10m` as the upstream workaround). Mitigations:

- The project **Timeout** (Web UI → repository → Settings → Timeout) accepts at
  most **120 minutes**; a full run can approach that.
- Collectors are **non-fatal**: `run_all.sh` records a failure and keeps going
  (so one broken script can't discard hours of work), then exits non-zero at the
  end if anything failed. The step is red, but the data is published.
- Fragments are **published incrementally**: `collect-and-publish` passes
  `PAGES_DIR=/build/pages` and `run_all.sh` publishes after every package. The
  first publish creates the `pages: refresh benchmark fragments` commit and
  later ones **amend** it, so a run leaves a single commit.
- Collections are **resumable** across runs: `collect-and-publish` passes
  `BENCH_RESUME=1` and seeds `results/` from the published `pages/data/*.json`,
  so a cancelled/timed-out run continues from the last publish instead of
  starting over. For a clean full run, delete `target/bench-data/results/` first
  (or unset `BENCH_RESUME`).

### Required secrets (Woodpecker → repository → Settings → Secrets)

- **`PAGES_TOKEN`** — a Codeberg access token used to clone/push the `pages`
  branch and to open the registry PR. Create it at
  *Codeberg → Settings → Applications → Generate new token* with the
  **`write:repository`** and **`write:issue`** scopes, then add it as the
  `PAGES_TOKEN` secret. (Use a dedicated token, not the release `FORGE_TOKEN`.)
- **`CODEBERG_PACKAGES_TOKEN`** — already required by `build-deps.yml` for
  publishing images to the Codeberg package registry; reused here for `:bench`.

If the `:bench` image is private, add Codeberg registry credentials under
*Settings → Registries* so the runner can pull it.

### Cron jobs (Woodpecker → repository → Settings → Cron)

Both crons run on branch **`main`** and are separated by their **name**, which
the workflow `when` blocks match with a glob (`cron: bench-*` / `cron: docker-*`)
so they do not fire each other:

- Docker toolchain image: name the cron **`docker-*`** (e.g. `docker-weekly`).
- Benchmark run: name the cron **`bench-*`** (e.g. `bench-biweekly`). The
  pipeline gates itself to every second ISO week (`BENCH_CRON_BIWEEKLY=1`), so
  it effectively runs bi-weekly. On an "off" week it does nothing.

### Fragment contract

Fragments are schema-v2 JSON written by `scripts/bench_packages/*` collectors;
`merge_results.py` stamps the run's hardware into every fragment, and each
fragment records the installed package version in `meta.packages`. The site
reads `pages/data/<package>.json` (missing files simply render as “not
collected”), and `ci_plan.py` uses the same version to decide staleness.
Publishing from a development machine is intentionally not done: numbers are
only comparable when collected on the CI runner.

### Third-party Rust collectors

`logp` and `entropium` are Rust comparison crates. They must never enter the
infomeasure-rs build (normal or dev), so they live in a standalone workspace
crate under `scripts/bench_packages/rust_collect/` (own `Cargo.toml` /
`Cargo.lock`) rather than as benches in the main crate. `run_all.sh` runs it via
`cargo run --release --manifest-path … --bin collect_logp` / `collect_entropium`.
It reads the same canonical datasets and writes the same schema-v2 fragments
(`results/logp.json`, `results/entropium.json`); the registry entries on
`pages` carry `ecosystem: crates`, so the weekly cron tracks their versions like
any other package.

Cross-grid coverage: both fill the discrete MLE cells (entropy, MI); `logp`
additionally fills KSG MI (`k = 4`, Algorithm 1). Cells a package does not
implement (e.g. CMI, TE/CTE, kernel) render as N/A — measures composed from
several package calls are not timed.

### Alphabet-scaling family

Alongside the main fragments, the discrete collectors emit one
alphabet-scaling fragment per package (`<pkg>_alphabet.json`): discrete MLE
timed across state counts at the cross sizes. The configuration lives in
`benches/detailed_grid.json` under `alphabet` (`states`, `sizes`, per-measure
`caps`, `budget_s`). The per-measure caps are a memory guard for the dense
tables (`entropy`/`mi`/`cmi`/`te` ≤ 200, `cte` ≤ 50, since its table is
`base^4`); collection then walks N ascending and skips larger N once a cell's
mean exceeds the budget. Datasets are
`<measure>_discrete_b<states>_s<seed>_n<n>.bin` (`gen_datasets`, `DATA_VERSION`
is part of the resume fingerprints). The site renders them in the "Alphabet
scaling" view, and they are excluded from the merged `cross_package.json`.
Only genuine discrete-MLE packages are collected: `syntropy` (kNN/continuous)
and `RTransferEntropy` (quantile-binned continuous) are excluded.

### GPU overlay

`infomeasure-rs` has GPU-accelerated kernel estimators (wgpu; Vulkan on the
runner). Alongside the full CPU fragment, the Rust collector emits a sparse
overlay, `infomeasure-rs_gpu.json`, containing only the kernel variants above
the dispatch gate (Gaussian ≥500, box ≥2000 points — resolved from the
detected adapter, so a software or absent adapter yields an empty overlay).
`run_all.sh` runs it as `BENCH_GPU=1 … cargo bench --features gpu --bench
collect_cross_package`; the CI `collect` step sources `init-nvidia-vulkan.sh`
and sets the NVIDIA env (mirroring the GPU test pipeline). The overlay is
excluded from `cross_package.json` and drives the site's CPU/GPU toggle, which
uses a GPU entry where one exists and the CPU number everywhere else. So the
toggle contrasts infomeasure-rs **GPU** against the CPU baseline rather than a
GPU-vs-GPU run: JIDT's CUDA path accelerates surrogate significance testing (not
a single estimate) and the other packages are CPU-only.

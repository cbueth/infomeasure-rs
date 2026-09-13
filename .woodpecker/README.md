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

### Test Pipeline (`.woodpecker/test.yml`)
Comprehensive testing with matrix strategy:

**Matrix Configuration:**
- **Toolchains**: `stable`, `beta` (removed nightly per requirement)
- **Features**:
  - No features (baseline)
  - `simd` only
  - `fast_exp` only
  - Skip `gpu` (not available in CI)

### GPU Test Pipeline (`.woodpecker/test-gpu.yml`)
Runs on the dedicated self-hosted GPU runner (`labels: provider=self-hosted, type=gpu`):

- **Toolchain**: `stable` only (conserves the scarce GPU runner)
- **Feature**: `gpu`
- Builds the workspace with `--features gpu`, then runs the GPU-accelerated correctness
  tests (name-filtered to `gpu`) plus the `#[ignore]`d discrete GPU smoke test
- Software-fallback guard: the GPU tests fail if wgpu only finds a CPU/software adapter
  (llvmpipe/lavapipe), so they never silently run on the CPU

**Test Steps:**
1. **Build**: Compile with feature-specific flags using pre-built CI image
2. **Test**: Run unit tests with appropriate features
3. **Python Validation**: Run validation tests with micromamba environment

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

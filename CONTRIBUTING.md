# Contributing to infomeasure-rs

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on the repository. Include a clear description of the problem, steps to reproduce, and any relevant error messages. Including a minimal reproduction case helps us fix the issue faster.

### Suggesting Features

Feature requests are welcome. Open an issue to describe the feature you'd like to see, why it's useful, and how you envision it working. We appreciate background on the use case.

### Pull Requests

1. Fork the repository and create a branch for your changes
2. Make your changes following the code style guidelines
3. Add tests for new functionality
4. Ensure all tests pass before submitting
5. Update documentation as needed
6. Open a pull request with a clear description of the changes

## Development Setup

### Prerequisites

- Rust 1.70 or later
- uv (for Python validation tests)

### Setting Up the Python Environment

Validation tests compare results against the infomeasure Python package. Set up the environment before running tests:

```bash
cd tests/validation_crate
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Running Tests

Run the full test suite including Python validation:

```bash
cargo test
```

For Rust-only unit tests:

```bash
cargo test --lib
```

Run a specific test:

```bash
cargo test -- <test_name>
```

## Profiling

The repository ships a sampling profiler harness for finding hotspots in the
estimator implementations. It samples CPU stacks on macOS, Linux and Windows
without kernel privileges.

```bash
RUSTFLAGS="-Cforce-frame-pointers" OPENBLAS_NUM_THREADS=1 \
    cargo run --profile profiling --features profiling --example profile_report
```

Both environment prefixes matter:

- `RUSTFLAGS="-Cforce-frame-pointers"` keeps frame pointers in compiled code so
  the sampler walks stacks directly. Without it, libunwind's DWARF unwinder is
  used from inside a signal handler, which is not async-signal-safe on Apple
  silicon and traps intermittently.
- `OPENBLAS_NUM_THREADS=1` stops OpenBLAS (which configures its worker pool
  before `main`) from busy-spinning through roughly a fifth of all samples.

Select what to profile with `PROFILE_ESTIMATOR`. Available workloads:
`discrete_entropy`, `mi_discrete`, `ordinal`, `renyi`, `tsallis`, `kl`,
`ksg_mi`, `ksg_cmi`, `kernel_gaussian_cpu`, `kernel_box_cpu`.

Useful knobs: `PROFILE_N` (dataset size), `PROFILE_K`, `PROFILE_BW`,
`PROFILE_ORDER`, `PROFILE_ALPHA`, `PROFILE_Q`, `PROFILE_SECONDS`,
`PROFILE_TOP` (table rows), and `PROFILE_FORMAT=json` for machine-readable
output. A flamegraph SVG lands in `target/profiling/` for visual inspection.

When optimising, capture a JSON profile before and after your change and
compare the ranked self-time entries. Wall-clock deltas belong to
criterion/Bencher.

## Code Style

The project follows standard Rust formatting. Run formatting before committing:

```bash
cargo fmt
```

Run clippy for linting:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Run type checking:

```bash
cargo check --all-features
```

## Benchmarks

Run performance benchmarks:

```bash
cargo bench
```

Benchmarks use Criterion and compare against the Python implementation.

## Documentation

Documentation is generated with KaTeX support for mathematical notation. Build the docs:

```bash
cargo doc --no-deps
```

For local doc serving:

```bash
cargo doc --no-deps --open
```

## License

By contributing to infomeasure-rs, you agree that your contributions will be licensed under the terms of the MIT OR Apache-2.0 license.

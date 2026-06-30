<!--
SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

[![status-badge](https://ci.codeberg.org/api/badges/16039/status.svg)](https://ci.codeberg.org/repos/16039)
[![CodSpeed](https://img.shields.io/endpoint?url=https://codspeed.io/badge.json)](https://app.codspeed.io/cbueth/infomeasure-rs?utm_source=badge)
[![docs.rs](https://docs.rs/infomeasure/badge.svg)](https://docs.rs/infomeasure)
[![crates.io](https://img.shields.io/crates/v/infomeasure.svg)](https://crates.io/crates/infomeasure)
[![rustc](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSES/MIT.txt)
[![Benchmarks](https://img.shields.io/badge/benchmarks-visit-8A2BE2)](https://cbueth.codeberg.page/infomeasure-rs/)

> **v0.3.0-beta.1 — We want your feedback!**
> All core features of the [infomeasure Python package](https://github.com/cbueth/infomeasure)
> have been reimplemented in Rust and are ready for testing.
> Try it out with our [Rust Guide](https://docs.rs/infomeasure/latest/infomeasure/guide/index.html)
> and [report issues or suggestions](https://codeberg.org/cbueth/infomeasure-rs/issues).
> Find the [benchmark and interactive Rust vs Python performance comparison here](https://cbueth.codeberg.page/infomeasure-rs/).

# infomeasure-rs

High-performance Rust library for information-theoretic measures with multiple estimation approaches.

## What This Does

`infomeasure-rs` computes **entropy**, **mutual information**, and **transfer entropy** from data using four different estimation strategies:

- **Discrete**: For categorical data with 11+ bias-corrected estimators
- **Kernel**: For continuous data with optional GPU acceleration
- **Ordinal**: For time series using permutation patterns
- **Exponential Family**: For high-dimensional data using k-NN

## Installation

```toml
[dependencies]
infomeasure = "0.3.0-beta.1"
```

## Quick Start

```rust
use infomeasure::estimators::entropy::Entropy;
use infomeasure::estimators::traits::GlobalValue;
use ndarray::array;

// Discrete entropy
let data = array![1, 2, 1, 3, 2, 1];
let entropy = Entropy::new_discrete(data).global_value();
println!("Entropy: {}", entropy);

// Kernel entropy for continuous data
let continuous = array![[1.0, 1.5], [2.0, 3.0], [4.0, 5.0]];
let kernel_entropy = Entropy::nd_kernel::<2>(continuous, 1.0).global_value();
println!("Kernel entropy: {}", kernel_entropy);
```

## Optional Features

- **`gpu`**: Enable GPU-accelerated kernel density estimation. Useful for large
  datasets with continuous variables (kernel approach) and batch processing.
- **`fast_exp`**: Use fast exponential approximations for improved performance
  in the exponential family (k-NN) estimators.

```toml
[dependencies]
infomeasure = { version = "0.3.0-beta.1", features = ["gpu"] }
```

```toml
[dependencies]
infomeasure = { version = "0.3.0-beta.1", features = ["fast_exp"] }
```

## Feature Status

All core measures from the Python package are implemented across all four
estimation approaches:

| Measure | Discrete | Kernel | Ordinal | k-NN (Exp. Family) |
|---------|----------|--------|---------|--------------------|
| **Entropy** $H(X)$ | ✅ | ✅ | ✅ | ✅ |
| **Joint Entropy** $H(X,Y)$ | ✅ | ✅ | ✅ | ✅ |
| **Conditional Entropy** $H(X\|Y)$ | ✅ | ✅ | ✅ | ✅ |
| **Cross-Entropy** $H_Q(P)$ | ✅[^1] | ✅ | ✅ | ✅ |
| **KLD** $D_{KL}(P\|Q)$ | ⚠️[^2] | ⚠️ | ⚠️ | ⚠️ |
| **JSD** $JSD(P\|Q)$ | ❌ | ❌ | ❌ | ❌ |
| **MI** $I(X;Y)$ | ✅ | ✅ | ✅ | ✅ |
| **CMI** $I(X;Y\|Z)$ | ✅ | ✅ | ✅ | ✅ |
| **TE** $T_{X \to Y}$ | ✅ | ✅ | ✅ | ✅ |
| **CTE** $T_{X \to Y\|Z}$ | ✅ | ✅ | ✅ | ✅ |

✅ = Implemented | ⚠️ = Via trait (see docs) | ❌ = Planned

[^1]: Discrete cross-entropy is available for MLE, Miller-Madow, and Bayesian estimators only.
[^2]: KLD is available via the `CrossEntropy` trait.

## Documentation

- **[Rust Guide](https://docs.rs/infomeasure/latest/infomeasure/guide/index.html)** — Comprehensive walkthrough of all estimators with mathematical background
- **[API Reference](https://docs.rs/infomeasure)** — Full API documentation with examples
- **[Examples](https://codeberg.org/cbueth/infomeasure-rs/src/branch/main/examples)** — Usage examples

## Rust vs Python

This crate reimplements the [infomeasure Python package](https://github.com/cbueth/infomeasure)
in Rust for users who need:

| Rust (`infomeasure-rs`)                                      | Python (`infomeasure`) |
|--------------------------------------------------------------|---|
| Compile-time type safety via Rust's type system              | Runtime string-based approach selection |
| Up to ~40x faster execution (detailled benchmarks to follow) | Flexible, scriptable interface |
| GPU acceleration for kernel estimators                       | GPU support via numba |
| Compile-time optimized estimator code                        | Runtime dispatch |
| `[dependencies]` in `Cargo.toml`                             | `pip install infomeasure` |

Choose **Rust** if you need maximum performance for production or large-scale analysis.
Choose **Python** if you need rapid prototyping, interactive analysis, or academic flexibility.

Full head-to-head benchmarks are being prepared and will be published soon.
See the [Rust Guide](https://docs.rs/infomeasure/latest/infomeasure/guide/index.html)
for a detailed comparison of the architecture.

## Python Compatibility

This crate maintains functional compatibility with the [infomeasure](https://github.com/cbueth/infomeasure)
Python package, implementing the same measures and estimation approaches, while providing 8-40x performance improvements.

## Repository Structure

- `src/` - Main source code
  - `estimators/` - Estimation techniques implementations
    - `approaches/` - Specific implementations (discrete, kernel, ...)
    - `traits/` - Shared interfaces for estimators
- `benches/` - Performance benchmarks using Criterion
- `tests/` - Unit and integration tests
- `examples/` - Example usage and demonstrations

## Development Setup

### Prerequisites
- **Rust** 1.70+ (for building)
- **uv** Python package manager (for validation tests)

### Python Environment Setup

The validation tests require a Python environment with `infomeasure` package.
Set it up once before running tests:

```bash
# Create virtual environment in validation crate directory
cd tests/validation_crate
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests (includes Python validation)
cargo test

# Run only Rust unit tests (skip Python validation)
cargo test --lib
```

## Testing and Validation

The project includes a validation crate that compares results with Python implementation to ensure compatibility and correctness.

## Benchmarks

Performance benchmarks are available for different estimation methods:

```bash
cargo bench
```

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT OR Apache-2.0

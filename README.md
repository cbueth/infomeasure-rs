<!--
SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>

SPDX-License-Identifier: MIT OR Apache-2.0
-->

[![status-badge](https://ci.codeberg.org/api/badges/16039/status.svg)](https://ci.codeberg.org/repos/16039)
[![docs.rs](https://docs.rs/infomeasure/badge.svg)](https://docs.rs/infomeasure)
[![crates.io](https://img.shields.io/crates/v/infomeasure.svg)](https://crates.io/crates/infomeasure)
[![rustc](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSES/MIT.txt)

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
infomeasure = "0.1.0"
```

### Optional Features

Enable GPU acceleration for large datasets:
```toml
infomeasure = { version = "0.1.0", features = ["gpu"] }
```

Enable fast exponential approximations:
```toml
infomeasure = { version = "0.1.0", features = ["fast_exp"] }
```

```rust
use infomeasure::estimators::entropy::Entropy;
use ndarray::array;

// Discrete entropy
let data = array!(1, 2, 1, 3, 2, 1);
let entropy = Entropy::new_discrete(data).global_value();
println!("Entropy: {}", entropy);

// Continuous data with kernel estimation
let continuous = array![[1.0, 1.5], [2.0, 3.0], [4.0, 5.0]];
let kernel_entropy = Entropy::nd_kernel::<2>(continuous, 1.0).global_value();
println!("Kernel entropy: {}", kernel_entropy);
```

## Feature Status

| Feature | Discrete | Kernel | Ordinal | k-NN |
|---------|----------|--------|---------|------|
| **Entropy** | ✅ | ✅ | ✅ | ✅ |
| **Mutual Information** | ✅ | ✅ | 🔄 | 🔄 |
| **Transfer Entropy** | ✅ | ✅ | 🔄 | 🔄 |

✅ = Available | 🔄 = In Development | ❌ = Planned

## Documentation

- **[API Reference](https://docs.rs/infomeasure)** - Complete documentation
- **[Examples](examples/)** - Usage examples

## Advanced Features

### GPU Acceleration
Enable GPU computation for large datasets:
```toml
infomeasure = { version = "0.1.0", features = ["gpu"] }
```

### Performance Optimizations
Fast exponential approximations:
```toml
infomeasure = { version = "0.1.0", features = ["fast_exp"] }
```

## Python Compatibility

This crate maintains API compatibility with the [infomeasure](https://github.com/cbueth/infomeasure) Python package while providing 10-100x performance improvements.

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

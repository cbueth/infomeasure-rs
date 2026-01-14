---
SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>

SPDX-License-Identifier: MIT OR Apache-2.0
---

# infomeasure-rs

`infomeasure` is a Rust library for computing information-theoretic measures such as entropy, mutual information, and transfer entropy. This project is a Rust implementation of the [infomeasure](https://github.com/cbueth/infomeasure), designed to provide superior performance while maintaining API compatibility.

> [!IMPORTANT]
> This crate is under development, features will be added step by step.

## Features

- **Multiple Estimation Techniques**: Supports discrete and kernel-based approaches
- **Future**: all features from the parent package, see [infomeasure introduction](https://infomeasure.readthedocs.io/en/latest/guide/introduction/)
- **High Performance**: Leverages Rust's zero-cost abstractions for efficient computation
- **Python Compatibility**: Designed to potentially serve as a backend for the Python package
- **Modular Design**: Easily extensible architecture for adding new estimators and methods


## Installation

> [!NOTE]
> As of now infomeasure is not on crates.io, yet.

Add this to your `Cargo.toml`:

```toml
[dependencies]
infomeasure = "0.1.0"
```

## Usage Examples

### Calculating Entropy with Gaussian Kernel

```rust
use infomeasure::estimators::entropy::Entropy;
use ndarray::Array2;

fn main() {
    // Create or load your data
    let data = Array2::from_shape_vec((1000, 2), vec![/* your data */]).unwrap();

    // Calculate entropy with Gaussian kernel
    let bandwidth = 0.5;
    let entropy = Entropy::nd_kernel_with_type::<2>(
        data.clone(),
        "gaussian".to_string(),
        bandwidth
    ).global_value();

    println!("Entropy: {}", entropy);
}
```

## Repository Structure

- `src/` - Main source code
  - `estimators/` - Estimation techniques implementations
    - `approaches/` - Specific implementations (discrete, kernel, ...)
    - `traits/` - Shared interfaces for estimators
- `benches/` - Performance benchmarks using Criterion
- `tests/` - Unit and integration tests
- `examples/` - Example usage and demonstrations

## Testing and Validation

The project includes a validation crate that compares results with the Python implementation to ensure compatibility and correctness. Run tests with:

```bash
cargo test
```

## Benchmarks

Performance benchmarks are available for different estimation methods:

```bash
cargo bench
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

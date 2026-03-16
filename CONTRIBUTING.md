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

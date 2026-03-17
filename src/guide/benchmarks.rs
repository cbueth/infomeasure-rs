// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Performance Benchmarks
//!
//! This guide covers performance benchmarks for the various estimators in `infomeasure-rs`.
//!
//! ## Overview
//!
//! Benchmarks are measured on various platforms. The benchmarks measure execution time
//! as a function of input data size. Each estimator has different scaling characteristics.
//!
//! ## Estimator Comparison
//!
//! ### Discrete Estimators
//!
//! Fastest for categorical data with few states.
//!
//! | Data Size | Typical Time |
//! |-----------|--------------|
//! | 100 | ~0.02ms |
//! | 1,000 | ~0.02ms |
//! | 10,000 | ~0.2ms |
//! | 100,000 | ~2ms |
//!
//! ### Kernel Estimators
//!
//! Non-parametric density estimation. Slower but works with continuous data.
//!
//! | Data Size | Typical Time (bandwidth=0.5) |
//! |-----------|----------------------------|
//! | 100 | ~0.1ms |
//! | 1,000 | ~2ms |
//! | 10,000 | ~20ms |
//! | 100,000 | ~200ms |
//!
//! ### KL (Kozachenko-Leonenko)
//!
//! k-nearest neighbor based entropy estimator.
//!
//! | Data Size | Typical Time (k=3) |
//! |-----------|---------------------|
//! | 100 | ~0.05ms |
//! | 1,000 | ~0.3ms |
//! | 10,000 | ~3ms |
//! | 100,000 | ~30ms |
//!
//! ### Ordinal Estimators
//!
//! Permutation/ordinal pattern based entropy.
//!
//! | Data Size | Typical Time (order=3) |
//! |-----------|------------------------|
//! | 100 | ~0.02ms |
//! | 1,000 | ~0.1ms |
//! | 10,000 | ~1ms |
//! | 100,000 | ~10ms |
//!
//! ## Interactive Viewer
//!
//! For more detailed benchmarking data, including:
//! - Multiple parameter combinations (k, bandwidth, order, history length)
//! - Scaling plots
//! - Error bars
//! - Comparison across different measures
//!
//! See the **[Interactive Benchmark Viewer](TBD)** (separate repository).
//!
//! ## Running Your Own Benchmarks
//!
//! To run benchmarks on your own machine:
//!
//! ```bash
//! # Install Python dependencies
//! cd tests/validation_crate
//! .venv/bin/pip install -r requirements.txt
//!
//! # Run benchmarks
//! python ../../scripts/python_benchmark.py \
//!     --group discrete \
//!     --measures entropy,mi,te \
//!     --sizes 100,1000,10000,100000 \
//!     --iterations 10 \
//!     --output results.json
//! ```
//!
//! ## Rust Benchmarks
//!
//! For Rust-specific benchmarks (comparing to Python implementation):
//!
//! ```bash
//! cargo bench
//! ```

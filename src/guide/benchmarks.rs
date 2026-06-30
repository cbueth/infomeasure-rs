// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Performance Benchmarks
//!
//! Interactive benchmark charts comparing Rust vs Python across all estimators and
//! approaches are available at:
//!
//! **<https://cbueth.codeberg.page/infomeasure-rs/>**
//!
//! The viewer includes scaling plots, per-approach parameter filters, log/linear
//! axes, hardware context, GPU toggle, version badges, and a sortable data table
//! with standard deviation per benchmark.
//!
//! ## Running Your Own Benchmarks
//!
//! Use the provided suite to reproduce or extend results on your hardware:
//!
//! ```bash
//! # Quick smoke test (faster but also takes time)
//! bash scripts/run_full_benchmark_suite.sh --quick
//!
//! # Full production run
//! bash scripts/run_full_benchmark_suite.sh
//!
//! # GPU-enabled run
//! INCLUDE_GPU=true bash scripts/run_full_benchmark_suite.sh
//!
//! # Regenerate viewer data from an existing run
//! bash scripts/run_full_benchmark_suite.sh --compare-only
//! ```
//!
//! If Rust has cashed benchmark results, it will not rerun them. To force re-running,
//! delete the `target/criterion` directory.
//!
//! > **Benchmark your own data.** Runtime depends on input characteristics: the `gpu`
//! > feature flag may speed up kernel estimators on large samples but add overhead for
//! > small ones; methods, bandwidth, and k-neighbors also affect throughput. Always
//! > profile on representative data with `--features gpu` both on and off.
//!
//! Running individual Rust benchmarks with [Criterion](https://github.com/bheisler/criterion.rs):
//!
//! ```bash
//! cargo bench --bench mi              # single binary
//! cargo bench                         # all benchmarks
//! cargo bench --features gpu          # GPU-accelerated variants
//! ```

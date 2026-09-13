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
//!
//! ## Dense Direct Paths and the Known-Alphabet Builder
//!
//! The discrete MLE estimators (`entropy`, `mutual_information`,
//! `transfer_entropy`) use an internal *dense direct* path whenever the joint
//! alphabet is small: each observation's mixed-radix code is built inline and
//! the joint/marginal counts are filled by direct indexing, then the measure is
//! summed over the cells. Above an internal cap of $2^{20}$ joint cells the
//! estimator falls back to the generic entropy-summation engine, so wide
//! alphabets and long histories do not regress.
//!
//! The public MLE constructors infer the alphabet with a min/max scan and retain
//! the inputs so that [local values](crate::estimators::traits::LocalValues) can
//! be requested. When the alphabet is known and only the average is needed, the
//! timing-optimised builders
//! [`DenseMiBuilder`](crate::estimators::approaches::discrete::DenseMiBuilder)
//! and
//! [`DenseCmiBuilder`](crate::estimators::approaches::discrete::DenseCmiBuilder)
//! (also used by the TE/CTE builders) borrow the raw code columns, skip the scan
//! with `with_alphabet`, and drop the retained inputs with `global_only`:
//!
//! ```rust
//! use infomeasure::estimators::entropy::GlobalValue;
//! use infomeasure::estimators::mutual_information::MutualInformation;
//!
//! let x = [0, 0, 1, 1, 0, 1, 0, 1];
//! let y = [0, 1, 0, 1, 1, 0, 1, 0];
//! // `with_alphabet` requires codes in `0..alphabet`.
//! let mi = MutualInformation::mi_discrete_mle(&[&x, &y])
//!     .with_alphabet(2)
//!     .global_only()
//!     .global_value();
//! assert!(mi >= 0.0);
//! ```
//!
//! The Criterion suite tracks both shapes per measure: `*_discrete/mle` uses the
//! inferring constructors, while `*_discrete/mle_alphabet` uses the builder
//! above.

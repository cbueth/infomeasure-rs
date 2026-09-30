// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Performance Benchmarks
//!
//! Interactive benchmark charts comparing Rust vs Python across all estimators and
//! approaches are available at:
//!
//! **<https://infomeasure.org/benchmarks/>**
//!
//! The viewer includes scaling plots, per-approach parameter filters, log/linear
//! axes, hardware context, a CPU/GPU toggle, version badges, and a sortable data
//! table with standard deviation per benchmark.
//!
//! The GPU toggle is a **sparse overlay**: the kernel estimators
//! (E/MI/CMI/TE/CTE) and the dense k-NN tier of the exponential-family
//! estimators can run on the GPU, and only above their size (and, for k-NN,
//! dimensionality) gates — so the toggle changes only the infomeasure-rs lines
//! where a GPU number exists and leaves every other line, and every smaller
//! size, on CPU. It therefore contrasts infomeasure-rs GPU against the CPU
//! baseline rather than a GPU-vs-GPU run — no other profiled toolkit offers a
//! comparable single-estimate GPU path (JIDT's CUDA path accelerates surrogate
//! significance testing, not one estimate).
//!
//! ## Headline Results
//!
//! Across the canonical cross-package run refreshed for each release
//! (single-threaded CPU track, one testbed), infomeasure-rs is roughly an
//! **order of magnitude faster than the Python reference overall**. The largest
//! margins are on the discrete and ordinal estimators (tens to over 100×),
//! followed by the k-NN / exponential-family family (~10×) and the kernel
//! estimators (a few ×). Enabling `gpu` adds further multiples on the kernel and
//! dense k-NN tiers above their gates, and `parallel` adds multi-core speed-ups
//! on the CPU fallback. Per-benchmark numbers, standard deviations and the
//! hardware context are on the page above and are regenerated every release.
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
//! `transfer_entropy`) use an internal *direct* path whenever the marginal
//! alphabet is small: each observation's mixed-radix code is built inline and
//! the joint/marginal counts are filled in a single pass, then the measure is
//! summed directly.
//!
//! The joint table is the product of the variable alphabets ($base^d$) and can
//! dominate the cost. It is stored in one of two ways, which **produce identical
//! values** and differ only in speed:
//!
//! - a **dense** `Vec` — direct indexing — while the joint is small relative to
//!   the number of samples (about 50 joint cells per observation, and never more
//!   than $2^{20}$ cells). This is the fastest path for the common small-alphabet
//!   case and the shapes covered by the Criterion suite.
//! - a **hash map** keyed by the packed code otherwise: it stores only occupied
//!   cells, so the cost grows with the number of observations $N$ rather than the
//!   alphabet. This is what lets wide alphabets and long histories scale, where a
//!   *dense* joint-count table over all $base^d$ combinations blows up in time
//!   and memory.
//!
//! The choice is a deterministic, performance-only heuristic — there is no
//! runtime timing or autotuning — so results are reproducible. The direct path
//! is abandoned for the generic entropy-summation engine only when the *marginal*
//! tables would be too large (memory bound for very wide alphabets).
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

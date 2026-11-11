// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Statistical Significance Testing
//!
//! Statistical tests answer: *"Is the observed value significantly different
//! from what chance would produce?"* This mirrors the Python `infomeasure`
//! package's `estimator.statistical_test()`.
//!
//! ## Permutation vs. bootstrap
//!
//! Both build a **null distribution** by resampling the source variable and
//! recomputing the measure `n_tests` times, but they break the dependence
//! differently:
//!
//! - **Permutation** ([`TestMethod::Permutation`], the default): shuffle the
//!   source. This destroys any dependence with the target while preserving each
//!   variable's marginal distribution, so it tests the null hypothesis of
//!   independence.
//! - **Bootstrap** ([`TestMethod::Bootstrap`]): resample the source with
//!   replacement. This breaks the source-target pairing, so it is a null-like
//!   resampling, not a joint bootstrap.
//!
//! [`TestMethod::Permutation`]: crate::estimators::stats::TestMethod::Permutation
//! [`TestMethod::Bootstrap`]: crate::estimators::stats::TestMethod::Bootstrap
//!
//! ## Surrogates for autocorrelated time series
//!
//! An iid permutation assumes exchangeable samples. For a time series that
//! assumption is false: shuffling a serially correlated source destroys its
//! autocorrelation as well as any coupling to the target, so the null
//! distribution is too narrow and the test rejects far more often than the
//! nominal level. Surrogate-data methods instead generate series that keep the
//! *linear serial structure* while breaking the coupling
//! [Theiler et al., 1992](super::references#theiler1992). Three opt-in
//! extensions are available:
//!
//! | Method | Builder | Null hypothesis | Keeps |
//! |--------|---------|-----------------|-------|
//! | [`BlockPermutation`] | [`StatisticalTestConfig::block_permutation`] | No source–target coupling beyond the serial structure inside a block | Within-block order and the exact marginal multiset |
//! | [`CircularShift`] | [`StatisticalTestConfig::circular_shift`] | No coupling; the series is stationary | Full circular autocorrelation |
//! | [`BlockBootstrap`] | [`StatisticalTestConfig::block_bootstrap`] | Source-resampling null, not a joint bootstrap | Contiguous-block structure and sample size |
//!
//! ### When to use which
//!
//! - **Block permutation** is the default choice for an autocorrelated but not
//!   strongly periodic source. It is the block analogue of the ordinary
//!   permutation test: the marginal distribution is preserved exactly, and
//!   only the ordering of blocks is randomised.
//! - **Circular shift** is appropriate for stationary series whose
//!   autocorrelation function is the structure to preserve. It is cheap (only
//!   $n$ distinct surrogates) and keeps the full circular autocorrelation
//!   exactly, but it also preserves any periodicity — including a period shared
//!   with the target, which can produce spurious significance. The identity
//!   rotation is a valid surrogate, so the observed series may appear in its own
//!   null distribution.
//! - **Block bootstrap** is a block-wise source-resampling null. It resamples
//!   blocks of the source with replacement while holding the target fixed, so
//!   it preserves short-range source structure but breaks the source-target
//!   pairing. It is not a joint bootstrap and should not be read as a
//!   confidence interval for the estimate.
//!
//! ### Choosing the block length
//!
//! The block length $b$ should be at least the decorrelation time of the
//! source. A practical recipe is to inspect the sample autocorrelation
//! $\hat\rho(k)$ and take the first lag where it falls below a small threshold
//! (for example $1/e$ or $0.05$), or use $b \approx n^{1/3}$ as a starting
//! point [Künsch, 1989](super::references#kunsch1989). Then check that
//! $n / b \gtrsim 10$: with only a handful of blocks, block permutation has few
//! distinct arrangements and the null distribution becomes coarse. Values
//! $b \ge n$ are clamped to one block, which makes block permutation the
//! identity and the test degenerate.
//!
//! ### Limitations
//!
//! - Block permutation breaks dependence on scales **longer** than $b$. If the
//!   process has long-range dependence, choose $b$ accordingly or use a
//!   long-range-preserving surrogate.
//! - Circular shift preserves periodicities exactly. A shared period in source
//!   and target is not evidence against independence under this null, but it
//!   will look significant.
//! - Block bootstrap is a source-resampling null, not a joint bootstrap. It
//!   does not give a confidence interval for the estimate, and a p-value
//!   computed from it is not a test of independence.
//! - All block methods assume approximate stationarity over the block length.
//!   Trends should be removed first.
//!
//! The construction follows the surrogate-data literature
//! [Theiler et al., 1992](super::references#theiler1992);
//! [Schreiber & Schmitz, 2000](super::references#schreiber_schmitz2000), with
//! blocks motivated by the moving-block bootstrap of
//! [Künsch, 1989](super::references#kunsch1989) and
//! [Politis & Romano, 1994](super::references#politis1994).
//!
//! [`BlockPermutation`]: crate::estimators::stats::TestMethod::BlockPermutation
//! [`CircularShift`]: crate::estimators::stats::TestMethod::CircularShift
//! [`BlockBootstrap`]: crate::estimators::stats::TestMethod::BlockBootstrap
//! [`StatisticalTestConfig::block_permutation`]: crate::estimators::stats::StatisticalTestConfig::block_permutation
//! [`StatisticalTestConfig::circular_shift`]: crate::estimators::stats::StatisticalTestConfig::circular_shift
//! [`StatisticalTestConfig::block_bootstrap`]: crate::estimators::stats::StatisticalTestConfig::block_bootstrap
//!
//! ## Result
//!
//! [`statistical_test`] returns a [`StatisticalTestResult`] with:
//!
//! - **`p_value`** — the proportion of null values strictly greater than the
//!   observed value (small ⇒ significant);
//! - **`t_score`** — the standardized effect size
//!   `(observed - null_mean) / null_std`;
//! - the full `test_values`, `null_mean`, `null_std`, and `observed_value`, with
//!   [`percentile`] and [`confidence_interval`] helpers.
//!
//! ## Effective transfer entropy
//!
//! For transfer-entropy measures, [`EffectiveValue::effective_value`] returns
//! the effective value of [Marschinski & Kantz, 2002](super::references#marschinski2002):
//! the observed value minus the value under a **single** resample of the source,
//! which removes the finite-sample bias that can make an uncoupled estimator
//! appear to transmit information. This single-draw definition matches the
//! Python package exactly: the number of resamples is not configurable, and the
//! call is `effective_value(method, seed)`, for example
//! `te.effective_value(TestMethod::Permutation, Some(42))`.
//!
//! A single surrogate draw is noisy, though. On high-dimensional or sparse data
//! the surrogate value itself has a large variance, so the reported effective
//! value can move substantially from draw to draw. The multi-surrogate
//! extension [`EffectiveValueStats::effective_value_with_resamples`] draws
//! `n_tests` surrogates, subtracts their **mean**, and returns an
//! [`EffectiveValueSummary`]:
//!
//! - `effective_value` = observed value − `null_mean`,
//! - `null_mean` — the mean surrogate value,
//! - `null_std` — its sample standard deviation (`ddof = 1`),
//! - `standard_error` — `null_std / sqrt(n_tests)`, the uncertainty of the
//!   bias correction (`NaN` for `n_tests < 2`),
//! - `n_tests`, `method`, and the individual `test_values`.
//!
//! Use the multi-surrogate form when the uncertainty of the correction matters;
//! keep `effective_value` for the Python-compatible single-draw number. With
//! `n_tests == 1` the summary reproduces `effective_value` under the same seed
//! exactly, and the spread fields are `NaN`.
//!
//! Surrogate-data resampling itself goes back to
//! [Theiler et al., 1992](super::references#theiler1992); here the surrogate is
//! the source series shuffled or bootstrapped in the same way as the null
//! distribution of [`statistical_test`].
//!
//! ## Usage
//!
//! ```
//! use ndarray::{array, Array1};
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::stats::{StatisticalTest, StatisticalTestConfig};
//!
//! let x: Array1<f64> = array![0., 1., 2., 3., 4., 5., 6., 7.];
//! let y: Array1<f64> = array![0., 1., 2., 3., 4., 5., 6., 7.];
//! let mi = MutualInformation::new_kernel(&[x, y], 2.0);
//!
//! let res = mi
//!     .statistical_test(&StatisticalTestConfig::permutation(200).seed(1))
//!     .unwrap();
//! println!("I = {}, p = {:.4}, t = {:.2}", res.observed_value, res.p_value, res.t_score);
//! let (lo, hi) = res.confidence_interval(95.0).unwrap();
//! println!("95% CI = {lo:.3}..{hi:.3}");
//! ```
//!
//! ```
//! use ndarray::{array, Array1};
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::stats::{EffectiveValue, TestMethod};
//!
//! let src: Array1<f64> = array![0., 1., 0., 1., 0., 1., 0., 1., 0., 1.];
//! let dst: Array1<f64> = array![0., 0., 1., 0., 1., 0., 1., 0., 1., 0.];
//! let te = TransferEntropy::new_kernel(&src, &dst, 1, 1, 1, 0.5);
//!
//! let ete = te.effective_value(TestMethod::Permutation, Some(1));
//! println!("effective TE = {ete}");
//! ```
//!
//! The same estimator with the multi-surrogate summary:
//!
//! ```
//! use ndarray::{array, Array1};
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::stats::{EffectiveValueStats, TestMethod};
//!
//! let source: Array1<i32> = array![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
//! let dest: Array1<i32> = array![0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
//! let te = TransferEntropy::new_discrete_mle(&source, &dest, 1, 1, 1);
//!
//! let summary = te
//!     .effective_value_with_resamples(TestMethod::Permutation, Some(42), 100)
//!     .unwrap();
//! println!(
//!     "eTE = {:.3} ± {:.3} (n = {})",
//!     summary.effective_value, summary.standard_error, summary.n_tests
//! );
//! assert!(summary.effective_value.is_finite());
//! assert!(summary.standard_error.is_finite());
//! ```
//!
//! For autocorrelated data, pick a time-series-aware surrogate instead of the
//! iid permutation. The example below uses short blocks purely to keep the
//! doctest fast; real series need a block length matched to their
//! decorrelation time.
//!
//! ```
//! use ndarray::{array, Array1};
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::stats::{StatisticalTest, StatisticalTestConfig};
//!
//! let x: Array1<f64> = array![0., 1., 2., 3., 4., 5., 6., 7.];
//! let y: Array1<f64> = array![0., 1., 2., 3., 4., 5., 6., 7.];
//! let mi = MutualInformation::new_kernel(&[x, y], 2.0);
//!
//! let res = mi
//!     .statistical_test(&StatisticalTestConfig::block_permutation(20, 2).seed(1))
//!     .unwrap();
//! println!("I = {}, p = {:.4}", res.observed_value, res.p_value);
//! ```
//!
//! ## Which measures?
//!
//! The test is defined wherever permuting the source is meaningful, that is the
//! mutual-information and transfer-entropy measures, and it is implemented for
//! every estimator family:
//!
//! | Approach | MI | CMI | TE | CTE |
//! |----------|----|-----|----|-----|
//! | Discrete | yes | yes | yes | yes |
//! | Kernel | yes | yes | yes | yes |
//! | Ordinal | yes | yes | yes | yes |
//! | kNN: KSG | yes | yes | yes | yes |
//! | kNN: Rényi | yes | yes | yes | yes |
//! | kNN: Tsallis | yes | yes | yes | yes |
//! | kNN: Kozachenko-Leonenko | yes | yes | yes | yes |
//!
//! Calling it on a single-variable entropy estimator is a compile error, rather
//! than the runtime error the Python package raises. The value-only
//! `DenseMiGlobal` and `DenseCmiGlobal` results returned by the `.global_only()`
//! dense builders are the exception: they deliberately retain no inputs, so use
//! `.build()` (or the `new_discrete_*` constructors) when you need the test.
//!
//! ## Reproducibility and parallelism
//!
//! Each draw is seeded deterministically from the configured `seed` (or from
//! entropy when none is given) together with the draw index, so a seeded test is
//! reproducible and independent of execution order. With the `parallel` feature
//! the draws run on a thread pool and produce exactly the same result as the
//! sequential build.
//!
//! The multi-surrogate effective value uses the same scheme, with one detail for
//! parity: draw zero uses the raw seed, so a one-draw summary is bit-for-bit the
//! single-draw `effective_value`; later draws use the mixed per-draw seeds. The
//! draws are prefix-stable, so raising `n_tests` never changes the earlier ones
//! and the parallel build still matches the serial one exactly.
//!
//! ## Cached marginal terms
//!
//! Permutation, block permutation, and circular shift only reorder the source,
//! so per-variable marginal terms such as $H(Y)$ are unchanged and can be
//! computed once and reused across draws. Bootstrap and block bootstrap resample
//! the source **with replacement**, which changes the empirical marginal; cached
//! marginal terms would then be stale. Those methods therefore rebuild the
//! estimator from the resampled data on every draw. The distinction matters for
//! correctness: reorder-invariant caches are not bootstrap-invariant.
//!
//! [`statistical_test`]: crate::estimators::stats::StatisticalTest::statistical_test
//! [`StatisticalTestResult`]: crate::estimators::stats::StatisticalTestResult
//! [`percentile`]: crate::estimators::stats::StatisticalTestResult::percentile
//! [`confidence_interval`]: crate::estimators::stats::StatisticalTestResult::confidence_interval
//! [`EffectiveValue::effective_value`]: crate::estimators::stats::EffectiveValue::effective_value
//! [`EffectiveValueStats::effective_value_with_resamples`]: crate::estimators::stats::EffectiveValueStats::effective_value_with_resamples
//! [`EffectiveValueSummary`]: crate::estimators::stats::EffectiveValueSummary
//!
//! ## References
//!
//! - [Schreiber, 2000](super::references#schreiber2000)
//! - [Marschinski & Kantz, 2002](super::references#marschinski2002)
//! - [Theiler et al., 1992](super::references#theiler1992)
//! - [Lizier, 2014](super::references#lizier2014)

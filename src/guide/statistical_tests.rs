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
//!   replacement, giving the sampling variability of the estimate.
//!
//! [`TestMethod::Permutation`]: crate::estimators::stats::TestMethod::Permutation
//! [`TestMethod::Bootstrap`]: crate::estimators::stats::TestMethod::Bootstrap
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
//! the observed value minus the value under a single permutation of the source
//! — the *effective transfer entropy*. This removes the finite-sample bias that
//! can make an uncoupled estimator appear to transmit information.
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
//! let (lo, hi) = res.confidence_interval(95.0);
//! println!("95% CI = {lo:.3}..{hi:.3}");
//! ```
//!
//! ```
//! use ndarray::{array, Array1};
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::stats::{EffectiveValue, StatisticalTestConfig};
//!
//! let src: Array1<f64> = array![0., 1., 0., 1., 0., 1., 0., 1., 0., 1.];
//! let dst: Array1<f64> = array![0., 0., 1., 0., 1., 0., 1., 0., 1., 0.];
//! let te = TransferEntropy::new_kernel(&src, &dst, 1, 1, 1, 0.5);
//!
//! let ete = te
//!     .effective_value(&StatisticalTestConfig::permutation(1).seed(1))
//!     .unwrap();
//! println!("effective TE = {ete}");
//! ```
//!
//! ## Which measures?
//!
//! The test is only defined where permuting the source is meaningful, i.e. the
//! mutual-information and transfer-entropy measures. It is a compile error to
//! call it on a single-variable entropy estimator, rather than a runtime panic
//! as in the Python package.
//!
//! [`statistical_test`]: crate::estimators::stats::StatisticalTest::statistical_test
//! [`StatisticalTestResult`]: crate::estimators::stats::StatisticalTestResult
//! [`percentile`]: crate::estimators::stats::StatisticalTestResult::percentile
//! [`confidence_interval`]: crate::estimators::stats::StatisticalTestResult::confidence_interval
//! [`EffectiveValue::effective_value`]: crate::estimators::stats::EffectiveValue::effective_value
//!
//! ## References
//!
//! - [Schreiber, 2000](super::references#schreiber2000)
//! - [Lizier, 2014](super::references#lizier2014)

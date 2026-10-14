// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Statistical Significance Tests
//!
//! Non-parametric significance testing for the information-theoretic measures,
//! matching the Python `infomeasure` package's `statistical_test` API.
//!
//! Given an estimator, [`StatisticalTest::statistical_test`] builds a null
//! distribution by resampling the **source** variable and recomputing the
//! measure, either by:
//!
//! - **Permutation** ([`TestMethod::Permutation`]): shuffle the source, which
//!   breaks any dependence with the target. The null hypothesis is that the
//!   observed measure is indistinguishable from chance.
//! - **Bootstrap** ([`TestMethod::Bootstrap`]): resample the source with
//!   replacement.
//!
//! The result carries the **p-value** (the fraction of null values strictly
//! greater than the observed value), the standardized **t-score**, the null
//! distribution, and helpers for percentiles and confidence intervals
//! ([`StatisticalTestResult`]).
//!
//! The test is only defined where permuting the source is meaningful, i.e. the
//! mutual-information and transfer-entropy measures; the trait is not
//! implemented for single-variable entropy estimators, so misuse is a compile
//! error rather than a runtime one.
//!
//! ```
//! use ndarray::{array, Array1};
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::stats::{StatisticalTest, StatisticalTestConfig};
//!
//! let x: Array1<f64> = array![0., 1., 2., 3., 4., 5., 6., 7.];
//! let y: Array1<f64> = array![0., 1., 2., 3., 4., 5., 6., 7.];
//! let mi = MutualInformation::new_kernel(&[x, y], 0.5);
//!
//! let res = mi
//!     .statistical_test(&StatisticalTestConfig::permutation(20).seed(1))
//!     .unwrap();
//! println!("I = {}, p = {}, t = {}", res.observed_value, res.p_value, res.t_score);
//! ```

mod config;
mod driver;
mod impls;
mod result;
mod traits;

pub use config::{StatisticalTestConfig, TestMethod};
pub use driver::{StatsError, run_resample_test};
pub use result::StatisticalTestResult;
pub use traits::{EffectiveValue, StatisticalTest};

// Internal glue reused by the approach modules' `Resample` impls.
pub(crate) use driver::{resample_1d_i32, resample_rows};
pub(crate) use traits::{EffectiveValueMeasure, PreparedDraw, Resample};

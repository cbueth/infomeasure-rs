// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Traits for statistical significance testing.

use crate::estimators::traits::GlobalValue;
use rand::rngs::SmallRng;

use super::config::{StatisticalTestConfig, TestMethod};
use super::driver::{StatsError, make_rng, run_resample_test};
use super::result::StatisticalTestResult;

/// Perform a permutation or bootstrap significance test on a measure.
///
/// Implemented only for the mutual-information and transfer-entropy measures
/// (which have a well-defined "shuffle the source" null hypothesis); calling it
/// on a single-variable entropy estimator is a compile error.
pub trait StatisticalTest {
    /// Run the test described by `cfg` against this estimator's observed value.
    fn statistical_test(
        &self,
        cfg: &StatisticalTestConfig,
    ) -> Result<StatisticalTestResult, StatsError>;
}

/// Effective value of a measure (e.g. effective transfer entropy).
///
/// The observed value minus the value under a single permutation of the source.
pub trait EffectiveValue {
    /// Compute the effective value with the method in `cfg`.
    fn effective_value(&self, cfg: &StatisticalTestConfig) -> Result<f64, StatsError>;
}

/// Internal: an estimator that can rebuild itself with its source resampled.
///
/// This is the only per-family glue a new measure needs to add to gain
/// [`StatisticalTest`]; the resampling and statistics live in the driver.
pub(crate) trait Resample: GlobalValue {
    /// Rebuild this estimator with the source variable resampled according to
    /// `method`.
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self;
}

impl<T: Resample> StatisticalTest for T {
    fn statistical_test(
        &self,
        cfg: &StatisticalTestConfig,
    ) -> Result<StatisticalTestResult, StatsError> {
        let observed = self.global_value();
        run_resample_test(observed, cfg, |rng| {
            self.resampled(cfg.method, rng).global_value()
        })
    }
}

/// Internal marker for measures where an effective value is meaningful
/// (transfer entropy and its conditional variant).
pub(crate) trait EffectiveValueMeasure: Resample {}

impl<T: EffectiveValueMeasure> EffectiveValue for T {
    fn effective_value(&self, cfg: &StatisticalTestConfig) -> Result<f64, StatsError> {
        let observed = self.global_value();
        let mut rng = make_rng(cfg.seed);
        let permuted = self.resampled(cfg.method, &mut rng).global_value();
        Ok(observed - permuted)
    }
}

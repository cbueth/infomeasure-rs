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
/// The observed value minus the value under a **single** resample of the source
/// — matching the Python package's `effective_val`, which uses exactly one draw.
/// Because one draw is the definition, the number of resamples is deliberately
/// not configurable here (a mean over many draws would be a different quantity).
pub trait EffectiveValue {
    /// Compute the effective value with the given resampling `method`, seeded
    /// with `seed` (`None` for a non-deterministic draw).
    fn effective_value(&self, method: TestMethod, seed: Option<u64>) -> f64;
}

/// A prepared, order-independent draw function returned by
/// [`Resample::prepared`]: it takes the draw index and an RNG and returns the
/// resampled measure value.
pub(crate) type PreparedDraw<'a> = Box<dyn Fn(usize, &mut SmallRng) -> f64 + Sync + 'a>;

/// Internal: an estimator that can rebuild itself with its source resampled.
///
/// This is the only per-family glue a new measure needs to add to gain
/// [`StatisticalTest`]; the resampling and statistics live in the driver.
pub(crate) trait Resample: GlobalValue {
    /// Rebuild this estimator with the source variable resampled according to
    /// `method`.
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self;

    /// Optional fast path: a draw function that caches the work invariant across
    /// resamples, for example the permutation-invariant marginal terms, and only
    /// recomputes what changes per draw. Returning `None` falls back to
    /// [`Resample::resampled`], which rebuilds the whole estimator each draw.
    ///
    /// Kernel estimators cache their invariant marginal densities. kNN
    /// estimators leave this as `None`: their marginal neighbour counts depend
    /// on the joint k-th neighbour radius `epsilon`, which changes on every
    /// permutation, so only the untouched marginals' sort order would be
    /// reusable, which is not worth the duplicated per-family formula.
    fn prepared(&self, _method: TestMethod) -> Option<PreparedDraw<'_>> {
        None
    }
}

impl<T: Resample + Sync> StatisticalTest for T {
    fn statistical_test(
        &self,
        cfg: &StatisticalTestConfig,
    ) -> Result<StatisticalTestResult, StatsError> {
        let observed = self.global_value();
        match self.prepared(cfg.method) {
            Some(draw) => run_resample_test(observed, cfg, draw),
            None => run_resample_test(observed, cfg, |_, rng| {
                self.resampled(cfg.method, rng).global_value()
            }),
        }
    }
}

/// Internal marker for measures where an effective value is meaningful
/// (transfer entropy and its conditional variant).
pub(crate) trait EffectiveValueMeasure: Resample {}

impl<T: EffectiveValueMeasure> EffectiveValue for T {
    fn effective_value(&self, method: TestMethod, seed: Option<u64>) -> f64 {
        let observed = self.global_value();
        let mut rng = make_rng(seed);
        observed - self.resampled(method, &mut rng).global_value()
    }
}

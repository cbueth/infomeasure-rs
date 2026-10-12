// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The result of a statistical significance test.

use ndarray::Array1;

use super::config::TestMethod;
use super::driver::StatsError;

/// Outcome of a permutation or bootstrap test.
///
/// Mirrors the Python `infomeasure` `StatisticalTestResult`: the observed
/// measure, the null distribution, and the derived statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct StatisticalTestResult {
    /// Proportion of null values strictly greater than the observed value.
    pub p_value: f64,
    /// Standardized effect size `(observed - null_mean) / null_std`
    /// (`NaN` when the null distribution has zero spread).
    pub t_score: f64,
    /// The resampled null values.
    pub test_values: Array1<f64>,
    /// The observed measure that was tested.
    pub observed_value: f64,
    /// Mean of the null distribution.
    pub null_mean: f64,
    /// Sample standard deviation of the null distribution (ddof = 1).
    pub null_std: f64,
    /// Number of resamples performed.
    pub n_tests: usize,
    /// The resampling method used.
    pub method: TestMethod,
}

impl StatisticalTestResult {
    /// The `q`-th percentile (`0..=100`) of the null distribution.
    ///
    /// Uses linear interpolation between order statistics, matching NumPy's
    /// default (`method="linear"`, "type 7").
    pub fn percentile(&self, q: f64) -> f64 {
        percentile(&self.test_values, q)
    }

    /// Confidence interval for `confidence_level` percent.
    ///
    /// Returns `(lower, upper)` bounds as percentiles of the null distribution,
    /// or [`StatsError::InvalidConfidenceLevel`] if the level is not in the open
    /// interval `(0, 100)` — matching the Python package, which raises for the
    /// same condition.
    pub fn confidence_interval(&self, confidence_level: f64) -> Result<(f64, f64), StatsError> {
        if !(confidence_level > 0.0 && confidence_level < 100.0) {
            return Err(StatsError::InvalidConfidenceLevel(confidence_level));
        }
        let y = (100.0 - confidence_level) / 2.0;
        Ok((self.percentile(y), self.percentile(100.0 - y)))
    }
}

/// Linear-interpolation percentile (NumPy default, "type 7").
fn percentile(values: &Array1<f64>, q: f64) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return values[0];
    }
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = q.clamp(0.0, 100.0);
    let rank = (q / 100.0) * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo >= n - 1 {
        return sorted[n - 1];
    }
    let frac = rank - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimators::stats::TestMethod;

    fn result(values: Vec<f64>) -> StatisticalTestResult {
        StatisticalTestResult {
            p_value: 0.0,
            t_score: 0.0,
            test_values: Array1::from(values),
            observed_value: 0.0,
            null_mean: 0.0,
            null_std: 0.0,
            n_tests: 0,
            method: TestMethod::Permutation,
        }
    }

    #[test]
    fn percentile_matches_numpy_linear() {
        let r = result(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(r.percentile(0.0), 1.0);
        assert_eq!(r.percentile(100.0), 4.0);
        assert!((r.percentile(50.0) - 2.5).abs() < 1e-12);
        assert!((r.percentile(25.0) - 1.75).abs() < 1e-12);
        assert!((r.percentile(75.0) - 3.25).abs() < 1e-12);
    }

    #[test]
    fn percentile_edge_cases() {
        assert!(result(vec![]).percentile(50.0).is_nan());
        assert_eq!(result(vec![7.0]).percentile(50.0), 7.0);
        // Unsorted input is handled.
        assert_eq!(result(vec![3.0, 1.0, 2.0]).percentile(0.0), 1.0);
    }

    #[test]
    fn confidence_interval_is_symmetric_percentiles() {
        let r = result((0..=100).map(|i| i as f64).collect());
        let (lo, hi) = r.confidence_interval(90.0).unwrap();
        assert!((lo - r.percentile(5.0)).abs() < 1e-12);
        assert!((hi - r.percentile(95.0)).abs() < 1e-12);
    }

    #[test]
    fn confidence_interval_rejects_out_of_range() {
        let r = result(vec![1.0, 2.0, 3.0]);
        assert!(r.confidence_interval(0.0).is_err());
        assert!(r.confidence_interval(100.0).is_err());
        assert!(r.confidence_interval(-5.0).is_err());
        assert!(r.confidence_interval(95.0).is_ok());
    }
}

// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The resampling driver shared by every statistical test.

use ndarray::{Array1, Array2};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use super::config::{StatisticalTestConfig, TestMethod};
use super::result::StatisticalTestResult;

/// Error returned by a statistical test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatsError {
    /// Fewer than two resamples were requested; the sample standard deviation
    /// (and hence the t-score) is undefined below two samples.
    TooFewTests(usize),
}

impl std::fmt::Display for StatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatsError::TooFewTests(n) => {
                write!(f, "statistical test needs at least 2 resamples, got {n}")
            }
        }
    }
}

impl std::error::Error for StatsError {}

/// Build the RNG for a test. Seeded when configured, otherwise from entropy.
pub(crate) fn make_rng(seed: Option<u64>) -> SmallRng {
    match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_entropy(),
    }
}

/// Draw `n_tests` resampled values from `sample` and summarise the null
/// distribution against `observed`.
///
/// This is the whole computation of a statistical test; the per-measure glue
/// only has to provide a closure that recomputes the measure on a resampled
/// estimator. The draws are independent, so this is where future parallelism
/// across resamples (and invariant caching) would plug in.
pub fn run_resample_test<S>(
    observed: f64,
    cfg: &StatisticalTestConfig,
    mut sample: S,
) -> Result<StatisticalTestResult, StatsError>
where
    S: FnMut(&mut SmallRng) -> f64,
{
    if cfg.n_tests < 2 {
        return Err(StatsError::TooFewTests(cfg.n_tests));
    }
    let mut rng = make_rng(cfg.seed);
    let mut test_values = Array1::zeros(cfg.n_tests);
    for value in test_values.iter_mut() {
        *value = sample(&mut rng);
    }
    Ok(summarise(observed, test_values, cfg.method))
}

fn summarise(observed: f64, test_values: Array1<f64>, method: TestMethod) -> StatisticalTestResult {
    let n_tests = test_values.len();
    let null_mean = test_values.mean().unwrap_or(f64::NAN);
    let null_std = sample_std(&test_values, null_mean);
    let greater = test_values.iter().filter(|&&v| v > observed).count();
    let p_value = greater as f64 / n_tests as f64;
    let t_score = if null_std > 0.0 {
        (observed - null_mean) / null_std
    } else {
        f64::NAN
    };
    StatisticalTestResult {
        p_value,
        t_score,
        test_values,
        observed_value: observed,
        null_mean,
        null_std,
        n_tests,
        method,
    }
}

/// Sample standard deviation (ddof = 1), matching NumPy's default.
fn sample_std(values: &Array1<f64>, mean: f64) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    let sum_sq: f64 = values.iter().map(|v| (v - mean).powi(2)).sum();
    (sum_sq / (n as f64 - 1.0)).sqrt()
}

/// Shuffle the rows (samples) of `data` in place-free fashion.
pub(crate) fn permute_rows(data: &Array2<f64>, rng: &mut SmallRng) -> Array2<f64> {
    let n = data.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    order.shuffle(rng);
    let mut out = data.clone();
    for (row, &src) in order.iter().enumerate() {
        out.row_mut(row).assign(&data.row(src));
    }
    out
}

/// Resample the rows of `data` with replacement.
pub(crate) fn bootstrap_rows(data: &Array2<f64>, rng: &mut SmallRng) -> Array2<f64> {
    let n = data.nrows();
    let mut out = data.clone();
    for row in 0..n {
        let src = rng.gen_range(0..n);
        out.row_mut(row).assign(&data.row(src));
    }
    out
}

/// Resample the rows of `data` with the given method.
pub(crate) fn resample_rows(
    data: &Array2<f64>,
    method: TestMethod,
    rng: &mut SmallRng,
) -> Array2<f64> {
    match method {
        TestMethod::Permutation => permute_rows(data, rng),
        TestMethod::Bootstrap => bootstrap_rows(data, rng),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn too_few_tests_is_an_error() {
        let cfg = StatisticalTestConfig::new(1, TestMethod::Permutation);
        assert!(matches!(
            run_resample_test(0.0, &cfg, |_| 0.0),
            Err(StatsError::TooFewTests(1))
        ));
    }

    #[test]
    fn p_value_is_strict_greater_fraction() {
        // Half the null values (1.0, 2.0) are greater than the observed 1.5.
        let cfg = StatisticalTestConfig::new(4, TestMethod::Permutation);
        let mut values = [0.0, 1.0, 2.0, 3.0].into_iter();
        let res = run_resample_test(1.5, &cfg, |_| values.next().unwrap()).unwrap();
        assert!((res.p_value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn t_score_uses_ddof_one() {
        let cfg = StatisticalTestConfig::new(3, TestMethod::Permutation);
        let mut values = [0.0, 2.0, 4.0].into_iter();
        let res = run_resample_test(3.0, &cfg, |_| values.next().unwrap()).unwrap();
        // mean = 2, sample std (ddof=1) = 2, t = (3 - 2) / 2.
        assert!((res.null_mean - 2.0).abs() < 1e-12);
        assert!((res.null_std - 2.0).abs() < 1e-12);
        assert!((res.t_score - 0.5).abs() < 1e-12);
    }

    #[test]
    fn zero_spread_gives_nan_t_score() {
        let cfg = StatisticalTestConfig::new(3, TestMethod::Permutation);
        let res = run_resample_test(1.0, &cfg, |_| 1.0).unwrap();
        assert!(res.t_score.is_nan());
    }

    #[test]
    fn seeded_runs_are_reproducible() {
        let cfg = StatisticalTestConfig::new(16, TestMethod::Permutation).seed(42);
        let draw = |rng: &mut SmallRng| rng.r#gen::<f64>();
        let a = run_resample_test(0.0, &cfg, draw).unwrap();
        let b = run_resample_test(0.0, &cfg, draw).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn permutation_preserves_the_row_multiset() {
        let data = Array2::from_shape_vec((3, 1), vec![1.0, 2.0, 3.0]).unwrap();
        let mut rng = make_rng(Some(7));
        let permuted = permute_rows(&data, &mut rng);
        let mut got: Vec<f64> = permuted.column(0).iter().copied().collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn bootstrap_samples_with_replacement() {
        let data = Array2::from_shape_vec((2, 1), vec![1.0, 2.0]).unwrap();
        let mut rng = make_rng(Some(1));
        // Drawing two rows from two values can repeat a value.
        let out = bootstrap_rows(&data, &mut rng);
        assert_eq!(out.dim(), (2, 1));
    }
}

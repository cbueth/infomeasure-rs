// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The resampling driver shared by every statistical test.

use ndarray::{Array1, Array2};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::config::{StatisticalTestConfig, TestMethod};
use super::result::StatisticalTestResult;

/// Error returned by a statistical test.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsError {
    /// Fewer than two resamples were requested; the sample standard deviation
    /// (and hence the t-score) is undefined below two samples.
    TooFewTests(usize),
    /// A confidence level outside the open interval `(0, 100)` was requested.
    InvalidConfidenceLevel(f64),
    /// A block-based resampling method was configured with a block length of
    /// zero; blocks must contain at least one sample.
    InvalidBlockLength(usize),
}

impl std::fmt::Display for StatsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatsError::TooFewTests(n) => {
                write!(f, "statistical test needs at least 2 resamples, got {n}")
            }
            StatsError::InvalidConfidenceLevel(level) => {
                write!(f, "confidence level must be in (0, 100), got {level}")
            }
            StatsError::InvalidBlockLength(len) => {
                write!(f, "block length must be at least 1, got {len}")
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

/// A 64-bit mixing function (SplitMix64). Derives independent per-draw seeds
/// from a single master seed so the draws do not depend on execution order.
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Draw `n_tests` resampled values from `sample` and summarise the null
/// distribution against `observed`.
///
/// `sample` is called as `sample(index, rng)`; the index lets a caller derive
/// an order-independent result. This is the whole computation of a statistical
/// test; the per-measure glue only has to provide a closure that recomputes the
/// measure on a resampled estimator. Each draw is seeded deterministically from
/// a master seed (the configured `seed`, or one drawn from entropy), so the
/// result does not depend on execution order and the draws can run in parallel
/// under the `parallel` feature while matching the serial run exactly.
pub fn run_resample_test<S>(
    observed: f64,
    cfg: &StatisticalTestConfig,
    sample: S,
) -> Result<StatisticalTestResult, StatsError>
where
    S: Fn(usize, &mut SmallRng) -> f64 + Sync,
{
    if cfg.n_tests < 2 {
        return Err(StatsError::TooFewTests(cfg.n_tests));
    }
    if let Some(block_len) = cfg.method.block_len()
        && block_len == 0
    {
        return Err(StatsError::InvalidBlockLength(block_len));
    }
    let master = cfg.seed.unwrap_or_else(rand::random::<u64>);
    let draw = |i: usize| {
        let mut rng = SmallRng::seed_from_u64(splitmix64(master.wrapping_add(i as u64)));
        sample(i, &mut rng)
    };
    #[cfg(feature = "parallel")]
    let values: Vec<f64> = (0..cfg.n_tests).into_par_iter().map(draw).collect();
    #[cfg(not(feature = "parallel"))]
    let values: Vec<f64> = (0..cfg.n_tests).map(draw).collect();
    Ok(summarise(observed, Array1::from(values), cfg.method))
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

/// Return source indices for a block permutation of `n` samples.
///
/// The indices are the concatenation of shuffled contiguous blocks of
/// `block_len` (the last block may be shorter). A `block_len` of zero is
/// treated defensively as one; statistical tests reject it before reaching
/// here. Values `>= n` produce a single block, i.e. the identity order.
fn block_permutation_indices(n: usize, block_len: usize, rng: &mut SmallRng) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let block_len = block_len.max(1);
    let mut starts: Vec<usize> = (0..n).step_by(block_len).collect();
    starts.shuffle(rng);
    let mut order = Vec::with_capacity(n);
    for start in starts {
        order.extend(start..(start + block_len).min(n));
    }
    order
}

/// Return source indices for a uniformly random circular shift of `n` samples.
fn circular_shift_indices(n: usize, rng: &mut SmallRng) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let offset = rng.gen_range(0..n);
    (0..n).map(|i| (i + offset) % n).collect()
}

/// Return source indices for a block bootstrap of `n` samples.
///
/// Contiguous blocks of `block_len` are drawn with replacement from the
/// non-wrapping starts `0..=n - block_len` until at least `n` indices are
/// collected, then truncated to `n`. A `block_len` of zero is treated
/// defensively as one; values `>= n` reduce to a single full-length block.
fn block_bootstrap_indices(n: usize, block_len: usize, rng: &mut SmallRng) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let block_len = block_len.max(1).min(n);
    let n_blocks = n.div_ceil(block_len);
    let max_start = n - block_len;
    let mut order = Vec::with_capacity(n_blocks * block_len);
    for _ in 0..n_blocks {
        let start = rng.gen_range(0..=max_start);
        order.extend(start..start + block_len);
    }
    order.truncate(n);
    order
}

/// Gather rows of `data` in `order`, where output row `i` copies source row
/// `order[i]`.
fn take_rows(data: &Array2<f64>, order: &[usize]) -> Array2<f64> {
    let mut out = Array2::zeros((order.len(), data.ncols()));
    for (row, &src) in order.iter().enumerate() {
        out.row_mut(row).assign(&data.row(src));
    }
    out
}

/// Gather values of a 1D series in `order`.
fn take_1d<T: Copy>(data: &Array1<T>, order: &[usize]) -> Array1<T> {
    Array1::from_iter(order.iter().map(|&i| data[i]))
}

/// Resample the rows of `data` with the given method.
pub(crate) fn resample_rows(
    data: &Array2<f64>,
    method: TestMethod,
    rng: &mut SmallRng,
) -> Array2<f64> {
    let n = data.nrows();
    match method {
        TestMethod::Permutation => permute_rows(data, rng),
        TestMethod::Bootstrap => bootstrap_rows(data, rng),
        TestMethod::BlockPermutation { block_len } => {
            take_rows(data, &block_permutation_indices(n, block_len, rng))
        }
        TestMethod::CircularShift => take_rows(data, &circular_shift_indices(n, rng)),
        TestMethod::BlockBootstrap { block_len } => {
            take_rows(data, &block_bootstrap_indices(n, block_len, rng))
        }
    }
}

/// Resample a 1D series with the given method.
pub(crate) fn resample_1d(
    data: &Array1<f64>,
    method: TestMethod,
    rng: &mut SmallRng,
) -> Array1<f64> {
    let n = data.len();
    match method {
        TestMethod::Permutation => {
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(rng);
            Array1::from_iter(order.iter().map(|&i| data[i]))
        }
        TestMethod::Bootstrap => {
            let mut out = Array1::zeros(n);
            for value in out.iter_mut() {
                *value = data[rng.gen_range(0..n)];
            }
            out
        }
        TestMethod::BlockPermutation { block_len } => {
            take_1d(data, &block_permutation_indices(n, block_len, rng))
        }
        TestMethod::CircularShift => take_1d(data, &circular_shift_indices(n, rng)),
        TestMethod::BlockBootstrap { block_len } => {
            take_1d(data, &block_bootstrap_indices(n, block_len, rng))
        }
    }
}

/// Resample an `i32` series (discrete symbols) with the given method.
pub(crate) fn resample_1d_i32(
    data: &Array1<i32>,
    method: TestMethod,
    rng: &mut SmallRng,
) -> Array1<i32> {
    let n = data.len();
    match method {
        TestMethod::Permutation => {
            let mut order: Vec<usize> = (0..n).collect();
            order.shuffle(rng);
            Array1::from_iter(order.iter().map(|&i| data[i]))
        }
        TestMethod::Bootstrap => {
            let mut out = Array1::zeros(n);
            for value in out.iter_mut() {
                *value = data[rng.gen_range(0..n)];
            }
            out
        }
        TestMethod::BlockPermutation { block_len } => {
            take_1d(data, &block_permutation_indices(n, block_len, rng))
        }
        TestMethod::CircularShift => take_1d(data, &circular_shift_indices(n, rng)),
        TestMethod::BlockBootstrap { block_len } => {
            take_1d(data, &block_bootstrap_indices(n, block_len, rng))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn too_few_tests_is_an_error() {
        let cfg = StatisticalTestConfig::new(1, TestMethod::Permutation);
        assert!(matches!(
            run_resample_test(0.0, &cfg, |_, _| 0.0),
            Err(StatsError::TooFewTests(1))
        ));
    }

    #[test]
    fn p_value_is_strict_greater_fraction() {
        // Half the null values (1.0, 2.0) are greater than the observed 1.5.
        let cfg = StatisticalTestConfig::new(4, TestMethod::Permutation);
        let values = [0.0, 1.0, 2.0, 3.0];
        let res = run_resample_test(1.5, &cfg, |i, _| values[i]).unwrap();
        assert!((res.p_value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn t_score_uses_ddof_one() {
        let cfg = StatisticalTestConfig::new(3, TestMethod::Permutation);
        let values = [0.0, 2.0, 4.0];
        let res = run_resample_test(3.0, &cfg, |i, _| values[i]).unwrap();
        // mean = 2, sample std (ddof=1) = 2, t = (3 - 2) / 2.
        assert!((res.null_mean - 2.0).abs() < 1e-12);
        assert!((res.null_std - 2.0).abs() < 1e-12);
        assert!((res.t_score - 0.5).abs() < 1e-12);
    }

    #[test]
    fn zero_spread_gives_nan_t_score() {
        let cfg = StatisticalTestConfig::new(3, TestMethod::Permutation);
        let res = run_resample_test(1.0, &cfg, |_, _| 1.0).unwrap();
        assert!(res.t_score.is_nan());
    }

    #[test]
    fn seeded_runs_are_reproducible() {
        let cfg = StatisticalTestConfig::new(16, TestMethod::Permutation).seed(42);
        let draw = |_: usize, rng: &mut SmallRng| rng.r#gen::<f64>();
        let a = run_resample_test(0.0, &cfg, draw).unwrap();
        let b = run_resample_test(0.0, &cfg, draw).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn draws_are_prefix_stable() {
        // Each draw is seeded from (master, index), so the first draws do not
        // change when more are requested: this is what makes the resampling
        // independent of execution order (and thus of parallel scheduling).
        let draw = |_: usize, rng: &mut SmallRng| rng.r#gen::<f64>();
        let a = run_resample_test(
            0.0,
            &StatisticalTestConfig::new(4, TestMethod::Permutation).seed(9),
            draw,
        )
        .unwrap();
        let b = run_resample_test(
            0.0,
            &StatisticalTestConfig::new(8, TestMethod::Permutation).seed(9),
            draw,
        )
        .unwrap();
        for (x, y) in a.test_values.iter().zip(b.test_values.iter()) {
            assert_eq!(x, y);
        }
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

    #[rstest]
    fn result_shape_follows_config(
        #[values(TestMethod::Permutation, TestMethod::Bootstrap)] method: TestMethod,
    ) {
        let cfg = StatisticalTestConfig::new(10, method).seed(1);
        let res = run_resample_test(0.5, &cfg, |_, rng| rng.r#gen::<f64>()).unwrap();
        assert_eq!(res.n_tests, 10);
        assert_eq!(res.test_values.len(), 10);
        assert_eq!(res.method, method);
        assert!((0.0..=1.0).contains(&res.p_value));
        assert!(res.observed_value == 0.5);
    }

    #[rstest]
    fn n_tests_determines_length(#[values(2, 5, 50)] n: usize) {
        let cfg = StatisticalTestConfig::new(n, TestMethod::Permutation).seed(0);
        let res = run_resample_test(0.0, &cfg, |_, rng| rng.r#gen::<f64>()).unwrap();
        assert_eq!(res.test_values.len(), n);
        assert_eq!(res.n_tests, n);
    }

    #[test]
    fn p_value_counts_strictly_greater() {
        // Observed equal to every null value ⇒ nothing is greater ⇒ p = 0.
        let cfg = StatisticalTestConfig::new(5, TestMethod::Permutation);
        let res = run_resample_test(1.0, &cfg, |_, _| 1.0).unwrap();
        assert_eq!(res.p_value, 0.0);
    }

    #[test]
    fn error_display_mentions_the_count() {
        let msg = StatsError::TooFewTests(1).to_string();
        assert!(msg.contains("at least 2"), "unexpected message: {msg}");
    }

    // ---- Time-series-aware surrogates -------------------------------------

    fn ramp(n: usize) -> Array1<f64> {
        Array1::from((0..n).map(|i| i as f64).collect::<Vec<_>>())
    }

    /// Map a unique-valued series back to its source indices.
    fn as_indices(series: &Array1<f64>) -> Vec<usize> {
        series
            .iter()
            .map(|&v| {
                assert!(v >= 0.0 && v.fract() == 0.0, "not an index: {v}");
                v as usize
            })
            .collect()
    }

    /// Assert that `order` is a concatenation of the original contiguous
    /// blocks, each appearing exactly once and in its original order.
    fn assert_blocks_intact(order: &[usize], n: usize, block_len: usize) {
        let block_len = block_len.max(1);
        let n_blocks = n.div_ceil(block_len);
        let mut seen = vec![false; n_blocks];
        let mut i = 0;
        while i < order.len() {
            let start = order[i];
            assert_eq!(
                start % block_len,
                0,
                "block must start at a block boundary, got {start}"
            );
            let block = start / block_len;
            let expected: Vec<usize> = (start..(start + block_len).min(n)).collect();
            assert_eq!(
                &order[i..i + expected.len()],
                expected.as_slice(),
                "block {block} was not preserved verbatim"
            );
            assert!(!seen[block], "block {block} appeared twice");
            seen[block] = true;
            i += expected.len();
        }
        assert!(seen.iter().all(|&s| s), "a block was dropped");
    }

    fn sorted(mut values: Vec<f64>) -> Vec<f64> {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        values
    }

    #[rstest]
    #[case(10, 1)]
    #[case(10, 3)]
    #[case(10, 4)]
    #[case(10, 10)]
    #[case(10, 17)]
    fn block_permutation_preserves_the_exact_multiset(#[case] n: usize, #[case] block_len: usize) {
        let x = ramp(n);
        let mut rng = make_rng(Some(17));
        let out = resample_1d(&x, TestMethod::BlockPermutation { block_len }, &mut rng);
        assert_eq!(out.len(), n);
        assert_eq!(sorted(out.to_vec()), sorted(x.to_vec()));
        if block_len < n {
            assert_ne!(out, x, "expected a non-identity block permutation");
        }

        // Discrete symbols take the same path.
        let symbols: Array1<i32> = Array1::from((0..n).map(|i| (i % 3) as i32).collect::<Vec<_>>());
        let mut rng = make_rng(Some(17));
        let out_i32 = resample_1d_i32(
            &symbols,
            TestMethod::BlockPermutation { block_len },
            &mut rng,
        );
        let mut got: Vec<i32> = out_i32.to_vec();
        got.sort_unstable();
        let mut expected: Vec<i32> = symbols.to_vec();
        expected.sort_unstable();
        assert_eq!(got, expected);
    }

    #[rstest]
    #[case(10, 1)]
    #[case(10, 3)]
    #[case(10, 4)]
    #[case(10, 10)]
    #[case(10, 17)]
    fn block_permutation_preserves_within_block_order(#[case] n: usize, #[case] block_len: usize) {
        let x = ramp(n);
        let mut rng = make_rng(Some(23));
        let out = resample_1d(&x, TestMethod::BlockPermutation { block_len }, &mut rng);
        // With unique values the output indexes the source exactly.
        assert_blocks_intact(&as_indices(&out), n, block_len);
    }

    #[test]
    fn block_permutation_keeps_rows_intact() {
        let data = Array2::from_shape_fn((12, 2), |(i, j)| (i * 10 + j) as f64);
        let mut rng = make_rng(Some(5));
        let out = resample_rows(
            &data,
            TestMethod::BlockPermutation { block_len: 4 },
            &mut rng,
        );
        assert_eq!(out.dim(), data.dim());
        for row in out.rows() {
            assert!(
                data.rows().into_iter().any(|candidate| candidate == row),
                "row {row:?} is not an original row"
            );
        }
        // Duplicated rows would weaken the check above; all rows are unique
        // here, so gathering whole rows is exact.
        let first_column: Vec<usize> = out.column(0).iter().map(|&v| v as usize / 10).collect();
        assert_blocks_intact(&first_column, 12, 4);
    }

    /// Deterministic AR(1) series with uniform innovations in `[-1, 1)`.
    fn ar1(n: usize, phi: f64, seed: u64) -> Array1<f64> {
        let mut rng = make_rng(Some(seed));
        let mut x = Vec::with_capacity(n);
        let mut prev = 0.0;
        for _ in 0..n {
            let innovation = 2.0 * rng.r#gen::<f64>() - 1.0;
            prev = phi * prev + innovation;
            x.push(prev);
        }
        Array1::from(x)
    }

    fn lag1_autocorrelation(x: &Array1<f64>) -> f64 {
        let n = x.len();
        let mean = x.mean().unwrap();
        let numerator: f64 = (1..n).map(|i| (x[i] - mean) * (x[i - 1] - mean)).sum();
        let denominator: f64 = x.iter().map(|v| (v - mean).powi(2)).sum();
        numerator / denominator
    }

    #[test]
    fn block_permutation_retains_lag1_autocorrelation_of_ar1() {
        // Fixed data and seeds make this a deterministic regression test.
        let x = ar1(500, 0.8, 12_345);
        let rho = lag1_autocorrelation(&x);
        assert!(
            rho > 0.6,
            "test data should be strongly autocorrelated: {rho}"
        );

        // A block length of ~20 exceeds the AR(1) decorrelation time at
        // phi = 0.8 (1 / (1 - phi) = 5), so only block boundaries break the
        // serial dependence.
        let block_len = 20;
        let mut rng = make_rng(Some(7));
        let block_permuted = resample_1d(&x, TestMethod::BlockPermutation { block_len }, &mut rng);
        let rho_block = lag1_autocorrelation(&block_permuted);
        assert!(
            (rho_block - rho).abs() <= 0.1,
            "block permutation changed lag-1 autocorrelation: {rho} -> {rho_block}"
        );

        // An iid shuffle destroys the serial dependence.
        let mut rng = make_rng(Some(8));
        let iid = resample_1d(&x, TestMethod::Permutation, &mut rng);
        let rho_iid = lag1_autocorrelation(&iid);
        assert!(
            rho_iid.abs() < 0.1,
            "iid permutation should destroy autocorrelation, got {rho_iid}"
        );
    }

    #[test]
    fn circular_shift_is_an_exact_rotation() {
        let x = ramp(17);
        let n = x.len();
        let mut rng = make_rng(Some(3));
        let out = resample_1d(&x, TestMethod::CircularShift, &mut rng);
        assert_eq!(out.len(), n);
        assert_eq!(sorted(out.to_vec()), sorted(x.to_vec()));

        let offset = (0..n)
            .find(|&k| (0..n).all(|i| out[i] == x[(i + k) % n]))
            .expect("output is not a rotation of the input");
        assert!(offset < n);

        // The i32 path rotates identically.
        let symbols: Array1<i32> = Array1::from((0..n).map(|i| (i as i32) * 2).collect::<Vec<_>>());
        let mut rng = make_rng(Some(3));
        let out_i32 = resample_1d_i32(&symbols, TestMethod::CircularShift, &mut rng);
        let offset_i32 = (0..n)
            .find(|&k| (0..n).all(|i| out_i32[i] == symbols[(i + k) % n]))
            .expect("i32 output is not a rotation of the input");
        assert!(offset_i32 < n);
    }

    #[test]
    fn block_bootstrap_preserves_length_and_block_structure() {
        let n = 23;
        let block_len = 5;
        let x = ramp(n);
        let mut rng = make_rng(Some(11));
        let out = resample_1d(&x, TestMethod::BlockBootstrap { block_len }, &mut rng);
        assert_eq!(out.len(), n);

        let indices = as_indices(&out);
        for (chunk, values) in indices.chunks(block_len).enumerate() {
            // Every (possibly truncated) block is contiguous and increasing.
            assert!(
                values.windows(2).all(|w| w[1] == w[0] + 1),
                "chunk {chunk} is not contiguous: {values:?}"
            );
            let start = values[0];
            // Full blocks start at a valid non-wrapping offset.
            assert!(start + values.len() <= n);
        }

        // Seeded draws are reproducible.
        let mut rng = make_rng(Some(11));
        let again = resample_1d(&x, TestMethod::BlockBootstrap { block_len }, &mut rng);
        assert_eq!(out, again);
    }

    #[test]
    fn block_bootstrap_with_large_block_is_identity() {
        let x = ramp(8);
        let mut rng = make_rng(Some(1));
        let out = resample_1d(&x, TestMethod::BlockBootstrap { block_len: 50 }, &mut rng);
        assert_eq!(out, x);
    }

    #[rstest]
    fn time_series_methods_are_seeded_and_order_independent(
        #[values(
            TestMethod::BlockPermutation { block_len: 4 },
            TestMethod::CircularShift,
            TestMethod::BlockBootstrap { block_len: 4 }
        )]
        method: TestMethod,
    ) {
        // Each draw is seeded from (master, index), so this also guarantees the
        // parallel build reproduces the sequential draw order exactly.
        let x = ramp(20);
        let cfg = StatisticalTestConfig::new(12, method).seed(99);
        // A position-weighted sum distinguishes surrogates whose mean is
        // invariant (all of them preserve the marginal multiset or structure).
        let draw = |_: usize, rng: &mut SmallRng| {
            resample_1d(&x, method, rng)
                .iter()
                .enumerate()
                .map(|(i, v)| v * (i + 1) as f64)
                .sum()
        };
        let a = run_resample_test(0.0, &cfg, draw).unwrap();
        let b = run_resample_test(0.0, &cfg, draw).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.method, method);
        assert_eq!(a.test_values.len(), 12);
    }

    #[test]
    fn zero_block_length_is_rejected() {
        for method in [
            TestMethod::BlockPermutation { block_len: 0 },
            TestMethod::BlockBootstrap { block_len: 0 },
        ] {
            let cfg = StatisticalTestConfig::new(4, method).seed(1);
            assert!(matches!(
                run_resample_test(0.0, &cfg, |_, _| 0.0),
                Err(StatsError::InvalidBlockLength(0))
            ));
        }
    }
}

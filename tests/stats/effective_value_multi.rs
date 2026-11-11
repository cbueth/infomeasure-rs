// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tests for the multi-surrogate effective value summary.
//!
//! The fixtures are deliberately small, deterministic, discrete random
//! sequences so that the assertions can state what they measure:
//!
//! - **Coupled**: `x` is i.i.d. uniform on four symbols and `y[t] = x[t-1]`
//!   with probability 0.9 (otherwise a fresh uniform symbol). Ignoring the
//!   finite-sample bias, `T_{X -> Y} = H(X) = ln 4 ~= 1.386` nats. The
//!   surrogate bias is about 0.05 nats at `n = 400`, so the effective value is
//!   about `1.0`; the test requires it to exceed `0.5`.
//! - **Independent**: `x` and `y` are i.i.d. uniform on four symbols drawn from
//!   separate seeds. The true transfer entropy is zero, and the observed value
//!   is essentially the positive finite-sample bias, which the surrogate mean
//!   cancels. The effective value is about `-0.01` nats at `n = 400`; the test
//!   requires `|eTE| < 0.05`.
//!
//! All tolerances below are stated in nats and cover the observed sampling
//! variation of these fixtures, not an arbitrary "small number".

use approx::assert_abs_diff_eq;
use infomeasure::estimators::GlobalValue;
use infomeasure::estimators::stats::{EffectiveValue, EffectiveValueStats, StatsError, TestMethod};
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rstest::rstest;

const N: usize = 400;
const ALPHABET: i32 = 4;

fn symbols(n: usize, seed: u64) -> Array1<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from(
        (0..n)
            .map(|_| rng.gen_range(0..ALPHABET))
            .collect::<Vec<_>>(),
    )
}

/// `y[t] = x[t-1]` with probability `1 - noise`, otherwise a fresh symbol.
fn coupled(n: usize, seed: u64, noise: f64) -> (Array1<i32>, Array1<i32>) {
    let x = symbols(n, seed);
    let mut rng = StdRng::seed_from_u64(seed ^ 0xDEAD_BEEF);
    let mut y = Array1::zeros(n);
    for t in 1..n {
        y[t] = if rng.gen_bool(noise) {
            rng.gen_range(0..ALPHABET)
        } else {
            x[t - 1]
        };
    }
    (x, y)
}

fn independent(n: usize) -> (Array1<i32>, Array1<i32>) {
    (symbols(n, 11), symbols(n, 12))
}

fn continuous(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from((0..n).map(|_| rng.gen_range(-1.0..1.0)).collect::<Vec<_>>())
}

fn coupled_continuous(n: usize) -> (Array1<f64>, Array1<f64>) {
    let x = continuous(n, 3);
    let mut y = Array1::zeros(n);
    for t in 1..n {
        y[t] = 0.8 * x[t - 1] + 0.1 * (t as f64 * 0.37).sin();
    }
    (x, y)
}

/// One-draw summaries must reproduce the Python-compatible single draw exactly.
#[rstest]
fn n_tests_one_matches_single_draw(
    #[values("discrete", "kernel", "ordinal", "ksg")] approach: &str,
) {
    let (summary_value, single) = match approach {
        "discrete" => {
            let (x, y) = coupled(N, 1, 0.1);
            let te = TransferEntropy::new_discrete_mle(&x, &y, 1, 1, 1);
            let summary = te
                .effective_value_with_resamples(TestMethod::Permutation, Some(42), 1)
                .unwrap();
            assert_eq!(summary.n_tests, 1);
            assert_eq!(summary.test_values.len(), 1);
            assert!(summary.null_std.is_nan());
            assert!(summary.standard_error.is_nan());
            (
                summary.effective_value,
                te.effective_value(TestMethod::Permutation, Some(42)),
            )
        }
        "kernel" => {
            let (x, y) = coupled_continuous(80);
            let te = TransferEntropy::new_kernel(&x, &y, 1, 1, 1, 0.5);
            let summary = te
                .effective_value_with_resamples(TestMethod::Permutation, Some(7), 1)
                .unwrap();
            assert!(summary.null_std.is_nan());
            (
                summary.effective_value,
                te.effective_value(TestMethod::Permutation, Some(7)),
            )
        }
        "ordinal" => {
            let (x, y) = coupled_continuous(80);
            let te = TransferEntropy::new_ordinal(&x, &y, 3, 1, 1, 1, false);
            let summary = te
                .effective_value_with_resamples(TestMethod::Permutation, Some(5), 1)
                .unwrap();
            assert!(summary.null_std.is_nan());
            (
                summary.effective_value,
                te.effective_value(TestMethod::Permutation, Some(5)),
            )
        }
        "ksg" => {
            let (x, y) = coupled_continuous(80);
            let te = TransferEntropy::new_ksg(&x, &y, 1, 1, 1, 4, 1e-10);
            let summary = te
                .effective_value_with_resamples(TestMethod::Permutation, Some(3), 1)
                .unwrap();
            assert!(summary.null_std.is_nan());
            (
                summary.effective_value,
                te.effective_value(TestMethod::Permutation, Some(3)),
            )
        }
        other => panic!("unknown approach {other}"),
    };
    assert_eq!(summary_value, single);
}

/// The conditional variant gets the same exact single-draw equivalence.
#[rstest]
fn conditional_n_tests_one_matches_single_draw(
    #[values("discrete", "kernel", "ksg")] approach: &str,
) {
    let (summary_value, single) = match approach {
        "discrete" => {
            let (x, y) = coupled(N, 1, 0.1);
            let z = symbols(N, 99);
            let cte = TransferEntropy::new_cte_discrete_mle(&x, &y, &z, 1, 1, 1, 1);
            let summary = cte
                .effective_value_with_resamples(TestMethod::Permutation, Some(13), 1)
                .unwrap();
            (
                summary.effective_value,
                cte.effective_value(TestMethod::Permutation, Some(13)),
            )
        }
        "kernel" => {
            let (x, y) = coupled_continuous(80);
            let z = continuous(80, 21);
            let cte = TransferEntropy::new_cte_kernel(&x, &y, &z, 1, 1, 1, 1, 0.5);
            let summary = cte
                .effective_value_with_resamples(TestMethod::Permutation, Some(13), 1)
                .unwrap();
            (
                summary.effective_value,
                cte.effective_value(TestMethod::Permutation, Some(13)),
            )
        }
        "ksg" => {
            let (x, y) = coupled_continuous(80);
            let z = continuous(80, 21);
            let cte = TransferEntropy::new_cte_ksg(&x, &y, &z, 1, 1, 1, 1, 4, 1e-10);
            let summary = cte
                .effective_value_with_resamples(TestMethod::Permutation, Some(13), 1)
                .unwrap();
            (
                summary.effective_value,
                cte.effective_value(TestMethod::Permutation, Some(13)),
            )
        }
        other => panic!("unknown approach {other}"),
    };
    assert_eq!(summary_value, single);
}

/// The multi-draw surrogate mean estimates the expectation of the single-draw
/// value. We estimate that expectation from 60 independent single draws and
/// compare it to the multi-draw mean within four combined standard errors.
#[test]
fn surrogate_mean_converges_to_single_draw_expectation() {
    let (x, y) = coupled(N, 1, 0.1);
    let te = TransferEntropy::new_discrete_mle(&x, &y, 1, 1, 1);
    let observed = te.global_value();

    const N_SINGLE: u64 = 60;
    let single_nulls: Vec<f64> = (0..N_SINGLE)
        .map(|seed| observed - te.effective_value(TestMethod::Permutation, Some(1_000 + seed)))
        .collect();
    let single_mean = single_nulls.iter().sum::<f64>() / N_SINGLE as f64;
    let single_var = single_nulls
        .iter()
        .map(|v| (v - single_mean).powi(2))
        .sum::<f64>()
        / (N_SINGLE as f64 - 1.0);
    let single_mean_se = (single_var / N_SINGLE as f64).sqrt();

    let summary = te
        .effective_value_with_resamples(TestMethod::Permutation, Some(7), 400)
        .unwrap();

    let combined_se = (single_mean_se.powi(2) + summary.standard_error.powi(2)).sqrt();
    assert!(
        (summary.null_mean - single_mean).abs() <= 4.0 * combined_se,
        "multi mean {} vs single-draw mean {} (combined SE {})",
        summary.null_mean,
        single_mean,
        combined_se
    );
    // The summary's effective value is by construction the observed value minus
    // its own surrogate mean.
    assert_abs_diff_eq!(
        summary.effective_value,
        observed - summary.null_mean,
        epsilon = 1e-12
    );
}

/// The spread is finite for more than one draw and the standard error of the
/// surrogate mean shrinks as `1 / sqrt(n)`. With a fixed seed the draws are
/// prefix-stable, so the large run contains the small run and the shrinkage is
/// not an artefact of resampling different data.
#[test]
fn standard_error_decreases_with_more_draws() {
    let (x, y) = coupled(N, 1, 0.1);
    let te = TransferEntropy::new_discrete_mle(&x, &y, 1, 1, 1);

    let few = te
        .effective_value_with_resamples(TestMethod::Permutation, Some(21), 100)
        .unwrap();
    let many = te
        .effective_value_with_resamples(TestMethod::Permutation, Some(21), 600)
        .unwrap();

    assert!(few.null_std.is_finite() && few.null_std > 0.0);
    assert!(few.standard_error.is_finite() && few.standard_error > 0.0);
    assert!(many.null_std.is_finite() && many.null_std > 0.0);
    assert!(many.standard_error.is_finite() && many.standard_error > 0.0);

    // The expected ratio is sqrt(100 / 600) ~= 0.41; allow generous sampling
    // slack.
    assert!(
        many.standard_error < 0.7 * few.standard_error,
        "SE(600) = {} is not clearly below SE(100) = {}",
        many.standard_error,
        few.standard_error
    );

    // Prefix stability: the first 100 values of the larger run are exactly the
    // small run's values.
    for (small, large) in few.test_values.iter().zip(many.test_values.iter()) {
        assert_eq!(small, large);
    }
}

/// An independent source has zero true transfer entropy after bias correction,
/// whereas a coupled source stays clearly positive.
#[test]
fn independent_source_is_near_zero_and_coupled_is_positive() {
    let (xi, yi) = independent(N);
    let independent = TransferEntropy::new_discrete_mle(&xi, &yi, 1, 1, 1)
        .effective_value_with_resamples(TestMethod::Permutation, Some(5), 200)
        .unwrap();
    assert!(
        independent.effective_value.abs() < 0.05,
        "independent eTE = {} should be within 0.05 nats of zero",
        independent.effective_value
    );

    let (xc, yc) = coupled(N, 1, 0.1);
    let coupled = TransferEntropy::new_discrete_mle(&xc, &yc, 1, 1, 1)
        .effective_value_with_resamples(TestMethod::Permutation, Some(5), 200)
        .unwrap();
    assert!(
        coupled.effective_value > 0.5,
        "coupled eTE = {} should be clearly positive",
        coupled.effective_value
    );
    assert!(coupled.effective_value > independent.effective_value);
}

/// Seeded summaries are bit-for-bit reproducible. Running this test under
/// `--features parallel` exercises the parallel draw path and asserts the same
/// deterministic result.
#[test]
fn seeded_summary_is_reproducible() {
    let (x, y) = coupled(N, 1, 0.1);
    let te = TransferEntropy::new_discrete_mle(&x, &y, 1, 1, 1);
    let a = te
        .effective_value_with_resamples(TestMethod::Permutation, Some(77), 128)
        .unwrap();
    let b = te
        .effective_value_with_resamples(TestMethod::Permutation, Some(77), 128)
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn zero_draws_is_an_error() {
    let (x, y) = coupled(50, 1, 0.1);
    let te = TransferEntropy::new_discrete_mle(&x, &y, 1, 1, 1);
    let err = te
        .effective_value_with_resamples(TestMethod::Permutation, Some(1), 0)
        .unwrap_err();
    assert_eq!(err, StatsError::NoResamples);
}

#[rstest]
fn zero_block_length_is_an_error(
    #[values(
        TestMethod::BlockPermutation { block_len: 0 },
        TestMethod::BlockBootstrap { block_len: 0 }
    )]
    method: TestMethod,
) {
    let (x, y) = coupled(50, 1, 0.1);
    let te = TransferEntropy::new_discrete_mle(&x, &y, 1, 1, 1);
    let err = te
        .effective_value_with_resamples(method, Some(1), 2)
        .unwrap_err();
    assert_eq!(err, StatsError::InvalidBlockLength(0));
}

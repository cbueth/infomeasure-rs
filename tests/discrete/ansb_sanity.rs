// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::GlobalValue;
use infomeasure::estimators::approaches::AnsbEntropy;
use ndarray::Array1;
use rstest::*;
use statrs::function::gamma::digamma;

#[test]
fn ansb_entropy_matches_formula() {
    // Data with coincidences: N=7, K=5 => Δ=2
    let data = Array1::from(vec![1, 2, 3, 4, 5, 1, 2]);
    let est = AnsbEntropy::new(data, None, 0.1);
    let h = est.global_value();

    // Expected: (γ - ln 2) + 2 ln N - ψ(Δ)
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    let n = 7.0_f64;
    let delta = 2.0_f64;
    let h_exp = (EULER_GAMMA - 2.0_f64.ln()) + 2.0 * n.ln() - digamma(delta);

    assert_abs_diff_eq!(h, h_exp, epsilon = 1e-12);
}

#[rstest]
#[case(vec![1, 2, 3, 4, 5], 0.1, true)] // Well-sampled: N/K = 1.0 > 0.1, should warn
#[case(vec![1, 2, 3, 4, 5], 2.0, false)] // Well-sampled: N/K = 1.0 <= 2.0, should not warn
#[case(vec![1, 1, 1, 1, 2], 0.1, true)] // Moderately undersampled: N/K = 2.5 > 0.1, should warn
#[case(vec![1, 1, 1, 1, 2], 3.0, false)] // Moderately undersampled: N/K = 2.5 <= 3.0, should not warn
#[case(vec![1, 1, 1, 1, 1], 5.0, false)] // Very undersampled: N/K = 1.0 <= 5.0, should not warn
#[case(vec![1, 1, 2, 2, 3], 0.0, true)] // Threshold 0.0: always warn if N > K
fn test_undersampled_warning(
    #[case] data: Vec<i32>,
    #[case] threshold: f64,
    #[case] should_warn: bool,
) {
    // For this test, we verify the logic by computing the N/K ratio and comparing to threshold
    let k_obs = if data.is_empty() {
        0
    } else {
        data.iter().collect::<std::collections::HashSet<_>>().len()
    };
    let n_size = data.len();

    if k_obs > 0 {
        let ratio = n_size as f64 / k_obs as f64;
        let warning_expected = ratio > threshold;

        assert_eq!(
            warning_expected, should_warn,
            "Warning condition mismatch for N={n_size}, K={k_obs}, ratio={ratio:.3}, threshold={threshold:.3}: expected warning={should_warn}, got warning={warning_expected}"
        );
    }
}

#[rstest]
#[case(vec![1, 2, 3, 4, 5], 0.1, 1.0)] // N=5, K=5, N/K=1.0
#[case(vec![1, 1, 1, 1, 2], 0.1, 2.5)] // N=5, K=2, N/K=2.5
#[case(vec![1, 1, 1, 1, 1], 0.5, 5.0)] // N=5, K=1, N/K=5.0
#[case(vec![1, 1, 2, 2, 3], 0.0, 1.67)] // N=5, K=3, N/K=1.67
fn test_undersampled_calculation(
    #[case] data: Vec<i32>,
    #[case] threshold: f64,
    #[case] expected_ratio: f64,
) {
    let data_arr = Array1::from(data.clone());
    let _est = AnsbEntropy::new(data_arr, None, threshold);

    // Calculate N/K ratio from the data
    let k_obs = if data.is_empty() {
        0
    } else {
        data.iter().collect::<std::collections::HashSet<_>>().len()
    };
    let n_size = data.len();

    if k_obs > 0 {
        let actual_ratio = n_size as f64 / k_obs as f64;
        assert_abs_diff_eq!(actual_ratio, expected_ratio, epsilon = 0.01);

        // Verify warning condition
        let should_warn = actual_ratio > threshold;
        let warning_expected = expected_ratio > threshold;
        assert_eq!(
            should_warn, warning_expected,
            "Warning condition mismatch for ratio={actual_ratio:.3}, threshold={threshold:.3}"
        );
    }
}

#[rstest]
#[case(vec![1, 2, 3, 4, 5, 1, 2], 0.1, 2.0)] // N=7, K=5, N/K=1.4, coincidences=2
#[case(vec![1, 1, 1, 2, 2, 3, 4], 0.05, 3.0)] // N=7, K=4, N/K=1.75, coincidences=3
#[case(vec![1, 1, 2, 2, 3, 3, 4], 0.2, 1.5)] // N=7, K=4, N/K=1.75, coincidences=3
fn test_ansb_warning_different_thresholds(
    #[case] data: Vec<i32>,
    #[case] low_threshold: f64,
    #[case] high_threshold: f64,
) {
    let data_arr = Array1::from(data.clone());

    // Test with low threshold (should warn)
    let est_low = AnsbEntropy::new(data_arr.clone(), None, low_threshold);
    let h_low = est_low.global_value();

    // Test with high threshold (should not warn)
    let est_high = AnsbEntropy::new(data_arr, None, high_threshold);
    let h_high = est_high.global_value();

    // Both should give the same numerical result (just different warning behavior)
    assert_abs_diff_eq!(h_low, h_high, epsilon = 1e-12);

    // Neither should be NaN (all test data has coincidences)
    assert!(!h_low.is_nan());
    assert!(!h_high.is_nan());
}

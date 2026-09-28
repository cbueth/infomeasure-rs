// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sanity tests for the Jensen–Shannon divergence.

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::bayes::AlphaParam;
use infomeasure::estimators::composite_measures::{jsd, jsd_kernel_1d, jsd_kernel_nd};
use infomeasure::estimators::entropy::Entropy;
use infomeasure::estimators::traits::{GlobalValue, ProbabilityMass};
use ndarray::{Array1, Array2, array};
use rstest::rstest;
use rustc_hash::FxHashMap;

#[test]
fn jsd_fewer_than_two_inputs_is_zero() {
    let p = Entropy::new_discrete(array![1, 1, 2, 3]);
    assert_abs_diff_eq!(jsd(&[p], None), 0.0, epsilon = 1e-12);
}

#[rstest]
fn jsd_is_symmetric(
    #[values(
        vec![1, 1, 2, 2, 3, 3, 3, 4],
        vec![0, 0, 1, 2, 2, 2, 3],
        vec![7, 7, 8, 8, 9, 9, 9]
    )]
    p: Vec<i32>,
    #[values(
        vec![1, 1, 1, 2, 3, 4, 4, 4],
        vec![0, 1, 1, 1, 2, 3, 3],
        vec![7, 8, 8, 9, 9, 9, 9]
    )]
    q: Vec<i32>,
) {
    let d_pq = jsd(
        &[
            Entropy::new_discrete(Array1::from(p.clone())),
            Entropy::new_discrete(Array1::from(q.clone())),
        ],
        None,
    );
    let d_qp = jsd(
        &[
            Entropy::new_discrete(Array1::from(q)),
            Entropy::new_discrete(Array1::from(p)),
        ],
        None,
    );
    assert_abs_diff_eq!(d_pq, d_qp, epsilon = 1e-12);
    assert!(d_pq >= -1e-12, "JSD must be non-negative, got {d_pq}");
    assert!(d_pq <= 2.0_f64.ln() + 1e-12);
}

#[rstest]
fn jsd_matches_manual_mixture_entropy(
    #[values(
        (vec![1, 1, 1, 2, 3, 3], vec![1, 2, 2, 2, 3, 4]),
        (vec![0, 0, 1, 2, 2, 3], vec![0, 1, 1, 1, 3, 3]),
        (vec![1, 2, 2, 3, 3, 4, 4], vec![1, 1, 2, 2, 4, 4, 4])
    )]
    pair: (Vec<i32>, Vec<i32>),
) {
    let p = Entropy::new_discrete(Array1::from(pair.0));
    let q = Entropy::new_discrete(Array1::from(pair.1));

    let mut mixture: FxHashMap<i32, f64> = FxHashMap::default();
    for (symbol, prob) in p.pmf() {
        *mixture.entry(symbol).or_insert(0.0) += 0.5 * prob;
    }
    for (symbol, prob) in q.pmf() {
        *mixture.entry(symbol).or_insert(0.0) += 0.5 * prob;
    }
    let h_mixture = -mixture
        .values()
        .filter(|&&m| m > 0.0)
        .map(|&m| m * m.ln())
        .sum::<f64>();
    let expected = h_mixture - 0.5 * (p.global_value() + q.global_value());

    assert_abs_diff_eq!(jsd(&[p, q], None), expected, epsilon = 1e-12);
}

#[rstest]
fn jsd_of_identical_estimators_is_zero(
    #[values(
        vec![1, 1, 2, 2, 3, 3, 3, 4],
        vec![0, 1, 2, 3, 4, 5, 5],
        vec![9, 9, 9, 9, 9]
    )]
    data: Vec<i32>,
) {
    let p = Entropy::new_discrete(Array1::from(data.clone()));
    let q = Entropy::new_discrete(Array1::from(data.clone()));
    assert_abs_diff_eq!(jsd(&[p, q], None), 0.0, epsilon = 1e-12);

    let bayes_p = Entropy::new_bayes(Array1::from(data.clone()), AlphaParam::Laplace, None);
    let bayes_q = Entropy::new_bayes(Array1::from(data.clone()), AlphaParam::Laplace, None);
    assert_abs_diff_eq!(jsd(&[bayes_p, bayes_q], None), 0.0, epsilon = 1e-12);

    let shrink_p = Entropy::new_shrink(Array1::from(data.clone()));
    let shrink_q = Entropy::new_shrink(Array1::from(data));
    assert_abs_diff_eq!(jsd(&[shrink_p, shrink_q], None), 0.0, epsilon = 1e-12);
}

#[rstest]
#[case(AlphaParam::Jeffrey)]
#[case(AlphaParam::Laplace)]
#[case(AlphaParam::SchGrass)]
fn jsd_bayes_self_is_zero(#[case] alpha: AlphaParam) {
    let p = Entropy::new_bayes(array![1, 1, 2, 2, 3, 3, 4], alpha.clone(), None);
    let q = Entropy::new_bayes(array![1, 1, 2, 2, 3, 3, 4], alpha, None);
    assert_abs_diff_eq!(jsd(&[p, q], None), 0.0, epsilon = 1e-12);
}

#[rstest]
fn jsd_ordinal_self_is_zero(#[values(2, 3, 4)] order: usize) {
    let p = Entropy::new_ordinal(array![1.0, 3.0, 2.0, 4.0, 2.5, 5.0, 3.5, 6.0], order);
    let q = Entropy::new_ordinal(array![1.0, 3.0, 2.0, 4.0, 2.5, 5.0, 3.5, 6.0], order);
    assert_abs_diff_eq!(jsd(&[p, q], None), 0.0, epsilon = 1e-12);
}

#[rstest]
fn jsd_of_disjoint_deterministic_distributions_is_log_n(#[values(2, 3, 4)] n: usize) {
    let dists: Vec<_> = (0..n)
        .map(|i| Entropy::new_discrete(array![i as i32, i as i32, i as i32]))
        .collect();
    assert_abs_diff_eq!(jsd(&dists, None), (n as f64).ln(), epsilon = 1e-12);
}

#[rstest]
fn jsd_weighted_two_deterministic_matches_binary_entropy(#[values(0.25, 0.5, 0.75)] w: f64) {
    let p = Entropy::new_discrete(array![0, 0, 0]);
    let q = Entropy::new_discrete(array![1, 1, 1]);
    let expected = -(w * w.ln() + (1.0 - w) * (1.0 - w).ln());
    assert_abs_diff_eq!(jsd(&[p, q], Some(&[w, 1.0 - w])), expected, epsilon = 1e-12);
}

#[test]
fn jsd_uniform_weights_match_none() {
    let build = || {
        [
            Entropy::new_discrete(array![1, 1, 2, 3, 3, 4]),
            Entropy::new_discrete(array![1, 2, 2, 2, 4, 4]),
            Entropy::new_discrete(array![1, 1, 1, 3, 4, 4]),
        ]
    };
    let uniform = jsd(&build(), Some(&[1.0 / 3.0; 3]));
    let default = jsd(&build(), None);
    assert_abs_diff_eq!(default, uniform, epsilon = 1e-12);
}

#[rstest]
#[case("box", 0.5)]
#[case("gaussian", 1.0)]
#[case("gaussian", 2.0)]
fn jsd_kernel_is_symmetric_and_finite(#[case] kernel: &str, #[case] bandwidth: f64) {
    let x = array![1.0, 2.0, 2.5, 4.0, 5.0, 5.5];
    let y = array![1.5, 2.2, 3.0, 4.4, 5.5, 6.0];

    let d_xy = jsd_kernel_1d(&[x.clone(), y.clone()], None, bandwidth, kernel);
    let d_yx = jsd_kernel_1d(&[y, x], None, bandwidth, kernel);
    assert!(d_xy.is_finite());
    assert_abs_diff_eq!(d_xy, d_yx, epsilon = 1e-12);
}

#[rstest]
#[case("box", 0.5)]
#[case("box", 1.0)]
#[case("gaussian", 1.0)]
fn jsd_kernel_nd_is_finite(#[case] kernel: &str, #[case] bandwidth: f64) {
    let x = Array2::from_shape_vec(
        (6, 2),
        vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.5, 1.5, 1.5, 0.5, 3.0, 3.0],
    )
    .unwrap();
    let y = Array2::from_shape_vec(
        (6, 2),
        vec![0.2, 0.1, 1.1, 0.9, 2.2, 1.8, 0.6, 1.4, 1.4, 0.6, 3.1, 2.9],
    )
    .unwrap();
    assert!(jsd_kernel_nd::<2>(&[x, y], None, bandwidth, kernel).is_finite());
}

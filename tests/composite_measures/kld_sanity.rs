// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sanity tests for the Kullback–Leibler divergence.

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::bayes::AlphaParam;
use infomeasure::estimators::composite_measures::{Kld, kld};
use infomeasure::estimators::entropy::Entropy;
use infomeasure::estimators::traits::{CrossEntropy, GlobalValue};
use ndarray::{Array1, array};
use rstest::rstest;

#[rstest]
fn kld_equals_cross_entropy_minus_entropy(
    #[values(
        vec![1, 1, 1, 2, 2, 3, 4, 5],
        vec![0, 0, 0, 1, 2, 3, 3, 3, 4],
        vec![5, 5, 6, 7, 8, 8, 8, 9]
    )]
    p: Vec<i32>,
    #[values(
        vec![1, 1, 2, 2, 2, 3, 3, 5],
        vec![0, 1, 1, 2, 2, 3, 4, 4, 4],
        vec![5, 6, 6, 7, 7, 8, 9, 9]
    )]
    q: Vec<i32>,
) {
    let ep = Entropy::new_discrete(Array1::from(p));
    let eq = Entropy::new_discrete(Array1::from(q));

    let expected = ep.cross_entropy(&eq) - ep.global_value();
    assert_abs_diff_eq!(kld(&ep, &eq), expected, epsilon = 1e-12);
    assert_abs_diff_eq!(ep.kld(&eq), expected, epsilon = 1e-12);
}

#[rstest]
fn kld_of_identical_distributions_is_zero(
    #[values(
        vec![1, 1, 2, 2, 3, 3, 3, 4],
        vec![0, 1, 2, 3, 4, 5],
        vec![9, 9, 9, 9, 9]
    )]
    data: Vec<i32>,
) {
    let p = Entropy::new_discrete(Array1::from(data));
    assert_abs_diff_eq!(p.kld(&p), 0.0, epsilon = 1e-12);
}

#[rstest]
fn kld_is_asymmetric(
    #[values(
        (vec![1, 1, 1, 1, 2, 3, 4, 5], vec![1, 2, 2, 2, 3, 3, 4, 5]),
        (vec![0, 0, 0, 0, 1, 2], vec![0, 1, 1, 1, 2, 2]),
        (vec![1, 1, 1, 2, 2, 3, 3, 3, 3], vec![1, 1, 2, 2, 2, 2, 3, 4, 4])
    )]
    pair: (Vec<i32>, Vec<i32>),
) {
    let p = Entropy::new_discrete(Array1::from(pair.0));
    let q = Entropy::new_discrete(Array1::from(pair.1));

    let d_pq = p.kld(&q);
    let d_qp = q.kld(&p);
    assert!(d_pq.is_finite() && d_qp.is_finite());
    assert!(
        (d_pq - d_qp).abs() > 1e-6,
        "KLD should be asymmetric: {d_pq} vs {d_qp}"
    );
}

#[rstest]
fn kld_miller_madow_and_bayes_are_finite(
    #[values(
        vec![1, 1, 1, 2, 2, 3, 4, 5],
        vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 5],
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    )]
    data: Vec<i32>,
) {
    let mm_p = Entropy::new_miller_madow(Array1::from(data.clone()));
    let mm_q = Entropy::new_miller_madow(Array1::from(vec![1, 1, 2, 2, 3, 4, 4, 5]));
    assert!(mm_p.kld(&mm_q).is_finite());

    let bayes_p = Entropy::new_bayes(Array1::from(data), AlphaParam::Laplace, None);
    let bayes_q = Entropy::new_bayes(array![1, 1, 2, 2, 3, 4, 4, 5], AlphaParam::Laplace, None);
    assert!(bayes_p.kld(&bayes_q).is_finite());
}

#[rstest]
#[case(AlphaParam::Jeffrey)]
#[case(AlphaParam::Laplace)]
#[case(AlphaParam::SchGrass)]
fn kld_bayes_all_priors_are_finite(#[case] alpha: AlphaParam) {
    let p = Entropy::new_bayes(array![1, 1, 1, 2, 2, 3, 4, 5], alpha.clone(), None);
    let q = Entropy::new_bayes(array![1, 1, 2, 2, 2, 3, 3, 5], alpha, None);
    assert!(p.kld(&q).is_finite());
}

#[rstest]
fn kld_ordinal_is_finite(#[values(2, 3, 4)] order: usize) {
    let x = Array1::from(vec![
        1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 4.5, 3.0, 5.0, 4.0, 6.0,
    ]);
    let y = Array1::from(vec![
        1.1, 1.9, 1.6, 3.1, 2.6, 4.2, 3.4, 4.6, 3.2, 5.1, 4.1, 6.2,
    ]);
    let p = Entropy::new_ordinal(x, order);
    let q = Entropy::new_ordinal(y, order);
    assert!(p.kld(&q).is_finite());
}

#[rstest]
#[case("box", 0.5)]
#[case("gaussian", 1.0)]
#[case("gaussian", 2.0)]
fn kld_kernel_is_finite(#[case] kernel: &str, #[case] bandwidth: f64) {
    let x = Array1::from(vec![1.0, 2.0, 2.5, 4.0, 5.0, 5.5, 6.0, 7.0]);
    let y = Array1::from(vec![1.5, 2.2, 3.0, 4.4, 5.5, 6.0, 6.8, 7.5]);
    let p = Entropy::new_kernel_with_type(x, kernel.into(), bandwidth);
    let q = Entropy::new_kernel_with_type(y, kernel.into(), bandwidth);
    assert!(p.kld(&q).is_finite());
}

#[rstest]
fn kld_kozachenko_leonenko_is_finite(#[values(1, 3, 5)] k: usize) {
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    let y = Array1::from(vec![0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]);
    let p = Entropy::new_kl_1d(x, k, 0.0);
    let q = Entropy::new_kl_1d(y, k, 0.0);
    assert!(p.kld(&q).is_finite());
}

#[rstest]
fn kld_renyi_is_finite(#[values(0.5, 0.8, 2.0)] alpha: f64) {
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    let y = Array1::from(vec![0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]);
    let p = Entropy::new_renyi_1d(x, 3, alpha, 0.0);
    let q = Entropy::new_renyi_1d(y, 3, alpha, 0.0);
    assert!(p.kld(&q).is_finite());
}

#[rstest]
fn kld_tsallis_is_finite(#[values(0.5, 0.9, 1.5)] q: f64) {
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    let y = Array1::from(vec![0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]);
    let p = Entropy::new_tsallis_1d(x, 3, q, 0.0);
    let q = Entropy::new_tsallis_1d(y, 3, q, 0.0);
    assert!(p.kld(&q).is_finite());
}

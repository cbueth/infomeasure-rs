// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Python parity tests for the Kullback–Leibler divergence.

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::bayes::AlphaParam;
use infomeasure::estimators::composite_measures::Kld;
use infomeasure::estimators::entropy::Entropy;
use ndarray::Array1;
use rstest::rstest;
use validation::python;

#[rstest]
#[case(vec![1, 1, 1, 2, 2, 3, 4, 5], vec![1, 1, 2, 2, 2, 3, 3, 5])]
#[case(vec![0, 0, 0, 1, 2, 3, 3, 3, 4], vec![0, 1, 1, 2, 2, 3, 4, 4, 4])]
#[case(vec![5, 5, 6, 7, 8, 8, 8, 9], vec![5, 6, 6, 7, 7, 8, 9, 9])]
fn kld_discrete_python_parity(#[case] p: Vec<i32>, #[case] q: Vec<i32>) {
    let d_rust = Entropy::new_discrete(Array1::from(p.clone()))
        .kld(&Entropy::new_discrete(Array1::from(q.clone())));
    let d_py = python::calculate_kld(&p, &q, "discrete", &[]).expect("python kld discrete");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 2, 2, 3, 4, 5], vec![1, 1, 2, 2, 2, 3, 3, 5])]
#[case(vec![0, 0, 0, 1, 2, 3, 3, 3, 4], vec![0, 1, 1, 2, 2, 3, 4, 4, 4])]
#[case(vec![5, 5, 6, 7, 8, 8, 8, 9], vec![5, 6, 6, 7, 7, 8, 9, 9])]
fn kld_miller_madow_python_parity(#[case] p: Vec<i32>, #[case] q: Vec<i32>) {
    let d_rust = Entropy::new_miller_madow(Array1::from(p.clone()))
        .kld(&Entropy::new_miller_madow(Array1::from(q.clone())));
    let d_py = python::calculate_kld(&p, &q, "miller_madow", &[]).expect("python kld mm");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 2, 2, 3, 4, 5], vec![1, 1, 2, 2, 2, 3, 3, 5])]
#[case(vec![0, 0, 0, 1, 2, 3, 3, 3, 4], vec![0, 1, 1, 2, 2, 3, 4, 4, 4])]
#[case(vec![5, 5, 6, 7, 8, 8, 8, 9], vec![5, 6, 6, 7, 7, 8, 9, 9])]
fn kld_bayes_python_parity(#[case] p: Vec<i32>, #[case] q: Vec<i32>) {
    let d_rust = Entropy::new_bayes(Array1::from(p.clone()), AlphaParam::Laplace, None).kld(
        &Entropy::new_bayes(Array1::from(q.clone()), AlphaParam::Laplace, None),
    );
    let kwargs = [("alpha".to_string(), "\"laplace\"".to_string())];
    let d_py = python::calculate_kld(&p, &q, "bayes", &kwargs).expect("python kld bayes");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-10);
}

#[rstest]
fn kld_ordinal_python_parity(#[values(2, 3, 4)] order: usize) {
    let x = Array1::from(vec![
        1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 4.5, 3.0, 5.0, 4.0, 6.0,
    ]);
    let y = Array1::from(vec![
        1.1, 1.9, 1.6, 3.1, 2.6, 4.2, 3.4, 4.6, 3.2, 5.1, 4.1, 6.2,
    ]);
    let d_rust =
        Entropy::new_ordinal(x.clone(), order).kld(&Entropy::new_ordinal(y.clone(), order));
    let kwargs = [
        ("embedding_dim".to_string(), order.to_string()),
        ("stable".to_string(), "True".to_string()),
    ];
    let d_py = python::calculate_kld(
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        "ordinal",
        &kwargs,
    )
    .expect("python kld ordinal");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-10);
}

#[rstest]
#[case("box", 0.5)]
#[case("gaussian", 1.0)]
#[case("gaussian", 2.0)]
fn kld_kernel_python_parity(#[case] kernel: &str, #[case] bandwidth: f64) {
    let x = Array1::from(vec![1.0, 2.0, 2.5, 4.0, 5.0, 5.5, 6.0, 7.0]);
    let y = Array1::from(vec![1.5, 2.2, 3.0, 4.4, 5.5, 6.0, 6.8, 7.5]);
    let d_rust = Entropy::new_kernel_with_type(x.clone(), kernel.into(), bandwidth).kld(
        &Entropy::new_kernel_with_type(y.clone(), kernel.into(), bandwidth),
    );
    let kwargs = [
        ("kernel".to_string(), format!("\"{kernel}\"")),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let d_py = python::calculate_kld(
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        "kernel",
        &kwargs,
    )
    .expect("python kld kernel");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-8);
}

#[rstest]
fn kld_kozachenko_leonenko_python_parity(#[values(2, 3, 5)] k: usize) {
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    let y = Array1::from(vec![0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]);
    let d_rust = Entropy::new_kl_1d(x.clone(), k, 0.0).kld(&Entropy::new_kl_1d(y.clone(), k, 0.0));
    let kwargs = [
        ("k".to_string(), k.to_string()),
        ("minkowski_p".to_string(), "inf".to_string()),
    ];
    let d_py = python::calculate_kld(
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        "metric",
        &kwargs,
    )
    .expect("python kld kl");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-8);
}

#[rstest]
fn kld_renyi_python_parity(#[values(0.5, 0.8, 2.0)] alpha: f64) {
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    let y = Array1::from(vec![0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]);
    let k = 3usize;
    let d_rust = Entropy::new_renyi_1d(x.clone(), k, alpha, 0.0).kld(&Entropy::new_renyi_1d(
        y.clone(),
        k,
        alpha,
        0.0,
    ));
    let kwargs = [
        ("k".to_string(), k.to_string()),
        ("alpha".to_string(), alpha.to_string()),
    ];
    let d_py = python::calculate_kld(
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        "renyi",
        &kwargs,
    )
    .expect("python kld renyi");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-8);
}

#[rstest]
fn kld_tsallis_python_parity(#[values(0.5, 0.9, 1.5)] q: f64) {
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    let y = Array1::from(vec![0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]);
    let k = 3usize;
    let d_rust = Entropy::new_tsallis_1d(x.clone(), k, q, 0.0).kld(&Entropy::new_tsallis_1d(
        y.clone(),
        k,
        q,
        0.0,
    ));
    let kwargs = [
        ("k".to_string(), k.to_string()),
        ("q".to_string(), q.to_string()),
    ];
    let d_py = python::calculate_kld(
        x.as_slice().unwrap(),
        y.as_slice().unwrap(),
        "tsallis",
        &kwargs,
    )
    .expect("python kld tsallis");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-8);
}

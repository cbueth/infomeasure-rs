// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Python parity tests for the Jensen–Shannon divergence.

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::bayes::AlphaParam;
use infomeasure::estimators::composite_measures::{jsd, jsd_kernel_1d};
use infomeasure::estimators::entropy::Entropy;
use ndarray::{Array1, array};
use rstest::rstest;
use validation::python;

#[rstest]
#[case(vec![1, 1, 2, 2, 3, 3, 3, 4], vec![1, 1, 1, 2, 3, 4, 4, 4])]
#[case(vec![0, 0, 1, 2, 2, 2, 3], vec![0, 1, 1, 1, 2, 3, 3])]
#[case(vec![5, 5, 6, 6, 7, 7, 7, 8], vec![5, 6, 6, 6, 7, 8, 8, 8])]
fn jsd_discrete_python_parity(#[case] p: Vec<i32>, #[case] q: Vec<i32>) {
    let d_rust = jsd(
        &[
            Entropy::new_discrete(Array1::from(p.clone())),
            Entropy::new_discrete(Array1::from(q.clone())),
        ],
        None,
    );
    let d_py = python::calculate_jsd(&[p, q], "discrete", &[]).expect("python jsd discrete");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 2, 2, 3, 3, 3, 4], vec![1, 1, 1, 2, 3, 4, 4, 4])]
#[case(vec![0, 0, 1, 2, 2, 2, 3], vec![0, 1, 1, 1, 2, 3, 3])]
#[case(vec![5, 5, 6, 6, 7, 7, 7, 8], vec![5, 6, 6, 6, 7, 8, 8, 8])]
fn jsd_bayes_python_parity(#[case] p: Vec<i32>, #[case] q: Vec<i32>) {
    let d_rust = jsd(
        &[
            Entropy::new_bayes(Array1::from(p.clone()), AlphaParam::Laplace, None),
            Entropy::new_bayes(Array1::from(q.clone()), AlphaParam::Laplace, None),
        ],
        None,
    );
    let kwargs = [("alpha".to_string(), "\"laplace\"".to_string())];
    let d_py = python::calculate_jsd(&[p, q], "bayes", &kwargs).expect("python jsd bayes");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 2, 2, 3, 3, 3, 4], vec![1, 1, 1, 2, 3, 4, 4, 4])]
#[case(vec![0, 0, 1, 2, 2, 2, 3], vec![0, 1, 1, 1, 2, 3, 3])]
#[case(vec![5, 5, 6, 6, 7, 7, 7, 8], vec![5, 6, 6, 6, 7, 8, 8, 8])]
fn jsd_shrink_python_parity(#[case] p: Vec<i32>, #[case] q: Vec<i32>) {
    let d_rust = jsd(
        &[
            Entropy::new_shrink(Array1::from(p.clone())),
            Entropy::new_shrink(Array1::from(q.clone())),
        ],
        None,
    );
    let d_py = python::calculate_jsd(&[p, q], "shrink", &[]).expect("python jsd shrink");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-10);
}

#[rstest]
fn jsd_ordinal_python_parity(#[values(2, 3, 4)] order: usize) {
    let x = Array1::from(vec![
        1.0, 3.0, 2.0, 4.5, 3.5, 5.0, 2.5, 6.0, 4.0, 5.5, 3.0, 6.5,
    ]);
    let y = Array1::from(vec![
        1.2, 2.8, 2.4, 4.1, 3.8, 5.3, 2.2, 6.4, 4.3, 5.1, 3.4, 6.2,
    ]);
    let d_rust = jsd(
        &[
            Entropy::new_ordinal(x.clone(), order),
            Entropy::new_ordinal(y.clone(), order),
        ],
        None,
    );
    let kwargs = [
        ("embedding_dim".to_string(), order.to_string()),
        ("stable".to_string(), "True".to_string()),
    ];
    let d_py = python::calculate_jsd(&[x.to_vec(), y.to_vec()], "ordinal", &kwargs)
        .expect("python jsd ordinal");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-10);
}

#[rstest]
#[case("box", 0.5)]
#[case("gaussian", 1.0)]
#[case("gaussian", 2.0)]
fn jsd_kernel_python_parity(#[case] kernel: &str, #[case] bandwidth: f64) {
    let x = array![1.0, 2.0, 2.5, 4.0, 5.0, 5.5, 6.0, 7.0];
    let y = array![1.5, 2.2, 3.0, 4.4, 5.5, 6.0, 6.8, 7.5];
    let d_rust = jsd_kernel_1d(&[x.clone(), y.clone()], None, bandwidth, kernel);
    let kwargs = [
        ("kernel".to_string(), format!("\"{kernel}\"")),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let d_py = python::calculate_jsd(&[x.to_vec(), y.to_vec()], "kernel", &kwargs)
        .expect("python jsd kernel");
    assert_abs_diff_eq!(d_rust, d_py, epsilon = 1e-8);
}

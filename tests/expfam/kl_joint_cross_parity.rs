// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
use infomeasure::estimators::{CrossEntropy, JointEntropy};
use ndarray::{Array1, Array2, array};
use rstest::rstest;
use validation::python;

fn flat_from_array2(a: &Array2<f64>) -> Vec<f64> {
    let mut v = Vec::with_capacity(a.len());
    for r in 0..a.nrows() {
        for c in 0..a.ncols() {
            v.push(a[(r, c)]);
        }
    }
    v
}

#[rstest]
#[case(3)]
#[case(5)]
fn kl_joint_python_parity_2d(#[case] k: usize) {
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    let y = Array1::from(vec![0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]);

    let series = [x.clone(), y.clone()];

    // Rust Joint Entropy
    let h_rust = KozachenkoLeonenkoEntropy::<2>::joint_entropy(&series, (k, 0.0));

    // Python Joint Entropy
    let mut joined = Array2::zeros((x.len(), 2));
    for i in 0..x.len() {
        joined[[i, 0]] = x[i];
        joined[[i, 1]] = y[i];
    }
    let flat = flat_from_array2(&joined);
    let kwargs = vec![
        ("k".to_string(), format!("{k}")),
        ("minkowski_p".to_string(), "2".to_string()),
    ];
    let h_py =
        python::calculate_entropy_float_nd(&flat, 2, "kl", &kwargs).expect("python kl failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
}

#[rstest]
#[case(3)]
#[case(4)]
fn kl_cross_python_parity_1d(#[case] k: usize) {
    let p_data = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let q_data = Array1::from(vec![0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]);

    let est_p = KozachenkoLeonenkoEntropy::<1>::new_1d(p_data.clone(), k, 0.0);
    let est_q = KozachenkoLeonenkoEntropy::<1>::new_1d(q_data.clone(), k, 0.0);

    let h_rust = est_p.cross_entropy(&est_q);

    let kwargs = vec![
        ("k".to_string(), format!("{k}")),
        ("minkowski_p".to_string(), "2".to_string()),
    ];
    let h_py = python::calculate_cross_entropy_float_nd(
        p_data.as_slice().unwrap(),
        q_data.as_slice().unwrap(),
        1,
        "kl",
        &kwargs,
    )
    .expect("python cross kl failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
}

#[rstest]
#[case(3)]
#[case(5)]
fn kl_cross_python_parity_2d(#[case] k: usize) {
    let p_data: Array2<f64> = array![
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
        [6.0, 6.0],
        [7.0, 7.0]
    ];
    let q_data: Array2<f64> = array![
        [0.1, 0.1],
        [1.1, 1.1],
        [2.1, 2.1],
        [3.1, 3.1],
        [4.1, 4.1],
        [5.1, 5.1],
        [6.1, 6.1],
        [7.1, 7.1]
    ];

    let est_p = KozachenkoLeonenkoEntropy::<2>::new(p_data.clone(), k, 0.0);
    let est_q = KozachenkoLeonenkoEntropy::<2>::new(q_data.clone(), k, 0.0);

    let h_rust = est_p.cross_entropy(&est_q);

    let flat_p = flat_from_array2(&p_data);
    let flat_q = flat_from_array2(&q_data);

    let kwargs = vec![
        ("k".to_string(), format!("{k}")),
        ("minkowski_p".to_string(), "2".to_string()),
    ];
    let h_py = python::calculate_cross_entropy_float_nd(&flat_p, &flat_q, 2, "kl", &kwargs)
        .expect("python cross kl failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
}

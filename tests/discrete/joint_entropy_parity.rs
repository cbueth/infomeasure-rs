// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::entropy::Entropy;
use ndarray::Array1;
use rstest::*;
use validation::python;

#[rstest]
#[case(vec![1, 2, 1, 3, 2, 1], vec![0, 1, 0, 1, 0, 1], "simple joint")]
#[case(vec![0, 0, 1, 1], vec![0, 1, 0, 1], "full combination")]
#[case(vec![1, 1, 1], vec![2, 2, 2], "constant joint")]
#[case(vec![1, 2, 3], vec![1, 2, 3], "identical joint")]
#[case(vec![1, 1, 2, 2, 3, 3], vec![1, 2, 3, 1, 2, 3], "independent variables")]
#[case(vec![0, 1, 0, 1, 0, 1, 0, 1], vec![0, 0, 1, 1, 2, 2, 3, 3], "binary and quad")]
#[case(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![8, 7, 6, 5, 4, 3, 2, 1], "all unique joint")]
fn joint_discrete_entropy_python_parity(
    #[case] x: Vec<i32>,
    #[case] y: Vec<i32>,
    #[case] _description: &str,
) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());

    // Rust calculation via Entropy facade
    let h_rust = Entropy::joint_discrete(&[x_arr, y_arr], ());

    // Python calculation
    // calculate_entropy_generic handles multiple series by passing them as a tuple to the Python estimator
    let h_py = python::calculate_entropy_generic(&[x, y], "discrete", &[])
        .expect("python joint discrete failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

#[rstest]
#[case(
    vec![
        vec![1, 2, 1, 3, 2, 1],
        vec![0, 1, 0, 1, 0, 1],
        vec![1, 1, 2, 2, 3, 3]
    ],
    "triple joint"
)]
#[case(
    vec![
        vec![0, 0, 1, 1, 0, 0, 1, 1],
        vec![0, 1, 0, 1, 0, 1, 0, 1],
        vec![0, 0, 0, 0, 1, 1, 1, 1]
    ],
    "full binary cube"
)]
#[case(
    vec![
        vec![1, 2, 3, 4],
        vec![1, 1, 2, 2],
        vec![1, 1, 1, 1]
    ],
    "mixed dependencies"
)]
#[case(
    vec![
        vec![1, 2, 3, 1, 2, 3, 1, 2, 3],
        vec![1, 1, 1, 2, 2, 2, 3, 3, 3],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2]
    ],
    "independent triple"
)]
fn joint_discrete_entropy_triple_python_parity(
    #[case] series: Vec<Vec<i32>>,
    #[case] _description: &str,
) {
    let rust_series: Vec<Array1<i32>> = series.iter().map(|v| Array1::from(v.clone())).collect();

    let h_rust = Entropy::joint_discrete(&rust_series, ());

    let h_py = python::calculate_entropy_generic(&series, "discrete", &[])
        .expect("python triple joint discrete failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

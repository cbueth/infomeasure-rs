// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};

use infomeasure::estimators::approaches::expfam::tsallis::TsallisEntropy;
use infomeasure::estimators::entropy::Entropy;
use infomeasure::estimators::{CrossEntropy, GlobalValue, JointEntropy, OptionalLocalValues};
use rstest::rstest;
use validation::python;

fn python_tsallis_entropy(data: &Array2<f64>, k: usize, q: f64) -> f64 {
    // Use validation crate to call Python infomeasure implementation (no numpy dependency here)
    use validation::python;
    // Flatten data row-major
    let mut flat: Vec<f64> = Vec::with_capacity(data.len());
    for r in 0..data.nrows() {
        for c in 0..data.ncols() {
            flat.push(data[(r, c)]);
        }
    }
    let dims = data.ncols();
    let kwargs = vec![
        ("k".to_string(), k.to_string()),
        ("q".to_string(), q.to_string()),
    ];
    python::calculate_entropy_float_nd(&flat, dims, "tsallis", &kwargs)
        .expect("python tsallis failed")
}

#[test]
fn tsallis_python_parity_1d() {
    // Simple 1D dataset
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0]);
    let data = x.into_shape_with_order((6, 1)).unwrap();

    for &(k, q) in &[(1usize, 0.5f64), (2, 0.5), (1, 2.0), (3, 2.0)] {
        let est = TsallisEntropy::<1>::new(data.clone(), k, q, 0.0);
        let h_rust = est.global_value();
        let h_py = python_tsallis_entropy(&data, k, q);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
        assert!(est.local_values_opt().is_err());
    }
}

#[test]
fn tsallis_python_parity_2d() {
    // Simple 2D dataset
    let data = Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0],
    )
    .unwrap();

    for &(k, q) in &[(1usize, 0.5f64), (2, 0.5), (1, 2.0)] {
        let est = TsallisEntropy::<2>::new(data.clone(), k, q, 0.0);
        let h_rust = est.global_value();
        let h_py = python_tsallis_entropy(&data, k, q);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    }
}

#[test]
fn facade_nd_smoke() {
    // Verify facade N-D constructors compile and produce same value as direct new()
    let data2 =
        Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();

    let k = 1usize;
    let alpha = 0.5f64;
    let q = 0.5f64;
    let renyi_direct = infomeasure::estimators::approaches::expfam::renyi::RenyiEntropy::<2>::new(
        data2.clone(),
        k,
        alpha,
        0.0,
    )
    .global_value();
    let renyi_facade = Entropy::renyi_nd::<2>(data2.clone(), k, alpha, 0.0).global_value();
    assert_abs_diff_eq!(renyi_direct, renyi_facade, epsilon = 1e-12);

    let tsallis_direct = TsallisEntropy::<2>::new(data2.clone(), k, q, 0.0).global_value();
    let tsallis_facade = Entropy::tsallis_nd::<2>(data2.clone(), k, q, 0.0).global_value();
    assert_abs_diff_eq!(tsallis_direct, tsallis_facade, epsilon = 1e-12);
}

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
#[case(3, 0.5)]
#[case(5, 2.0)]
fn tsallis_joint_python_parity_2d(#[case] k: usize, #[case] q: f64) {
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]);
    let y = Array1::from(vec![0.1, 1.2, 2.9, 6.1, 9.8, 15.2, 21.1, 27.9]);

    let series = [x.clone(), y.clone()];

    // Rust Joint Entropy
    let h_rust = TsallisEntropy::<2>::joint_entropy(&series, (k, q, 0.0));

    // Python Joint Entropy
    let mut joined = Array2::zeros((x.len(), 2));
    for i in 0..x.len() {
        joined[[i, 0]] = x[i];
        joined[[i, 1]] = y[i];
    }
    let flat = flat_from_array2(&joined);
    let kwargs = vec![
        ("k".to_string(), format!("{k}")),
        ("q".to_string(), format!("{q}")),
    ];
    let h_py = python::calculate_entropy_float_nd(&flat, 2, "tsallis", &kwargs)
        .expect("python tsallis failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
}

#[rstest]
#[case(3, 0.5)]
#[case(4, 2.0)]
fn tsallis_cross_python_parity_1d(#[case] k: usize, #[case] q: f64) {
    let p_data = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let q_data = Array1::from(vec![0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]);

    let est_p = TsallisEntropy::<1>::new_1d(p_data.clone(), k, q, 0.0);
    let est_q = TsallisEntropy::<1>::new_1d(q_data.clone(), k, q, 0.0);

    let h_rust = est_p.cross_entropy(&est_q);

    let kwargs = vec![
        ("k".to_string(), format!("{k}")),
        ("q".to_string(), format!("{q}")),
    ];
    let h_py = python::calculate_cross_entropy_float_nd(
        p_data.as_slice().unwrap(),
        q_data.as_slice().unwrap(),
        1,
        "tsallis",
        &kwargs,
    )
    .expect("python cross tsallis failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
}

#[rstest]
#[case(3, 0.5)]
#[case(5, 2.0)]
fn tsallis_cross_python_parity_2d(#[case] k: usize, #[case] q: f64) {
    let p_data: Array2<f64> = ndarray::array![
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
        [6.0, 6.0],
        [7.0, 7.0]
    ];
    let q_data: Array2<f64> = ndarray::array![
        [0.1, 0.1],
        [1.1, 1.1],
        [2.1, 2.1],
        [3.1, 3.1],
        [4.1, 4.1],
        [5.1, 5.1],
        [6.1, 6.1],
        [7.1, 7.1]
    ];

    let est_p = TsallisEntropy::<2>::new(p_data.clone(), k, q, 0.0);
    let est_q = TsallisEntropy::<2>::new(q_data.clone(), k, q, 0.0);

    let h_rust = est_p.cross_entropy(&est_q);

    let flat_p = flat_from_array2(&p_data);
    let flat_q = flat_from_array2(&q_data);

    let kwargs = vec![
        ("k".to_string(), format!("{k}")),
        ("q".to_string(), format!("{q}")),
    ];
    let h_py = python::calculate_cross_entropy_float_nd(&flat_p, &flat_q, 2, "tsallis", &kwargs)
        .expect("python cross tsallis failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
}

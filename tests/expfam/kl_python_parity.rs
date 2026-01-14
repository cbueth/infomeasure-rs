// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};

use infomeasure::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
use infomeasure::estimators::{GlobalValue, LocalValues};
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

#[test]
fn kl_python_parity_param_grid_nd() {
    // 1D data
    let data_1d = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0]);
    let data_1d_2d = data_1d
        .clone()
        .into_shape_with_order((data_1d.len(), 1))
        .unwrap();

    // 2D data
    let data_2d: Array2<f64> = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 5.0],
    ];

    // 3D data
    let data_3d: Array2<f64> = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 3.0],
    ];

    let ks = [1usize, 2, 3, 4];

    // 1D
    for &k in &ks {
        if k > data_1d_2d.nrows() - 1 {
            continue;
        }
        let est = KozachenkoLeonenkoEntropy::<1>::new(data_1d_2d.clone(), k, 0.0);
        let h_rust = est.global_value();
        let flat = flat_from_array2(&data_1d_2d);
        let kwargs = vec![
            ("k".to_string(), format!("{}", k)),
            ("minkowski_p".to_string(), "2".to_string()),
        ];
        let h_py =
            python::calculate_entropy_float_nd(&flat, 1, "kl", &kwargs).expect("python KL failed");
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
    }

    // 2D
    for &k in &ks {
        if k > data_2d.nrows() - 1 {
            continue;
        }
        let est = KozachenkoLeonenkoEntropy::<2>::new(data_2d.clone(), k, 0.0);
        let h_rust = est.global_value();
        let flat = flat_from_array2(&data_2d);
        let kwargs = vec![
            ("k".to_string(), format!("{}", k)),
            ("minkowski_p".to_string(), "2".to_string()),
        ];
        let h_py =
            python::calculate_entropy_float_nd(&flat, 2, "kl", &kwargs).expect("python KL failed");
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
    }

    // 3D
    for &k in &ks {
        if k > data_3d.nrows() - 1 {
            continue;
        }
        let est = KozachenkoLeonenkoEntropy::<3>::new(data_3d.clone(), k, 0.0);
        let h_rust = est.global_value();
        let flat = flat_from_array2(&data_3d);
        let kwargs = vec![
            ("k".to_string(), format!("{}", k)),
            ("minkowski_p".to_string(), "2".to_string()),
        ];
        let h_py =
            python::calculate_entropy_float_nd(&flat, 3, "kl", &kwargs).expect("python KL failed");
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
    }
}

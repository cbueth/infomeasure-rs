use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2, array};

use infomeasure::estimators::approaches::expfam::renyi::RenyiEntropy;
use infomeasure::estimators::approaches::expfam::tsallis::TsallisEntropy;
use infomeasure::estimators::GlobalValue;
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
fn renyi_python_validation_param_grid_nd() {
    // Datasets for dims 1, 2, 3
    let data_1d = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0]);
    let data_1d_2d = data_1d.clone().into_shape_with_order((data_1d.len(), 1)).unwrap();

    let data_2d: Array2<f64> = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 5.0],
    ];

    let data_3d: Array2<f64> = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 3.0],
    ];

    // Parameter grids
    let ks = [1usize, 2, 3];
    let alphas = [0.5_f64, 2.0_f64];
    // alpha = 1 (Shannon limit) covered separately with tighter tolerance

    // 1D
    for &k in &ks {
        if k > data_1d_2d.nrows() - 1 { continue; }
        for &alpha in &alphas {
            let est = RenyiEntropy::<1>::new(data_1d_2d.clone(), k, alpha, 0.0);
            let h_rust = est.global_value();
            // call Python via validation crate
            let flat = flat_from_array2(&data_1d_2d);
            let kwargs = vec![
                ("k".to_string(), format!("{}", k)),
                ("alpha".to_string(), format!("{}", alpha)),
            ];
            let h_py = python::calculate_entropy_float_nd(&flat, 1, "renyi", &kwargs)
                .expect("python renyi failed");
            assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
        }
        // Shannon limit alpha->1
        let alpha1 = 1.0_f64;
        let est = RenyiEntropy::<1>::new(data_1d_2d.clone(), k, alpha1, 0.0);
        let h_rust = est.global_value();
        let flat = flat_from_array2(&data_1d_2d);
        let kwargs = vec![
            ("k".to_string(), format!("{}", k)),
            ("alpha".to_string(), format!("{}", alpha1)),
        ];
        let h_py = python::calculate_entropy_float_nd(&flat, 1, "renyi", &kwargs)
            .expect("python renyi failed");
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-7);
    }

    // 2D
    for &k in &ks {
        if k > data_2d.nrows() - 1 { continue; }
        for &alpha in &alphas {
            let est = RenyiEntropy::<2>::new(data_2d.clone(), k, alpha, 0.0);
            let h_rust = est.global_value();
            let flat = flat_from_array2(&data_2d);
            let kwargs = vec![
                ("k".to_string(), format!("{}", k)),
                ("alpha".to_string(), format!("{}", alpha)),
            ];
            let h_py = python::calculate_entropy_float_nd(&flat, 2, "renyi", &kwargs)
                .expect("python renyi failed");
            assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
        }
        let alpha1 = 1.0_f64;
        let est = RenyiEntropy::<2>::new(data_2d.clone(), k, alpha1, 0.0);
        let h_rust = est.global_value();
        let flat = flat_from_array2(&data_2d);
        let kwargs = vec![
            ("k".to_string(), format!("{}", k)),
            ("alpha".to_string(), format!("{}", alpha1)),
        ];
        let h_py = python::calculate_entropy_float_nd(&flat, 2, "renyi", &kwargs)
            .expect("python renyi failed");
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-7);
    }

    // 3D
    for &k in &ks {
        if k > data_3d.nrows() - 1 { continue; }
        for &alpha in &alphas {
            let est = RenyiEntropy::<3>::new(data_3d.clone(), k, alpha, 0.0);
            let h_rust = est.global_value();
            let flat = flat_from_array2(&data_3d);
            let kwargs = vec![
                ("k".to_string(), format!("{}", k)),
                ("alpha".to_string(), format!("{}", alpha)),
            ];
            let h_py = python::calculate_entropy_float_nd(&flat, 3, "renyi", &kwargs)
                .expect("python renyi failed");
            assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-7);
        }
        let alpha1 = 1.0_f64;
        let est = RenyiEntropy::<3>::new(data_3d.clone(), k, alpha1, 0.0);
        let h_rust = est.global_value();
        let flat = flat_from_array2(&data_3d);
        let kwargs = vec![
            ("k".to_string(), format!("{}", k)),
            ("alpha".to_string(), format!("{}", alpha1)),
        ];
        let h_py = python::calculate_entropy_float_nd(&flat, 3, "renyi", &kwargs)
            .expect("python renyi failed");
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-6);
    }
}

#[test]
fn tsallis_python_validation_param_grid_nd() {
    // Reuse datasets above
    let data_1d = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0]);
    let data_1d_2d = data_1d.clone().into_shape_with_order((data_1d.len(), 1)).unwrap();

    let data_2d: Array2<f64> = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 5.0],
    ];

    let data_3d: Array2<f64> = array![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 3.0],
    ];

    let ks = [1usize, 2, 3];
    let qs = [0.5_f64, 2.0_f64];

    // 1D
    for &k in &ks {
        if k > data_1d_2d.nrows() - 1 { continue; }
        for &q in &qs {
            let est = TsallisEntropy::<1>::new(data_1d_2d.clone(), k, q, 0.0);
            let h_rust = est.global_value();
            let flat = flat_from_array2(&data_1d_2d);
            let kwargs = vec![
                ("k".to_string(), format!("{}", k)),
                ("q".to_string(), format!("{}", q)),
            ];
            let h_py = python::calculate_entropy_float_nd(&flat, 1, "tsallis", &kwargs)
                .expect("python tsallis failed");
            assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
        }
        // Shannon limit q->1
        let q1 = 1.0_f64;
        let est = TsallisEntropy::<1>::new(data_1d_2d.clone(), k, q1, 0.0);
        let h_rust = est.global_value();
        let flat = flat_from_array2(&data_1d_2d);
        let kwargs = vec![
            ("k".to_string(), format!("{}", k)),
            ("q".to_string(), format!("{}", q1)),
        ];
        let h_py = python::calculate_entropy_float_nd(&flat, 1, "tsallis", &kwargs)
            .expect("python tsallis failed");
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-7);
    }

    // 2D
    for &k in &ks {
        if k > data_2d.nrows() - 1 { continue; }
        for &q in &qs {
            let est = TsallisEntropy::<2>::new(data_2d.clone(), k, q, 0.0);
            let h_rust = est.global_value();
            let flat = flat_from_array2(&data_2d);
            let kwargs = vec![
                ("k".to_string(), format!("{}", k)),
                ("q".to_string(), format!("{}", q)),
            ];
            let h_py = python::calculate_entropy_float_nd(&flat, 2, "tsallis", &kwargs)
                .expect("python tsallis failed");
            assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
        }
        let q1 = 1.0_f64;
        let est = TsallisEntropy::<2>::new(data_2d.clone(), k, q1, 0.0);
        let h_rust = est.global_value();
        let flat = flat_from_array2(&data_2d);
        let kwargs = vec![
            ("k".to_string(), format!("{}", k)),
            ("q".to_string(), format!("{}", q1)),
        ];
        let h_py = python::calculate_entropy_float_nd(&flat, 2, "tsallis", &kwargs)
            .expect("python tsallis failed");
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-7);
    }

    // 3D
    for &k in &ks {
        if k > data_3d.nrows() - 1 { continue; }
        for &q in &qs {
            let est = TsallisEntropy::<3>::new(data_3d.clone(), k, q, 0.0);
            let h_rust = est.global_value();
            let flat = flat_from_array2(&data_3d);
            let kwargs = vec![
                ("k".to_string(), format!("{}", k)),
                ("q".to_string(), format!("{}", q)),
            ];
            let h_py = python::calculate_entropy_float_nd(&flat, 3, "tsallis", &kwargs)
                .expect("python tsallis failed");
            assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-7);
        }
        let q1 = 1.0_f64;
        let est = TsallisEntropy::<3>::new(data_3d.clone(), k, q1, 0.0);
        let h_rust = est.global_value();
        let flat = flat_from_array2(&data_3d);
        let kwargs = vec![
            ("k".to_string(), format!("{}", k)),
            ("q".to_string(), format!("{}", q1)),
        ];
        let h_py = python::calculate_entropy_float_nd(&flat, 3, "tsallis", &kwargs)
            .expect("python tsallis failed");
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-6);
    }
}

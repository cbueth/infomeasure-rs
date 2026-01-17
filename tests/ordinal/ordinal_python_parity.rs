// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use ndarray::Array1;

use infomeasure::estimators::approaches::ordinal::ordinal_estimator::OrdinalEntropy;
use infomeasure::estimators::approaches::ordinal::ordinal_utils::symbolize_series_u64;
use infomeasure::estimators::{GlobalValue, LocalValues};

fn python_ordinal_entropy(series: &Array1<f64>, order: usize) -> (f64, Option<Vec<f64>>) {
    // Use validation crate to call Python infomeasure ordinal estimator directly (no numpy dependency)
    use validation::python;
    let data = series.as_slice().unwrap();
    // Ordinal is 1D; pass approach "ordinal" with kwargs (Python ignores step_size)
    let kwargs = vec![
        ("embedding_dim".to_string(), order.to_string()),
        ("stable".to_string(), "True".to_string()),
    ];
    let h =
        python::calculate_entropy_float(data, "ordinal", &kwargs).expect("python ordinal failed");

    // Python implementation fails for order=1 local values
    let locals = if order > 1 {
        Some(
            python::calculate_local_entropy_float(data, "ordinal", &kwargs)
                .expect("python local ordinal failed"),
        )
    } else {
        None
    };
    (h, locals)
}

#[test]
fn ordinal_python_parity_basic_sets() {
    let cases: Vec<(Vec<f64>, usize, usize)> = vec![
        (vec![1., 2., 3., 2., 1.], 1, 1),
        (vec![1., 2., 3., 2., 1.], 2, 1),
        (vec![1., 2., 3., 2., 1.], 3, 1),
        (vec![0., 2., 4., 3., 1.], 3, 1),
        (vec![0., 1., 0., 1., 0.], 2, 1),
        (vec![3., 1., 2., 5., 4.], 3, 1),
        (vec![0., 1., 2., 3., 4., 5.], 2, 1),
        (
            vec![0., 7., 2., 3., 45., 7., 1., 8., 4., 5., 2., 7., 8.],
            2,
            1,
        ),
    ];

    for (data, order, _step_size) in cases.into_iter() {
        let series = Array1::from(data.clone());
        let rust_est = OrdinalEntropy::new(series.clone(), order);
        let h_rust = rust_est.global_value();
        let (h_py, locals_py_opt) = python_ordinal_entropy(&series, order);

        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);

        if let Some(locals_py) = locals_py_opt {
            let locals_rust = rust_est.local_values();
            assert_eq!(locals_rust.len(), locals_py.len());
            for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
                assert_abs_diff_eq!(*lr, *lp, epsilon = 1e-10);
            }

            // local mean parity
            if !locals_rust.is_empty() {
                assert_abs_diff_eq!(h_rust, locals_rust.mean().unwrap(), epsilon = 1e-12);
            }
        }
    }
}

#[test]
fn ordinal_python_parity_param_grid() {
    // Parameterized grid over different orders
    let series = Array1::from(vec![
        46., 43., 9., 17., 48., 34., 8., 17., 15., 23., 17., 1., 13., 43., 40., 28., 12., 45., 37.,
        20., 25., 44., 25., 26., 12., 33., 36., 11., 25., 23.,
    ]);
    let orders = [2usize, 3, 4, 5, 6];

    for &m in &orders {
        let rust_est = OrdinalEntropy::new(series.clone(), m);
        let h_rust = rust_est.global_value();
        let (h_py, locals_py_opt) = python_ordinal_entropy(&series, m);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);

        if let Some(locals_py) = locals_py_opt {
            let locals_rust = rust_est.local_values();
            assert_eq!(locals_rust.len(), locals_py.len());
            for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
                assert_abs_diff_eq!(*lr, *lp, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn parity_symbolize_series_grid_stable_true() {
    use validation::python;

    // Series without ties to avoid ambiguity when stable=False; here we use stable=True
    let series_list: Vec<Vec<f64>> = vec![
        vec![0., 1., 2., 3., 4., 5., 6., 7.],
        vec![0., 7., 2., 3., 45., 7.5, 1., 8., 4., 5., 2.5, 7.2, 8.1],
        vec![
            3., 1., 4., 1.5, 5., 9., 2., 6., 2., 5., 3.5, 5., 8., 9.7, 3.,
        ],
    ];
    let embedding_dims = [2usize, 3, 4, 5];
    let steps = [1usize, 2, 3];

    for data in series_list.iter() {
        let series = Array1::from(data.clone());
        for &m in &embedding_dims {
            for &tau in &steps {
                // Skip invalid combos where not enough length
                let span = (m - 1) * tau;
                if series.len() <= span {
                    continue;
                }

                let rust_codes = symbolize_series_u64(&series, m, tau, true)
                    .iter()
                    .map(|&x| x as i64)
                    .collect::<Vec<_>>();
                let py_codes = python::calculate_symbolize_series(
                    series.as_slice().unwrap(),
                    m,
                    tau,
                    true,
                    true,
                )
                .unwrap();
                assert_eq!(
                    rust_codes.len(),
                    py_codes.len(),
                    "length mismatch for m={m}, tau={tau}"
                );
                for (i, (r, p)) in rust_codes.iter().zip(py_codes.iter()).enumerate() {
                    assert_eq!(*r, *p, "code mismatch at idx {i} for m={m}, tau={tau}");
                }
            }
        }
    }
}

#[test]
fn parity_symbolize_series_stable_false_no_ties() {
    use validation::python;

    // data without ties to avoid ambiguity when stable=False
    let series = Array1::from(vec![4., 1., 3.2, 99., 3., 2., 9., 6., 7., 3.1]);
    let embedding_dims = [2usize, 3, 4];
    let steps = [1usize, 2, 3];

    for &m in &embedding_dims {
        for &tau in &steps {
            if series.len() <= (m - 1) * tau {
                continue;
            }
            let rust_codes = symbolize_series_u64(&series, m, tau, false)
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>();
            let py_codes =
                python::calculate_symbolize_series(series.as_slice().unwrap(), m, tau, true, false)
                    .unwrap();
            assert_eq!(
                rust_codes.len(),
                py_codes.len(),
                "length mismatch for m={m}, tau={tau}"
            );
            for (i, (r, p)) in rust_codes.iter().zip(py_codes.iter()).enumerate() {
                assert_eq!(*r, *p, "code mismatch at idx {i} for m={m}, tau={tau}");
            }
        }
    }
}

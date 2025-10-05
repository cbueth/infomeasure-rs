use approx::assert_abs_diff_eq;
use ndarray::Array1;

use infomeasure::estimators::approaches::ordinal::ordinal::OrdinalEntropy;
use infomeasure::estimators::traits::LocalValues;

fn python_ordinal_entropy(series: &Array1<f64>, order: usize) -> f64 {
    // Use validation crate to call Python infomeasure ordinal estimator directly (no numpy dependency)
    use validation::python;
    let data = series.as_slice().unwrap();
    // Ordinal is 1D; pass approach "ordinal" with kwargs (Python ignores step_size)
    let kwargs = vec![
        ("embedding_dim".to_string(), order.to_string()),
        ("stable".to_string(), "True".to_string()),
    ];
    python::calculate_entropy_float(data, "ordinal", &kwargs)
        .expect("python ordinal failed")
}

#[test]
fn ordinal_python_parity_basic_sets() {
    let cases: Vec<(Vec<f64>, usize, usize)> = vec![
        (vec![1.,2.,3.,2.,1.], 1, 1),
        (vec![1.,2.,3.,2.,1.], 2, 1),
        (vec![1.,2.,3.,2.,1.], 3, 1),
        (vec![0., 2., 4., 3., 1.], 3, 1),
        (vec![0., 1., 0., 1., 0.], 2, 1),
        (vec![3., 1., 2., 5., 4.], 3, 1),
        (vec![0., 1., 2., 3., 4., 5.], 2, 1),
        (vec![0., 7., 2., 3., 45., 7., 1., 8., 4., 5., 2., 7., 8.], 2, 1),
    ];

    for (data, order, _step_size) in cases.into_iter() {
        let series = Array1::from(data.clone());
        let rust_est = OrdinalEntropy::new(series.clone(), order);
        let h_rust = rust_est.global_value();
        let h_py = python_ordinal_entropy(&series, order);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
        // local mean parity
        if rust_est.local_values().len() > 0 {
            assert_abs_diff_eq!(h_rust, rust_est.local_values().mean().unwrap(), epsilon = 1e-12);
        }
    }
}

#[test]
fn ordinal_python_parity_param_grid() {
    // Parameterized grid over different orders
    let series = Array1::from(vec![
        46., 43.,  9., 17., 48., 34.,  8., 17., 15., 23., 17.,  1., 13., 43., 40., 28., 12., 45., 37., 20., 25., 44., 25., 26., 12., 33., 36., 11., 25., 23.]);
    let orders = [2usize, 3, 4, 5, 6];

    for &m in &orders {
        let rust_est = OrdinalEntropy::new(series.clone(), m);
        let h_rust = rust_est.global_value();
        let h_py = python_ordinal_entropy(&series, m);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    }
}

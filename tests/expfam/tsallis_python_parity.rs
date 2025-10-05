use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};
use std::process::{Command, Stdio};

use infomeasure::estimators::approaches::expfam::tsallis::TsallisEntropy;
use infomeasure::estimators::entropy::Entropy;
use infomeasure::estimators::traits::LocalValues;

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
    let data = x.into_shape_with_order((6,1)).unwrap();

    for &(k, q) in &[(1usize, 0.5f64), (2, 0.5), (1, 2.0), (3, 2.0)] {
        let est = TsallisEntropy::<1>::new(data.clone(), k, q);
        let h_rust = est.global_value();
        let h_py = python_tsallis_entropy(&data, k, q);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
        assert_eq!(est.local_values().len(), 0);
    }
}

#[test]
fn tsallis_python_parity_2d() {
    // Simple 2D dataset
    let data = Array2::from_shape_vec((5, 2), vec![
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 2.0,
    ]).unwrap();

    for &(k, q) in &[(1usize, 0.5f64), (2, 0.5), (1, 2.0)] {
        let est = TsallisEntropy::<2>::new(data.clone(), k, q);
        let h_rust = est.global_value();
        let h_py = python_tsallis_entropy(&data, k, q);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    }
}

#[test]
fn facade_nd_smoke() {
    // Verify facade N-D constructors compile and produce same value as direct new()
    let data2 = Array2::from_shape_vec((4, 2), vec![
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
    ]).unwrap();

    let k = 1usize; let alpha = 0.5f64; let q = 0.5f64;
    let renyi_direct = infomeasure::estimators::approaches::expfam::renyi::RenyiEntropy::<2>::new(data2.clone(), k, alpha).global_value();
    let renyi_facade = Entropy::renyi_nd::<2>(data2.clone(), k, alpha).global_value();
    assert_abs_diff_eq!(renyi_direct, renyi_facade, epsilon = 1e-12);

    let tsallis_direct = TsallisEntropy::<2>::new(data2.clone(), k, q).global_value();
    let tsallis_facade = Entropy::tsallis_nd::<2>(data2.clone(), k, q).global_value();
    assert_abs_diff_eq!(tsallis_direct, tsallis_facade, epsilon = 1e-12);
}

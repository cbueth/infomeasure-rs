use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};
use validation::python;

use infomeasure::estimators::approaches::expfam::renyi::RenyiEntropy;
use infomeasure::estimators::traits::LocalValues;

fn python_renyi_entropy(data: &Array2<f64>, k: usize, alpha: f64) -> f64 {
    // Serialize data as JSON 2D array
    let rows = data.nrows();
    let mut vec2d: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for i in 0..rows { vec2d.push(data.row(i).to_vec()); }
    let data_json = serde_json::to_string(&vec2d).unwrap();

    {
        let mut flat: Vec<f64> = Vec::with_capacity(data.len());
        for r in 0..data.nrows() {
            for c in 0..data.ncols() {
                flat.push(data[(r, c)]);
            }
        }
        let dims = data.ncols();
        let kwargs = vec![
            ("k".to_string(), k.to_string()),
            ("alpha".to_string(), alpha.to_string()),
        ];
        return python::calculate_entropy_float_nd(&flat, dims, "renyi", &kwargs)
            .expect("python renyi failed");
    }
}

#[test]
fn renyi_python_parity_1d() {
    // Simple 1D dataset
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0]);
    let data = x.into_shape_with_order((6,1)).unwrap();

    for &(k, alpha) in &[(1usize, 0.5f64), (2, 0.5), (1, 2.0), (3, 2.0)] {
        let est = RenyiEntropy::<1>::new(data.clone(), k, alpha);
        let h_rust = est.global_value();
        let h_py = python_renyi_entropy(&data, k, alpha);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
        // local values are not provided; ensure it's empty as per current design
        assert_eq!(est.local_values().len(), 0);
    }
}

#[test]
fn renyi_python_parity_2d() {
    // Simple 2D dataset
    let data = Array2::from_shape_vec((5, 2), vec![
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 2.0,
    ]).unwrap();

    for &(k, alpha) in &[(1usize, 0.5f64), (2, 0.5), (1, 2.0)] {
        let est = RenyiEntropy::<2>::new(data.clone(), k, alpha);
        let h_rust = est.global_value();
        let h_py = python_renyi_entropy(&data, k, alpha);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    }
}

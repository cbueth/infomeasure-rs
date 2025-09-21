use approx::assert_abs_diff_eq;
use ndarray::Array1;
use std::process::{Command, Stdio};
use std::io::Read;

use infomeasure::estimators::approaches::ordinal::ordinal::OrdinalEntropy;
use infomeasure::estimators::traits::LocalValues;

fn python_ordinal_entropy(series: &Array1<f64>, order: usize, step_size: usize) -> f64 {
    // Prepare JSON for Python
    let data_json = serde_json::to_string(series.as_slice().unwrap()).unwrap();

    let py_code = r#"
import sys, json
import numpy as np
from infomeasure.estimators.utils.ordinal import symbolize_series

data = json.loads(sys.argv[1])
order = int(sys.argv[2])
step_size = int(sys.argv[3])
arr = np.asarray(data, dtype=float)
patterns = symbolize_series(arr, order, step_size=step_size, to_int=True, stable=True)
if patterns.size == 0:
    print("0.0")
else:
    _, counts = np.unique(patterns, return_counts=True)
    p = counts / counts.sum()
    h = -np.sum(p * np.log(p))
    print(repr(float(h)))
"#;

    let mut cmd = Command::new("python3");
    cmd.arg("-c").arg(py_code).arg(&data_json).arg(order.to_string()).arg(step_size.to_string());
    // Ensure Python can import the bundled infomeasure-python package
    cmd.env("PYTHONPATH", "infomeasure-python");
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().expect("Failed to spawn python3");
    let output = child.wait_with_output().expect("Failed to wait on python3");

    if !output.status.success() {
        panic!(
            "Python parity helper failed. Status: {:?}\nSTDERR:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let s = String::from_utf8_lossy(&output.stdout);
    s.trim().parse::<f64>().expect("Failed to parse Python output as f64")
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

    for (data, order, step_size) in cases.into_iter() {
        let series = Array1::from(data.clone());
        let rust_est = OrdinalEntropy::new(series.clone(), order, step_size);
        let h_rust = rust_est.global_value();
        let h_py = python_ordinal_entropy(&series, order, step_size);
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
        // local mean parity
        if rust_est.local_values().len() > 0 {
            assert_abs_diff_eq!(h_rust, rust_est.local_values().mean().unwrap(), epsilon = 1e-12);
        }
    }
}

#[test]
fn ordinal_python_parity_param_grid() {
    // Parameterized grid over different orders and step sizes
    let series = Array1::from(vec![0., 7., 2., 3., 45., 7., 1., 8., 4., 5., 2., 7., 8., 5., 8., 0., 7., 1., 3., 51., 6., 7.]);
    let orders = [2usize, 3, 4];
    let steps = [1usize, 2];

    for &m in &orders {
        for &tau in &steps {
            let rust_est = OrdinalEntropy::new(series.clone(), m, tau);
            let h_rust = rust_est.global_value();
            let h_py = python_ordinal_entropy(&series, m, tau);
            assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
        }
    }
}

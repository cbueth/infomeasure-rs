use approx::assert_abs_diff_eq;
use ndarray::{Array1, Array2};
use std::process::{Command, Stdio};

use infomeasure::estimators::approaches::expfam::renyi::RenyiEntropy;
use infomeasure::estimators::traits::LocalValues;

fn python_renyi_entropy(data: &Array2<f64>, k: usize, alpha: f64) -> f64 {
    // Serialize data as JSON 2D array
    let rows = data.nrows();
    let mut vec2d: Vec<Vec<f64>> = Vec::with_capacity(rows);
    for i in 0..rows { vec2d.push(data.row(i).to_vec()); }
    let data_json = serde_json::to_string(&vec2d).unwrap();

    let py_code = r#"
import sys, json, math
import numpy as np

data = json.loads(sys.argv[1])
k = int(sys.argv[2])
alpha = float(sys.argv[3])
arr = np.asarray(data, dtype=float)
if arr.ndim == 1:
    arr = arr.reshape((-1, 1))
N, m = arr.shape
if N == 0 or k > N-1:
    print("0.0")
    raise SystemExit
# Pairwise Euclidean distances
D = np.sqrt(((arr[:, None, :] - arr[None, :, :])**2).sum(axis=2))
# Exclude self by setting diagonal to inf
np.fill_diagonal(D, np.inf)
# k-th smallest (0-indexed => k-1)
rho_k = np.partition(D, k-1, axis=1)[:, k-1]
# Unit ball volume
V_m = math.pi ** (m / 2.0) / math.gamma(m / 2.0 + 1.0)
# Special case where C_k becomes problematic
if abs(alpha - (k + 1)) < 1e-12:
    print("0.0")
    raise SystemExit
C_k = (math.gamma(k) / math.gamma(k + 1.0 - alpha)) ** (1.0 / (1.0 - alpha))
I_q = (N * C_k * V_m) ** (1.0 - alpha) * np.sum((rho_k[rho_k > 0] ** m) ** (1.0 - alpha)) / len(rho_k)
H = math.log(I_q) / (1.0 - alpha)
print(repr(float(H)))
"#;

    let mut cmd = Command::new("python3");
    cmd.arg("-c").arg(py_code).arg(&data_json).arg(k.to_string()).arg(alpha.to_string());
    cmd.env("PYTHONPATH", "infomeasure-python");
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let output = cmd.output().expect("Failed to run python3");
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
fn renyi_python_parity_1d() {
    // Simple 1D dataset
    let x = Array1::from(vec![0.0, 1.0, 3.0, 6.0, 10.0, 15.0]);
    let data = x.into_shape((6,1)).unwrap();

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

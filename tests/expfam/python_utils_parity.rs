use approx::assert_abs_diff_eq;
use ndarray::{array, Array2};

use infomeasure::estimators::approaches::expfam::utils::{
    unit_ball_volume,
    knn_radii,
    calculate_common_entropy_components,
};
use std::process::Command;

fn run_py(args: &[&str]) -> String {
    let status = Command::new("micromamba")
        .args(["run", "-n", "infomeasure-rs-validation", "python"])
        .args(args)
        .output()
        .expect("failed to run micromamba python");
    if !status.status.success() {
        panic!(
            "Python failed: {}",
            String::from_utf8_lossy(&status.stderr)
        );
    }
    String::from_utf8_lossy(&status.stdout).to_string()
}

fn py_unit_ball_volume(m: usize) -> f64 {
    let code = format!(
        "from infomeasure.estimators.utils.unit_ball_volume import unit_ball_volume as ubv\nprint(ubv({}))",
        m
    );
    let out = run_py(&["-c", &code]);
    out.trim().parse().expect("parse py unit_ball_volume")
}

fn py_knn_radii(data: &Array2<f64>, k: usize) -> Vec<f64> {
    let mut rows = Vec::with_capacity(data.nrows());
    for r in 0..data.nrows() { rows.push(data.row(r).to_vec()); }
    let json = serde_json::to_string(&rows).unwrap();
    let code = r#"
import sys, json
import numpy as np
from infomeasure.estimators.utils.exponential_family import knn_radii as py_knn
X = np.asarray(json.loads(sys.argv[1]), dtype=float)
k = int(sys.argv[2])
res = py_knn(X, k)
print(json.dumps(list(map(float, res))))
"#;
    let out = run_py(&["-c", code, &json, &k.to_string()]);
    serde_json::from_str(out.trim()).expect("parse py knn_radii")
}

fn py_common_components(data: &Array2<f64>, k: usize) -> (f64, Vec<f64>, usize, usize) {
    let mut rows = Vec::with_capacity(data.nrows());
    for r in 0..data.nrows() { rows.push(data.row(r).to_vec()); }
    let json = serde_json::to_string(&rows).unwrap();
    let code = r#"
import sys, json
import numpy as np
from infomeasure.estimators.utils.exponential_family import calculate_common_entropy_components as calc
X = np.asarray(json.loads(sys.argv[1]), dtype=float)
k = int(sys.argv[2])
V_m, rho_k, N, m = calc(X, k)
print(json.dumps({"V_m": float(V_m), "rho_k": list(map(float, rho_k)), "N": int(N), "m": int(m)}))
"#;
    let out = run_py(&["-c", code, &json, &k.to_string()]);
    let v: serde_json::Value = serde_json::from_str(out.trim()).expect("parse py common");
    let v_m = v.get("V_m").and_then(|x| x.as_f64()).expect("V_m f64");
    let n = v.get("N").and_then(|x| x.as_u64()).unwrap() as usize;
    let m = v.get("m").and_then(|x| x.as_u64()).unwrap() as usize;
    let rho_k = v.get("rho_k").and_then(|x| x.as_array()).unwrap()
        .iter().map(|e| e.as_f64().unwrap()).collect::<Vec<_>>();
    (v_m, rho_k, n, m)
}

#[test]
fn parity_unit_ball_volume_against_python() {
    for m in 1..=6 {
        let rust = unit_ball_volume(m);
        let py = py_unit_ball_volume(m);
        assert_abs_diff_eq!(rust, py, epsilon = 1e-12);
    }
}


#[test]
fn parity_common_components_against_python() {
    // 1D sample
    let d1: Array2<f64> = array![[0.0],[1.0],[3.0],[6.0],[10.0]];
    let (v_m_r, rho_r, n_r, m_r) = calculate_common_entropy_components::<1>(d1.view(), 2);
    let (v_m_p, rho_p, n_p, m_p) = py_common_components(&d1, 2);
    assert_abs_diff_eq!(v_m_r, v_m_p, epsilon = 1e-12);
    assert_eq!(n_r, n_p);
    assert_eq!(m_r, m_p);
    assert_eq!(rho_r.len(), rho_p.len());
    for (a,b) in rho_r.iter().zip(rho_p.iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
    }

    // 2D sample
    let d2: Array2<f64> = array![[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]];
    let (v_m_r2, rho_r2, n_r2, m_r2) = calculate_common_entropy_components::<2>(d2.view(), 2);
    let (v_m_p2, rho_p2, n_p2, m_p2) = py_common_components(&d2, 2);
    assert_abs_diff_eq!(v_m_r2, v_m_p2, epsilon = 1e-12);
    assert_eq!(n_r2, n_p2);
    assert_eq!(m_r2, m_p2);
    assert_eq!(rho_r2.len(), rho_p2.len());
    for (a,b) in rho_r2.iter().zip(rho_p2.iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-12);
    }
}

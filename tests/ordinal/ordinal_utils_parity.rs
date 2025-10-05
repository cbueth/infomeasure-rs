use approx::assert_abs_diff_eq;
use ndarray::Array1;

use infomeasure::estimators::approaches::ordinal::ordinal_utils::{symbolize_series, symbolize_series_u64};

fn run_python_symbolize(series: &[f64], emb_dim: usize, step_size: usize, to_int: bool, stable: bool) -> Vec<i64> {
    // Call Python's utils.symbolize_series via micromamba environment and return integer codes
    let json = serde_json::to_string(series).unwrap();
    let code = r#"
import sys, json
import numpy as np
from infomeasure.estimators.utils.ordinal import symbolize_series

series = np.asarray(json.loads(sys.argv[1]), dtype=float)
emb_dim = int(sys.argv[2])
step = int(sys.argv[3])
to_int = sys.argv[4].lower() == 'true'
stable = sys.argv[5].lower() == 'true'
res = symbolize_series(series, emb_dim, step, to_int=to_int, stable=stable)
print(json.dumps(list(map(int, res.tolist()))))
"#;
    let out = std::process::Command::new("micromamba")
        .args(["run", "-n", "infomeasure-rs-validation", "python", "-c", code, &json, &emb_dim.to_string(), &step_size.to_string(), &to_int.to_string(), &stable.to_string()])
        .output()
        .expect("failed to run micromamba python");
    if !out.status.success() {
        panic!(
            "Python failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let s = String::from_utf8_lossy(&out.stdout).to_string();
    serde_json::from_str::<Vec<i64>>(s.trim()).expect("parse python symbolize_series result")
}

#[test]
fn parity_symbolize_series_grid_stable_true() {
    // Series without ties to avoid ambiguity when stable=False; here we use stable=True
    let series_list: Vec<Vec<f64>> = vec![
        vec![0., 1., 2., 3., 4., 5., 6., 7.],
        vec![0., 7., 2., 3., 45., 7.5, 1., 8., 4., 5., 2.5, 7.2, 8.1],
        vec![3., 1., 4., 1.5, 5., 9., 2., 6., 5., 3.5, 5., 8., 9.7, 3.],
    ];
    let embedding_dims = [2usize, 3, 4, 5];
    let steps = [1usize, 2, 3];

    for data in series_list.iter() {
        let series = Array1::from(data.clone());
        for &m in &embedding_dims {
            for &tau in &steps {
                // Skip invalid combos where not enough length
                let span = (m - 1) * tau;
                if series.len() <= span { continue; }

                let rust_codes = symbolize_series_u64(&series, m, tau, true)
                    .iter().map(|&x| x as i64).collect::<Vec<_>>();
                let py_codes = run_python_symbolize(series.as_slice().unwrap(), m, tau, true, true);
                assert_eq!(rust_codes.len(), py_codes.len(), "length mismatch for m={}, tau={}", m, tau);
                for (i, (r, p)) in rust_codes.iter().zip(py_codes.iter()).enumerate() {
                    assert_eq!(*r, *p, "code mismatch at idx {} for m={}, tau={}", i, m, tau);
                }
            }
        }
    }
}

#[test]
fn parity_symbolize_series_stable_false_no_ties() {
    // Use strictly increasing data to avoid ties; stable flag should not matter
    let series = Array1::from(vec![0., 1., 2., 3., 4., 5., 6.]);
    let embedding_dims = [2usize, 3, 4];
    let steps = [1usize, 2];

    for &m in &embedding_dims {
        for &tau in &steps {
            if series.len() <= (m - 1) * tau { continue; }
            let rust_codes = symbolize_series(&series, m, tau, false)
                .iter().map(|&x| x as i64).collect::<Vec<_>>();
            let py_codes = run_python_symbolize(series.as_slice().unwrap(), m, tau, true, false);
            assert_eq!(rust_codes.len(), py_codes.len());
            for (r, p) in rust_codes.iter().zip(py_codes.iter()) {
                assert_eq!(*r, *p);
            }
        }
    }
}

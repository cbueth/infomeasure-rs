use infomeasure::estimators::approaches::ordinal::ordinal_utils::{
    remap_u64_to_i32, symbolize_series_u64,
};
use ndarray::{Array1, array};

fn run_python_symbolize(
    series: &[f64],
    emb_dim: usize,
    step_size: usize,
    to_int: bool,
    stable: bool,
) -> Vec<i64> {
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
        .args([
            "run",
            "-n",
            "infomeasure-rs-validation",
            "python",
            "-c",
            code,
            &json,
            &emb_dim.to_string(),
            &step_size.to_string(),
            &to_int.to_string(),
            &stable.to_string(),
        ])
        .output()
        .expect("failed to run micromamba python");
    if !out.status.success() {
        panic!("Python failed: {}", String::from_utf8_lossy(&out.stderr));
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
                let py_codes = run_python_symbolize(series.as_slice().unwrap(), m, tau, true, true);
                assert_eq!(
                    rust_codes.len(),
                    py_codes.len(),
                    "length mismatch for m={}, tau={}",
                    m,
                    tau
                );
                for (i, (r, p)) in rust_codes.iter().zip(py_codes.iter()).enumerate() {
                    assert_eq!(
                        *r, *p,
                        "code mismatch at idx {} for m={}, tau={}",
                        i, m, tau
                    );
                }
            }
        }
    }
}

#[test]
fn parity_symbolize_series_stable_false_no_ties() {
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
            let py_codes = run_python_symbolize(series.as_slice().unwrap(), m, tau, true, false);
            assert_eq!(
                rust_codes.len(),
                py_codes.len(),
                "length mismatch for m={}, tau={}",
                m,
                tau
            );
            for (i, (r, p)) in rust_codes.iter().zip(py_codes.iter()).enumerate() {
                assert_eq!(
                    *r, *p,
                    "code mismatch at idx {} for m={}, tau={}",
                    i, m, tau
                );
            }
        }
    }
}

#[test]
fn test_remap_u64_to_i32_parametrized() {
    let test_cases: [(&str, Array1<u64>, Array1<i32>); 6] = [
        (
            "basic",
            array![100, 200, 100, 300, 200, 400],
            array![0, 1, 0, 2, 1, 3],
        ),
        ("empty", Array1::<u64>::zeros(0), Array1::<i32>::zeros(0)),
        (
            "all_same",
            Array1::from_elem(5, 42u64),
            Array1::from_elem(5, 0i32),
        ),
        (
            "all_unique",
            array![10, 20, 30, 40, 50],
            array![0, 1, 2, 3, 4],
        ),
        (
            "first_occurrence_order",
            array![50, 10, 50, 30, 10, 30],
            array![0, 1, 0, 2, 1, 2],
        ),
        (
            "large_values",
            array![u64::MAX, u64::MIN, u64::MAX, 1234567890123456789u64],
            array![0, 1, 0, 2],
        ),
    ];

    for (name, input, expected) in test_cases.iter() {
        assert_eq!(
            remap_u64_to_i32(input),
            *expected,
            "test case {:?} failed",
            name
        );
    }
}

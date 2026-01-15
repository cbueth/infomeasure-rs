// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::GlobalValue;
use infomeasure::estimators::approaches::discrete::bayes::{AlphaParam, BayesEntropy};
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rstest::*;
use validation::python;

#[rstest]
#[case(vec![1, 1, 2, 3, 3, 4, 5], AlphaParam::Jeffrey, None, "mixed Jeffrey")]
#[case(vec![1, 1, 2], AlphaParam::Laplace, None, "simple Laplace")]
#[case(vec![1, 0, 1, 0], AlphaParam::Laplace, Some(3), "binary K=3")]
#[case(vec![1, 0, 1, 0], AlphaParam::Laplace, Some(5), "binary K=5")]
#[case(vec![1, 0, 1, 0], AlphaParam::Value(0.1), None, "binary alpha=0.1")]
#[case(vec![1, 0, 1, 0], AlphaParam::Value(2.0), None, "binary alpha=2.0")]
#[case(vec![1, 1, 1, 1, 1], AlphaParam::Laplace, None, "uniform Laplace")]
#[case(vec![0, 1, 0, 1, 2, 2], AlphaParam::SchGrass, None, "mixed SchGrass")]
#[case(vec![0, 1, 0, 1, 2, 2], AlphaParam::MinMax, None, "mixed MinMax")]
#[case(vec![1, 2, 3], AlphaParam::Laplace, None, "three symbols alpha=1.0")]
#[case(vec![1, 2, 3], AlphaParam::Jeffrey, None, "three symbols alpha=0.5")]
fn bayes_entropy_python_parity(
    #[case] data: Vec<i32>,
    #[case] alpha_param: AlphaParam,
    #[case] k_override: Option<usize>,
    #[case] _description: &str,
) {
    let arr = Array1::from(data.clone());
    let rust_est = BayesEntropy::new(arr, alpha_param.clone(), k_override);
    let h_rust = rust_est.global_value();

    let mut kwargs = Vec::new();

    let alpha_str = match alpha_param {
        AlphaParam::Value(v) => v.to_string(),
        AlphaParam::Jeffrey => "0.5".to_string(),
        AlphaParam::Laplace => "1.0".to_string(),
        AlphaParam::SchGrass => "\"sch-grass\"".to_string(),
        AlphaParam::MinMax => "\"min-max\"".to_string(),
    };
    kwargs.push(("alpha".to_string(), alpha_str));

    if let Some(k) = k_override {
        kwargs.push(("K".to_string(), k.to_string()));
    }

    let h_py = python::calculate_entropy(&data, "bayes", &kwargs).expect("python bayes failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], 0.0)]
fn test_bayes_mi_parity(#[case] x: Vec<i32>, #[case] y: Vec<i32>, #[case] ___expected: f64) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let mi_est = MutualInformation::new_discrete_bayes(&[x_arr, y_arr]);
    let res = mi_est.global_value();

    let kwargs = vec![
        ("alpha".to_string(), "0.5".to_string()), // Jeffrey is default in Rust new_discrete_bayes
    ];
    let mi_py = python::calculate_mi(&[x, y], "bayes", &kwargs).unwrap();
    assert_abs_diff_eq!(res, mi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![1, 0, 1, 0], 0.0)]
fn test_bayes_te_parity(
    #[case] source: Vec<i32>,
    #[case] dest: Vec<i32>,
    #[case] ___expected: f64,
) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let te_est = TransferEntropy::new_discrete_bayes(&s_arr, &d_arr, 1, 1, 1);
    let res = te_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
        ("alpha".to_string(), "0.5".to_string()),
    ];
    let te_py = python::calculate_te(&source, &dest, "bayes", &kwargs).unwrap();
    assert_abs_diff_eq!(res, te_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_bayes_cmi_parity(
    #[case] x: Vec<i32>,
    #[case] y: Vec<i32>,
    #[case] z: Vec<i32>,
    #[case] __expected: f64,
) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let z_arr = Array1::from(z.clone());
    let cmi_est = MutualInformation::new_cmi_discrete_bayes(&[x_arr, y_arr], &z_arr);
    let res = cmi_est.global_value();

    let kwargs = vec![("alpha".to_string(), "0.5".to_string())];
    let cmi_py = python::calculate_cmi(&[x, y], &z, "bayes", &kwargs).unwrap();
    assert_abs_diff_eq!(res, cmi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_bayes_cte_parity(
    #[case] source: Vec<i32>,
    #[case] dest: Vec<i32>,
    #[case] cond: Vec<i32>,
    #[case] __expected: f64,
) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let c_arr = Array1::from(cond.clone());
    let cte_est = TransferEntropy::new_cte_discrete_bayes(&s_arr, &d_arr, &c_arr, 1, 1, 1, 1);
    let res = cte_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
        ("cond_hist_len".to_string(), "1".to_string()),
        ("alpha".to_string(), "0.5".to_string()),
    ];
    let cte_py = python::calculate_cte(&source, &dest, &cond, "bayes", &kwargs).unwrap();
    assert_abs_diff_eq!(res, cte_py, epsilon = 1e-10);
}

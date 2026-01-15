// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::chao_wang_jost::ChaoWangJostEntropy;
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::traits::GlobalValue;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rstest::*;
use validation::python;

#[rstest]
#[case(vec![1, 1, 1, 1, 1], "uniform 5")]
#[case(vec![1, 0, 1, 0], "binary")]
#[case(vec![1, 2, 3, 4, 5], "all singletons 5")]
#[case(vec![1, 1, 2], "simple [2, 1]")]
#[case(vec![0, 0, 1, 1, 2, 2], "uniform counts [2, 2, 2]")]
#[case(vec![1, 2, 2, 3, 3, 3], "mixed counts")]
#[case(vec![1, 1, 1, 2], "skewed [3, 1]")]
#[case(vec![1, 2, 3, 3, 4, 4, 4], "tripleton")]
#[case(vec![1, 1, 2, 2, 3, 3, 4, 4, 5], "mostly doubletons")]
#[case(vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4], "increasing frequency")]
#[case(vec![1, 1, 1, 1, 2, 2, 3], "dominant group")]
#[case(vec![1, 2, 3, 4, 5, 6, 7, 8], "all singletons 8")]
fn chao_wang_jost_entropy_python_parity(#[case] data: Vec<i32>, #[case] _description: &str) {
    let arr = Array1::from(data.clone());
    let rust_est = ChaoWangJostEntropy::new(arr);
    let h_rust = rust_est.global_value();

    let h_py = python::calculate_entropy(&data, "chao_wang_jost", &[])
        .expect("python chao_wang_jost failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], 0.0)]
fn test_chao_wang_jost_mi_parity(
    #[case] x: Vec<i32>,
    #[case] y: Vec<i32>,
    #[case] __expected: f64,
) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let mi_est = MutualInformation::new_discrete_chao_wang_jost(&[x_arr, y_arr]);
    let res = mi_est.global_value();

    let mi_py = python::calculate_mi(&[x, y], "chao_wang_jost", &[]).unwrap();
    assert_abs_diff_eq!(res, mi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![1, 0, 1, 0], 0.0)]
fn test_chao_wang_jost_te_parity(
    #[case] source: Vec<i32>,
    #[case] dest: Vec<i32>,
    #[case] _expected: f64,
) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let te_est = TransferEntropy::new_discrete_chao_wang_jost(&s_arr, &d_arr, 1, 1, 1);
    let res = te_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
    ];
    let te_py = python::calculate_te(&source, &dest, "chao_wang_jost", &kwargs).unwrap();
    assert_abs_diff_eq!(res, te_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_chao_wang_jost_cmi_parity(
    #[case] x: Vec<i32>,
    #[case] y: Vec<i32>,
    #[case] z: Vec<i32>,
    #[case] _expected: f64,
) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let z_arr = Array1::from(z.clone());
    let cmi_est = MutualInformation::new_cmi_discrete_chao_wang_jost(&[x_arr, y_arr], &z_arr);
    let res = cmi_est.global_value();

    let cmi_py = python::calculate_cmi(&[x, y], &z, "chao_wang_jost", &[]).unwrap();
    assert_abs_diff_eq!(res, cmi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_chao_wang_jost_cte_parity(
    #[case] source: Vec<i32>,
    #[case] dest: Vec<i32>,
    #[case] cond: Vec<i32>,
    #[case] _expected: f64,
) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let c_arr = Array1::from(cond.clone());
    let cte_est =
        TransferEntropy::new_cte_discrete_chao_wang_jost(&s_arr, &d_arr, &c_arr, 1, 1, 1, 1);
    let res = cte_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
        ("cond_hist_len".to_string(), "1".to_string()),
    ];
    let cte_py = python::calculate_cte(&source, &dest, &cond, "chao_wang_jost", &kwargs).unwrap();
    assert_abs_diff_eq!(res, cte_py, epsilon = 1e-10);
}

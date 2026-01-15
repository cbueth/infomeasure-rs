// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::ansb::AnsbEntropy;
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::traits::GlobalValue;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rstest::*;
use validation::python;

#[rstest]
#[case(vec![1, 1, 2, 3, 4], None, "one coincidence")]
#[case(vec![1, 1, 1, 2, 2], None, "multiple coincidences")]
#[case(vec![1, 1, 2, 2, 3, 3], None, "pairs")]
#[case(vec![1, 1, 2, 3, 4, 5], None, "one pair mixed")]
#[case(vec![1, 1, 2, 3, 4], Some(3), "K=3 < observed K=4")]
#[case(vec![1, 1, 2, 2, 3, 3], Some(4), "K=4 < observed K=3 (pairs)")]
#[case(vec![1, 1, 1, 1], Some(1), "all same K=1")]
#[case(vec![1, 1, 2, 3], Some(2), "K=2 < observed K=3")]
#[case(vec![1, 2, 3, 4, 5], None, "no coincidences")]
#[case(vec![1, 1, 2, 3], Some(5), "K=5 > N=4 (negative coincidences)")]
#[case(vec![1, 2, 3, 4], Some(6), "no coincidences K=6 > N=4")]
fn ansb_entropy_python_parity(
    #[case] data: Vec<i32>,
    #[case] k_override: Option<usize>,
    #[case] _description: &str,
) {
    let arr = Array1::from(data.clone());
    let rust_est = AnsbEntropy::new(arr, k_override, 0.0);
    let h_rust = rust_est.global_value();

    let mut kwargs = Vec::new();
    if let Some(k) = k_override {
        kwargs.push(("K".to_string(), k.to_string()));
    }

    let h_py = python::calculate_entropy(&data, "ansb", &kwargs).expect("python ansb failed");

    if h_rust.is_nan() {
        assert!(
            h_py.is_nan(),
            "Rust returned NaN but Python returned {h_py}"
        );
    } else {
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    }
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], 0.0)]
fn test_ansb_mi_parity(#[case] x: Vec<i32>, #[case] y: Vec<i32>, #[case] __expected: f64) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let mi_est = MutualInformation::new_discrete_ansb(&[x_arr, y_arr]);
    let res = mi_est.global_value();

    let mi_py = python::calculate_mi(&[x, y], "ansb", &[]).unwrap();
    if res.is_nan() && mi_py.is_nan() {
        return;
    }
    assert_abs_diff_eq!(res, mi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 2, 0.0)]
#[case(vec![1, 0, 1, 0], vec![1, 0, 1, 0], 2, 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], 2, 0.0)]
fn test_ansb_te_parity(
    #[case] source: Vec<i32>,
    #[case] dest: Vec<i32>,
    #[case] base: i32,
    #[case] _expected: f64,
) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let te_est = TransferEntropy::new_discrete_ansb(&s_arr, &d_arr, 1, 1, 1);
    let res = te_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
        ("base".to_string(), base.to_string()),
    ];
    let te_py = python::calculate_te(&source, &dest, "ansb", &kwargs).unwrap();
    if res.is_nan() && te_py.is_nan() {
        return;
    }
    assert_abs_diff_eq!(res, te_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_ansb_cmi_parity(
    #[case] x: Vec<i32>,
    #[case] y: Vec<i32>,
    #[case] z: Vec<i32>,
    #[case] _expected: f64,
) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let z_arr = Array1::from(z.clone());
    let cmi_est = MutualInformation::new_cmi_discrete_ansb(&[x_arr, y_arr], &z_arr);
    let res = cmi_est.global_value();

    let cmi_py = python::calculate_cmi(&[x, y], &z, "ansb", &[]).unwrap();
    if res.is_nan() && cmi_py.is_nan() {
        return;
    }
    assert_abs_diff_eq!(res, cmi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_ansb_cte_parity(
    #[case] source: Vec<i32>,
    #[case] dest: Vec<i32>,
    #[case] cond: Vec<i32>,
    #[case] _expected: f64,
) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let c_arr = Array1::from(cond.clone());
    let cte_est = TransferEntropy::new_cte_discrete_ansb(&s_arr, &d_arr, &c_arr, 1, 1, 1, 1);
    let res = cte_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
        ("cond_hist_len".to_string(), "1".to_string()),
    ];
    let cte_py = python::calculate_cte(&source, &dest, &cond, "ansb", &kwargs).unwrap();
    if res.is_nan() && cte_py.is_nan() {
        return;
    }
    assert_abs_diff_eq!(res, cte_py, epsilon = 1e-10);
}

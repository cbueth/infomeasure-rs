// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use ndarray::Array1;
use rstest::rstest;
use validation::python;

use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::traits::GlobalValue;
use infomeasure::estimators::transfer_entropy::TransferEntropy;

#[rstest]
#[case(2, true)]
#[case(3, true)]
#[case(2, false)]
fn ordinal_mi_parity(#[case] order: usize, #[case] stable: bool) {
    let x = Array1::from(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
    let y = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

    let rust_est = MutualInformation::new_ordinal(&[x.clone(), y.clone()], order, 1, stable);
    let h_rust = rust_est.global_value();

    let stable_str = if stable { "True" } else { "False" };
    let kwargs = vec![
        ("embedding_dim".to_string(), order.to_string()),
        ("stable".to_string(), stable_str.to_string()),
    ];
    let h_py = python::calculate_mi_float(&[x.to_vec(), y.to_vec()], "ordinal", &kwargs).unwrap();

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

#[rstest]
#[case(2, true)]
#[case(3, true)]
#[case(2, false)]
fn ordinal_cmi_parity(#[case] order: usize, #[case] stable: bool) {
    let x = Array1::from(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
    let y = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    let z = Array1::from(vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);

    let rust_est =
        MutualInformation::new_cmi_ordinal(&[x.clone(), y.clone()], &z, order, 1, stable);
    let h_rust = rust_est.global_value();

    let stable_str = if stable { "True" } else { "False" };
    let kwargs = vec![
        ("embedding_dim".to_string(), order.to_string()),
        ("stable".to_string(), stable_str.to_string()),
    ];
    let h_py = python::calculate_cmi_float(
        &[x.to_vec(), y.to_vec()],
        z.as_slice().unwrap(),
        "ordinal",
        &kwargs,
    )
    .unwrap();

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

#[rstest]
#[case(2, true, 1, 1)]
#[case(3, true, 1, 1)]
#[case(2, false, 1, 1)]
#[case(2, true, 2, 1)]
#[case(2, true, 1, 2)]
fn ordinal_te_parity(
    #[case] order: usize,
    #[case] stable: bool,
    #[case] src_hist: usize,
    #[case] dest_hist: usize,
) {
    let source = Array1::from(vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]);
    let dest = Array1::from(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

    let rust_est =
        TransferEntropy::new_ordinal(&source, &dest, order, src_hist, dest_hist, 1, stable);
    let h_rust = rust_est.global_value();

    let stable_str = if stable { "True" } else { "False" };
    let kwargs = vec![
        ("embedding_dim".to_string(), order.to_string()),
        ("stable".to_string(), stable_str.to_string()),
        ("src_hist_len".to_string(), src_hist.to_string()),
        ("dest_hist_len".to_string(), dest_hist.to_string()),
    ];
    let h_py = python::calculate_te_float(
        source.as_slice().unwrap(),
        dest.as_slice().unwrap(),
        "ordinal",
        &kwargs,
    )
    .unwrap();

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

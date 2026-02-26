// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::disallowed_names, clippy::unnecessary_to_owned)]

use approx::assert_abs_diff_eq;
use ndarray::Array1;
use rstest::rstest;

use infomeasure::estimators::approaches::ordinal::ordinal_estimator::{
    OrdinalConditionalMutualInformation, OrdinalConditionalTransferEntropy,
    OrdinalMutualInformation, OrdinalTransferEntropy,
};
use infomeasure::estimators::approaches::ordinal::ordinal_utils::symbolize_series_u64;
use infomeasure::estimators::traits::GlobalValue;
use validation::python;

#[rstest]
fn ordinal_mi_parity(#[values(3, 4)] order: usize, #[values(1, 2)] step_size: usize) {
    let n = 100;
    let data1 = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1).sin()));
    let data2 = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1 + 0.5).cos()));

    let est =
        OrdinalMutualInformation::new(&[data1.clone(), data2.clone()], order, step_size, true);
    let mi_rust = est.global_value();

    let data1_vec = data1.to_vec();
    let data2_vec = data2.to_vec();

    let mi_python = python::calculate_mi(
        &[data1_vec, data2_vec],
        "ordinal",
        &[
            ("embedding_dim".to_string(), order.to_string()),
            ("step_size".to_string(), step_size.to_string()),
            ("stable".to_string(), "True".to_string()),
        ],
    )
    .unwrap();

    let codes1 = symbolize_series_u64(&data1, order, step_size, true);
    let codes2 = symbolize_series_u64(&data2, order, step_size, true);
    println!("MI Rust: {}, MI Python: {}", mi_rust, mi_python);
    println!("Codes1 len: {}, Codes2 len: {}", codes1.len(), codes2.len());

    use validation::python as py;
    let py_codes1 =
        py::calculate_symbolize_series(&data1.to_vec(), order, step_size, true, true).unwrap();
    let py_codes2 =
        py::calculate_symbolize_series(&data2.to_vec(), order, step_size, true, true).unwrap();

    // Joint Rust
    let joint_codes_rust =
        infomeasure::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact(
            &[
                infomeasure::estimators::approaches::ordinal::ordinal_utils::remap_u64_to_i32(
                    &codes1,
                ),
                infomeasure::estimators::approaches::ordinal::ordinal_utils::remap_u64_to_i32(
                    &codes2,
                ),
            ],
        );

    let _py_uid = format!(
        "{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        0
    );
    let _py_joint_script = format!(
        "import infomeasure as im, numpy as np; from infomeasure.estimators.utils.ordinal import reduce_joint_space; print(reduce_joint_space((np.array({:?}), np.array({:?}))).tolist())",
        py_codes1, py_codes2
    );
    // Actually using a simpler way to see joint in Python
    println!(
        "Joint Codes Rust (first 10): {:?}",
        &joint_codes_rust.as_slice().unwrap()[..10.min(joint_codes_rust.len())]
    );

    assert_abs_diff_eq!(mi_rust, mi_python, epsilon = 1e-10);
}

#[rstest]
fn ordinal_cmi_parity(#[values(3, 4)] order: usize, #[values(1, 2)] step_size: usize) {
    let n = 100;
    let data1 = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1).sin()));
    let data2 = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1 + 0.5).cos()));
    let dataz = Array1::from_iter((0..n).map(|i| (i as f64 * 0.2).sin()));

    let est = OrdinalConditionalMutualInformation::new(
        &[data1.clone(), data2.clone()],
        &dataz,
        order,
        step_size,
        true,
    );
    let cmi_rust = est.global_value();

    let data1_vec = data1.to_vec();
    let data2_vec = data2.to_vec();
    let dataz_vec = dataz.to_vec();

    let cmi_python = python::calculate_cmi(
        &[data1_vec, data2_vec],
        &dataz_vec,
        "ordinal",
        &[
            ("embedding_dim".to_string(), order.to_string()),
            ("step_size".to_string(), step_size.to_string()),
            ("stable".to_string(), "True".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(cmi_rust, cmi_python, epsilon = 1e-10);
}

#[rstest]
fn ordinal_te_parity(#[values(3, 4)] order: usize, #[values(1, 2)] step_size: usize) {
    let n = 100;
    let source = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1).sin()));
    let dest = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1 + 0.5).cos()));

    let est = OrdinalTransferEntropy::new(&source, &dest, order, 1, 1, step_size, true);
    let te_rust = est.global_value();

    let source_vec = source.to_vec();
    let dest_vec = dest.to_vec();

    let te_python = python::calculate_te(
        &source_vec,
        &dest_vec,
        "ordinal",
        &[
            ("embedding_dim".to_string(), order.to_string()),
            ("step_size".to_string(), step_size.to_string()),
            ("stable".to_string(), "True".to_string()),
            ("src_hist_len".to_string(), "1".to_string()),
            ("dest_hist_len".to_string(), "1".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(te_rust, te_python, epsilon = 1e-10);
}

#[rstest]
fn ordinal_cte_parity(#[values(3, 4)] order: usize, #[values(1, 2)] step_size: usize) {
    let n = 100;
    let source = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1).sin()));
    let dest = Array1::from_iter((0..n).map(|i| (i as f64 * 0.1 + 0.5).cos()));
    let cond = Array1::from_iter((0..n).map(|i| (i as f64 * 0.2).sin()));

    let est = OrdinalConditionalTransferEntropy::new(
        &source, &dest, &cond, order, 1, 1, 1, step_size, true,
    );
    let cte_rust = est.global_value();

    let source_vec = source.to_vec();
    let dest_vec = dest.to_vec();
    let cond_vec = cond.to_vec();

    let cte_python = python::calculate_cte(
        &source_vec,
        &dest_vec,
        &cond_vec,
        "ordinal",
        &[
            ("embedding_dim".to_string(), order.to_string()),
            ("step_size".to_string(), step_size.to_string()),
            ("stable".to_string(), "True".to_string()),
            ("src_hist_len".to_string(), "1".to_string()),
            ("dest_hist_len".to_string(), "1".to_string()),
            ("cond_hist_len".to_string(), "1".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(cte_rust, cte_python, epsilon = 1e-10);
}

// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use infomeasure::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use ndarray::Array1;
use rstest::*;

#[rstest]
#[case(
    vec![],
    vec![]
)]
#[case(
    vec![vec![10, 20, 10, 30]],
    vec![0, 1, 0, 2]
)]
#[case(
    vec![
        vec![1, 1, 2, 2],
        vec![1, 2, 1, 2]
    ],
    vec![0, 1, 2, 3]
)]
#[case(
    vec![
        vec![1, 1, 1, 1],
        vec![2, 2, 2, 2]
    ],
    vec![0, 0, 0, 0]
)]
#[case(
    vec![
        vec![1, 2, 1, 2],
        vec![1, 2, 1, 2]
    ],
    vec![0, 1, 0, 1]
)]
#[case(
    vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9]
    ],
    vec![0, 1, 2]
)]
fn test_reduce_joint_space_compact(#[case] inputs: Vec<Vec<i32>>, #[case] expected: Vec<i32>) {
    let code_arrays: Vec<Array1<i32>> = inputs.into_iter().map(Array1::from).collect();
    let result = reduce_joint_space_compact(&code_arrays);
    assert_eq!(result, Array1::from(expected));
}

#[test]
#[should_panic(expected = "All code arrays must have the same length for joint reduction")]
fn test_reduce_joint_space_compact_mismatch_length() {
    let arr1 = Array1::from(vec![1, 2, 3]);
    let arr2 = Array1::from(vec![1, 2]);
    reduce_joint_space_compact(&[arr1, arr2]);
}

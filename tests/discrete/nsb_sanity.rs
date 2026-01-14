// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::GlobalValue;
use infomeasure::estimators::approaches::NsbEntropy;
use ndarray::array;

#[test]
fn nsb_entropy_python_example() {
    // Python doc example: data = [1, 2, 3, 4, 5, 1, 2]
    let data = array![1, 2, 3, 4, 5, 1, 2];
    let est = NsbEntropy::new(data, None);
    let h = est.global_value();

    // Expected from Python example (nats): ~1.4526460202102247
    let expected = 1.4526460202102247_f64;

    // Allow a modest tolerance due to numerical integration differences
    assert_abs_diff_eq!(h, expected, epsilon = 5e-3);
}

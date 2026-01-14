// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::{
    miller_madow::MillerMadowEntropy, mle::DiscreteEntropy,
};
use infomeasure::estimators::{GlobalValue, LocalValues};
use ndarray::Array1;

#[test]
fn miller_madow_known_example() {
    let data = Array1::from(vec![1, 1, 2, 3, 3, 4, 5]);
    let mle = DiscreteEntropy::new(data.clone());
    let mm = MillerMadowEntropy::new(data.clone());

    let n = data.len() as f64;
    // K = number of unique symbols
    let k = 5.0;
    let correction = (k - 1.0) / (2.0 * n);

    let expected = mle.global_value() + correction;
    assert_abs_diff_eq!(mm.global_value(), expected, epsilon = 1e-12);

    // Local values should be mle locals + correction
    let mle_locals = mle.local_values();
    let mm_locals = mm.local_values();
    for (a, b) in mle_locals.iter().zip(mm_locals.iter()) {
        assert_abs_diff_eq!(*b, *a + correction, epsilon = 1e-12);
    }
}

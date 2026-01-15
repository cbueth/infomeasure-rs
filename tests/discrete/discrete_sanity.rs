// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use ndarray::Array1;
// use infomeasure::estimators::entropy::Entropy; // facade (not needed here)
use infomeasure::estimators::approaches::discrete::mle::DiscreteEntropy; // direct
use infomeasure::estimators::{GlobalValue, LocalValues, OptionalLocalValues};

#[test]
fn discrete_entropy_known_example() {
    // Example from Python docs: [1,1,2,3,3,4,5]
    let data = Array1::from(vec![1, 1, 2, 3, 3, 4, 5]);

    // Use the approaches::DiscreteEntropy directly to avoid API coupling
    let est = DiscreteEntropy::new(data.clone());

    // Expected global entropy in nats: H = ln(7) - (4/7) ln(2)
    let expected_h = 7f64.ln() - (4.0 / 7.0) * 2f64.ln();
    let h = est.global_value();
    assert_abs_diff_eq!(h, expected_h, epsilon = 1e-12);

    // Local values: -ln p(x)
    let locals = est.local_values();
    // For values with count 2 -> p=2/7 => -ln p ≈ 1.2527629685
    let ln_2_7 = -((2.0f64 / 7.0).ln());
    // For values with count 1 -> p=1/7 => -ln p ≈ 1.9459101491
    let ln_1_7 = -((1.0f64 / 7.0).ln());

    let expected_locals = [ln_2_7, ln_2_7, ln_1_7, ln_2_7, ln_2_7, ln_1_7, ln_1_7];
    for (i, &val) in locals.iter().enumerate() {
        assert_abs_diff_eq!(val, expected_locals[i], epsilon = 1e-12);
    }

    // OptionalLocalValues should report support
    assert!(est.supports_local());
    let opt = est.local_values_opt().unwrap();
    assert_eq!(opt.len(), locals.len());
}

#[test]
fn discrete_entropy_uniform() {
    // Uniform distribution over 4 symbols
    let data = Array1::from(vec![0, 1, 2, 3, 0, 1, 2, 3]);
    let est = DiscreteEntropy::new(data);
    // H should be ln(4) = 1.3862943611 nats
    let expected_h = (4.0f64).ln();
    assert_abs_diff_eq!(est.global_value(), expected_h, epsilon = 1e-12);

    let locals = est.local_values();
    for val in locals.iter() {
        assert_abs_diff_eq!(*val, expected_h, epsilon = 1e-12);
    }
}

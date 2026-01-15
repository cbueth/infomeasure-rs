// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::ZhangEntropy;
use infomeasure::estimators::{GlobalValue, LocalValues};
use ndarray::Array1;
use std::collections::HashMap;

#[test]
fn zhang_local_and_global_consistency() {
    let data = Array1::from(vec![1, 1, 2, 3, 3, 4, 5]);
    let est = ZhangEntropy::new(data.clone());

    // Local mapping should match per-count computed t2 contributions
    // Compute counts first
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &v in data.iter() {
        *counts.entry(v).or_insert(0) += 1;
    }
    let n_total = data.len();

    /// Computes the Zhang estimator's t2 contribution for a given count.
    ///
    /// # Arguments
    ///
    /// * `n` - The frequency count of a specific symbol in the dataset
    /// * `n_total` - The total number of samples in the dataset
    ///
    /// # Returns
    ///
    /// The t2 contribution value for the given count, used in Zhang's entropy estimator
    fn t2_for_count(n: usize, n_total: usize) -> f64 {
        if n == 0 || n >= n_total {
            return 0.0;
        }
        let nf = n as f64;
        let n_minus_1 = nf - 1.0;
        let n_total_f = n_total as f64;
        let mut t2 = 0.0_f64;
        let mut prod = 1.0_f64;
        for k in 1..=(n_total - n) {
            let denom = n_total_f - (k as f64);
            let factor = 1.0 - (n_minus_1 / denom);
            prod *= factor;
            t2 += prod / (k as f64);
        }
        t2
    }

    let locals = est.local_values();
    for (i, &v) in data.iter().enumerate() {
        let n = counts[&v];
        let expected = t2_for_count(n, n_total);
        assert_abs_diff_eq!(locals[i], expected, epsilon = 1e-12);
    }

    // Global should equal mean of locals
    let mean_locals = locals.mean().unwrap();
    assert_abs_diff_eq!(est.global_value(), mean_locals, epsilon = 1e-12);
}

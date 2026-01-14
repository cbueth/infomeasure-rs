// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::GlobalValue;
use infomeasure::estimators::approaches::ChaoShenEntropy;
use ndarray::array;

#[test]
fn chao_shen_entropy_basic() {
    let data = array![1, 1, 2, 3, 3, 4, 5];
    let est = ChaoShenEntropy::new(data);
    let h = est.global_value();

    // Manual compute
    // counts and ML probabilities
    use std::collections::HashMap;
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &x in [1, 1, 2, 3, 3, 4, 5].iter() {
        *counts.entry(x).or_insert(0) += 1;
    }
    let n = 7.0_f64;
    let mut f1 = 0usize;
    for &c in counts.values() {
        if c == 1 {
            f1 += 1;
        }
    }
    if (f1 as f64) == n {
        if f1 > 0 {
            f1 -= 1;
        }
    }
    let c_cov = 1.0 - (f1 as f64) / n;

    let mut h_exp = 0.0_f64;
    for &c in counts.values() {
        let p_ml = (c as f64) / n;
        let pa = c_cov * p_ml;
        if pa <= 0.0 {
            continue;
        }
        let la = 1.0 - (1.0 - pa).powf(n);
        if la <= 0.0 {
            continue;
        }
        h_exp -= pa * pa.ln() / la;
    }

    assert_abs_diff_eq!(h, h_exp, epsilon = 1e-12);
}

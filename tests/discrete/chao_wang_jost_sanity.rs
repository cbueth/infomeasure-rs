use approx::assert_abs_diff_eq;
use infomeasure::estimators::GlobalValue;
use infomeasure::estimators::approaches::ChaoWangJostEntropy;
use ndarray::array;
use statrs::function::gamma::digamma;

#[test]
fn chao_wang_jost_entropy_basic() {
    // Example dataset similar to doc discussion
    let data = array![1, 1, 1, 2, 2, 3, 5, 8];
    let est = ChaoWangJostEntropy::new(data);
    let h = est.global_value();

    // Manual compute expected using the formula
    use std::collections::HashMap;
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &x in [1, 1, 1, 2, 2, 3, 5, 8].iter() {
        *counts.entry(x).or_insert(0) += 1;
    }
    let n = 8usize;
    let mut f1 = 0usize;
    let mut f2 = 0usize;
    for &c in counts.values() {
        if c == 1 {
            f1 += 1;
        }
        if c == 2 {
            f2 += 1;
        }
    }

    let a = if f2 > 0 {
        (2.0 * f2 as f64) / (((n - 1) as f64) * (f1 as f64) + 2.0 * (f2 as f64))
    } else if f1 > 0 {
        2.0 / (((n - 1) as f64) * ((f1 - 1) as f64) + 2.0)
    } else {
        1.0
    };

    let dg_n = digamma(n as f64);
    let mut cwj = 0.0_f64;
    for &c in counts.values() {
        if c >= 1 && c < n {
            cwj += (c as f64) * (dg_n - digamma(c as f64));
        }
    }
    cwj /= n as f64;
    if a != 1.0 && f1 > 0 {
        let one_minus_a = 1.0 - a;
        let mut p2 = 0.0_f64;
        for r in 1..n {
            p2 += (one_minus_a.powi(r as i32)) / (r as f64);
        }
        let correction = (f1 as f64) / (n as f64) * one_minus_a.powi(1 - n as i32) * (-a.ln() - p2);
        cwj += correction;
    }

    assert_abs_diff_eq!(h, cwj, epsilon = 1e-12);
}

use ndarray::Array1;
use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::GrassbergerEntropy;
use infomeasure::estimators::{GlobalValue, LocalValues};
use std::collections::HashMap;

#[test]
fn grassberger_local_values_match_formula() {
    let data = Array1::from(vec![1,1,2,3,3,4,5]);
    let est = GrassbergerEntropy::new(data.clone());

    // Compute expected locals manually from counts
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &v in data.iter() { *counts.entry(v).or_insert(0) += 1; }
    let n = data.len() as f64;
    let n_ln = n.ln();

    let locals = est.local_values();
    for (i, &v) in data.iter().enumerate() {
        let c = counts[&v] as i64;
        let cf = c as f64;
        let sign = if c % 2 == 0 { 1.0 } else { -1.0 };
        let expected = n_ln - statrs::function::gamma::digamma(cf) - sign / (cf + 1.0);
        assert_abs_diff_eq!(locals[i], expected, epsilon = 1e-12);
    }

    // Global equals mean of locals (via trait default)
    let mean_locals = locals.mean().unwrap();
    assert_abs_diff_eq!(est.global_value(), mean_locals, epsilon = 1e-12);
}
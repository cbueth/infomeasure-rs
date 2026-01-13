use approx::assert_abs_diff_eq;
use ndarray::array;
use infomeasure::estimators::approaches::BonachelaEntropy;
use infomeasure::estimators::GlobalValue;

#[test]
fn bonachela_entropy_basic() {
    let data = array![1,1,2,3,3,4,5];
    let est = BonachelaEntropy::new(data);
    let h = est.global_value();

    // Manual computation per formula
    let counts = {
        use std::collections::HashMap;
        let mut m = HashMap::new();
        for &x in [1,1,2,3,3,4,5].iter() { *m.entry(x).or_insert(0usize) += 1; }
        m
    };
    let n = 7usize;
    let mut acc = 0.0_f64;
    for &cnt in counts.values() {
        let ni = cnt + 1;
        let mut inner = 0.0_f64;
        for j in (cnt + 2)..=(n + 2) { inner += 1.0 / (j as f64); }
        acc += (ni as f64) * inner;
    }
    let h_exp = acc / ((n + 2) as f64);

    assert_abs_diff_eq!(h, h_exp, epsilon = 1e-12);
}

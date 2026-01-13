use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::bayes::{AlphaParam, BayesEntropy};
use infomeasure::estimators::{CrossEntropy, GlobalValue};
use ndarray::array;

#[test]
fn bayes_entropy_laplace_basic() {
    // Data: [1,1,2,3,3,4,5] => N=7, uniq={1,2,3,4,5} K=5
    let data = array![1, 1, 2, 3, 3, 4, 5];
    let est = BayesEntropy::new(data, AlphaParam::Laplace, None);
    let h = est.global_value();

    // Expected: compute directly
    let n = 7.0_f64;
    let k = 5.0_f64;
    let alpha = 1.0_f64; // Laplace
    let denom = n + k * alpha; // 12
    // counts: 1->2, 2->1, 3->2, 4->1, 5->1
    let probs = vec![
        (2.0 + alpha) / denom,
        (1.0 + alpha) / denom,
        (2.0 + alpha) / denom,
        (1.0 + alpha) / denom,
        (1.0 + alpha) / denom,
    ];
    let mut h_exp = 0.0;
    for p in probs {
        h_exp -= p * p.ln();
    }

    assert_abs_diff_eq!(h, h_exp, epsilon = 1e-12);
}

#[test]
fn bayes_cross_entropy_intersection() {
    // P: [1,1,2,2] -> uniq={1,2} K=2, N=4
    // Q: [1,2,2,3] -> uniq={1,2,3} K=3, N=4
    let p = array![1, 1, 2, 2];
    let q = array![1, 2, 2, 3];
    let est_p = BayesEntropy::new(p, AlphaParam::Laplace, None);
    let est_q = BayesEntropy::new(q, AlphaParam::Laplace, None);

    let h_cx = est_p.cross_entropy(&est_q);

    // Expected manual calculation (intersection {1,2})
    // P probs with Kp=2: counts (1:2,2:2) => p = (2+1)/(4+2) = 0.5 for both
    // Q probs with Kq=3: counts (1:1,2:2,3:1) => denom 4+3=7
    // 1 -> (1+1)/7 = 2/7; 2 -> (2+1)/7 = 3/7
    let p1 = 0.5_f64;
    let p2 = 0.5_f64;
    let q1: f64 = 2.0 / 7.0;
    let q2: f64 = 3.0 / 7.0;
    let h_exp = -(p1 * q1.ln() + p2 * q2.ln());

    assert_abs_diff_eq!(h_cx, h_exp, epsilon = 1e-12);
}

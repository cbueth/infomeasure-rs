use approx::assert_abs_diff_eq;
use ndarray::array;
use infomeasure::estimators::approaches::MillerMadowEntropy;
use infomeasure::estimators::CrossEntropy;

#[test]
fn miller_madow_cross_entropy_basic() {
    // P: [1,1,2,2] -> Kp=2, Np=4; Q: [1,2,2,3] -> Kq=3, Nq=4
    let p = array![1,1,2,2];
    let q = array![1,2,2,3];
    let est_p = MillerMadowEntropy::new(p);
    let est_q = MillerMadowEntropy::new(q);

    let h_cx = est_p.cross_entropy(&est_q);

    // Manual ML cross-entropy + correction
    // ML: P: p(1)=0.5, p(2)=0.5; Q: q(1)=0.25, q(2)=0.5, q(3)=0.25
    let h_ml = -(0.5_f64 * (0.25_f64).ln() + 0.5_f64 * (0.5_f64).ln());
    let correction = (((2 + 3) as f64) / 2.0 - 1.0) / ((4 + 4) as f64); // 1.5 / 8 = 0.1875
    let h_exp = h_ml + correction;

    assert_abs_diff_eq!(h_cx, h_exp, epsilon = 1e-12);
}

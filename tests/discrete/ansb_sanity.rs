use approx::assert_abs_diff_eq;
use infomeasure::estimators::GlobalValue;
use infomeasure::estimators::approaches::AnsbEntropy;
use ndarray::array;
use statrs::function::gamma::digamma;

#[test]
fn ansb_entropy_matches_formula() {
    // Data with coincidences: N=7, K=5 => Δ=2
    let data = array![1, 2, 3, 4, 5, 1, 2];
    let est = AnsbEntropy::new(data, None, 0.1);
    let h = est.global_value();

    // Expected: (γ - ln 2) + 2 ln N - ψ(Δ)
    const EULER_GAMMA: f64 = 0.577215_664_901_532_9;
    let n = 7.0_f64;
    let delta = 2.0_f64;
    let h_exp = (EULER_GAMMA - 2.0_f64.ln()) + 2.0 * n.ln() - digamma(delta);

    assert_abs_diff_eq!(h, h_exp, epsilon = 1e-12);
}

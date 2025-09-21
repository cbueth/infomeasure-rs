use ndarray::Array1;
use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::ordinal::ordinal::OrdinalEntropy;
use infomeasure::estimators::approaches::discrete::mle::DiscreteEntropy;
use infomeasure::estimators::traits::LocalValues;
use infomeasure::estimators::approaches::ordinal::ordinal_utils::symbolize_series;

#[test]
fn ordinal_basic_patterns_and_entropy() {
    // Series: [1,2,3,2,1], order=2, step_size=1
    // Windows: [1,2] -> [0,1] code 0
    //          [2,3] -> [0,1] code 0
    //          [3,2] -> [1,0] code 1
    //          [2,1] -> [1,0] code 1
    // codes -> [0,0,1,1] => p={0:0.5, 1:0.5} => H=ln 2
    let series: Array1<f64> = Array1::from(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    let ord = OrdinalEntropy::new(series.clone(), 2, 1);
    let h = ord.global_value();
    assert_abs_diff_eq!(h, (2.0_f64).ln(), epsilon = 1e-12);

    // Local mean equals global
    let lm = ord.local_values().mean().unwrap();
    assert_abs_diff_eq!(h, lm, epsilon = 1e-12);
}

#[test]
fn ordinal_tie_handling_stable() {
    // Ties: expect stable argsort to keep index order
    // Series: [1,1,2,2], order=2, step_size=1
    // Windows: [1,1]-> [0,1] code 0; [1,2]->0; [2,2]->0
    let series: Array1<f64> = Array1::from(vec![1.0, 1.0, 2.0, 2.0]);
    let ord = OrdinalEntropy::new(series, 2, 1);
    let h = ord.global_value();
    assert_abs_diff_eq!(h, 0.0, epsilon = 1e-12);
}

#[test]
fn ordinal_insufficient_length_yields_zero_entropy() {
    // N=3, order=4, step_size=1 => no windows
    let series: Array1<f64> = Array1::from(vec![1.0, 2.0, 3.0]);
    // Should not panic; entropy should be 0 and local values empty
    let ord = OrdinalEntropy::new(series, 4, 1);
    assert_eq!(ord.local_values().len(), 0);
    assert_abs_diff_eq!(ord.global_value(), 0.0, epsilon = 1e-12);
}

#[test]
#[should_panic]
fn ordinal_order_limit_panics_over_12() {
    let series: Array1<f64> = Array1::from(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                                                7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    // order=13 should panic due to temporary i32 limitation
    let _ord = OrdinalEntropy::new(series, 13, 1);
}


#[test]
fn ordinal_equivalence_with_discrete_on_codes() {
    // For a given series, the ordinal entropy equals the discrete entropy of its codes
    let series: Array1<f64> = Array1::from(vec![3.0, 1.0, 2.0, 5.0, 4.0]);
    let ord = OrdinalEntropy::new(series.clone(), 3, 1);
    let local = ord.local_values();
    let h_ord = ord.global_value();

    // Recompute codes the same way and feed into DiscreteEntropy
    // Use the public facade to ensure API surface
    let codes = symbolize_series(&series, 3, 1, true);
    let disc = DiscreteEntropy::new(codes);
    let h_disc = disc.global_value();
    assert_abs_diff_eq!(h_ord, h_disc, epsilon = 1e-12);
    assert_abs_diff_eq!(h_ord, local.mean().unwrap_or(0.0), epsilon = 1e-12);
}

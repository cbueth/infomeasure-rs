use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::{mle::DiscreteEntropy, shrink::ShrinkEntropy};
use infomeasure::estimators::{GlobalValue, LocalValues};
use ndarray::Array1;

#[test]
fn shrink_basic_behaviour() {
    // Non-uniform distribution to exercise shrinkage
    let data = Array1::from(vec![1, 1, 1, 2, 2, 3, 4]); // counts: 3,2,1,1 (K=4, N=7)
    let mle = DiscreteEntropy::new(data.clone());
    let shr = ShrinkEntropy::new(data.clone());

    // Shrinkage should move probabilities toward uniform, so entropy should be >= MLE
    let h_mle = mle.global_value();
    let h_shr = shr.global_value();
    assert!(h_shr >= h_mle - 1e-12);

    // Local values are finite and same length as data
    let locals = shr.local_values();
    assert_eq!(locals.len(), data.len());
    for v in locals.iter() {
        assert!(v.is_finite());
    }

    // On perfectly uniform data, shrinkage should equal MLE
    let uniform = Array1::from(vec![0, 1, 2, 3, 0, 1, 2, 3]); // uniform over 4 symbols
    let mle_u = DiscreteEntropy::new(uniform.clone());
    let shr_u = ShrinkEntropy::new(uniform.clone());
    assert_abs_diff_eq!(mle_u.global_value(), shr_u.global_value(), epsilon = 1e-12);
}

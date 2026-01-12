use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::mle::DiscreteEntropy;
use infomeasure::estimators::traits::GlobalValue;
use infomeasure::estimators::traits::LocalValues;
use ndarray::Array1;
use rstest::*;
use validation::python;

#[rstest]
#[case(vec![1, 1, 1, 1, 1], "uniform")]
#[case(vec![1, 0, 1, 0], "binary")]
#[case(vec![1, 2, 3, 4, 5], "all singletons")]
#[case(vec![1, 1, 7, 2, 3, 6, 6, 3], "mixed")]
#[case(vec![2, 3, 6, 6, 3, 6, 5, 7], "mixed 2")]
#[case(vec![1, 1, 2, 3, 3, 4, 5], "mixed 3")]
#[case(vec![0, 1, 0, 1, 0], "binary long")]
#[case(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "large uniform")]
#[case(vec![1, 1, 1, 2, 2, 3], "skewed")]
fn discrete_entropy_python_parity(#[case] data: Vec<i32>, #[case] _description: &str) {
    let arr = Array1::from(data.clone());
    let rust_est = DiscreteEntropy::new(arr);
    let h_rust = rust_est.global_value();
    let locals_rust = rust_est.local_values();

    let h_py = python::calculate_entropy(&data, "discrete", &[]).expect("python discrete failed");
    let locals_py = python::calculate_local_entropy(&data, "discrete", &[]).expect("python local discrete failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    assert_eq!(locals_rust.len(), locals_py.len());
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_abs_diff_eq!(*lr, *lp, epsilon = 1e-10);
    }
}

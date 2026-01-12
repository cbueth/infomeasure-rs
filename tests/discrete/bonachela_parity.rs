use approx::assert_abs_diff_eq;
use ndarray::Array1;
use infomeasure::estimators::approaches::discrete::bonachela::BonachelaEntropy;
use infomeasure::estimators::traits::GlobalValue;
use validation::python;
use rstest::*;

#[rstest]
#[case(vec![1, 1, 2], "simple [2, 1]")]
#[case(vec![1, 1, 1], "uniform 3")]
#[case(vec![1, 1, 1, 1, 1], "uniform 5")]
#[case(vec![1, 2, 3, 4], "all singletons 4")]
#[case(vec![1, 1, 2, 2], "balanced binary")]
#[case(vec![1, 2, 3], "all singletons 3")]
#[case(vec![1, 1, 1, 2], "skewed binary")]
#[case(vec![1, 2, 1, 2, 1], "alternating")]
#[case(vec![1, 1, 2, 2, 3, 3, 4], "mostly balanced")]
#[case(vec![1, 1, 2, 2, 3], "mixed counts")]
#[case(vec![1, 1, 2, 2, 3, 3], "balanced ternary")]
#[case(vec![1, 1, 2, 2, 3, 3, 1, 2], "larger dataset")]
fn bonachela_entropy_python_parity(#[case] data: Vec<i32>, #[case] _description: &str) {
    let arr = Array1::from(data.clone());
    let rust_est = BonachelaEntropy::new(arr);
    let h_rust = rust_est.global_value();

    let h_py = python::calculate_entropy(&data, "bonachela", &[])
        .expect("python bonachela failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

use approx::assert_abs_diff_eq;
use ndarray::Array1;
use infomeasure::estimators::approaches::discrete::zhang::ZhangEntropy;
use infomeasure::estimators::traits::LocalValues;
use validation::python;
use rstest::*;

#[rstest]
#[case(vec![1, 1, 2], "simple [2, 1]")]
#[case(vec![1, 1, 1], "uniform")]
#[case(vec![1, 2, 3, 4], "all singletons")]
#[case(vec![1, 1, 2, 2], "balanced binary")]
#[case(vec![1, 2, 3], "all singletons 3")]
#[case(vec![1, 1, 1, 2], "skewed binary")]
#[case(vec![1, 2, 1, 2, 1], "alternating")]
#[case(vec![1, 1, 2, 2, 3, 3, 4], "mostly balanced")]
#[case(vec![1, 1, 2, 2, 3], "mixed counts")]
#[case(vec![1, 1, 2, 2, 3, 3], "balanced ternary")]
#[case(vec![1, 1, 2, 2, 3, 3, 1, 2], "larger dataset")]
fn zhang_entropy_python_parity(#[case] data: Vec<i32>, #[case] _description: &str) {
    let arr = Array1::from(data.clone());
    let rust_est = ZhangEntropy::new(arr);
    let h_rust = rust_est.global_value();
    let locals_rust = rust_est.local_values();

    let h_py = python::calculate_entropy(&data, "zhang", &[])
        .expect("python zhang failed");
    let locals_py = python::calculate_local_entropy(&data, "zhang", &[])
        .expect("python local zhang failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    assert_eq!(locals_rust.len(), locals_py.len());
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_abs_diff_eq!(*lr, *lp, epsilon = 1e-10);
    }
}

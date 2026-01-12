use approx::assert_abs_diff_eq;
use ndarray::Array1;
use infomeasure::estimators::approaches::discrete::shrink::ShrinkEntropy;
use infomeasure::estimators::traits::LocalValues;
use validation::python;
use rstest::*;

#[rstest]
#[case(vec![1, 1, 1, 1, 1], "uniform")]
#[case(vec![1, 0, 1, 0], "binary")]
#[case(vec![1, 2, 3, 4, 5], "all singletons")]
#[case(vec![1, 1, 2], "simple [2, 1]")]
#[case(vec![0, 0, 1, 1, 2, 2], "uniform counts [2, 2, 2]")]
#[case(vec![1, 2, 2, 3, 3, 3], "mixed counts")]
#[case(vec![1, 1, 1, 2], "skewed [3, 1]")]
#[case(vec![1, 2, 3, 4, 5, 4, 3, 4, 5], "mixed 2")]
fn shrink_entropy_python_parity(#[case] data: Vec<i32>, #[case] _description: &str) {
    let arr = Array1::from(data.clone());
    let rust_est = ShrinkEntropy::new(arr);
    let h_rust = rust_est.global_value();
    let locals_rust = rust_est.local_values();

    let h_py = python::calculate_entropy(&data, "shrink", &[])
        .expect("python shrink failed");
    let locals_py = python::calculate_local_entropy(&data, "shrink", &[])
        .expect("python local shrink failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    assert_eq!(locals_rust.len(), locals_py.len());
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_abs_diff_eq!(*lr, *lp, epsilon = 1e-10);
    }
}

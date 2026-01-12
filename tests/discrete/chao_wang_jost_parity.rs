use approx::assert_abs_diff_eq;
use ndarray::Array1;
use infomeasure::estimators::approaches::discrete::chao_wang_jost::ChaoWangJostEntropy;
use infomeasure::estimators::traits::GlobalValue;
use validation::python;
use rstest::*;

#[rstest]
#[case(vec![1, 1, 1, 1, 1], "uniform 5")]
#[case(vec![1, 0, 1, 0], "binary")]
#[case(vec![1, 2, 3, 4, 5], "all singletons 5")]
#[case(vec![1, 1, 2], "simple [2, 1]")]
#[case(vec![0, 0, 1, 1, 2, 2], "uniform counts [2, 2, 2]")]
#[case(vec![1, 2, 2, 3, 3, 3], "mixed counts")]
#[case(vec![1, 1, 1, 2], "skewed [3, 1]")]
#[case(vec![1, 2, 3, 3, 4, 4, 4], "tripleton")]
#[case(vec![1, 1, 2, 2, 3, 3, 4, 4, 5], "mostly doubletons")]
#[case(vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4], "increasing frequency")]
#[case(vec![1, 1, 1, 1, 2, 2, 3], "dominant group")]
#[case(vec![1, 2, 3, 4, 5, 6, 7, 8], "all singletons 8")]
fn chao_wang_jost_entropy_python_parity(#[case] data: Vec<i32>, #[case] _description: &str) {
    let arr = Array1::from(data.clone());
    let rust_est = ChaoWangJostEntropy::new(arr);
    let h_rust = rust_est.global_value();

    let h_py = python::calculate_entropy(&data, "chao_wang_jost", &[])
        .expect("python chao_wang_jost failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

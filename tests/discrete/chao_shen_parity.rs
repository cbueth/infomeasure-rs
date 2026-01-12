use approx::assert_abs_diff_eq;
use ndarray::Array1;
use infomeasure::estimators::approaches::discrete::chao_shen::ChaoShenEntropy;
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
#[case(vec![1, 1, 2, 3, 3, 4, 5], "larger mixed")]
fn chao_shen_entropy_python_parity(#[case] data: Vec<i32>, #[case] _description: &str) {
    let arr = Array1::from(data.clone());
    let rust_est = ChaoShenEntropy::new(arr);
    let h_rust = rust_est.global_value();

    let h_py = python::calculate_entropy(&data, "chao_shen", &[])
        .expect("python chao_shen failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

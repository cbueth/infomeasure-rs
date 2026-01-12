use approx::assert_abs_diff_eq;
use ndarray::Array1;
use infomeasure::estimators::approaches::discrete::ansb::AnsbEntropy;
use infomeasure::estimators::traits::GlobalValue;
use validation::python;
use rstest::*;

#[rstest]
#[case(vec![1, 1, 2, 3, 4], None, "one coincidence")]
#[case(vec![1, 1, 1, 2, 2], None, "multiple coincidences")]
#[case(vec![1, 1, 2, 2, 3, 3], None, "pairs")]
#[case(vec![1, 1, 2, 3, 4, 5], None, "one pair mixed")]
#[case(vec![1, 1, 2, 3, 4], Some(3), "K=3 < observed K=4")]
#[case(vec![1, 1, 2, 2, 3, 3], Some(4), "K=4 < observed K=3 (pairs)")]
#[case(vec![1, 1, 1, 1], Some(1), "all same K=1")]
#[case(vec![1, 1, 2, 3], Some(2), "K=2 < observed K=3")]
#[case(vec![1, 2, 3, 4, 5], None, "no coincidences")]
#[case(vec![1, 1, 2, 3], Some(5), "K=5 > N=4 (negative coincidences)")]
#[case(vec![1, 2, 3, 4], Some(6), "no coincidences K=6 > N=4")]
fn ansb_entropy_python_parity(
    #[case] data: Vec<i32>,
    #[case] k_override: Option<usize>,
    #[case] _description: &str,
) {
    let arr = Array1::from(data.clone());
    let rust_est = AnsbEntropy::new(arr, k_override, 0.0);
    let h_rust = rust_est.global_value();

    let mut kwargs = Vec::new();
    if let Some(k) = k_override {
        kwargs.push(("K".to_string(), k.to_string()));
    }

    let h_py = python::calculate_entropy(&data, "ansb", &kwargs)
        .expect("python ansb failed");

    if h_rust.is_nan() {
        assert!(h_py.is_nan(), "Rust returned NaN but Python returned {}", h_py);
    } else {
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    }
}

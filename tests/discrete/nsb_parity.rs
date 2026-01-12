use approx::assert_abs_diff_eq;
use ndarray::Array1;
use infomeasure::estimators::approaches::discrete::nsb::NsbEntropy;
use infomeasure::estimators::traits::GlobalValue;
use validation::python;
use rstest::*;

#[rstest]
#[case(vec![1, 1, 2], None, "simple [2, 1]")]
#[case(vec![1, 2, 3, 4, 5, 1, 2], None, "more complex")]
#[case(vec![1, 1, 1, 2, 2, 3], None, "mixed [3, 2, 1]")]
#[case(vec![1, 1, 2, 2], None, "balanced binary")]
#[case(vec![1, 1, 1, 2], None, "unbalanced binary")]
#[case(vec![1, 2, 1, 2, 1], None, "alternating binary")]
#[case(vec![1, 1, 2, 2, 3, 3, 4], None, "mostly balanced")]
#[case(vec![1, 1, 2, 2, 3, 3, 1, 2], None, "larger dataset")]
#[case(vec![1, 1, 2, 2, 3], Some(4), "K=4 > observed K=3")]
#[case(vec![1, 1, 2, 2, 3], Some(10), "K=10 >> observed K=3")]
#[case(vec![1, 1, 2, 2, 3, 3], Some(2), "K=2 < observed K=3")]
#[case(vec![1, 2, 3, 4, 5], Some(10), "no coincidences K=10")]
#[case(vec![1, 2, 3, 4, 5], None, "no coincidences")]
#[case(vec![1, 1, 1, 1], None, "all same values")]
#[case(vec![1, 1, 2, 2, 3], Some(5), "K=N=5 case")]
fn nsb_entropy_python_parity(
    #[case] data: Vec<i32>,
    #[case] k_override: Option<usize>,
    #[case] _description: &str,
) {
    let arr = Array1::from(data.clone());
    let rust_est = NsbEntropy::new(arr, k_override);
    let h_rust = rust_est.global_value();

    let mut kwargs = Vec::new();
    if let Some(k) = k_override {
        kwargs.push(("K".to_string(), k.to_string()));
    }

    let h_py = python::calculate_entropy(&data, "nsb", &kwargs)
        .expect("python nsb failed");

    if h_rust.is_nan() {
        assert!(h_py.is_nan(), "Rust returned NaN but Python returned {}", h_py);
    } else {
        // NSB uses numerical integration; use a looser tolerance
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-5);
    }
}

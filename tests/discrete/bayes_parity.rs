use approx::assert_abs_diff_eq;
use ndarray::Array1;
use infomeasure::estimators::approaches::discrete::bayes::{BayesEntropy, AlphaParam};
use infomeasure::estimators::traits::{LocalValues, GlobalValue};
use validation::python;
use rstest::*;

#[rstest]
#[case(vec![1, 1, 2, 3, 3, 4, 5], AlphaParam::Jeffrey, None, "mixed Jeffrey")]
#[case(vec![1, 1, 2], AlphaParam::Laplace, None, "simple Laplace")]
#[case(vec![1, 0, 1, 0], AlphaParam::Laplace, Some(3), "binary K=3")]
#[case(vec![1, 0, 1, 0], AlphaParam::Laplace, Some(5), "binary K=5")]
#[case(vec![1, 0, 1, 0], AlphaParam::Value(0.1), None, "binary alpha=0.1")]
#[case(vec![1, 0, 1, 0], AlphaParam::Value(2.0), None, "binary alpha=2.0")]
#[case(vec![1, 1, 1, 1, 1], AlphaParam::Laplace, None, "uniform Laplace")]
#[case(vec![0, 1, 0, 1, 2, 2], AlphaParam::SchGrass, None, "mixed SchGrass")]
#[case(vec![0, 1, 0, 1, 2, 2], AlphaParam::MinMax, None, "mixed MinMax")]
#[case(vec![1, 2, 3], AlphaParam::Laplace, None, "three symbols alpha=1.0")]
#[case(vec![1, 2, 3], AlphaParam::Jeffrey, None, "three symbols alpha=0.5")]
fn bayes_entropy_python_parity(
    #[case] data: Vec<i32>,
    #[case] alpha_param: AlphaParam,
    #[case] k_override: Option<usize>,
    #[case] _description: &str,
) {
    let arr = Array1::from(data.clone());
    let rust_est = BayesEntropy::new(arr, alpha_param.clone(), k_override);
    let h_rust = rust_est.global_value();

    let mut kwargs = Vec::new();

    let alpha_str = match alpha_param {
        AlphaParam::Value(v) => v.to_string(),
        AlphaParam::Jeffrey => "0.5".to_string(),
        AlphaParam::Laplace => "1.0".to_string(),
        AlphaParam::SchGrass => "\"sch-grass\"".to_string(),
        AlphaParam::MinMax => "\"min-max\"".to_string(),
    };
    kwargs.push(("alpha".to_string(), alpha_str));

    if let Some(k) = k_override {
        kwargs.push(("K".to_string(), k.to_string()));
    }

    let h_py = python::calculate_entropy(&data, "bayes", &kwargs).expect("python bayes failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
}

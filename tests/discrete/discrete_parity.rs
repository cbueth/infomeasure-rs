use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::mle::DiscreteEntropy;
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use infomeasure::estimators::{GlobalValue, LocalValues};
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
    let locals_py = python::calculate_local_entropy(&data, "discrete", &[])
        .expect("python local discrete failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    assert_eq!(locals_rust.len(), locals_py.len());
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_abs_diff_eq!(*lr, *lp, epsilon = 1e-10);
    }
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], 0.0)]
fn test_mle_mi_parity(#[case] x: Vec<i32>, #[case] y: Vec<i32>, #[case] expected: f64) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let mi_est = MutualInformation::new_discrete_mle(&[x_arr, y_arr]);
    let res = mi_est.global_value();

    let mi_py = python::calculate_mi(&[x, y], "discrete", &[]).unwrap();
    assert_abs_diff_eq!(res, mi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![1, 0, 1, 0], 0.0)]
fn test_mle_te_parity(#[case] source: Vec<i32>, #[case] dest: Vec<i32>, #[case] expected: f64) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let te_est = TransferEntropy::new_discrete_mle(&s_arr, &d_arr, 1, 1, 1);
    let res = te_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
    ];
    let te_py = python::calculate_te(&source, &dest, "discrete", &kwargs).unwrap();
    assert_abs_diff_eq!(res, te_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_mle_cmi_parity(
    #[case] x: Vec<i32>,
    #[case] y: Vec<i32>,
    #[case] z: Vec<i32>,
    #[case] expected: f64,
) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let z_arr = Array1::from(z.clone());
    let cmi_est = MutualInformation::new_cmi_discrete_mle(&[x_arr, y_arr], &z_arr);
    let res = cmi_est.global_value();

    let cmi_py = python::calculate_cmi(&[x, y], &z, "discrete", &[]).unwrap();
    assert_abs_diff_eq!(res, cmi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_mle_cte_parity(
    #[case] source: Vec<i32>,
    #[case] dest: Vec<i32>,
    #[case] cond: Vec<i32>,
    #[case] expected: f64,
) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let c_arr = Array1::from(cond.clone());
    let cte_est = TransferEntropy::new_cte_discrete_mle(&s_arr, &d_arr, &c_arr, 1, 1, 1, 1);
    let res = cte_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
        ("cond_hist_len".to_string(), "1".to_string()),
    ];
    let cte_py = python::calculate_cte(&source, &dest, &cond, "discrete", &kwargs).unwrap();
    assert_abs_diff_eq!(res, cte_py, epsilon = 1e-10);
}

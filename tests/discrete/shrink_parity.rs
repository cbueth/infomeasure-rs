use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::shrink::ShrinkEntropy;
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::traits::{GlobalValue, LocalValues};
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rstest::*;
use validation::python;

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

    let h_py = python::calculate_entropy(&data, "shrink", &[]).expect("python shrink failed");
    let locals_py =
        python::calculate_local_entropy(&data, "shrink", &[]).expect("python local shrink failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    assert_eq!(locals_rust.len(), locals_py.len());
    for (lr, lp) in locals_rust.iter().zip(locals_py.iter()) {
        assert_abs_diff_eq!(*lr, *lp, epsilon = 1e-10);
    }
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], 0.0)]
fn test_shrink_mi_parity(#[case] x: Vec<i32>, #[case] y: Vec<i32>, #[case] expected: f64) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let mi_est = MutualInformation::new_discrete_shrink(&[x_arr, y_arr]);
    let res = mi_est.global_value();

    let mi_py = python::calculate_mi(&[x, y], "shrink", &[]).unwrap();
    assert_abs_diff_eq!(res, mi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![1, 0, 1, 0], 0.0)]
fn test_shrink_te_parity(#[case] source: Vec<i32>, #[case] dest: Vec<i32>, #[case] expected: f64) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let te_est = TransferEntropy::new_discrete_shrink(&s_arr, &d_arr, 1, 1, 1);
    let res = te_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
    ];
    let te_py = python::calculate_te(&source, &dest, "shrink", &kwargs).unwrap();
    assert_abs_diff_eq!(res, te_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_shrink_cmi_parity(
    #[case] x: Vec<i32>,
    #[case] y: Vec<i32>,
    #[case] z: Vec<i32>,
    #[case] expected: f64,
) {
    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let z_arr = Array1::from(z.clone());
    let cmi_est = MutualInformation::new_cmi_discrete_shrink(&[x_arr, y_arr], &z_arr);
    let res = cmi_est.global_value();

    let cmi_py = python::calculate_cmi(&[x, y], &z, "shrink", &[]).unwrap();
    assert_abs_diff_eq!(res, cmi_py, epsilon = 1e-10);
}

#[rstest]
#[case(vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], vec![1, 1, 1, 1, 1], 0.0)]
#[case(vec![1, 0, 1, 0], vec![0, 1, 0, 1], vec![1, 1, 0, 0], 0.0)]
fn test_shrink_cte_parity(
    #[case] source: Vec<i32>,
    #[case] dest: Vec<i32>,
    #[case] cond: Vec<i32>,
    #[case] expected: f64,
) {
    let s_arr = Array1::from(source.clone());
    let d_arr = Array1::from(dest.clone());
    let c_arr = Array1::from(cond.clone());
    let cte_est = TransferEntropy::new_cte_discrete_shrink(&s_arr, &d_arr, &c_arr, 1, 1, 1, 1);
    let res = cte_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), "1".to_string()),
        ("dest_hist_len".to_string(), "1".to_string()),
        ("cond_hist_len".to_string(), "1".to_string()),
    ];
    let cte_py = python::calculate_cte(&source, &dest, &cond, "shrink", &kwargs).unwrap();
    assert_abs_diff_eq!(res, cte_py, epsilon = 1e-10);
}

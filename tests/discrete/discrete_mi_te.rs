// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::traits::{GlobalValue, LocalValues};
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rstest::rstest;
use validation::python;

fn generate_random_data(size: usize, alphabet_size: i32, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(0..alphabet_size)).collect()
}

#[rstest]
#[case(vec![0, 0, 1, 1, 0, 1, 0, 1], vec![0, 1, 0, 1, 0, 1, 0, 1])]
#[case(vec![1, 1, 2, 2, 3, 3], vec![1, 2, 1, 2, 1, 2])]
#[case(generate_random_data(100, 5, 42), generate_random_data(100, 5, 43))]
fn test_discrete_mi_mle_parity(#[case] x_vec: Vec<i32>, #[case] y_vec: Vec<i32>) {
    let x = Array1::from(x_vec.clone());
    let y = Array1::from(y_vec.clone());

    let mi_est = MutualInformation::new_discrete_mle(&[x, y]);
    let mi_rust = mi_est.global_value();

    let mi_py = python::calculate_mi(&[x_vec.clone(), y_vec.clone()], "discrete", &[]).unwrap();

    println!("MI Rust: {}, MI Python: {}", mi_rust, mi_py);
    assert!((mi_rust - mi_py).abs() < 1e-10);

    let locals_rust = mi_est.local_values();
    let locals_py = python::calculate_local_mi(&[x_vec, y_vec], "discrete", &[]).unwrap();

    for (r, p) in locals_rust.iter().zip(locals_py.iter()) {
        assert!((r - p).abs() < 1e-10);
    }
}

#[rstest]
#[case(generate_random_data(50, 3, 44), generate_random_data(50, 3, 45))]
fn test_discrete_mi_chao_shen_parity(#[case] x_vec: Vec<i32>, #[case] y_vec: Vec<i32>) {
    let x = Array1::from(x_vec.clone());
    let y = Array1::from(y_vec.clone());

    let mi_est = MutualInformation::new_discrete_chao_shen(&[x, y]);
    let mi_rust = mi_est.global_value();

    let mi_py = python::calculate_mi(&[x_vec.clone(), y_vec.clone()], "chao_shen", &[]).unwrap();

    println!("MI Chao-Shen Rust: {}, MI Python: {}", mi_rust, mi_py);
    assert!((mi_rust - mi_py).abs() < 1e-10);
}

#[rstest]
#[case(generate_random_data(100, 10, 46), generate_random_data(100, 10, 47))]
fn test_discrete_mi_nsb_parity(#[case] x_vec: Vec<i32>, #[case] y_vec: Vec<i32>) {
    let x = Array1::from(x_vec.clone());
    let y = Array1::from(y_vec.clone());

    let mi_est = MutualInformation::new_discrete_nsb(&[x, y]);
    let mi_rust = mi_est.global_value();

    let mi_py = python::calculate_mi(&[x_vec.clone(), y_vec.clone()], "nsb", &[]).unwrap();

    println!("MI NSB Rust: {}, MI Python: {}", mi_rust, mi_py);
    assert!((mi_rust - mi_py).abs() < 1e-7); // NSB might have slightly more numerical diff
}

#[rstest]
#[case(vec![0, 0, 1, 1, 0, 1, 0, 1], vec![0, 1, 0, 1, 0, 1, 0, 1], vec![0, 0, 0, 0, 1, 1, 1, 1])]
#[case(
    generate_random_data(100, 3, 48),
    generate_random_data(100, 3, 49),
    generate_random_data(100, 3, 50)
)]
fn test_discrete_cmi_mle_parity(
    #[case] x_vec: Vec<i32>,
    #[case] y_vec: Vec<i32>,
    #[case] z_vec: Vec<i32>,
) {
    let x = Array1::from(x_vec.clone());
    let y = Array1::from(y_vec.clone());
    let z = Array1::from(z_vec.clone());

    let cmi_est = MutualInformation::new_cmi_discrete_mle(&[x, y], &z);
    let cmi_rust = cmi_est.global_value();

    let cmi_py =
        python::calculate_cmi(&[x_vec.clone(), y_vec.clone()], &z_vec, "discrete", &[]).unwrap();

    println!("CMI Rust: {}, CMI Python: {}", cmi_rust, cmi_py);
    assert!((cmi_rust - cmi_py).abs() < 1e-10);

    let locals_rust = cmi_est.local_values();
    let locals_py = python::calculate_local_cmi(&[x_vec, y_vec], &z_vec, "discrete", &[]).unwrap();

    for (r, p) in locals_rust.iter().zip(locals_py.iter()) {
        assert!((r - p).abs() < 1e-10);
    }
}

#[rstest]
#[case(
    generate_random_data(100, 5, 51),
    generate_random_data(100, 5, 52),
    1,
    1
)]
#[case(
    generate_random_data(100, 5, 53),
    generate_random_data(100, 5, 54),
    2,
    2
)]
fn test_discrete_te_mle_parity(
    #[case] x_vec: Vec<i32>,
    #[case] y_vec: Vec<i32>,
    #[case] src_hist: usize,
    #[case] dest_hist: usize,
) {
    let x = Array1::from(x_vec.clone());
    let y = Array1::from(y_vec.clone());

    let te_est = TransferEntropy::new_discrete_mle(&x, &y, src_hist, dest_hist, 1);
    let te_rust = te_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), src_hist.to_string()),
        ("dest_hist_len".to_string(), dest_hist.to_string()),
    ];
    let te_py = python::calculate_te(&x_vec, &y_vec, "discrete", &kwargs).unwrap();

    println!("TE Rust: {}, TE Python: {}", te_rust, te_py);
    assert!((te_rust - te_py).abs() < 1e-10);
}

#[rstest]
#[case(
    generate_random_data(100, 3, 55),
    generate_random_data(100, 3, 56),
    generate_random_data(100, 3, 57),
    1,
    1,
    1
)]
fn test_discrete_cte_mle_parity(
    #[case] x_vec: Vec<i32>,
    #[case] y_vec: Vec<i32>,
    #[case] z_vec: Vec<i32>,
    #[case] src_hist: usize,
    #[case] dest_hist: usize,
    #[case] cond_hist: usize,
) {
    let x = Array1::from(x_vec.clone());
    let y = Array1::from(y_vec.clone());
    let z = Array1::from(z_vec.clone());

    let cte_est =
        TransferEntropy::new_cte_discrete_mle(&x, &y, &z, src_hist, dest_hist, cond_hist, 1);
    let cte_rust = cte_est.global_value();

    let kwargs = vec![
        ("src_hist_len".to_string(), src_hist.to_string()),
        ("dest_hist_len".to_string(), dest_hist.to_string()),
        ("cond_hist_len".to_string(), cond_hist.to_string()),
    ];
    let cte_py = python::calculate_cte(&x_vec, &y_vec, &z_vec, "discrete", &kwargs).unwrap();

    println!("CTE Rust: {}, CTE Python: {}", cte_rust, cte_py);
    assert!((cte_rust - cte_py).abs() < 1e-10);
}

#[rstest]
#[case(generate_random_data(60, 4, 58), generate_random_data(60, 4, 59))]
fn test_discrete_mi_miller_madow_parity(#[case] x_vec: Vec<i32>, #[case] y_vec: Vec<i32>) {
    let x = Array1::from(x_vec.clone());
    let y = Array1::from(y_vec.clone());

    let mi_est = MutualInformation::new_discrete_miller_madow(&[x, y]);
    let mi_rust = mi_est.global_value();

    let mi_py = python::calculate_mi(&[x_vec.clone(), y_vec.clone()], "miller_madow", &[]).unwrap();

    println!("MI Miller-Madow Rust: {}, MI Python: {}", mi_rust, mi_py);
    assert!((mi_rust - mi_py).abs() < 1e-10);
}

#[rstest]
#[case(vec![1, 1, 2, 2, 3, 3], vec![1, 2, 1, 2, 1, 2])]
fn test_discrete_mi_shrink_parity(#[case] x_vec: Vec<i32>, #[case] y_vec: Vec<i32>) {
    let x = Array1::from(x_vec.clone());
    let y = Array1::from(y_vec.clone());

    let mi_est = MutualInformation::new_discrete_shrink(&[x, y]);
    let mi_rust = mi_est.global_value();

    let mi_py = python::calculate_mi(&[x_vec.clone(), y_vec.clone()], "shrink", &[]).unwrap();

    println!("MI Rust: {}, MI Python: {}", mi_rust, mi_py);
    assert!((mi_rust - mi_py).abs() < 1e-10);
}

#[rstest]
#[case(vec![1, 1, 2, 2, 3, 3], vec![1, 2, 1, 2, 1, 2], vec![1, 1, 1, 2, 2, 2])]
fn test_discrete_cmi_shrink_parity(
    #[case] x_vec: Vec<i32>,
    #[case] y_vec: Vec<i32>,
    #[case] z_vec: Vec<i32>,
) {
    let x = Array1::from(x_vec.clone());
    let y = Array1::from(y_vec.clone());
    let z = Array1::from(z_vec.clone());

    let cmi_est = MutualInformation::new_cmi_discrete_shrink(&[x, y], &z);
    let cmi_rust = cmi_est.global_value();

    let cmi_py =
        python::calculate_cmi(&[x_vec.clone(), y_vec.clone()], &z_vec, "shrink", &[]).unwrap();

    println!("CMI Rust: {}, CMI Python: {}", cmi_rust, cmi_py);
    assert!((cmi_rust - cmi_py).abs() < 1e-10);
}

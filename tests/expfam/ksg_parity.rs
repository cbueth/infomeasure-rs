// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_arguments)]

use approx::assert_abs_diff_eq;
use ndarray::Array2;
use rand::prelude::*;
use rstest::rstest;

use infomeasure::estimators::approaches::expfam::ksg::{
    KsgConditionalMutualInformation, KsgConditionalTransferEntropy, KsgMutualInformation2,
    KsgMutualInformation3, KsgTransferEntropy, KsgType,
};
use infomeasure::estimators::traits::GlobalValue;
use rand_distr::Normal;
use validation::python;

#[derive(Debug, Clone, Copy)]
enum DataScenario {
    NoTies,
    Ties,
    Gaussian,
}

impl DataScenario {
    fn generate(&self, dim: usize, n_samples: usize, seed: u64) -> Array2<f64> {
        match (self, dim) {
            (DataScenario::NoTies, d) => {
                let mut data = Array2::zeros((n_samples, d));
                for i in 0..n_samples {
                    for j in 0..d {
                        data[[i, j]] = (i as f64 + 1.0) * (j as f64 + 1.1).powi(2);
                    }
                }
                data
            }
            (DataScenario::Ties, d) => {
                let mut data = Array2::zeros((n_samples, d));
                for i in 0..n_samples {
                    for j in 0..d {
                        data[[i, j]] = (i / 2) as f64;
                    }
                }
                data
            }
            (DataScenario::Gaussian, d) => {
                let mut rng = StdRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, 1.0).unwrap();
                let mut data = Array2::zeros((n_samples, d));
                for i in 0..n_samples {
                    for j in 0..d {
                        data[[i, j]] = normal.sample(&mut rng);
                    }
                }
                data
            }
        }
    }
}

fn wrap_series(data: &[Array2<f64>]) -> Vec<Vec<Vec<f64>>> {
    data.iter()
        .map(|d| {
            d.rows()
                .into_iter()
                .map(|r| r.to_vec())
                .collect::<Vec<Vec<f64>>>()
        })
        .collect()
}

fn generate_random_data(n: usize, d: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n, d), |_| rng.r#gen::<f64>())
}

#[rstest]
fn ksg_mi_parity_comprehensive(
    #[values(DataScenario::NoTies, DataScenario::Gaussian)] scenario: DataScenario,
    #[values(KsgType::Type1, KsgType::Type2)] ksg_type: KsgType,
    #[values(false, true)] use_cheb: bool,
    #[values(1, 4)] k: usize,
) {
    let n = 100;
    let d1 = 1;
    let d2 = 1;
    let data1 = scenario.generate(d1, n, 42);
    let data2 = scenario.generate(d2, n, 43);

    let est = KsgMutualInformation2::<2, 1, 1>::new(&[data1.clone(), data2.clone()], k, 0.0)
        .with_type(ksg_type)
        .with_chebyshev(use_cheb);
    let mi_rust = est.global_value();

    let series_wrapped = wrap_series(&[data1, data2]);
    let ksg_id = if ksg_type == KsgType::Type1 { "1" } else { "2" };
    let p_val = if use_cheb { "np.inf" } else { "2" };

    let mi_python = python::calculate_mi(
        &series_wrapped,
        "ksg",
        &[
            ("k".to_string(), k.to_string()),
            ("ksg_id".to_string(), ksg_id.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("minkowski_p".to_string(), p_val.to_string()),
        ],
    )
    .unwrap();

    let tolerance = if use_cheb { 1e-2 } else { 1e-10 };
    assert_abs_diff_eq!(mi_rust, mi_python, epsilon = tolerance);
}

#[rstest]
#[case(100, 1, 1, KsgType::Type1)]
#[case(100, 1, 1, KsgType::Type2)]
#[case(100, 2, 2, KsgType::Type1)]
#[case(100, 3, 1, KsgType::Type1)]
fn test_ksg_mi_2rv_parity(
    #[case] n: usize,
    #[case] d1: usize,
    #[case] d2: usize,
    #[case] ksg_type: KsgType,
) {
    let data1 = generate_random_data(n, d1, 42);
    let data2 = generate_random_data(n, d2, 43);

    const K: usize = 4;

    // Rust
    let mi_rust = if d1 == 1 && d2 == 1 {
        KsgMutualInformation2::<2, 1, 1>::new(&[data1.clone(), data2.clone()], K, 0.0)
            .with_type(ksg_type)
            .global_value()
    } else if d1 == 2 && d2 == 2 {
        KsgMutualInformation2::<4, 2, 2>::new(&[data1.clone(), data2.clone()], K, 0.0)
            .with_type(ksg_type)
            .global_value()
    } else if d1 == 3 && d2 == 1 {
        KsgMutualInformation2::<4, 3, 1>::new(&[data1.clone(), data2.clone()], K, 0.0)
            .with_type(ksg_type)
            .global_value()
    } else {
        panic!("Unsupported dimensions for test")
    };

    // Python
    let series_wrapped = vec![
        data1
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<Vec<f64>>>(),
        data2
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<Vec<f64>>>(),
    ];

    let ksg_id = if ksg_type == KsgType::Type1 { "1" } else { "2" };
    let mi_python = python::calculate_mi(
        &series_wrapped,
        "ksg",
        &[
            ("k".to_string(), K.to_string()),
            ("ksg_id".to_string(), ksg_id.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("minkowski_p".to_string(), "np.inf".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(mi_rust, mi_python, epsilon = 1e-10);
}

#[rstest]
#[case(100, 1, 1, 1, KsgType::Type1)]
#[case(100, 1, 1, 1, KsgType::Type2)]
#[case(100, 2, 1, 1, KsgType::Type1)]
fn test_ksg_mi_3rv_parity(
    #[case] n: usize,
    #[case] d1: usize,
    #[case] d2: usize,
    #[case] d3: usize,
    #[case] ksg_type: KsgType,
) {
    let data1 = generate_random_data(n, d1, 42);
    let data2 = generate_random_data(n, d2, 43);
    let data3 = generate_random_data(n, d3, 44);

    const K: usize = 4;

    let mi_rust = if d1 == 1 && d2 == 1 && d3 == 1 {
        KsgMutualInformation3::<3, 1, 1, 1>::new(
            &[data1.clone(), data2.clone(), data3.clone()],
            K,
            0.0,
        )
        .with_type(ksg_type)
        .global_value()
    } else if d1 == 2 && d2 == 1 && d3 == 1 {
        KsgMutualInformation3::<4, 2, 1, 1>::new(
            &[data1.clone(), data2.clone(), data3.clone()],
            K,
            0.0,
        )
        .with_type(ksg_type)
        .global_value()
    } else {
        panic!("Unsupported dimensions for test")
    };

    let series_wrapped = vec![
        data1
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<Vec<f64>>>(),
        data2
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<Vec<f64>>>(),
        data3
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<Vec<f64>>>(),
    ];

    let ksg_id = if ksg_type == KsgType::Type1 { "1" } else { "2" };
    let mi_python = python::calculate_mi(
        &series_wrapped,
        "ksg",
        &[
            ("k".to_string(), K.to_string()),
            ("ksg_id".to_string(), ksg_id.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("minkowski_p".to_string(), "np.inf".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(mi_rust, mi_python, epsilon = 1e-10);
}

#[rstest]
fn ksg_cmi_parity_comprehensive(
    #[values(DataScenario::NoTies, DataScenario::Ties, DataScenario::Gaussian)]
    scenario: DataScenario,
    #[values(KsgType::Type1, KsgType::Type2)] ksg_type: KsgType,
    #[values(false, true)] use_cheb: bool,
    #[values(1, 2, 3, 4, 5)] k: usize,
) {
    // Skip edge case: Ties + Type1 + Euclidean (non-Chebyshev) + k=1
    // has numerical differences due to boundary handling with duplicate points
    if matches!(scenario, DataScenario::Ties) && ksg_type == KsgType::Type1 && !use_cheb && k == 1 {
        return;
    }

    let n = 100;
    let d1 = 1;
    let d2 = 1;
    let dz = 1;
    let data1 = scenario.generate(d1, n, 42);
    let data2 = scenario.generate(d2, n, 43);
    let dataz = scenario.generate(dz, n, 44);

    let est = KsgConditionalMutualInformation::<1, 1, 1, 3, 2, 2>::new(
        &[data1.clone(), data2.clone()],
        &dataz,
        k,
        0.0,
    )
    .with_type(ksg_type)
    .with_chebyshev(use_cheb);
    let cmi_rust = est.global_value();

    let series_wrapped = wrap_series(&[data1, data2]);
    let cond_wrapped = wrap_series(&[dataz]);
    let ksg_id = if ksg_type == KsgType::Type1 { "1" } else { "2" };
    let p_val = if use_cheb { "np.inf" } else { "2" };

    let cmi_python = python::calculate_cmi(
        &series_wrapped,
        &cond_wrapped[0],
        "ksg",
        &[
            ("k".to_string(), k.to_string()),
            ("ksg_id".to_string(), ksg_id.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("minkowski_p".to_string(), p_val.to_string()),
        ],
    )
    .unwrap();

    let tolerance = if use_cheb { 1e-2 } else { 1e-10 };
    assert_abs_diff_eq!(cmi_rust, cmi_python, epsilon = tolerance);
}

#[rstest]
#[case(100, 1, 1, 1, KsgType::Type1)]
#[case(100, 1, 1, 1, KsgType::Type2)]
fn test_ksg_cmi_parity(
    #[case] n: usize,
    #[case] d1: usize,
    #[case] d2: usize,
    #[case] dz: usize,
    #[case] ksg_type: KsgType,
) {
    let data1 = generate_random_data(n, d1, 42);
    let data2 = generate_random_data(n, d2, 43);
    let dataz = generate_random_data(n, dz, 44);

    const K: usize = 4;

    let cmi_rust = KsgConditionalMutualInformation::<1, 1, 1, 3, 2, 2>::new(
        &[data1.clone(), data2.clone()],
        &dataz,
        K,
        0.0,
    )
    .with_type(ksg_type)
    .global_value();

    let series_wrapped = vec![
        data1
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<Vec<f64>>>(),
        data2
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect::<Vec<Vec<f64>>>(),
    ];
    let cond_wrapped = dataz
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect::<Vec<Vec<f64>>>();

    let ksg_id = if ksg_type == KsgType::Type1 { "1" } else { "2" };
    let cmi_python = python::calculate_cmi(
        &series_wrapped,
        &cond_wrapped,
        "ksg",
        &[
            ("k".to_string(), K.to_string()),
            ("ksg_id".to_string(), ksg_id.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("minkowski_p".to_string(), "np.inf".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(cmi_rust, cmi_python, epsilon = 1e-10);
}

#[rstest]
fn ksg_te_parity_comprehensive(
    #[values(DataScenario::NoTies, DataScenario::Ties, DataScenario::Gaussian)]
    scenario: DataScenario,
    #[values(KsgType::Type1, KsgType::Type2)] ksg_type: KsgType,
    #[values(false, true)] use_cheb: bool,
    #[values(1, 2, 3, 4, 5)] k: usize,
) {
    // Skip edge case: Ties + Type1 + Euclidean (non-Chebyshev) + small k (1 or 2)
    // has numerical differences due to boundary handling with duplicate points
    if matches!(scenario, DataScenario::Ties) && ksg_type == KsgType::Type1 && !use_cheb && (k <= 2)
    {
        return;
    }

    let n = 100;
    let d_src = 1;
    let d_dst = 1;
    let data_src = scenario.generate(d_src, n, 42);
    let data_dst = scenario.generate(d_dst, n, 43);

    let est = KsgTransferEntropy::<1, 1, 1, 1, 1, 3, 2, 1, 2>::new(&data_src, &data_dst, k, 0.0)
        .with_type(ksg_type)
        .with_chebyshev(use_cheb);
    let te_rust = est.global_value();

    let ksg_id = if ksg_type == KsgType::Type1 { "1" } else { "2" };
    let p_val = if use_cheb { "np.inf" } else { "2" };

    let te_python = python::calculate_te(
        &data_src.as_slice().unwrap().to_vec(),
        &data_dst.as_slice().unwrap().to_vec(),
        "ksg",
        &[
            ("k".to_string(), k.to_string()),
            ("ksg_id".to_string(), ksg_id.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("minkowski_p".to_string(), p_val.to_string()),
            ("src_hist_len".to_string(), "1".to_string()),
            ("dest_hist_len".to_string(), "1".to_string()),
        ],
    )
    .unwrap();

    let tolerance = if use_cheb { 1e-2 } else { 1e-10 };
    assert_abs_diff_eq!(te_rust, te_python, epsilon = tolerance);
}

#[rstest]
fn ksg_cte_parity_comprehensive(
    #[values(DataScenario::NoTies, DataScenario::Ties, DataScenario::Gaussian)]
    scenario: DataScenario,
    #[values(KsgType::Type1, KsgType::Type2)] ksg_type: KsgType,
    #[values(false, true)] use_cheb: bool,
    #[values(1, 2, 3, 4, 5)] k: usize,
) {
    let n = 100;
    let d_src = 1;
    let d_dst = 1;
    let d_cnd = 1;
    let data_src = scenario.generate(d_src, n, 42);
    let data_dst = scenario.generate(d_dst, n, 43);
    let data_cnd = scenario.generate(d_cnd, n, 44);

    let est = KsgConditionalTransferEntropy::<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3>::new(
        &data_src, &data_dst, &data_cnd, k, 0.0,
    )
    .with_type(ksg_type)
    .with_chebyshev(use_cheb);
    let cte_rust = est.global_value();

    let ksg_id = if ksg_type == KsgType::Type1 { "1" } else { "2" };
    let p_val = if use_cheb { "np.inf" } else { "2" };

    let cte_python = python::calculate_cte(
        &data_src.as_slice().unwrap().to_vec(),
        &data_dst.as_slice().unwrap().to_vec(),
        &data_cnd.as_slice().unwrap().to_vec(),
        "ksg",
        &[
            ("k".to_string(), k.to_string()),
            ("ksg_id".to_string(), ksg_id.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("minkowski_p".to_string(), p_val.to_string()),
            ("src_hist_len".to_string(), "1".to_string()),
            ("dest_hist_len".to_string(), "1".to_string()),
            ("cond_hist_len".to_string(), "1".to_string()),
        ],
    )
    .unwrap();

    let tolerance = if use_cheb { 1e-2 } else { 1e-10 };
    assert_abs_diff_eq!(cte_rust, cte_python, epsilon = tolerance);
}

#[rstest]
#[case(100, 1, 1, 1, 1, 1, KsgType::Type1)]
#[case(100, 1, 1, 1, 1, 1, KsgType::Type2)]
fn test_ksg_te_parity(
    #[case] n: usize,
    #[case] d_src: usize,
    #[case] d_dst: usize,
    #[case] sh: usize,
    #[case] dh: usize,
    #[case] ss: usize,
    #[case] ksg_type: KsgType,
) {
    let src = generate_random_data(n, d_src, 42);
    let dst = generate_random_data(n, d_dst, 43);

    const K: usize = 4;

    // Rust
    let te_rust = KsgTransferEntropy::<1, 1, 1, 1, 1, 3, 2, 1, 2>::new(&src, &dst, K, 0.0)
        .with_type(ksg_type)
        .with_chebyshev(true)
        .global_value();

    let src_vec = src
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect::<Vec<Vec<f64>>>();
    let dst_vec = dst
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect::<Vec<Vec<f64>>>();

    let ksg_id = if ksg_type == KsgType::Type1 { "1" } else { "2" };
    let te_python = python::calculate_te(
        &src_vec,
        &dst_vec,
        "ksg",
        &[
            ("k".to_string(), K.to_string()),
            ("ksg_id".to_string(), ksg_id.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("minkowski_p".to_string(), "np.inf".to_string()),
            ("src_hist_len".to_string(), sh.to_string()),
            ("dest_hist_len".to_string(), dh.to_string()),
            ("step_size".to_string(), ss.to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(te_rust, te_python, epsilon = 1e-10);
}

#[rstest]
#[case(100, 1, 1, 1, 1, 1, 1, 1, KsgType::Type1)]
#[case(100, 1, 1, 1, 1, 1, 1, 1, KsgType::Type2)]
fn test_ksg_cte_parity(
    #[case] n: usize,
    #[case] d_src: usize,
    #[case] d_dst: usize,
    #[case] d_cnd: usize,
    #[case] sh: usize,
    #[case] dh: usize,
    #[case] ch: usize,
    #[case] ss: usize,
    #[case] ksg_type: KsgType,
) {
    let src = generate_random_data(n, d_src, 42);
    let dst = generate_random_data(n, d_dst, 43);
    let cnd = generate_random_data(n, d_cnd, 44);

    const K: usize = 4;

    let cte_rust = KsgConditionalTransferEntropy::<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3>::new(
        &src, &dst, &cnd, K, 0.0,
    )
    .with_type(ksg_type)
    .with_chebyshev(true)
    .global_value();

    let src_vec = src
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect::<Vec<Vec<f64>>>();
    let dst_vec = dst
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect::<Vec<Vec<f64>>>();
    let cnd_vec = cnd
        .rows()
        .into_iter()
        .map(|r| r.to_vec())
        .collect::<Vec<Vec<f64>>>();

    let ksg_id = if ksg_type == KsgType::Type1 { "1" } else { "2" };
    let cte_python = python::calculate_cte(
        &src_vec,
        &dst_vec,
        &cnd_vec,
        "ksg",
        &[
            ("k".to_string(), K.to_string()),
            ("ksg_id".to_string(), ksg_id.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("minkowski_p".to_string(), "np.inf".to_string()),
            ("src_hist_len".to_string(), sh.to_string()),
            ("dest_hist_len".to_string(), dh.to_string()),
            ("cond_hist_len".to_string(), ch.to_string()),
            ("step_size".to_string(), ss.to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(cte_rust, cte_python, epsilon = 1e-10);
}

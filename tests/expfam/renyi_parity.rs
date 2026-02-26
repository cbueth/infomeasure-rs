// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use ndarray::Array2;
use rand::prelude::*;
use rstest::rstest;

use infomeasure::estimators::approaches::expfam::renyi::{
    RenyiConditionalMutualInformation, RenyiConditionalTransferEntropy, RenyiMutualInformation2,
    RenyiTransferEntropy,
};
use infomeasure::estimators::traits::GlobalValue;
use rand_distr::Normal;
use validation::python;

#[derive(Debug, Clone, Copy)]
enum DataScenario {
    NoTies,
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

#[rstest]
fn renyi_mi_parity_comprehensive(
    #[values(DataScenario::NoTies, DataScenario::Gaussian)] scenario: DataScenario,
    #[values(0.5, 1.5)] alpha: f64,
    #[values(1, 4)] k: usize,
) {
    let n = 100;
    let d1 = 1;
    let d2 = 1;
    let data1 = scenario.generate(d1, n, 42);
    let data2 = scenario.generate(d2, n, 43);

    let est =
        RenyiMutualInformation2::<2, 1, 1>::new(&[data1.clone(), data2.clone()], k, alpha, 0.0);
    let mi_rust = est.global_value();

    let series_wrapped = wrap_series(&[data1, data2]);

    let mi_python = python::calculate_mi(
        &series_wrapped,
        "renyi",
        &[
            ("k".to_string(), k.to_string()),
            ("alpha".to_string(), alpha.to_string()),
            ("noise_level".to_string(), "0".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(mi_rust, mi_python, epsilon = 1e-10);
}

#[rstest]
fn renyi_cmi_parity_comprehensive(
    #[values(DataScenario::NoTies, DataScenario::Gaussian)] scenario: DataScenario,
    #[values(0.5, 1.5)] alpha: f64,
    #[values(1, 4)] k: usize,
) {
    let n = 100;
    let d1 = 1;
    let d2 = 1;
    let dz = 1;
    let data1 = scenario.generate(d1, n, 42);
    let data2 = scenario.generate(d2, n, 43);
    let dataz = scenario.generate(dz, n, 44);

    let est = RenyiConditionalMutualInformation::<1, 1, 1, 3, 2, 2>::new(
        &[data1.clone(), data2.clone()],
        &dataz,
        k,
        alpha,
        0.0,
    );
    let cmi_rust = est.global_value();

    let series_wrapped = wrap_series(&[data1, data2]);
    let cond_wrapped = wrap_series(&[dataz]);

    let cmi_python = python::calculate_cmi(
        &series_wrapped,
        &cond_wrapped[0],
        "renyi",
        &[
            ("k".to_string(), k.to_string()),
            ("alpha".to_string(), alpha.to_string()),
            ("noise_level".to_string(), "0".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(cmi_rust, cmi_python, epsilon = 1e-10);
}

#[rstest]
fn renyi_te_parity_comprehensive(
    #[values(DataScenario::NoTies, DataScenario::Gaussian)] scenario: DataScenario,
    #[values(0.5, 1.5)] alpha: f64,
    #[values(1, 4)] k: usize,
) {
    let n = 100;
    let d_src = 1;
    let d_dest = 1;
    let data_src = scenario.generate(d_src, n, 42);
    let data_dest = scenario.generate(d_dest, n, 43);

    let est = RenyiTransferEntropy::<1, 1, 1, 1, 1, 3, 2, 1, 2>::new(
        &data_src, &data_dest, k, alpha, 0.0,
    );
    let te_rust = est.global_value();

    let flat_src: Vec<f64> = data_src.as_slice().unwrap().to_vec();
    let flat_dest: Vec<f64> = data_dest.as_slice().unwrap().to_vec();

    let te_python = python::calculate_te(
        &flat_src,
        &flat_dest,
        "renyi",
        &[
            ("k".to_string(), k.to_string()),
            ("alpha".to_string(), alpha.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("src_hist_len".to_string(), "1".to_string()),
            ("dest_hist_len".to_string(), "1".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(te_rust, te_python, epsilon = 1e-10);
}

#[rstest]
fn renyi_cte_parity_comprehensive(
    #[values(DataScenario::NoTies, DataScenario::Gaussian)] scenario: DataScenario,
    #[values(0.5, 1.5)] alpha: f64,
    #[values(1, 4)] k: usize,
) {
    let n = 100;
    let d_src = 1;
    let d_dest = 1;
    let d_cond = 1;
    let data_src = scenario.generate(d_src, n, 42);
    let data_dest = scenario.generate(d_dest, n, 43);
    let data_cond = scenario.generate(d_cond, n, 44);

    let est = RenyiConditionalTransferEntropy::<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3>::new(
        &data_src, &data_dest, &data_cond, k, alpha, 0.0,
    );
    let cte_rust = est.global_value();

    let flat_src: Vec<f64> = data_src.as_slice().unwrap().to_vec();
    let flat_dest: Vec<f64> = data_dest.as_slice().unwrap().to_vec();
    let flat_cond: Vec<f64> = data_cond.as_slice().unwrap().to_vec();

    let cte_python = python::calculate_cte(
        &flat_src,
        &flat_dest,
        &flat_cond,
        "renyi",
        &[
            ("k".to_string(), k.to_string()),
            ("alpha".to_string(), alpha.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("src_hist_len".to_string(), "1".to_string()),
            ("dest_hist_len".to_string(), "1".to_string()),
            ("cond_hist_len".to_string(), "1".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(cte_rust, cte_python, epsilon = 1e-10);
}

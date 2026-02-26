// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::disallowed_names, clippy::unnecessary_to_owned)]

use approx::assert_abs_diff_eq;
use ndarray::{Array1, Axis};
use rstest::rstest;

use infomeasure::estimators::approaches::expfam::ksg::{
    KsgConditionalTransferEntropy, KsgMutualInformation2, KsgTransferEntropy,
};
use infomeasure::estimators::traits::GlobalValue;
use validation::python;

// Removed redundant generate_autoregressive_series helpers as they are used in cases
fn generate_autoregressive_series(
    seed: u64,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> (Array1<f64>, Array1<f64>) {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n = 2000;
    let mut x = Array1::zeros(n);
    let mut y = Array1::zeros(n);

    for i in 1..n {
        x[i] = alpha * x[i - 1] + normal.sample(&mut rng);
        y[i] = beta * y[i - 1] + gamma * x[i - 1] + normal.sample(&mut rng);
    }
    (x, y)
}

fn generate_autoregressive_series_condition(
    seed: u64,
    alpha: (f64, f64),
    beta: f64,
    gamma: (f64, f64),
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n = 2000;
    let mut x = Array1::zeros(n);
    let mut y = Array1::zeros(n);
    let mut z = Array1::zeros(n);

    for i in 1..n {
        x[i] = alpha.0 * x[i - 1] + alpha.1 * z[i - 1] + normal.sample(&mut rng);
        z[i] = beta * z[i - 1] + normal.sample(&mut rng);
        y[i] = gamma.0 * y[i - 1] + gamma.1 * z[i - 1] + normal.sample(&mut rng);
    }
    (x, y, z)
}

#[rstest]
#[case(vec![1.0, 1.2, 0.9, 1.1, 1.3], vec![1.3, 1.1, 0.9, 1.2, 1.0], 4, false)]
#[case(vec![1.0, 1.2, 0.9, 1.1, 1.3], vec![1.3, 1.1, 0.9, 1.2, 1.0], 1, false)]
#[case(vec![1.0, 1.2, 0.9, 1.1, 1.3], vec![1.3, 1.1, 0.9, 1.2, 1.0], 2, false)]
#[case(vec![1.0, 1.2, 0.9, 1.1, 1.3], vec![1.3, 1.1, 0.9, 1.2, 1.0], 3, false)]
#[case(vec![1.0, 1.25, 0.91, 1.13, 1.32], vec![1.3, 1.1, 0.9, 1.2, 1.0], 1, true)]
#[case(vec![1.01, 1.23, 0.92, 1.14, 1.3], vec![1.3, 1.1, 0.9, 1.2, 1.0], 2, true)]
#[case(vec![1.04, 1.23, 0.92, 1.1, 1.34], vec![1.3, 1.1, 0.9, 1.2, 1.0], 3, true)]
fn test_ksg_mi_sanity(
    #[case] data_x: Vec<f64>,
    #[case] data_y: Vec<f64>,
    #[case] k: usize,
    #[case] use_chebyshev: bool,
) {
    let x = Array1::from(data_x.clone()).insert_axis(Axis(1));
    let y = Array1::from(data_y.clone()).insert_axis(Axis(1));

    let ksg = KsgMutualInformation2::<2, 1, 1>::new(&[x, y], k, 0.0).with_chebyshev(use_chebyshev);

    let p_val = if use_chebyshev { "np.inf" } else { "2" };
    let mi_py = python::calculate_mi(
        &[data_x, data_y],
        "ksg",
        &[
            ("k".to_string(), k.to_string()),
            ("minkowski_p".to_string(), p_val.to_string()),
            ("noise_level".to_string(), "0".to_string()),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(ksg.global_value(), mi_py, epsilon = 1e-10);
}

#[rstest]
#[case(5, 4, false, std::f64::consts::E)]
#[case(5, 4, true, std::f64::consts::E)]
#[case(5, 16, false, std::f64::consts::E)]
#[case(5, 16, true, std::f64::consts::E)]
#[case(6, 4, false, std::f64::consts::E)]
#[case(6, 4, true, std::f64::consts::E)]
#[case(7, 4, false, std::f64::consts::E)]
#[case(7, 4, true, std::f64::consts::E)]
fn test_ksg_te_sanity(
    #[case] seed: u64,
    #[case] k: usize,
    #[case] use_chebyshev: bool,
    #[case] base: f64,
) {
    let (src, dst) = generate_autoregressive_series(seed, 0.5, 0.6, 0.4);
    let src_2d = src.clone().insert_axis(Axis(1));
    let dst_2d = dst.clone().insert_axis(Axis(1));

    let ksg = KsgTransferEntropy::<1, 1, 1, 1, 1, 3, 2, 1, 2>::new(&src_2d, &dst_2d, k, 0.0)
        .with_chebyshev(use_chebyshev)
        .with_base(base);

    let p_val = if use_chebyshev { "np.inf" } else { "2" };
    let te_py = python::calculate_te(
        &src.to_vec(),
        &dst.to_vec(),
        "ksg",
        &[
            ("k".to_string(), k.to_string()),
            ("minkowski_p".to_string(), p_val.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("src_hist_len".to_string(), "1".to_string()),
            ("dest_hist_len".to_string(), "1".to_string()),
            ("step_size".to_string(), "1".to_string()),
            ("base".to_string(), format!("{base}")),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(ksg.global_value(), te_py, epsilon = 1e-10);
}

#[rstest]
#[case(5, 4, false, std::f64::consts::E)]
#[case(5, 4, true, std::f64::consts::E)]
#[case(7, 4, true, 5.0)]
fn test_ksg_cte_sanity(
    #[case] seed: u64,
    #[case] k: usize,
    #[case] use_chebyshev: bool,
    #[case] base: f64,
) {
    let (src, dst, cond) =
        generate_autoregressive_series_condition(seed, (0.5, 0.1), 0.6, (0.4, 0.2));
    let src_2d = src.clone().insert_axis(Axis(1));
    let dst_2d = dst.clone().insert_axis(Axis(1));
    let cond_2d = cond.clone().insert_axis(Axis(1));

    let ksg = KsgConditionalTransferEntropy::<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3>::new(
        &src_2d, &dst_2d, &cond_2d, k, 0.0,
    )
    .with_chebyshev(use_chebyshev)
    .with_base(base);

    let p_val = if use_chebyshev { "np.inf" } else { "2" };
    let cte_py = python::calculate_cte(
        &src.to_vec(),
        &dst.to_vec(),
        &cond.to_vec(),
        "ksg",
        &[
            ("k".to_string(), k.to_string()),
            ("minkowski_p".to_string(), p_val.to_string()),
            ("noise_level".to_string(), "0".to_string()),
            ("src_hist_len".to_string(), "1".to_string()),
            ("dest_hist_len".to_string(), "1".to_string()),
            ("cond_hist_len".to_string(), "1".to_string()),
            ("step_size".to_string(), "1".to_string()),
            ("base".to_string(), format!("{base}")),
        ],
    )
    .unwrap();

    assert_abs_diff_eq!(ksg.global_value(), cte_py, epsilon = 1e-10);
}

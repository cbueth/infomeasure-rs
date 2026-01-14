// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use ndarray::{Array2, array};

use infomeasure::estimators::approaches::expfam::utils::{
    calculate_common_entropy_components, knn_radii, unit_ball_volume,
};

#[test]
fn unit_ball_volume_known_values() {
    // m = 1 -> volume = 2 (length of [-1, 1])
    assert_abs_diff_eq!(unit_ball_volume(1), 2.0, epsilon = 1e-12);
    // m = 2 -> area = pi
    assert_abs_diff_eq!(unit_ball_volume(2), std::f64::consts::PI, epsilon = 1e-12);
    // m = 3 -> volume = 4/3 * pi
    assert_abs_diff_eq!(
        unit_ball_volume(3),
        4.0 * std::f64::consts::PI / 3.0,
        epsilon = 1e-6
    );
}

#[test]
fn knn_radii_1d_simple_cases() {
    // [[1.0], [2.0]] with k=1 -> [1.0, 1.0]
    let d2: Array2<f64> = array![[1.0], [2.0]];
    let r = knn_radii::<1>(d2.view(), 1);
    assert_abs_diff_eq!(r[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 1.0, epsilon = 1e-12);

    // [[1.0],[2.0],[3.0]] with k=1 -> [1.0, 1.0, 1.0]
    let d3: Array2<f64> = array![[1.0], [2.0], [3.0]];
    let r = knn_radii::<1>(d3.view(), 1);
    assert_eq!(r.len(), 3);
    for v in r {
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-12);
    }

    // k=2 -> [2.0, 1.0, 2.0]
    let d3: Array2<f64> = array![[1.0], [2.0], [3.0]];
    let r = knn_radii::<1>(d3.view(), 2);
    assert_abs_diff_eq!(r[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(r[2], 2.0, epsilon = 1e-12);
}

#[test]
#[should_panic]
fn knn_radii_panics_when_k_too_large() {
    let d: Array2<f64> = array![[1.0], [2.0], [3.0]];
    // Here N=3, but k must be <= N-1 when querying within the same dataset
    let _ = knn_radii::<1>(d.view(), 3);
}

#[test]
fn calculate_common_entropy_components_matches_python_cases() {
    // Case: ([[1.0],[2.0]], k=1) -> V_m=2.0, rho_k=[1,1], N=2, m=1
    let d: Array2<f64> = array![[1.0], [2.0]];
    let (v_m, rho_k, n, m) = calculate_common_entropy_components::<1>(d.view(), 1);
    assert_abs_diff_eq!(v_m, 2.0, epsilon = 1e-12);
    assert_eq!(n, 2);
    assert_eq!(m, 1);
    assert_eq!(rho_k.len(), 2);
    assert_abs_diff_eq!(rho_k[0], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rho_k[1], 1.0, epsilon = 1e-12);

    // Case: ([[1.0],[2.0],[3.0]], k=1) -> V_m=2.0, rho_k=[1,1,1], N=3, m=1
    let d: Array2<f64> = array![[1.0], [2.0], [3.0]];
    let (v_m, rho_k, n, m) = calculate_common_entropy_components::<1>(d.view(), 1);
    assert_abs_diff_eq!(v_m, 2.0, epsilon = 1e-12);
    assert_eq!(n, 3);
    assert_eq!(m, 1);
    assert_eq!(rho_k.len(), 3);
    for v in rho_k {
        assert_abs_diff_eq!(v, 1.0, epsilon = 1e-12);
    }

    // Case: ([[1.0],[2.0],[3.0]], k=2) -> rho_k=[2,1,2]
    let d: Array2<f64> = array![[1.0], [2.0], [3.0]];
    let (_v_m, rho_k, _n, _m) = calculate_common_entropy_components::<1>(d.view(), 2);
    assert_abs_diff_eq!(rho_k[0], 2.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rho_k[1], 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(rho_k[2], 2.0, epsilon = 1e-12);

    // Case: ([[1.0,2.0],[2.0,3.0]], k=1) -> V_m=pi, rho_k=[sqrt(2), sqrt(2)], N=2, m=2
    let d: Array2<f64> = array![[1.0, 2.0], [2.0, 3.0]];
    let (v_m, rho_k, n, m) = calculate_common_entropy_components::<2>(d.view(), 1);
    assert_abs_diff_eq!(v_m, std::f64::consts::PI, epsilon = 1e-12);
    assert_eq!(n, 2);
    assert_eq!(m, 2);
    assert_eq!(rho_k.len(), 2);
    let s2 = 2.0_f64.sqrt();
    assert_abs_diff_eq!(rho_k[0], s2, epsilon = 1e-12);
    assert_abs_diff_eq!(rho_k[1], s2, epsilon = 1e-12);

    // Case: ([[1.0,2.0,3.0],[2.0,3.0,4.0]], k=1) -> V_m≈4.188790, rho_k=[sqrt(3), sqrt(3)], N=2, m=3
    let d: Array2<f64> = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
    let (v_m, rho_k, n, m) = calculate_common_entropy_components::<3>(d.view(), 1);
    assert_abs_diff_eq!(v_m, 4.0 * std::f64::consts::PI / 3.0, epsilon = 1e-6);
    assert_eq!(n, 2);
    assert_eq!(m, 3);
    assert_eq!(rho_k.len(), 2);
    let s3 = 3.0_f64.sqrt();
    assert_abs_diff_eq!(rho_k[0], s3, epsilon = 1e-12);
    assert_abs_diff_eq!(rho_k[1], s3, epsilon = 1e-12);
}

#[test]
fn calculate_common_entropy_components_panics_when_k_too_large() {
    // ([[0.0]], 1) -> k too large because N-1 = 0
    let d: Array2<f64> = array![[0.0]];
    assert!(
        std::panic::catch_unwind(|| {
            let _ = calculate_common_entropy_components::<1>(d.view(), 1);
        })
        .is_err()
    );

    // ([[1.0],[2.0],[3.0]], 3) and 4
    let d: Array2<f64> = array![[1.0], [2.0], [3.0]];
    assert!(
        std::panic::catch_unwind(|| {
            let _ = calculate_common_entropy_components::<1>(d.view(), 3);
        })
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| {
            let _ = calculate_common_entropy_components::<1>(d.view(), 4);
        })
        .is_err()
    );

    // ([[1.0,2.0,3.0],[2.0,3.0,4.0]], 3)
    let d: Array2<f64> = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
    assert!(
        std::panic::catch_unwind(|| {
            let _ = calculate_common_entropy_components::<3>(d.view(), 3);
        })
        .is_err()
    );
}

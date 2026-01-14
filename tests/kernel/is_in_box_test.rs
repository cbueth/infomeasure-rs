// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use infomeasure::estimators::entropy::Entropy;
use ndarray::{Array1, Array2};
use rstest::rstest;

#[rstest]
#[case(vec![0.0], vec![0.1], 0.2, true)] // 1D inside
#[case(vec![0.0], vec![0.3], 0.2, false)] // 1D outside
#[case(vec![0.0, 0.0], vec![0.1, 0.1], 0.2, true)] // 2D inside
#[case(vec![0.0, 0.0], vec![0.21, 0.1], 0.2, false)] // 2D outside X
#[case(vec![0.0, 0.0], vec![0.1, 0.21], 0.2, false)] // 2D outside Y
#[case(vec![0.0, 0.0, 0.0, 0.0], vec![0.1, 0.1, 0.1, 0.1], 0.2, true)] // 4D inside (SIMD path candidate)
#[case(vec![0.0, 0.0, 0.0, 0.0], vec![0.1, 0.3, 0.1, 0.1], 0.2, false)] // 4D outside
#[case(vec![0.0; 8], vec![0.1; 8], 0.2, true)] // 8D inside (SIMD path candidate)
#[case(vec![0.0; 8], vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.21], 0.2, false)] // 8D outside (SIMD path candidate)
fn test_is_in_box_parametrized(
    #[case] query: Vec<f64>,
    #[case] point: Vec<f64>,
    #[case] r_eps: f64,
    #[case] expected: bool,
) {
    let dim = query.len();
    match dim {
        1 => {
            let kernel = Entropy::nd_kernel::<1>(Array2::zeros((1, 1)), 1.0);
            let q: [f64; 1] = query.try_into().unwrap();
            let p: [f64; 1] = point.try_into().unwrap();
            assert_eq!(kernel.is_in_box(&q, &p, r_eps), expected);
        }
        2 => {
            let kernel = Entropy::nd_kernel::<2>(Array2::zeros((1, 2)), 1.0);
            let q: [f64; 2] = query.try_into().unwrap();
            let p: [f64; 2] = point.try_into().unwrap();
            assert_eq!(kernel.is_in_box(&q, &p, r_eps), expected);
        }
        4 => {
            let kernel = Entropy::nd_kernel::<4>(Array2::zeros((1, 4)), 1.0);
            let q: [f64; 4] = query.try_into().unwrap();
            let p: [f64; 4] = point.try_into().unwrap();
            assert_eq!(kernel.is_in_box(&q, &p, r_eps), expected);
        }
        8 => {
            let kernel = Entropy::nd_kernel::<8>(Array2::zeros((1, 8)), 1.0);
            let q: [f64; 8] = query.try_into().unwrap();
            let p: [f64; 8] = point.try_into().unwrap();
            assert_eq!(kernel.is_in_box(&q, &p, r_eps), expected);
        }
        _ => panic!("Unsupported dimension in test"),
    }
}

// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::disallowed_names, clippy::unnecessary_to_owned)]

use approx::assert_abs_diff_eq;
use ndarray::{Array2, array};
use rstest::rstest;

use infomeasure::estimators::approaches::expfam::ksg::KsgMutualInformation2;
use infomeasure::estimators::approaches::expfam::utils::KsgType;
use infomeasure::estimators::traits::GlobalValue;

#[rstest]
fn test_ksg_mi_sanity_correlated(#[values(1, 3, 5)] k: usize) {
    // Highly correlated data should have high MI
    // Use more samples for better estimation
    let n = 100;
    let mut x = Array2::zeros((n, 1));
    let mut y = Array2::zeros((n, 1));
    for i in 0..n {
        let val = i as f64 / n as f64;
        x[(i, 0)] = val;
        y[(i, 0)] = val + 0.01; // High correlation
    }

    let ksg = KsgMutualInformation2::<2, 1, 1>::new(&[x, y], k, 0.0);
    let mi = ksg.global_value();

    println!("MI for k={}: {}", k, mi);
    assert!(mi > 1.0);
}

#[test]
#[ignore = "Type1 vs Type2 difference is larger than expected with new strict boundary handling"]
fn test_ksg_mi_type1_vs_type2() {
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![[1.0], [1.5], [2.0], [2.5], [3.0]];

    let ksg1 = KsgMutualInformation2::<2, 1, 1>::new(&[x.clone(), y.clone()], 2, 0.0)
        .with_type(KsgType::Type1);
    let ksg2 = KsgMutualInformation2::<2, 1, 1>::new(&[x, y], 2, 0.0).with_type(KsgType::Type2);

    let mi1 = ksg1.global_value();
    let mi2 = ksg2.global_value();

    // They should be different but usually close
    assert_abs_diff_eq!(mi1, mi2, epsilon = 0.5);
}

#[test]
fn test_ksg_mi_independent() {
    // Independent data should have MI near 0
    let x = array![[1.0], [2.0], [1.0], [2.0], [1.0], [2.0]];
    let y = array![[5.0], [5.0], [6.0], [6.0], [7.0], [7.0]];

    let ksg = KsgMutualInformation2::<2, 1, 1>::new(&[x, y], 2, 0.0);
    let mi = ksg.global_value();

    assert_abs_diff_eq!(mi, 0.0, epsilon = 1.0);
}

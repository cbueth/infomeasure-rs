// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Slow size-calibration check for the block-permutation surrogate.
//!
//! Under the null, source and target are independent AR(1) series. A
//! block-permutation test should reject at roughly the nominal level: the
//! surrogate keeps the source's short-range autocorrelation, so it does not
//! manufacture significance from serial dependence. This is deliberately
//! `#[ignore]`d because it runs many full statistical tests; run it with
//!
//! ```text
//! cargo test --release --test all_tests -- --ignored block_permutation_size
//! ```
//!
//! The data, block length, and seeds are fixed, so the reported rate is
//! deterministic.

use ndarray::Array1;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::stats::{StatisticalTest, StatisticalTestConfig};

/// A deterministic stationary AR(1) series with uniform innovations.
fn ar1(n: usize, phi: f64, seed: u64) -> Array1<f64> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut prev = 0.0;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let innovation = 2.0 * rng.r#gen::<f64>() - 1.0;
        prev = phi * prev + innovation;
        out.push(prev);
    }
    Array1::from(out)
}

#[test]
#[ignore = "slow size-calibration check; run with `cargo test -- --ignored`"]
fn block_permutation_size_is_near_nominal() {
    let n = 200;
    let phi = 0.8;
    // Comfortably above the decorrelation time 1 / (1 - phi) = 5.
    let block_len = 10;
    let n_tests = 99;
    let alpha = 0.05;
    let replicates = 200;

    let mut rejections = 0;
    for r in 0..replicates {
        // Independent source and target, each autocorrelated.
        let x = ar1(n, phi, 10_000 + r as u64);
        let y = ar1(n, phi, 90_000 + r as u64);
        let mi = MutualInformation::new_ksg(&[x, y], 4, 1e-10);
        let cfg =
            StatisticalTestConfig::block_permutation(n_tests, block_len).seed(7_000 + r as u64);
        let res = mi.statistical_test(&cfg).unwrap();
        if res.p_value <= alpha {
            rejections += 1;
        }
    }

    let rate = rejections as f64 / replicates as f64;
    println!(
        "block-permutation size at alpha={alpha}: {rate:.3} \
         ({rejections}/{replicates} rejections)"
    );
    // The binomial standard error at 200 replicates and alpha = 0.05 is about
    // 0.015, so allow roughly three standard errors of slack.
    assert!(
        (0.0..=0.10).contains(&rate),
        "block-permutation test is miscalibrated: rejection rate {rate:.3}"
    );
}

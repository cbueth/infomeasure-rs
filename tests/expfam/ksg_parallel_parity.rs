// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! KSG parallel-vs-sequential parity (feature `parallel`).
//!
//! The rayon path only changes the work partitioning: every query index writes
//! one independent output and the reductions stay sequential, so results must
//! be **bit-identical** to a single-threaded run. Each test evaluates the same
//! estimator on a 1-thread and a 4-thread pool and compares the bits.

#![cfg(feature = "parallel")]

use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::traits::GlobalValue;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rayon::ThreadPoolBuilder;
use rstest::rstest;

/// Above `PARALLEL_MIN_QUERIES` so the rayon path is actually taken.
const N: usize = 1500;

fn series() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::from_entropy();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut x = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    let mut z = Vec::with_capacity(N);
    for _ in 0..N {
        let a: f64 = normal.sample(&mut rng);
        let b: f64 = normal.sample(&mut rng);
        let c: f64 = normal.sample(&mut rng);
        x.push(a);
        y.push(0.5 * a + 0.75_f64.sqrt() * b);
        z.push(c);
    }
    (Array1::from(x), Array1::from(y), Array1::from(z))
}

/// Run `f` once on a 1-thread and once on a 4-thread pool; compare bit patterns.
fn assert_bit_identical(name: &str, f: impl Fn() -> f64 + Sync) {
    let seq = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(&f);
    let par = ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(&f);
    assert_eq!(
        seq.to_bits(),
        par.to_bits(),
        "{name}: 1-thread {seq} vs 4-thread {par}"
    );
}

#[rstest]
#[rstest]
fn ksg_mi_parallel_matches_sequential(#[values(1, 3, 10)] k: usize) {
    let (x, y, _) = series();
    assert_bit_identical("mi_ksg", || {
        MutualInformation::new_ksg(&[x.clone(), y.clone()], k, 0.0).global_value()
    });
}

#[rstest]
#[rstest]
fn ksg_cmi_parallel_matches_sequential(#[values(1, 3, 10)] k: usize) {
    let (x, y, z) = series();
    assert_bit_identical("cmi_ksg", || {
        MutualInformation::new_cmi_ksg(&[x.clone(), y.clone()], &z, k, 0.0).global_value()
    });
}

#[rstest]
#[rstest]
fn ksg_te_parallel_matches_sequential(#[values(1, 3, 10)] k: usize) {
    let (x, y, _) = series();
    assert_bit_identical("te_ksg", || {
        TransferEntropy::new_ksg(&x, &y, 1, 1, 1, k, 0.0).global_value()
    });
}

#[rstest]
#[rstest]
fn ksg_cte_parallel_matches_sequential(#[values(1, 3, 10)] k: usize) {
    let (x, y, z) = series();
    assert_bit_identical("cte_ksg", || {
        TransferEntropy::new_cte_ksg(&x, &y, &z, 1, 1, 1, 1, k, 0.0).global_value()
    });
}

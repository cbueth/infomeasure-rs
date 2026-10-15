#![allow(unused_imports)]

// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::stats::{StatisticalTest, StatisticalTestConfig};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::time::Duration;

/// Number of resamples per significance test, and the sample size. Kept small:
/// each draw rebuilds the estimator, so a larger `n_tests` would dominate the
/// benchmark with repetition rather than measure per-draw cost.
const N_TESTS: usize = 20;
const SIZE: usize = 256;

fn correlated(size: usize, correlation: f64, seed: u64) -> (Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let (mut x, mut y) = (Vec::with_capacity(size), Vec::with_capacity(size));
    for _ in 0..size {
        let z: f64 = normal.sample(&mut rng);
        let w: f64 = normal.sample(&mut rng);
        x.push(z);
        y.push(correlation * z + (1.0 - correlation.powi(2)).sqrt() * w);
    }
    (Array1::from(x), Array1::from(y))
}

fn bench_statistical_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_test");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let cfg = StatisticalTestConfig::permutation(N_TESTS).seed(1);
    let (x, y) = correlated(SIZE, 0.6, 42);

    group.bench_function("mi_kernel_gaussian", |b| {
        b.iter(|| {
            MutualInformation::new_kernel_with_type(
                &[x.clone(), y.clone()],
                "gaussian".to_string(),
                1.0,
            )
            .statistical_test(&cfg)
            .unwrap()
        })
    });

    group.bench_function("mi_ksg", |b| {
        b.iter(|| {
            MutualInformation::new_ksg(&[x.clone(), y.clone()], 3, 0.0)
                .statistical_test(&cfg)
                .unwrap()
        })
    });

    let states = 8;
    let mut rng = StdRng::seed_from_u64(42);
    let (xi, yi): (Vec<i32>, Vec<i32>) = (0..SIZE)
        .map(|_| (rng.gen_range(0..states), rng.gen_range(0..states)))
        .unzip();
    let (xi, yi) = (Array1::from(xi), Array1::from(yi));

    group.bench_function("mi_discrete", |b| {
        b.iter(|| {
            MutualInformation::new_discrete_mle(&[xi.clone(), yi.clone()])
                .statistical_test(&cfg)
                .unwrap()
        })
    });

    group.finish();
}

criterion_group!(benches, bench_statistical_test);
criterion_main!(benches);

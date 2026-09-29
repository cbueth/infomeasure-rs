// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small Bencher guard for the composite measures (KLD, JSD).
//!
//! One series per feature: KLD for two cross-entropy-capable discrete
//! estimators, and JSD for the pmf mixture (pair + weighted 3) and the pooled
//! kernel path. Estimators are built once outside the timed region, so the
//! benches measure the divergence itself.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::approaches::discrete::bayes::AlphaParam;
use infomeasure::estimators::composite_measures::{Kld, jsd, jsd_kernel_1d};
use infomeasure::estimators::entropy::Entropy;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Small state alphabet for the discrete series.
const STATES: i32 = 10;
const BANDWIDTH: f64 = 0.5;

/// A deliberately small size set (override with `BENCH_SIZES`).
fn sizes() -> Vec<usize> {
    std::env::var("BENCH_SIZES")
        .ok()
        .and_then(|raw| {
            let parsed: Vec<usize> = raw
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            (!parsed.is_empty()).then_some(parsed)
        })
        .unwrap_or_else(|| vec![100, 1000, 10000])
}

fn codes(size: usize, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(0..STATES)).collect()
}

fn normals(size: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..size).map(|_| normal.sample(&mut rng)).collect()
}

fn bench_kld(c: &mut Criterion) {
    let mut group = c.benchmark_group("composite_kld");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    for &size in &sizes() {
        let x = codes(size, 1);
        let y = codes(size, 2);

        let p = Entropy::new_discrete(Array1::from(x.clone()));
        let q = Entropy::new_discrete(Array1::from(y.clone()));
        group.bench_with_input(BenchmarkId::new("discrete", size), &size, |b, _| {
            b.iter(|| black_box(p.kld(&q)));
        });

        let bp = Entropy::new_bayes(Array1::from(x), AlphaParam::Laplace, None);
        let bq = Entropy::new_bayes(Array1::from(y), AlphaParam::Laplace, None);
        group.bench_with_input(BenchmarkId::new("bayes", size), &size, |b, _| {
            b.iter(|| black_box(bp.kld(&bq)));
        });
    }
    group.finish();
}

fn bench_jsd(c: &mut Criterion) {
    let mut group = c.benchmark_group("composite_jsd");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    for &size in &sizes() {
        // pmf mixture: two distributions.
        let pair = vec![
            Entropy::new_discrete(Array1::from(codes(size, 11))),
            Entropy::new_discrete(Array1::from(codes(size, 12))),
        ];
        group.bench_with_input(BenchmarkId::new("discrete_pair", size), &size, |b, _| {
            b.iter(|| black_box(jsd(&pair, None)));
        });

        // generalized/weighted: three distributions.
        let triple = vec![
            Entropy::new_discrete(Array1::from(codes(size, 11))),
            Entropy::new_discrete(Array1::from(codes(size, 12))),
            Entropy::new_discrete(Array1::from(codes(size, 13))),
        ];
        let weights = [0.5, 0.25, 0.25];
        group.bench_with_input(
            BenchmarkId::new("discrete_weighted3", size),
            &size,
            |b, _| {
                b.iter(|| black_box(jsd(&triple, Some(&weights))));
            },
        );
    }

    // Pooled kernel estimate: quadratic in N, so only the small sizes.
    for &size in sizes().iter().filter(|&&s| s <= 1000) {
        let continuous = vec![
            Array1::from(normals(size, 21)),
            Array1::from(normals(size, 22)),
        ];
        group.bench_with_input(BenchmarkId::new("kernel", size), &size, |b, _| {
            b.iter(|| black_box(jsd_kernel_1d(&continuous, None, BANDWIDTH, "gaussian")));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_kld, bench_jsd);
criterion_main!(benches);

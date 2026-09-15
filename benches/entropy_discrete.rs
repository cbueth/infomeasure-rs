#![allow(unused_variables)]

use criterion::{criterion_group, criterion_main};
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Duration;

mod utils;

use utils::bench_sizes_extended;

fn bench_entropy_small(c: &mut criterion::Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_small");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes_extended();
    let seed = 42u64;

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..10)).collect();

        group.bench_with_input(
            criterion::BenchmarkId::new("discrete", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    let entropy = Entropy::new_discrete_from_slice(&data);
                    black_box(entropy.global_value())
                });
            },
        );
    }

    // Known-alphabet single-pass dense histogram (the collector path), at a
    // small and a large base: a Bencher guard for the alphabet lever.
    for &base in &[10i32, 200] {
        for &size in &sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..base)).collect();

            group.bench_with_input(
                criterion::BenchmarkId::new(format!("alphabet_b{base}"), size),
                &size,
                |b, _| {
                    b.iter(|| {
                        let entropy =
                            Entropy::new_discrete_from_slice_with_alphabet(&data, base as usize);
                        black_box(entropy.global_value())
                    });
                },
            );
        }
    }

    group.finish();
}

fn black_box<T>(t: T) -> T {
    use std::hint::black_box;
    black_box(t)
}

criterion_group!(benches, bench_entropy_small);
criterion_main!(benches);

#![allow(unused_imports)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::entropy::GlobalValue;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Duration;

mod utils;

fn generate_lagged_series(
    size: usize,
    coupling: f64,
    lag: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut source = Vec::with_capacity(size + lag);
    let mut target = Vec::with_capacity(size);

    for _ in 0..size + lag {
        source.push(rng.sample(Normal::new(0.0, 1.0).unwrap()));
    }

    for i in 0..size {
        let noise = rng.sample(Normal::new(0.0, 1.0).unwrap());
        target.push(coupling * source[i + lag] + (1.0 - coupling.powi(2)).sqrt() * noise);
    }

    let source = source[..size].to_vec();
    (source, target)
}

fn bench_discrete_cte(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let num_states = 5;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let source: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let mut target: Vec<i32> = Vec::with_capacity(size);
        for i in 1..size {
            target.push(if source[i - 1] == source[i] {
                source[i]
            } else {
                rng.gen_range(0..num_states)
            });
        }
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("mle", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_mle(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_kernel_cte(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_kernel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let bandwidths = [0.1, 0.5, 1.0];
    let lag = 1;
    let seed = 42u64;

    for bw in bandwidths {
        for size in sizes {
            let (source, target) = generate_lagged_series(size, 0.5, lag, seed);
            let mut rng = StdRng::seed_from_u64(seed + 1);
            let cond: Vec<f64> = (0..size)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect();
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);
            let cond_arr = Array1::from(cond);

            let id = BenchmarkId::new(format!("bw_{}", bw.to_string().replace('.', "_")), size);
            group.bench_with_input(id, &(bw, size), |b, _| {
                b.iter(|| {
                    let cte = TransferEntropy::new_cte_kernel(
                        &source_arr,
                        &target_arr,
                        &cond_arr,
                        1,
                        1,
                        1,
                        1,
                        bw,
                    );
                    black_box(cte.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_ksg_cte(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_ksg");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for size in sizes {
            let (source, target) = generate_lagged_series(size, 0.5, 1, seed);
            let mut rng = StdRng::seed_from_u64(seed + 1);
            let cond: Vec<f64> = (0..size)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect();
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);
            let cond_arr = Array1::from(cond);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let cte = TransferEntropy::new_cte_ksg(
                        &source_arr,
                        &target_arr,
                        &cond_arr,
                        1,
                        1,
                        1,
                        1,
                        k,
                        noise_level,
                    );
                    black_box(cte.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_renyi_cte(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_renyi");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let ks = [1, 3];
    let alphas = [0.5, 1.5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for alpha in alphas {
            for size in sizes {
                let (source, target) = generate_lagged_series(size, 0.5, 1, seed);
                let mut rng = StdRng::seed_from_u64(seed + 1);
                let cond: Vec<f64> = (0..size)
                    .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                    .collect();
                let source_arr = Array1::from(source);
                let target_arr = Array1::from(target);
                let cond_arr = Array1::from(cond);

                let id = BenchmarkId::new(format!("k{}_alpha{}", k, alpha), size);
                group.bench_with_input(id, &(k, alpha, size), |b, _| {
                    b.iter(|| {
                        let cte = TransferEntropy::new_cte_renyi(
                            &source_arr,
                            &target_arr,
                            &cond_arr,
                            k,
                            alpha,
                            noise_level,
                        );
                        black_box(cte.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_tsallis_cte(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_tsallis");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let ks = [1, 3];
    let qs = [0.5, 1.5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for q in qs {
            for size in sizes {
                let (source, target) = generate_lagged_series(size, 0.5, 1, seed);
                let mut rng = StdRng::seed_from_u64(seed + 1);
                let cond: Vec<f64> = (0..size)
                    .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                    .collect();
                let source_arr = Array1::from(source);
                let target_arr = Array1::from(target);
                let cond_arr = Array1::from(cond);

                let id = BenchmarkId::new(format!("k{}_q{}", k, q), size);
                group.bench_with_input(id, &(k, q, size), |b, _| {
                    b.iter(|| {
                        let cte = TransferEntropy::new_cte_tsallis(
                            &source_arr,
                            &target_arr,
                            &cond_arr,
                            k,
                            q,
                            noise_level,
                        );
                        black_box(cte.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_kl_cte(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_kl");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for size in sizes {
            let (source, target) = generate_lagged_series(size, 0.5, 1, seed);
            let mut rng = StdRng::seed_from_u64(seed + 1);
            let cond: Vec<f64> = (0..size)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect();
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);
            let cond_arr = Array1::from(cond);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let cte = TransferEntropy::new_cte_kl(
                        &source_arr,
                        &target_arr,
                        &cond_arr,
                        k,
                        noise_level,
                    );
                    black_box(cte.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_ordinal_cte(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_ordinal");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let orders = [2, 3, 4];
    let seed = 42u64;

    for order in orders {
        for size in sizes {
            let (source, target) = generate_lagged_series(size, 0.5, 1, seed);
            let mut rng = StdRng::seed_from_u64(seed + 1);
            let cond: Vec<f64> = (0..size)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect();
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);
            let cond_arr = Array1::from(cond);

            let id = BenchmarkId::new(format!("order_{}", order), size);
            group.bench_with_input(id, &(order, size), |b, _| {
                b.iter(|| {
                    let cte = TransferEntropy::new_cte_ordinal(
                        &source_arr,
                        &target_arr,
                        &cond_arr,
                        order,
                        1,
                        1,
                        1,
                        1,
                        false,
                    );
                    black_box(cte.global_value())
                });
            });
        }
    }

    group.finish();
}

fn black_box<T>(t: T) -> T {
    use std::hint::black_box;
    black_box(t)
}

criterion_group!(
    benches,
    bench_discrete_cte,
    bench_kernel_cte,
    bench_ksg_cte,
    bench_renyi_cte,
    bench_tsallis_cte,
    bench_kl_cte,
    bench_ordinal_cte
);
criterion_main!(benches);

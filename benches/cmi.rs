#![allow(unused_imports)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::entropy::GlobalValue;
use infomeasure::estimators::mutual_information::MutualInformation;
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Duration;

mod utils;

fn generate_correlated(size: usize, correlation: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(size);
    let mut y = Vec::with_capacity(size);

    for _ in 0..size {
        let z: f64 = rng.sample(Normal::new(0.0, 1.0).unwrap());
        let w: f64 = rng.sample(Normal::new(0.0, 1.0).unwrap());
        let xi = z;
        let yi = correlation * z + (1.0 - correlation.powi(2)).sqrt() * w;
        x.push(xi);
        y.push(yi);
    }

    (x, y)
}

fn bench_discrete_cmi(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmi_discrete");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let z: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);
        let z_arr = Array1::from(z);

        let id = BenchmarkId::new("mle", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cmi = MutualInformation::new_cmi_discrete_mle(
                    &[x_arr.clone(), y_arr.clone()],
                    &z_arr,
                );
                black_box(cmi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_kernel_cmi(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmi_kernel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let bandwidths = [0.1, 0.5, 1.0];
    let seed = 42u64;

    for bw in bandwidths {
        for size in sizes {
            let (x, y) = generate_correlated(size, 0.5, seed);
            let mut rng = StdRng::seed_from_u64(seed + 1);
            let z: Vec<f64> = (0..size)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect();
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);
            let z_arr = Array1::from(z);

            let id = BenchmarkId::new(format!("bw_{}", bw.to_string().replace('.', "_")), size);
            group.bench_with_input(id, &(bw, size), |b, _| {
                b.iter(|| {
                    let cmi = MutualInformation::new_cmi_kernel(
                        &[x_arr.clone(), y_arr.clone()],
                        &z_arr,
                        bw,
                    );
                    black_box(cmi.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_ksg_cmi(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmi_ksg");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for size in sizes {
            let (x, y) = generate_correlated(size, 0.5, seed);
            let mut rng = StdRng::seed_from_u64(seed + 1);
            let z: Vec<f64> = (0..size)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect();
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);
            let z_arr = Array1::from(z);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let cmi = MutualInformation::new_cmi_ksg(
                        &[x_arr.clone(), y_arr.clone()],
                        &z_arr,
                        k,
                        noise_level,
                    );
                    black_box(cmi.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_renyi_cmi(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmi_renyi");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let ks = [1, 3];
    let alphas = [0.5, 1.5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for alpha in alphas {
            for size in sizes {
                let (x, y) = generate_correlated(size, 0.5, seed);
                let mut rng = StdRng::seed_from_u64(seed + 1);
                let z: Vec<f64> = (0..size)
                    .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                    .collect();
                let x_arr = Array1::from(x);
                let y_arr = Array1::from(y);
                let z_arr = Array1::from(z);

                let id = BenchmarkId::new(format!("k{}_alpha{}", k, alpha), size);
                group.bench_with_input(id, &(k, alpha, size), |b, _| {
                    b.iter(|| {
                        let cmi = MutualInformation::new_cmi_renyi(
                            &[x_arr.clone(), y_arr.clone()],
                            &z_arr,
                            k,
                            alpha,
                            noise_level,
                        );
                        black_box(cmi.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_tsallis_cmi(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmi_tsallis");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let ks = [1, 3];
    let qs = [0.5, 1.5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for q in qs {
            for size in sizes {
                let (x, y) = generate_correlated(size, 0.5, seed);
                let mut rng = StdRng::seed_from_u64(seed + 1);
                let z: Vec<f64> = (0..size)
                    .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                    .collect();
                let x_arr = Array1::from(x);
                let y_arr = Array1::from(y);
                let z_arr = Array1::from(z);

                let id = BenchmarkId::new(format!("k{}_q{}", k, q), size);
                group.bench_with_input(id, &(k, q, size), |b, _| {
                    b.iter(|| {
                        let cmi = MutualInformation::new_cmi_tsallis(
                            &[x_arr.clone(), y_arr.clone()],
                            &z_arr,
                            k,
                            q,
                            noise_level,
                        );
                        black_box(cmi.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_kl_cmi(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmi_kl");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 500, 1000];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for size in sizes {
            let (x, y) = generate_correlated(size, 0.5, seed);
            let mut rng = StdRng::seed_from_u64(seed + 1);
            let z: Vec<f64> = (0..size)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect();
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);
            let z_arr = Array1::from(z);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let cmi = MutualInformation::new_cmi_kl(
                        &[x_arr.clone(), y_arr.clone()],
                        &z_arr,
                        k,
                        noise_level,
                    );
                    black_box(cmi.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_ordinal_cmi(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmi_ordinal");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let orders = [2, 3, 4];
    let seed = 42u64;

    for order in orders {
        for size in sizes {
            let (x, y) = generate_correlated(size, 0.5, seed);
            let mut rng = StdRng::seed_from_u64(seed + 1);
            let z: Vec<f64> = (0..size)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect();
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);
            let z_arr = Array1::from(z);

            let id = BenchmarkId::new(format!("order_{}", order), size);
            group.bench_with_input(id, &(order, size), |b, _| {
                b.iter(|| {
                    let cmi = MutualInformation::new_cmi_ordinal(
                        &[x_arr.clone(), y_arr.clone()],
                        &z_arr,
                        order,
                        1,
                        false,
                    );
                    black_box(cmi.global_value())
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
    bench_discrete_cmi,
    bench_kernel_cmi,
    bench_ksg_cmi,
    bench_renyi_cmi,
    bench_tsallis_cmi,
    bench_kl_cmi,
    bench_ordinal_cmi
);
criterion_main!(benches);

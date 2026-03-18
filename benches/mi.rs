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

fn bench_discrete_mi(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("mle", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi = MutualInformation::new_discrete_mle(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_kernel_mi(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_kernel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 5000];
    let bandwidths = [0.1, 0.5, 1.0];
    let seed = 42u64;

    for bw in bandwidths {
        for size in sizes {
            let (x, y) = generate_correlated(size, 0.5, seed);
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);

            let id = BenchmarkId::new(format!("bw_{}", bw.to_string().replace('.', "_")), size);
            group.bench_with_input(id, &(bw, size), |b, _| {
                b.iter(|| {
                    let mi = MutualInformation::new_kernel(&[x_arr.clone(), y_arr.clone()], bw);
                    black_box(mi.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_ksg_mi(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_ksg");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 5000];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for size in sizes {
            let (x, y) = generate_correlated(size, 0.5, seed);
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let mi =
                        MutualInformation::new_ksg(&[x_arr.clone(), y_arr.clone()], k, noise_level);
                    black_box(mi.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_ordinal_mi(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_ordinal");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let orders = [2, 3, 4];
    let seed = 42u64;

    for order in orders {
        for size in sizes {
            let (x, y) = generate_correlated(size, 0.5, seed);
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);

            let id = BenchmarkId::new(format!("order_{}", order), size);
            group.bench_with_input(id, &(order, size), |b, _| {
                b.iter(|| {
                    let mi = MutualInformation::new_ordinal(
                        &[x_arr.clone(), y_arr.clone()],
                        order,
                        1,
                        false,
                    );
                    black_box(mi.global_value())
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
    bench_discrete_mi,
    bench_kernel_mi,
    bench_ksg_mi,
    bench_ordinal_mi
);
criterion_main!(benches);

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

use utils::{bench_bandwidths, bench_k_values, bench_orders, bench_sizes, bench_sizes_extended};

/// Fixed worker count for the `*_parallel` kernel benchmarks. The global rayon
/// pool is pinned to 1 in the benchmark environment, so the existing benchmarks
/// stay single-threaded and only these variants use a fixed multi-thread pool.
#[cfg(feature = "parallel")]
const PARALLEL_BENCH_THREADS: usize = 4;

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

    let sizes = bench_sizes_extended();
    let seed = 42u64;
    // Small base = dense direct table; base^2 > 2^20 exercises the fallback to
    // the generic entropy engine (not the dense-or-hash joint; MI has no hash
    // joint). Kept to one extra base so Bencher tracks both shapes cheaply.
    let states: [(i32, &str); 2] = [(10, ""), (1500, "_b1500")];

    for &(num_states, suffix) in &states {
        for &size in &sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
            let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);

            let id = BenchmarkId::new(format!("mle{suffix}"), size);
            group.bench_with_input(id, &size, |b, _| {
                b.iter(|| {
                    let mi = MutualInformation::new_discrete_mle(&[x_arr.clone(), y_arr.clone()]);
                    black_box(mi.global_value())
                });
            });

            // Known-alphabet, global-only builder (the collector path): borrows
            // the columns and skips the alphabet scan and input retention.
            let id = BenchmarkId::new(format!("mle_alphabet{suffix}"), size);
            group.bench_with_input(id, &size, |b, _| {
                b.iter(|| {
                    black_box(
                        MutualInformation::mi_discrete_mle(&[
                            x_arr.as_slice().unwrap(),
                            y_arr.as_slice().unwrap(),
                        ])
                        .with_alphabet(num_states as usize)
                        .global_only()
                        .global_value(),
                    )
                });
            });
        }
    }

    group.finish();
}

fn bench_kernel_mi(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_kernel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let bandwidths = bench_bandwidths();
    let kernel_types = ["box", "gaussian"];
    let seed = 42u64;

    for &kernel_type in &kernel_types {
        for &bw in &bandwidths {
            for &size in &sizes {
                let (x, y) = generate_correlated(size, 0.5, seed);
                let x_arr = Array1::from(x);
                let y_arr = Array1::from(y);

                let kt = kernel_type.to_string();
                let bw_str = bw.to_string().replace('.', "_");
                let id = BenchmarkId::new(format!("{}/bw_{}", kernel_type, bw_str), size);
                group.bench_with_input(id, &(kt, bw), |b, (kt, bw)| {
                    b.iter(|| {
                        let mi = MutualInformation::new_kernel_with_type(
                            &[x_arr.clone(), y_arr.clone()],
                            kt.clone(),
                            *bw,
                        );
                        black_box(mi.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

/// Same kernel MI sweep as `bench_kernel_mi`, but executed on a fixed
/// multi-thread rayon pool (see [`PARALLEL_BENCH_THREADS`]). Bencher tracks these
/// as distinct `mi_kernel_parallel/...` benchmarks.
#[cfg(feature = "parallel")]
fn bench_kernel_mi_parallel(c: &mut Criterion) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(PARALLEL_BENCH_THREADS)
        .build()
        .unwrap();
    let mut group = c.benchmark_group("mi_kernel_parallel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let bandwidths = bench_bandwidths();
    let kernel_types = ["box", "gaussian"];
    let seed = 42u64;

    for &kernel_type in &kernel_types {
        for &bw in &bandwidths {
            for &size in &sizes {
                let (x, y) = generate_correlated(size, 0.5, seed);
                let x_arr = Array1::from(x);
                let y_arr = Array1::from(y);

                let kt = kernel_type.to_string();
                let bw_str = bw.to_string().replace('.', "_");
                let id = BenchmarkId::new(format!("{}/bw_{}", kernel_type, bw_str), size);
                group.bench_with_input(id, &(kt, bw), |b, (kt, bw)| {
                    b.iter(|| {
                        pool.install(|| {
                            let mi = MutualInformation::new_kernel_with_type(
                                &[x_arr.clone(), y_arr.clone()],
                                kt.clone(),
                                *bw,
                            );
                            black_box(mi.global_value())
                        })
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_ksg_mi(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_ksg");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let ks = bench_k_values();
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &size in &sizes {
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

    let sizes = bench_sizes_extended();
    let orders = bench_orders();
    let seed = 42u64;

    for &order in &orders {
        for &size in &sizes {
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

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_discrete_mi,
    bench_kernel_mi,
    bench_kernel_mi_parallel,
    bench_ksg_mi,
    bench_ordinal_mi
);
#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    bench_discrete_mi,
    bench_kernel_mi,
    bench_ksg_mi,
    bench_ordinal_mi
);
criterion_main!(benches);

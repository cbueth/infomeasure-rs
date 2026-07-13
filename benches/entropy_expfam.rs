#![allow(unused_imports)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Duration;

mod utils;

use utils::{bench_alphas, bench_bandwidths, bench_k_values, bench_q_values, bench_sizes};

fn bench_renyi_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_renyi");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let alphas = bench_alphas();
    let ks = bench_k_values();
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &alpha in &alphas {
            for &size in &sizes {
                let mut rng = StdRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, 1.0).unwrap();
                let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
                let arr = Array1::from(data);

                let id = BenchmarkId::new(
                    format!("k{}_alpha{}", k, alpha.to_string().replace('.', "_")),
                    size,
                );
                group.bench_with_input(id, &(k, alpha, size), |b, _| {
                    b.iter(|| {
                        let entropy = Entropy::new_renyi_1d(arr.clone(), k, alpha, noise_level);
                        black_box(entropy.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_tsallis_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_tsallis");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let qs = bench_q_values();
    let ks = bench_k_values();
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &q in &qs {
            for &size in &sizes {
                let mut rng = StdRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, 1.0).unwrap();
                let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
                let arr = Array1::from(data);

                let id =
                    BenchmarkId::new(format!("k{}_q{}", k, q.to_string().replace('.', "_")), size);
                group.bench_with_input(id, &(k, q, size), |b, _| {
                    b.iter(|| {
                        let entropy = Entropy::new_tsallis_1d(arr.clone(), k, q, noise_level);
                        black_box(entropy.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_kl_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_kl");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let ks = bench_k_values();
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &size in &sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
            let arr = Array1::from(data);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let entropy = Entropy::new_kl_1d(arr.clone(), k, noise_level);
                    black_box(entropy.global_value())
                });
            });
        }
    }

    group.finish();
}

/// KL entropy with Chebyshev (L-infinity) metric for tie-breaking isolation.
fn bench_kl_entropy_cheb(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_kl_cheb");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let ks = bench_k_values();
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &size in &sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
            let arr = Array1::from(data);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, (k, _size)| {
                let k = *k;
                b.iter_with_setup(
                    || arr.clone(),
                    |data| {
                        let entropy = Entropy::new_kl_1d(data, k, noise_level).with_chebyshev(true);
                        black_box(entropy.global_value())
                    },
                );
            });
        }
    }

    group.finish();
}

/// KL entropy with k=1 (nearest_one path, no collection overhead) and k=10
/// (larger sorted collection) to measure the scaling of collection overhead.
fn bench_kl_entropy_k_extended(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_kl_k");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let ks = [1usize, 10];
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &size in &sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
            let arr = Array1::from(data);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let entropy = Entropy::new_kl_1d(arr.clone(), k, noise_level);
                    black_box(entropy.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_kl_nd_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_kl_nd");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let dims = [2, 4, 8];
    let k = 3;
    let seed = 42u64;
    let noise_level = 1e-10;

    for dim in dims {
        for &size in &sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let data: Vec<f64> = (0..size * dim).map(|_| normal.sample(&mut rng)).collect();
            let arr = Array2::from_shape_vec((size, dim), data).unwrap();

            let id = BenchmarkId::new(format!("{}d", dim), size);
            group.bench_with_input(id, &(dim, size), |b, _| {
                b.iter(|| {
                    let entropy = match dim {
                        2 => Entropy::kl_nd::<2>(arr.clone(), k, noise_level).global_value(),
                        4 => Entropy::kl_nd::<4>(arr.clone(), k, noise_level).global_value(),
                        8 => Entropy::kl_nd::<8>(arr.clone(), k, noise_level).global_value(),
                        _ => panic!("Unsupported dimension: {dim}"),
                    };
                    black_box(entropy)
                });
            });
        }
    }

    group.finish();
}

/// KL ND entropy with Chebyshev metric (4D, as representative for scaling analysis).
fn bench_kl_nd_entropy_cheb(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_kl_nd_cheb");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let dim = 4;
    let k = 3;
    let seed = 42u64;
    let noise_level = 1e-10;

    for &size in &sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f64> = (0..size * dim).map(|_| normal.sample(&mut rng)).collect();
        let arr = Array2::from_shape_vec((size, dim), data).unwrap();

        let id = BenchmarkId::new("4d", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = KozachenkoLeonenkoEntropy::<4>::new(arr.clone(), k, noise_level)
                    .with_chebyshev(true);
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_kernel_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_kernel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let kernel_types = ["box", "gaussian"];
    let bandwidths = bench_bandwidths();
    let seed = 42u64;

    for &kernel_type in &kernel_types {
        for &bandwidth in &bandwidths {
            for &size in &sizes {
                let mut rng = StdRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, 1.0).unwrap();
                let arr: Array1<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();

                let kt = kernel_type.to_string();
                let bw_str = bandwidth.to_string().replace('.', "_");
                let id = BenchmarkId::new(format!("{}/bw{}", kernel_type, bw_str), size);
                group.bench_with_input(id, &(kt, bandwidth), |b, (kt, bw)| {
                    b.iter(|| {
                        let entropy = Entropy::new_kernel_with_type(arr.clone(), kt.clone(), *bw);
                        black_box(entropy.global_value())
                    });
                });
            }
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
    bench_renyi_entropy,
    bench_tsallis_entropy,
    bench_kl_entropy,
    bench_kl_entropy_cheb,
    bench_kl_entropy_k_extended,
    bench_kl_nd_entropy,
    bench_kl_nd_entropy_cheb,
    bench_kernel_entropy
);
criterion_main!(benches);

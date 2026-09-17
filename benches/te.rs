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

use utils::{
    bench_alphas, bench_bandwidths, bench_k_values, bench_orders, bench_q_values, bench_sizes,
    bench_sizes_extended,
};

/// Fixed worker count for the `*_parallel` kernel benchmarks. The global rayon
/// pool is pinned to 1 in the benchmark environment, so the existing benchmarks
/// stay single-threaded and only these variants use a fixed multi-thread pool.
#[cfg(feature = "parallel")]
const PARALLEL_BENCH_THREADS: usize = 4;

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

fn bench_discrete_te(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_discrete");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes_extended();
    let seed = 42u64;
    // Small base = dense joint; base^3 > ~50·N exercises the sparse hash joint.
    let states: [(i32, &str); 2] = [(5, ""), (100, "_b100")];

    for &(num_states, suffix) in &states {
        for &size in &sizes {
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
            let source_arr = Array1::from(source[..size - 1].to_vec());
            let target_arr = Array1::from(target);

            let id = BenchmarkId::new(format!("mle{suffix}"), size);
            group.bench_with_input(id, &size, |b, _| {
                b.iter(|| {
                    let te = TransferEntropy::new_discrete_mle(&source_arr, &target_arr, 1, 1, 1);
                    black_box(te.global_value())
                });
            });

            // Known-alphabet, global-only builder (the collector path): borrows
            // the columns and skips the alphabet scan and input retention.
            let id = BenchmarkId::new(format!("mle_alphabet{suffix}"), size);
            group.bench_with_input(id, &size, |b, _| {
                b.iter(|| {
                    black_box(
                        TransferEntropy::te_discrete_mle(
                            source_arr.as_slice().unwrap(),
                            target_arr.as_slice().unwrap(),
                            1,
                            1,
                            1,
                        )
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

fn bench_kernel_te(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_kernel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let bandwidths = bench_bandwidths();
    let kernel_types = ["box", "gaussian"];
    let lag = 1;
    let seed = 42u64;

    for &kernel_type in &kernel_types {
        for &bw in &bandwidths {
            for &size in &sizes {
                let (source, target) = generate_lagged_series(size, 0.5, lag, seed);
                let source_arr = Array1::from(source);
                let target_arr = Array1::from(target);

                let kt = kernel_type.to_string();
                let bw_str = bw.to_string().replace('.', "_");
                let id = BenchmarkId::new(format!("{}/bw_{}", kernel_type, bw_str), size);
                group.bench_with_input(id, &(kt, bw), |b, (kt, bw)| {
                    b.iter(|| {
                        let te = TransferEntropy::new_kernel_with_type(
                            &source_arr,
                            &target_arr,
                            1,
                            1,
                            1,
                            kt.clone(),
                            *bw,
                        );
                        black_box(te.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

/// Same kernel TE sweep as `bench_kernel_te`, executed on a fixed multi-thread
/// rayon pool (see [`PARALLEL_BENCH_THREADS`]) and tracked as
/// `te_kernel_parallel/...`.
#[cfg(feature = "parallel")]
fn bench_kernel_te_parallel(c: &mut Criterion) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(PARALLEL_BENCH_THREADS)
        .build()
        .unwrap();
    let mut group = c.benchmark_group("te_kernel_parallel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let bandwidths = bench_bandwidths();
    let kernel_types = ["box", "gaussian"];
    let lag = 1;
    let seed = 42u64;

    for &kernel_type in &kernel_types {
        for &bw in &bandwidths {
            for &size in &sizes {
                let (source, target) = generate_lagged_series(size, 0.5, lag, seed);
                let source_arr = Array1::from(source);
                let target_arr = Array1::from(target);

                let kt = kernel_type.to_string();
                let bw_str = bw.to_string().replace('.', "_");
                let id = BenchmarkId::new(format!("{}/bw_{}", kernel_type, bw_str), size);
                group.bench_with_input(id, &(kt, bw), |b, (kt, bw)| {
                    b.iter(|| {
                        pool.install(|| {
                            let te = TransferEntropy::new_kernel_with_type(
                                &source_arr,
                                &target_arr,
                                1,
                                1,
                                1,
                                kt.clone(),
                                *bw,
                            );
                            black_box(te.global_value())
                        })
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_ksg_te(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_ksg");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let ks = bench_k_values();
    let lag = 1;
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &size in &sizes {
            let (source, target) = generate_lagged_series(size, 0.5, lag, seed);
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let te =
                        TransferEntropy::new_ksg(&source_arr, &target_arr, 1, 1, 1, k, noise_level);
                    black_box(te.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_te_renyi(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_renyi");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let ks = bench_k_values();
    let alphas = bench_alphas();
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &alpha in &alphas {
            for &size in &sizes {
                let (source, target) = generate_lagged_series(size, 0.5, 1, seed);
                let source_arr = Array1::from(source);
                let target_arr = Array1::from(target);

                let id = BenchmarkId::new(
                    format!("k{}_alpha{}", k, alpha.to_string().replace('.', "_")),
                    size,
                );
                group.bench_with_input(id, &(k, alpha, size), |b, _| {
                    b.iter(|| {
                        let te = TransferEntropy::new_renyi(
                            &source_arr,
                            &target_arr,
                            k,
                            alpha,
                            noise_level,
                        );
                        black_box(te.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_te_tsallis(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_tsallis");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let ks = bench_k_values();
    let qs = bench_q_values();
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &q in &qs {
            for &size in &sizes {
                let (source, target) = generate_lagged_series(size, 0.5, 1, seed);
                let source_arr = Array1::from(source);
                let target_arr = Array1::from(target);

                let id =
                    BenchmarkId::new(format!("k{}_q{}", k, q.to_string().replace('.', "_")), size);
                group.bench_with_input(id, &(k, q, size), |b, _| {
                    b.iter(|| {
                        let te = TransferEntropy::new_tsallis(
                            &source_arr,
                            &target_arr,
                            k,
                            q,
                            noise_level,
                        );
                        black_box(te.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_ordinal_te(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_ordinal");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes_extended();
    let orders = bench_orders();
    let lag = 1;
    let seed = 42u64;

    for &order in &orders {
        for &size in &sizes {
            let (source, target) = generate_lagged_series(size, 0.5, lag, seed);
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);

            let id = BenchmarkId::new(format!("order_{}", order), size);
            group.bench_with_input(id, &(order, size), |b, _| {
                b.iter(|| {
                    let te = TransferEntropy::new_ordinal(
                        &source_arr,
                        &target_arr,
                        order,
                        1,
                        1,
                        1,
                        false,
                    );
                    black_box(te.global_value())
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

/// Same KSG TE sweep on a fixed multi-thread rayon pool (see
/// [`PARALLEL_BENCH_THREADS`]); tracked as `te_ksg_parallel/...`.
#[cfg(feature = "parallel")]
fn bench_ksg_te_parallel(c: &mut Criterion) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(PARALLEL_BENCH_THREADS)
        .build()
        .unwrap();
    let mut group = c.benchmark_group("te_ksg_parallel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = bench_sizes();
    let ks = bench_k_values();
    let lag = 1;
    let seed = 42u64;
    let noise_level = 1e-10;

    for &k in &ks {
        for &size in &sizes {
            let (source, target) = generate_lagged_series(size, 0.5, lag, seed);
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    pool.install(|| {
                        let te = TransferEntropy::new_ksg(
                            &source_arr,
                            &target_arr,
                            1,
                            1,
                            1,
                            k,
                            noise_level,
                        );
                        black_box(te.global_value())
                    })
                });
            });
        }
    }

    group.finish();
}

#[cfg(feature = "parallel")]
criterion_group!(
    benches,
    bench_discrete_te,
    bench_kernel_te,
    bench_kernel_te_parallel,
    bench_ksg_te,
    bench_ksg_te_parallel,
    bench_te_renyi,
    bench_te_tsallis,
    bench_ordinal_te
);
#[cfg(not(feature = "parallel"))]
criterion_group!(
    benches,
    bench_discrete_te,
    bench_kernel_te,
    bench_ksg_te,
    bench_te_renyi,
    bench_te_tsallis,
    bench_ordinal_te
);
criterion_main!(benches);

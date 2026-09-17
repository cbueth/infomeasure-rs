// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measures the CPU-vs-GPU crossover per machine.
//!
//! Each size is timed twice for both kernels: once forced onto the CPU path
//! (`set_force_cpu(true)`) and once forced onto the GPU path (dispatch gate
//! overridden to zero). Comparing the two curves per testbed yields the
//! crossover points that the adaptive gates in `estimators::gpu` should use;
//! the Bencher history of this benchmark documents each machine's crossover
//! over time.
//!
//! Run: `cargo bench --bench gpu_crossover --features gpu`
//! Sizes/bandwidths follow the shared `BENCH_SIZES`/`BENCH_BANDWIDTHS` env
//! overrides via `benches/utils`.

#![allow(unused_imports)]
#[cfg(feature = "gpu")]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(feature = "gpu")]
use infomeasure::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
#[cfg(feature = "gpu")]
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
#[cfg(feature = "gpu")]
use infomeasure::estimators::gpu::set_gpu_min_points_override;
#[cfg(feature = "gpu")]
use ndarray::{Array1, Array2};
#[cfg(feature = "gpu")]
use rand::Rng;
#[cfg(feature = "gpu")]
use rand::SeedableRng;
#[cfg(feature = "gpu")]
use rand::rngs::StdRng;
#[cfg(feature = "gpu")]
use rand_distr::{Distribution, Normal};
#[cfg(feature = "gpu")]
use std::time::Duration;

#[cfg(feature = "gpu")]
mod utils;

#[cfg(feature = "gpu")]
use utils::{bench_sizes, bench_sizes_extended};

/// Sizes to sweep: the standard set plus the extended set. CI runs the bench
/// with `BENCH_SIZES=1000` and `BENCH_SIZES_EXTENDED=5000`, so the union keeps
/// the small crossover points and adds the 5000-point one the expfam tier needs
/// (its gate is 4000).
#[cfg(feature = "gpu")]
fn crossover_sizes() -> Vec<usize> {
    let mut sizes = bench_sizes();
    sizes.extend(bench_sizes_extended());
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// Bandwidth held fixed: the crossover against N is the quantity of interest,
/// and a single bandwidth halves the runtime.
#[cfg(feature = "gpu")]
const BANDWIDTH: f64 = 0.9;

#[cfg(feature = "gpu")]
fn gaussian_sample(size: usize) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array1::from(
        (0..size)
            .map(|_| normal.sample(&mut rng))
            .collect::<Vec<f64>>(),
    )
}

#[cfg(feature = "gpu")]
fn gaussian_sample_nd(dims: usize, size: usize) -> Array2<f64> {
    let mut rng = StdRng::from_entropy();
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn((size, dims), |_| normal.sample(&mut rng))
}

#[cfg(feature = "gpu")]
fn bench_gaussian_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover_gaussian");
    group.measurement_time(Duration::from_secs(4));

    // Force every call through the GPU path regardless of size.
    set_gpu_min_points_override(Some(0), Some(0), Some(0), Some(0));

    for size in crossover_sizes() {
        let data = gaussian_sample(size);

        group.bench_with_input(BenchmarkId::new("cpu", size), &data, |b, data| {
            b.iter(|| {
                let mut est =
                    Entropy::new_kernel_with_type(data.clone(), "gaussian".to_string(), BANDWIDTH);
                est.set_force_cpu(true);
                black_box(est.global_value())
            });
        });
        group.bench_with_input(BenchmarkId::new("gpu", size), &data, |b, data| {
            b.iter(|| {
                let mut est =
                    Entropy::new_kernel_with_type(data.clone(), "gaussian".to_string(), BANDWIDTH);
                est.set_force_cpu(false);
                black_box(est.global_value())
            });
        });
    }

    set_gpu_min_points_override(None, None, None, None);
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_box_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover_box");
    group.measurement_time(Duration::from_secs(4));

    set_gpu_min_points_override(Some(0), Some(0), Some(0), Some(0));

    for size in crossover_sizes() {
        let data = gaussian_sample(size);

        group.bench_with_input(BenchmarkId::new("cpu", size), &data, |b, data| {
            b.iter(|| {
                let mut est =
                    Entropy::new_kernel_with_type(data.clone(), "box".to_string(), BANDWIDTH);
                est.set_force_cpu(true);
                black_box(est.global_value())
            });
        });
        group.bench_with_input(BenchmarkId::new("gpu", size), &data, |b, data| {
            b.iter(|| {
                let mut est =
                    Entropy::new_kernel_with_type(data.clone(), "box".to_string(), BANDWIDTH);
                est.set_force_cpu(false);
                black_box(est.global_value())
            });
        });
    }

    set_gpu_min_points_override(None, None, None, None);
    group.finish();
}

#[cfg(feature = "gpu")]
fn black_box<T>(t: T) -> T {
    use std::hint::black_box;
    black_box(t)
}

/// k for the expfam kNN series (matches the classifier benchmarks).
#[cfg(feature = "gpu")]
const EXPFAM_K: usize = 3;

/// Expfam KL entropy crossover per dimensionality. expfam radii are irregular
/// kiddo traversal on the CPU vs a dense O(N²) per-row k-selection on the GPU,
/// so the crossover moves with `D`.
#[cfg(feature = "gpu")]
fn bench_expfam_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover_expfam_kl");
    group.measurement_time(Duration::from_secs(4));

    set_gpu_min_points_override(Some(0), Some(0), Some(0), Some(0));

    macro_rules! dim_series {
        ($dim:literal) => {
            for size in crossover_sizes() {
                let data = gaussian_sample_nd($dim, size);

                group.bench_with_input(
                    BenchmarkId::new(format!("cpu_d{}", $dim), size),
                    &data,
                    |b, data| {
                        b.iter(|| {
                            let mut est =
                                KozachenkoLeonenkoEntropy::<$dim>::new(data.clone(), EXPFAM_K, 0.0);
                            est.set_force_cpu(true);
                            black_box(est.global_value())
                        });
                    },
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("gpu_d{}", $dim), size),
                    &data,
                    |b, data| {
                        b.iter(|| {
                            let mut est =
                                KozachenkoLeonenkoEntropy::<$dim>::new(data.clone(), EXPFAM_K, 0.0);
                            est.set_force_cpu(false);
                            black_box(est.global_value())
                        });
                    },
                );
            }
        };
    }

    dim_series!(1);
    dim_series!(2);
    dim_series!(4);
    dim_series!(6);
    dim_series!(8);
    dim_series!(16);

    set_gpu_min_points_override(None, None, None, None);
    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    bench_gaussian_crossover,
    bench_box_crossover,
    bench_expfam_crossover
);

#[cfg(feature = "gpu")]
criterion_main!(benches);

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmarks require the 'gpu' feature to be enabled.");
    println!("Run with: cargo bench --bench gpu_crossover --features gpu");
}

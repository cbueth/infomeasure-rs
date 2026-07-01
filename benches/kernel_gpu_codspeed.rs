// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Minimal CodSpeed GPU regression benchmarks.
//!
//! Unlike `kernel_gpu.rs` (parameter exploration), this file uses hardcoded
//! single-parameter benchmarks to track GPU kernel path regressions in CI.
//! Gaussian kernel: all 5 measures at N=3000, bw=0.5.
//! Box kernel: entropy at N=10000 (Box GPU threshold is N >= 2000).
//! No loops, no env var dependency.

#![allow(unused_imports)]

#[cfg(feature = "gpu")]
use criterion::{Criterion, black_box, criterion_group, criterion_main};
#[cfg(feature = "gpu")]
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
#[cfg(feature = "gpu")]
use infomeasure::estimators::mutual_information::MutualInformation;
#[cfg(feature = "gpu")]
use infomeasure::estimators::transfer_entropy::TransferEntropy;
#[cfg(feature = "gpu")]
use ndarray::Array1;
#[cfg(feature = "gpu")]
use rand::Rng;
#[cfg(feature = "gpu")]
use rand::SeedableRng;
#[cfg(feature = "gpu")]
use rand::rngs::StdRng;
#[cfg(feature = "gpu")]
use rand_distr::{Distribution, Normal};

#[cfg(feature = "gpu")]
const N: usize = 3000;
#[cfg(feature = "gpu")]
const BW: f64 = 0.5;
#[cfg(feature = "gpu")]
const KERNEL_TYPE: &str = "gaussian";
#[cfg(feature = "gpu")]
const SEED: u64 = 42;
#[cfg(feature = "gpu")]
const N_BOX: usize = 10000;

#[cfg(feature = "gpu")]
fn generate_random(size: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..size).map(|_| normal.sample(&mut rng)).collect()
}

#[cfg(feature = "gpu")]
fn generate_correlated(size: usize, correlation: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut x = Vec::with_capacity(size);
    let mut y = Vec::with_capacity(size);
    for _ in 0..size {
        let z: f64 = rng.sample(normal);
        let w: f64 = rng.sample(normal);
        x.push(z);
        y.push(correlation * z + (1.0 - correlation.powi(2)).sqrt() * w);
    }
    (x, y)
}

#[cfg(feature = "gpu")]
fn bench_kernel_entropy_gpu(c: &mut Criterion) {
    let data = generate_random(N, SEED);
    let arr = Array1::from(data);
    c.bench_function("entropy_kernel_gpu", |b| {
        b.iter(|| {
            let entropy = Entropy::new_kernel_with_type(arr.clone(), KERNEL_TYPE.to_string(), BW);
            black_box(entropy.global_value())
        });
    });
}

#[cfg(feature = "gpu")]
fn bench_kernel_mi_gpu(c: &mut Criterion) {
    let (x, y) = generate_correlated(N, 0.5, SEED);
    let x_arr = Array1::from(x);
    let y_arr = Array1::from(y);
    c.bench_function("mi_kernel_gpu", |b| {
        b.iter(|| {
            let mi = MutualInformation::new_kernel_with_type(
                &[x_arr.clone(), y_arr.clone()],
                KERNEL_TYPE.to_string(),
                BW,
            );
            black_box(mi.global_value())
        });
    });
}

#[cfg(feature = "gpu")]
fn bench_kernel_cmi_gpu(c: &mut Criterion) {
    let (x, y) = generate_correlated(N, 0.5, SEED);
    let z = generate_random(N, SEED + 1);
    let x_arr = Array1::from(x);
    let y_arr = Array1::from(y);
    let z_arr = Array1::from(z);
    c.bench_function("cmi_kernel_gpu", |b| {
        b.iter(|| {
            let cmi = MutualInformation::new_cmi_kernel_with_type(
                &[x_arr.clone(), y_arr.clone()],
                &z_arr,
                KERNEL_TYPE.to_string(),
                BW,
            );
            black_box(cmi.global_value())
        });
    });
}

#[cfg(feature = "gpu")]
fn bench_kernel_te_gpu(c: &mut Criterion) {
    let (source, target) = generate_correlated(N, 0.5, SEED);
    let source_arr = Array1::from(source);
    let target_arr = Array1::from(target);
    c.bench_function("te_kernel_gpu", |b| {
        b.iter(|| {
            let te = TransferEntropy::new_kernel_with_type(
                &source_arr,
                &target_arr,
                1,
                1,
                1,
                KERNEL_TYPE.to_string(),
                BW,
            );
            black_box(te.global_value())
        });
    });
}

#[cfg(feature = "gpu")]
fn bench_kernel_cte_gpu(c: &mut Criterion) {
    let (source, target) = generate_correlated(N, 0.5, SEED);
    let cond = generate_random(N, SEED + 1);
    let source_arr = Array1::from(source);
    let target_arr = Array1::from(target);
    let cond_arr = Array1::from(cond);
    c.bench_function("cte_kernel_gpu", |b| {
        b.iter(|| {
            let cte = TransferEntropy::new_cte_kernel_with_type(
                &source_arr,
                &target_arr,
                &cond_arr,
                1,
                1,
                1,
                1,
                KERNEL_TYPE.to_string(),
                BW,
            );
            black_box(cte.global_value())
        });
    });
}

#[cfg(feature = "gpu")]
fn bench_kernel_entropy_box_gpu(c: &mut Criterion) {
    let data = generate_random(N_BOX, SEED);
    let arr = Array1::from(data);
    c.bench_function("entropy_kernel_box_gpu", |b| {
        b.iter(|| {
            let entropy = Entropy::new_kernel_with_type(arr.clone(), "box".to_string(), BW);
            black_box(entropy.global_value())
        });
    });
}

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    bench_kernel_entropy_gpu,
    bench_kernel_entropy_box_gpu,
    bench_kernel_mi_gpu,
    bench_kernel_cmi_gpu,
    bench_kernel_te_gpu,
    bench_kernel_cte_gpu,
);

#[cfg(feature = "gpu")]
criterion_main!(benches);

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmarks require the 'gpu' feature to be enabled.");
    println!("Run with: cargo bench --bench kernel_gpu_codspeed --features gpu");
}

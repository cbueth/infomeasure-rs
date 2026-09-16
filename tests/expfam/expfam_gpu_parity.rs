// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU vs CPU parity tests for the exponential-family (expfam) kNN estimators.
//!
//! Mirrors the kernel GPU parity suite: each case builds the estimator twice —
//! once with the dense GPU kNN tier enabled (the default) and once forced onto
//! the CPU via `set_force_cpu(true)` — then compares the results. Sizes are
//! chosen above the expfam dispatch gate (see `estimators::gpu` defaults) so
//! the GPU path is actually exercised.
//!
//! The shader selects the k-th neighbour from f32 distances, so the comparison
//! uses the same broad tolerance the kernel suite uses rather than bit equality;
//! continuous data keeps exact ties out of the k-th boundary. Inputs are drawn
//! from a fresh entropy-seeded RNG so the suite is not tied to one sample.

// The whole suite is GPU-only; compiling it away without the feature keeps the
// parity macros from becoming unused definitions.
#![cfg(feature = "gpu")]

#[cfg(feature = "gpu")]
use crate::test_helpers::assert_hardware_gpu_adapter;
#[cfg(feature = "gpu")]
use infomeasure::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
#[cfg(feature = "gpu")]
use infomeasure::estimators::approaches::expfam::renyi::RenyiEntropy;
#[cfg(feature = "gpu")]
use infomeasure::estimators::approaches::expfam::tsallis::TsallisEntropy;
#[cfg(feature = "gpu")]
use infomeasure::estimators::traits::{CrossEntropy, GlobalValue, LocalValues};
#[cfg(feature = "gpu")]
use ndarray::Array2;
#[cfg(feature = "gpu")]
use rand::SeedableRng;
#[cfg(feature = "gpu")]
use rand::rngs::StdRng;
#[cfg(feature = "gpu")]
use rand_distr::{Distribution, Normal};
#[cfg(feature = "gpu")]
use rstest::rstest;

/// Above the shipped expfam gate so the dense tier is actually dispatched.
#[cfg(feature = "gpu")]
const SIZE: usize = 6000;
/// Small k, as used by the classifier workloads.
#[cfg(feature = "gpu")]
const K: usize = 3;

/// Gaussian cloud drawn from an entropy-seeded RNG (no fixed seed).
#[cfg(feature = "gpu")]
fn gaussian_nd(dims: usize, size: usize) -> Array2<f64> {
    let mut rng = StdRng::from_entropy();
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn((size, dims), |_| normal.sample(&mut rng))
}

/// Tolerance used by the kernel GPU suite: absolute plus relative part.
#[cfg(feature = "gpu")]
fn assert_parity(name: &str, gpu: f64, cpu: f64) {
    let allowed = 5e-3 + 1e-2 * cpu.abs();
    assert!(
        (gpu - cpu).abs() <= allowed,
        "{name} mismatch: GPU {gpu} vs CPU {cpu} (allowed {allowed})"
    );
}

#[cfg(feature = "gpu")]
fn assert_local_parity(name: &str, gpu: &[f64], cpu: &[f64]) {
    assert_eq!(gpu.len(), cpu.len(), "{name} length mismatch");
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let allowed = 5e-3 + 1e-2 * c.abs();
        assert!(
            (g - c).abs() <= allowed,
            "{name} local value mismatch at index {i}: GPU {g} vs CPU {c} (allowed {allowed})"
        );
    }
}

/// KL entropy parity per dimension and metric.
macro_rules! kl_parity_test {
    ($name:ident, $dim:literal) => {
        #[rstest]
        #[cfg(feature = "gpu")]
        fn $name(#[values(true, false)] use_chebyshev: bool) {
            assert_hardware_gpu_adapter();
            let data = gaussian_nd($dim, SIZE);

            let mut gpu = KozachenkoLeonenkoEntropy::<$dim>::new(data.clone(), K, 0.0)
                .with_chebyshev(use_chebyshev);
            gpu.set_force_cpu(false);
            let mut cpu = KozachenkoLeonenkoEntropy::<$dim>::new(data.clone(), K, 0.0)
                .with_chebyshev(use_chebyshev);
            cpu.set_force_cpu(true);

            let name = format!("kl_d{}_cheb{use_chebyshev}", $dim);
            assert_parity(&name, gpu.global_value(), cpu.global_value());
            assert_local_parity(
                &name,
                &gpu.local_values().to_vec(),
                &cpu.local_values().to_vec(),
            );
        }
    };
}

#[cfg(feature = "gpu")]
kl_parity_test!(kl_gpu_parity_d8, 8);
#[cfg(feature = "gpu")]
kl_parity_test!(kl_gpu_parity_d16, 16);

/// KL cross-entropy (asymmetric query/data clouds) parity per dimension.
macro_rules! kl_cross_parity_test {
    ($name:ident, $dim:literal) => {
        #[rstest]
        #[cfg(feature = "gpu")]
        fn $name(#[values(true, false)] use_chebyshev: bool) {
            assert_hardware_gpu_adapter();
            let mut p = gaussian_nd($dim, SIZE);
            if let Some(slice) = p.as_slice_mut() {
                for v in slice.iter_mut() {
                    *v += 0.25;
                }
            }
            let q = gaussian_nd($dim, SIZE);

            let mut gpu = KozachenkoLeonenkoEntropy::<$dim>::new(p.clone(), K, 0.0)
                .with_chebyshev(use_chebyshev);
            gpu.set_force_cpu(false);
            let mut cpu = KozachenkoLeonenkoEntropy::<$dim>::new(p.clone(), K, 0.0)
                .with_chebyshev(use_chebyshev);
            cpu.set_force_cpu(true);

            let gpu_q = KozachenkoLeonenkoEntropy::<$dim>::new(q.clone(), K, 0.0)
                .with_chebyshev(use_chebyshev);
            let cpu_q = KozachenkoLeonenkoEntropy::<$dim>::new(q.clone(), K, 0.0)
                .with_chebyshev(use_chebyshev);

            assert_parity(
                &format!("kl_cross_d{}_cheb{use_chebyshev}", $dim),
                gpu.cross_entropy(&gpu_q),
                cpu.cross_entropy(&cpu_q),
            );
        }
    };
}

#[cfg(feature = "gpu")]
kl_cross_parity_test!(kl_cross_gpu_parity_d8, 8);
#[cfg(feature = "gpu")]
kl_cross_parity_test!(kl_cross_gpu_parity_d16, 16);

/// Rényi entropy parity per dimension (Euclidean metric, both the q→1 Shannon
/// limit and the general q branch).
macro_rules! renyi_parity_test {
    ($name:ident, $dim:literal) => {
        #[rstest]
        #[cfg(feature = "gpu")]
        fn $name(#[values(1.0, 0.5)] alpha: f64) {
            assert_hardware_gpu_adapter();
            let data = gaussian_nd($dim, SIZE);

            let mut gpu = RenyiEntropy::<$dim>::new(data.clone(), K, alpha, 0.0);
            gpu.set_force_cpu(false);
            let mut cpu = RenyiEntropy::<$dim>::new(data.clone(), K, alpha, 0.0);
            cpu.set_force_cpu(true);

            assert_parity(
                &format!("renyi_d{}_alpha{alpha}", $dim),
                gpu.global_value(),
                cpu.global_value(),
            );
        }
    };
}

#[cfg(feature = "gpu")]
renyi_parity_test!(renyi_gpu_parity_d8, 8);
#[cfg(feature = "gpu")]
renyi_parity_test!(renyi_gpu_parity_d16, 16);

/// Tsallis entropy parity per dimension (Euclidean metric).
macro_rules! tsallis_parity_test {
    ($name:ident, $dim:literal) => {
        #[rstest]
        #[cfg(feature = "gpu")]
        fn $name(#[values(1.0, 0.5)] q: f64) {
            assert_hardware_gpu_adapter();
            let data = gaussian_nd($dim, SIZE);

            let mut gpu = TsallisEntropy::<$dim>::new(data.clone(), K, q, 0.0);
            gpu.set_force_cpu(false);
            let mut cpu = TsallisEntropy::<$dim>::new(data.clone(), K, q, 0.0);
            cpu.set_force_cpu(true);

            assert_parity(
                &format!("tsallis_d{}_q{q}", $dim),
                gpu.global_value(),
                cpu.global_value(),
            );
        }
    };
}

#[cfg(feature = "gpu")]
tsallis_parity_test!(tsallis_gpu_parity_d8, 8);
#[cfg(feature = "gpu")]
tsallis_parity_test!(tsallis_gpu_parity_d16, 16);

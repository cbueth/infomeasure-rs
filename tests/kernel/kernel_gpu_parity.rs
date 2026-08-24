// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU vs CPU parity tests for the kernel estimators.
//!
//! Each test case builds the estimator twice — once with GPU enabled (default
//! when built with `--features gpu`) and once forced to CPU via
//! `set_force_cpu(true)` — then asserts the local values match. Sizes are
//! chosen above the GPU dispatch gates (see `estimators::gpu` defaults) so
//! the GPU path is actually exercised.

#[cfg(feature = "gpu")]
use crate::test_helpers::assert_hardware_gpu_adapter;
#[cfg(feature = "gpu")]
use infomeasure::estimators::LocalValues;
#[cfg(feature = "gpu")]
use infomeasure::estimators::mutual_information::MutualInformation;
#[cfg(feature = "gpu")]
use infomeasure::estimators::transfer_entropy::TransferEntropy;
#[cfg(feature = "gpu")]
use ndarray::Array1;
#[cfg(feature = "gpu")]
use rand::rngs::StdRng;
#[cfg(feature = "gpu")]
use rand::{Rng, SeedableRng};
#[cfg(feature = "gpu")]
use rstest::rstest;

#[cfg(feature = "gpu")]
const SIZE: usize = 6000; // above both GPU dispatch gates (box 4000, gaussian 1200)
#[cfg(feature = "gpu")]
const BANDWIDTH: f64 = 1.0;

/// Build the GPU and forced-CPU variants of an estimator and assert parity.
#[cfg(feature = "gpu")]
fn assert_gpu_matches_cpu(name: &str, build: impl Fn(bool) -> Array1<f64>) {
    let gpu = build(false);
    let cpu = build(true);

    assert_eq!(gpu.len(), cpu.len(), "{name} length mismatch");
    assert!(
        gpu.iter().all(|&v| v.is_finite()),
        "{name} GPU produced non-finite values"
    );

    // GPU path computes in f32 then converts back, so allow a modest tolerance.
    let epsilon = 1e-3;
    let max_relative = 1e-2;
    let gpu_mean = gpu.mean().unwrap();
    let cpu_mean = cpu.mean().unwrap();
    let allowed = epsilon + max_relative * cpu_mean.abs();
    assert!(
        (gpu_mean - cpu_mean).abs() <= allowed,
        "{name} global value mismatch GPU vs CPU: GPU {gpu_mean} vs CPU {cpu_mean} (allowed {allowed})"
    );

    // Spot-check a subset of local values (keep the test fast).
    let sample_size = gpu.len().min(10);
    let step = gpu.len() / sample_size.max(1);
    for i in (0..gpu.len()).step_by(step.max(1)) {
        if gpu[i].abs() > 1e-6 && cpu[i].abs() > 1e-6 {
            let allowed = epsilon + max_relative * cpu[i].abs();
            assert!(
                (gpu[i] - cpu[i]).abs() <= allowed,
                "{name} local value mismatch at index {i}: GPU {} vs CPU {} (allowed {})",
                gpu[i],
                cpu[i],
                allowed
            );
        }
    }
}

/// 2D correlated series (same pattern as the kernel benches).
#[cfg(feature = "gpu")]
fn generate_correlated(size: usize, correlation: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    use rand_distr::Normal;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(size);
    let mut y = Vec::with_capacity(size);
    for _ in 0..size {
        let z: f64 = rng.sample(Normal::new(0.0, 1.0).unwrap());
        let w: f64 = rng.sample(Normal::new(0.0, 1.0).unwrap());
        x.push(z);
        y.push(correlation * z + (1.0 - correlation.powi(2)).sqrt() * w);
    }
    (x, y)
}

/// Uniform conditioning series.
#[cfg(feature = "gpu")]
fn generate_cond(size: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(0.0..1.0)).collect()
}

#[rstest]
#[cfg(feature = "gpu")]
fn test_kernel_mi_gpu_parity(#[values("box", "gaussian")] kernel_type: &str) {
    assert_hardware_gpu_adapter();
    let (x, y) = generate_correlated(SIZE, 0.5, 42);
    let x_arr = Array1::from(x);
    let y_arr = Array1::from(y);

    assert_gpu_matches_cpu(&format!("mi_kernel/{kernel_type}"), |force_cpu| {
        let mut est = MutualInformation::new_kernel_with_type(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            BANDWIDTH,
        );
        est.set_force_cpu(force_cpu);
        est.local_values()
    });
}

#[rstest]
#[cfg(feature = "gpu")]
fn test_kernel_cmi_gpu_parity(#[values("box", "gaussian")] kernel_type: &str) {
    assert_hardware_gpu_adapter();
    let (x, y) = generate_correlated(SIZE, 0.5, 42);
    let z = Array1::from(generate_cond(SIZE, 43));

    assert_gpu_matches_cpu(&format!("cmi_kernel/{kernel_type}"), |force_cpu| {
        let mut est = MutualInformation::new_cmi_kernel_with_type(
            &[Array1::from(x.clone()), Array1::from(y.clone())],
            &z,
            kernel_type.to_string(),
            BANDWIDTH,
        );
        est.set_force_cpu(force_cpu);
        est.local_values()
    });
}

#[rstest]
#[cfg(feature = "gpu")]
fn test_kernel_te_gpu_parity(#[values("box", "gaussian")] kernel_type: &str) {
    assert_hardware_gpu_adapter();
    let (src, dst) = generate_correlated(SIZE, 0.5, 44);
    let src_arr = Array1::from(src);
    let dst_arr = Array1::from(dst);

    assert_gpu_matches_cpu(&format!("te_kernel/{kernel_type}"), |force_cpu| {
        let mut est = TransferEntropy::new_kernel_with_type(
            &src_arr.clone(),
            &dst_arr.clone(),
            1,
            1,
            1,
            kernel_type.to_string(),
            BANDWIDTH,
        );
        est.set_force_cpu(force_cpu);
        est.local_values()
    });
}

#[rstest]
#[cfg(feature = "gpu")]
fn test_kernel_cte_gpu_parity(#[values("box", "gaussian")] kernel_type: &str) {
    assert_hardware_gpu_adapter();
    let (src, dst) = generate_correlated(SIZE, 0.5, 44);
    let cond = Array1::from(generate_cond(SIZE, 45));

    assert_gpu_matches_cpu(&format!("cte_kernel/{kernel_type}"), |force_cpu| {
        let mut est = TransferEntropy::new_cte_kernel_with_type(
            &Array1::from(src.clone()),
            &Array1::from(dst.clone()),
            &cond,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            BANDWIDTH,
        );
        est.set_force_cpu(force_cpu);
        est.local_values()
    });
}

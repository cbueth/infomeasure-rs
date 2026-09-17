// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU vs CPU parity for the KSG count tier (feature `gpu`).
//!
//! Each case builds the estimator twice, once with the GPU count path enabled
//! and once forced onto the CPU, then compares the results. Sizes are above the
//! shipped KSG gate (4000 integrated, 1000 discrete) so the GPU path is actually
//! exercised.
//!
//! The counts are integers but the shader evaluates f32 distances, so a
//! candidate exactly at the radius boundary could be classified differently.
//! `add_noise` and continuous data keep exact ties out, and the comparison uses
//! the kernel suite's tolerance rather than bit equality.

#![cfg(feature = "gpu")]

use crate::test_helpers::assert_hardware_gpu_adapter;
use infomeasure::estimators::gpu::set_gpu_min_points_override;
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::traits::{GlobalValue, LocalValues};
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use rstest::rstest;

/// Above the shipped KSG gate on both integrated (4000) and discrete (1000).
const SIZE: usize = 6000;

fn series() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::from_entropy();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut x = Vec::with_capacity(SIZE);
    let mut y = Vec::with_capacity(SIZE);
    let mut z = Vec::with_capacity(SIZE);
    for _ in 0..SIZE {
        let a: f64 = normal.sample(&mut rng);
        let b: f64 = normal.sample(&mut rng);
        let c: f64 = normal.sample(&mut rng);
        x.push(a);
        y.push(0.5 * a + 0.75_f64.sqrt() * b);
        z.push(c);
    }
    (Array1::from(x), Array1::from(y), Array1::from(z))
}

/// Force the KSG count gate open so the GPU path is exercised even where the
/// shipped profile disables it (integrated GPUs).
fn force_ksg_gpu() {
    set_gpu_min_points_override(None, None, None, None, Some(0));
}

fn assert_parity(name: &str, gpu: f64, cpu: f64) {
    let allowed = 5e-3 + 1e-2 * cpu.abs();
    assert!(
        (gpu - cpu).abs() <= allowed,
        "{name} mismatch: GPU {gpu} vs CPU {cpu} (allowed {allowed})"
    );
}

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

#[rstest]
#[case(3, true)]
#[case(3, false)]
#[case(10, true)]
fn ksg_mi_gpu_parity(#[case] k: usize, #[case] use_chebyshev: bool) {
    assert_hardware_gpu_adapter();
    force_ksg_gpu();
    let (x, y, _) = series();

    let mut gpu =
        MutualInformation::new_ksg(&[x.clone(), y.clone()], k, 0.0).with_chebyshev(use_chebyshev);
    gpu.set_force_cpu(false);
    let mut cpu =
        MutualInformation::new_ksg(&[x.clone(), y.clone()], k, 0.0).with_chebyshev(use_chebyshev);
    cpu.set_force_cpu(true);

    let name = format!("mi_ksg_k{k}_cheb{use_chebyshev}");
    assert_parity(&name, gpu.global_value(), cpu.global_value());
    assert_local_parity(
        &name,
        &gpu.local_values().to_vec(),
        &cpu.local_values().to_vec(),
    );
}

#[rstest]
#[case(3, true)]
#[case(3, false)]
#[case(10, true)]
fn ksg_cmi_gpu_parity(#[case] k: usize, #[case] use_chebyshev: bool) {
    assert_hardware_gpu_adapter();
    force_ksg_gpu();
    let (x, y, z) = series();

    let mut gpu = MutualInformation::new_cmi_ksg(&[x.clone(), y.clone()], &z, k, 0.0)
        .with_chebyshev(use_chebyshev);
    gpu.set_force_cpu(false);
    let mut cpu = MutualInformation::new_cmi_ksg(&[x.clone(), y.clone()], &z, k, 0.0)
        .with_chebyshev(use_chebyshev);
    cpu.set_force_cpu(true);

    let name = format!("cmi_ksg_k{k}_cheb{use_chebyshev}");
    assert_parity(&name, gpu.global_value(), cpu.global_value());
    assert_local_parity(
        &name,
        &gpu.local_values().to_vec(),
        &cpu.local_values().to_vec(),
    );
}

#[rstest]
#[case(3, true)]
#[case(3, false)]
#[case(10, true)]
fn ksg_te_gpu_parity(#[case] k: usize, #[case] use_chebyshev: bool) {
    assert_hardware_gpu_adapter();
    force_ksg_gpu();
    let (x, y, _) = series();

    let mut gpu = TransferEntropy::new_ksg(&x, &y, 1, 1, 1, k, 0.0).with_chebyshev(use_chebyshev);
    gpu.set_force_cpu(false);
    let mut cpu = TransferEntropy::new_ksg(&x, &y, 1, 1, 1, k, 0.0).with_chebyshev(use_chebyshev);
    cpu.set_force_cpu(true);

    let name = format!("te_ksg_k{k}_cheb{use_chebyshev}");
    assert_parity(&name, gpu.global_value(), cpu.global_value());
}

#[rstest]
#[case(3, true)]
#[case(3, false)]
#[case(10, true)]
fn ksg_cte_gpu_parity(#[case] k: usize, #[case] use_chebyshev: bool) {
    assert_hardware_gpu_adapter();
    force_ksg_gpu();
    let (x, y, z) = series();

    let mut gpu =
        TransferEntropy::new_cte_ksg(&x, &y, &z, 1, 1, 1, 1, k, 0.0).with_chebyshev(use_chebyshev);
    gpu.set_force_cpu(false);
    let mut cpu =
        TransferEntropy::new_cte_ksg(&x, &y, &z, 1, 1, 1, 1, k, 0.0).with_chebyshev(use_chebyshev);
    cpu.set_force_cpu(true);

    let name = format!("cte_ksg_k{k}_cheb{use_chebyshev}");
    assert_parity(&name, gpu.global_value(), cpu.global_value());
}

// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use infomeasure::estimators::{Entropy, GlobalValue};
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

// Import test helper functions
use crate::test_helpers::{generate_random_nd_data, measure_execution_time};

/// Measure the performance of the Gaussian kernel entropy calculation
fn measure_gaussian_kernel_performance(
    size: usize,
    dims: usize,
    bandwidth: f64,
    seed: u64,
) -> Duration {
    // Generate random data
    let data = generate_random_nd_data(size, dims, seed);

    // Measure performance using the centralized function
    measure_execution_time(|| match dims {
        1 => {
            let _ =
                Entropy::nd_kernel_with_type::<1>(data.clone(), "gaussian".to_string(), bandwidth)
                    .global_value();
        }
        2 => {
            let _ =
                Entropy::nd_kernel_with_type::<2>(data.clone(), "gaussian".to_string(), bandwidth)
                    .global_value();
        }
        3 => {
            let _ =
                Entropy::nd_kernel_with_type::<3>(data.clone(), "gaussian".to_string(), bandwidth)
                    .global_value();
        }
        4 => {
            let _ =
                Entropy::nd_kernel_with_type::<4>(data.clone(), "gaussian".to_string(), bandwidth)
                    .global_value();
        }
        8 => {
            let _ =
                Entropy::nd_kernel_with_type::<8>(data.clone(), "gaussian".to_string(), bandwidth)
                    .global_value();
        }
        16 => {
            let _ =
                Entropy::nd_kernel_with_type::<16>(data.clone(), "gaussian".to_string(), bandwidth)
                    .global_value();
        }
        32 => {
            let _ =
                Entropy::nd_kernel_with_type::<32>(data.clone(), "gaussian".to_string(), bandwidth)
                    .global_value();
        }
        _ => panic!("Unsupported number of dimensions: {dims}"),
    })
}

/// Measure the performance of the Box kernel entropy calculation
fn measure_box_kernel_performance(size: usize, dims: usize, bandwidth: f64, seed: u64) -> Duration {
    // Generate random data
    let data = generate_random_nd_data(size, dims, seed);

    // Measure performance
    let start = Instant::now();

    match dims {
        1 => Entropy::nd_kernel_with_type::<1>(data.clone(), "box".to_string(), bandwidth)
            .global_value(),
        2 => Entropy::nd_kernel_with_type::<2>(data.clone(), "box".to_string(), bandwidth)
            .global_value(),
        3 => Entropy::nd_kernel_with_type::<3>(data.clone(), "box".to_string(), bandwidth)
            .global_value(),
        4 => Entropy::nd_kernel_with_type::<4>(data.clone(), "box".to_string(), bandwidth)
            .global_value(),
        8 => Entropy::nd_kernel_with_type::<8>(data.clone(), "box".to_string(), bandwidth)
            .global_value(),
        16 => Entropy::nd_kernel_with_type::<16>(data.clone(), "box".to_string(), bandwidth)
            .global_value(),
        32 => Entropy::nd_kernel_with_type::<32>(data.clone(), "box".to_string(), bandwidth)
            .global_value(),
        _ => panic!("Unsupported number of dimensions: {dims}"),
    };

    start.elapsed()
}

#[ignore]
#[test]
fn test_gaussian_kernel_performance() {
    // Define test parameters
    let sizes = [100, 500, 1000, 5000, 10000];
    let dimensions = [1, 2, 4, 8, 16, 32];
    let bandwidth = 0.5;
    let seed = 42;
    let num_runs = 3; // Reduced number of runs to speed up testing

    // Create a file to store the results
    #[cfg(all(feature = "gpu", feature = "fast_exp"))]
    let mut file = File::create("gaussian_gpu_fast_exp_performance.md").unwrap();
    #[cfg(all(feature = "gpu", not(feature = "fast_exp")))]
    let mut file = File::create("gaussian_gpu_performance.md").unwrap();
    #[cfg(all(not(feature = "gpu"), feature = "fast_exp"))]
    let mut file = File::create("gaussian_fast_exp_performance.md").unwrap();
    #[cfg(all(not(feature = "gpu"), not(feature = "fast_exp")))]
    let mut file = File::create("gaussian_baseline_performance.md").unwrap();

    #[cfg(all(feature = "gpu", feature = "fast_exp"))]
    writeln!(file, "# Gaussian Kernel GPU with Fast Exp Performance\n").unwrap();
    #[cfg(all(feature = "gpu", not(feature = "fast_exp")))]
    writeln!(file, "# Gaussian Kernel GPU Performance\n").unwrap();
    #[cfg(all(not(feature = "gpu"), feature = "fast_exp"))]
    writeln!(file, "# Gaussian Kernel Fast Exp Performance\n").unwrap();
    #[cfg(all(not(feature = "gpu"), not(feature = "fast_exp")))]
    writeln!(file, "# Gaussian Kernel Baseline Performance\n").unwrap();

    writeln!(file, "| Data Size | Dimensions | Time (ms) |").unwrap();
    writeln!(file, "|-----------|------------|-----------|").unwrap();

    for &size in &sizes {
        for &dims in &dimensions {
            // Run multiple times and take the average
            let mut total_duration = Duration::new(0, 0);

            for _ in 0..num_runs {
                let duration = measure_gaussian_kernel_performance(size, dims, bandwidth, seed);
                total_duration += duration;
            }

            let avg_duration = total_duration / num_runs as u32;
            let avg_ms = avg_duration.as_millis();

            println!("Gaussian - Size: {size}, Dims: {dims}, Time: {avg_ms} ms");
            writeln!(file, "| {size} | {dims} | {avg_ms} |").unwrap();
        }
    }

    #[cfg(all(feature = "gpu", feature = "fast_exp"))]
    println!(
        "Gaussian kernel GPU with Fast Exp performance results have been saved to gaussian_gpu_fast_exp_performance.md"
    );
    #[cfg(all(feature = "gpu", not(feature = "fast_exp")))]
    println!(
        "Gaussian kernel GPU performance results have been saved to gaussian_gpu_performance.md"
    );
    #[cfg(all(not(feature = "gpu"), feature = "fast_exp"))]
    println!(
        "Gaussian kernel Fast Exp performance results have been saved to gaussian_fast_exp_performance.md"
    );
    #[cfg(all(not(feature = "gpu"), not(feature = "fast_exp")))]
    println!(
        "Gaussian kernel baseline performance results have been saved to gaussian_baseline_performance.md"
    );
}

#[test]
#[ignore]
fn test_box_kernel_performance() {
    // Define test parameters
    let sizes = [100, 500, 1000, 5000, 10000];
    let dimensions = [1, 2, 4, 8, 16, 32];
    let bandwidth = 0.5;
    let seed = 42;
    let num_runs = 3; // Reduced number of runs to speed up testing

    // Create a file to store the results
    #[cfg(all(feature = "gpu", feature = "fast_exp"))]
    let mut file = File::create("box_gpu_fast_exp_performance.md").unwrap();
    #[cfg(all(feature = "gpu", not(feature = "fast_exp")))]
    let mut file = File::create("box_gpu_performance.md").unwrap();
    #[cfg(all(not(feature = "gpu"), feature = "fast_exp"))]
    let mut file = File::create("box_fast_exp_performance.md").unwrap();
    #[cfg(all(not(feature = "gpu"), not(feature = "fast_exp")))]
    let mut file = File::create("box_baseline_performance.md").unwrap();

    #[cfg(all(feature = "gpu", feature = "fast_exp"))]
    writeln!(file, "# Box Kernel GPU with Fast Exp Performance\n").unwrap();
    #[cfg(all(feature = "gpu", not(feature = "fast_exp")))]
    writeln!(file, "# Box Kernel GPU Performance\n").unwrap();
    #[cfg(all(not(feature = "gpu"), feature = "fast_exp"))]
    writeln!(file, "# Box Kernel Fast Exp Performance\n").unwrap();
    #[cfg(all(not(feature = "gpu"), not(feature = "fast_exp")))]
    writeln!(file, "# Box Kernel Baseline Performance\n").unwrap();

    writeln!(file, "| Data Size | Dimensions | Time (ms) |").unwrap();
    writeln!(file, "|-----------|------------|-----------|").unwrap();

    for &size in &sizes {
        for &dims in &dimensions {
            // Run multiple times and take the average
            let mut total_duration = Duration::new(0, 0);

            for _ in 0..num_runs {
                let duration = measure_box_kernel_performance(size, dims, bandwidth, seed);
                total_duration += duration;
            }

            let avg_duration = total_duration / num_runs as u32;
            let avg_ms = avg_duration.as_millis();

            println!("Box - Size: {size}, Dims: {dims}, Time: {avg_ms} ms");
            writeln!(file, "| {size} | {dims} | {avg_ms} |").unwrap();
        }
    }

    #[cfg(all(feature = "gpu", feature = "fast_exp"))]
    println!(
        "Box kernel GPU with Fast Exp performance results have been saved to box_gpu_fast_exp_performance.md"
    );
    #[cfg(all(feature = "gpu", not(feature = "fast_exp")))]
    println!("Box kernel GPU performance results have been saved to box_gpu_performance.md");
    #[cfg(all(not(feature = "gpu"), feature = "fast_exp"))]
    println!(
        "Box kernel Fast Exp performance results have been saved to box_fast_exp_performance.md"
    );
    #[cfg(all(not(feature = "gpu"), not(feature = "fast_exp")))]
    println!(
        "Box kernel baseline performance results have been saved to box_baseline_performance.md"
    );
}

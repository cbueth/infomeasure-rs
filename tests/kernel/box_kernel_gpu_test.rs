// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_relative_eq;
use infomeasure::estimators::entropy::{Entropy, LocalValues};
use ndarray::Array2;

// Import test helper functions
use crate::test_helpers::{generate_random_nd_data, measure_execution_time};

/// Test that compares the CPU and GPU implementations of the box kernel
///
/// This test is only run when the `gpu_support` feature is enabled.
/// It verifies that the GPU implementation produces the same results as the CPU implementation.
#[test]
#[cfg(feature = "gpu_support")]
fn test_box_kernel_cpu_vs_gpu() {
    // Set up test parameters
    let seed = 42;
    let size = 1000; // Use a larger size to make GPU acceleration worthwhile
    let bandwidths = [0.5, 1.0, 2.0];
    let dimensions = [1, 2, 3, 4, 8];

    println!("Testing box kernel CPU vs GPU implementation");

    for &dim in &dimensions {
        for &bandwidth in &bandwidths {
            // Generate random data
            let data = generate_random_nd_data(size, dim, seed);

            // Create test name
            let test_name = format!("Box Kernel (dim={dim}, bandwidth={bandwidth}, size={size})");
            println!("Testing {test_name}");

            // Compare CPU and GPU implementations based on dimension
            match dim {
                1 => compare_box_kernel_cpu_vs_gpu::<1>(data, bandwidth, &test_name),
                2 => compare_box_kernel_cpu_vs_gpu::<2>(data, bandwidth, &test_name),
                3 => compare_box_kernel_cpu_vs_gpu::<3>(data, bandwidth, &test_name),
                4 => compare_box_kernel_cpu_vs_gpu::<4>(data, bandwidth, &test_name),
                8 => compare_box_kernel_cpu_vs_gpu::<8>(data, bandwidth, &test_name),
                _ => panic!("Unsupported dimension: {dim}"),
            };
        }
    }
}

/// Helper function to compare CPU and GPU implementations of the box kernel
fn compare_box_kernel_cpu_vs_gpu<const K: usize>(
    data: Array2<f64>,
    bandwidth: f64,
    test_name: &str,
) {
    // Create a KernelEntropy instance
    let kernel = Entropy::nd_kernel_with_type::<K>(data.clone(), "box".to_string(), bandwidth);

    // Calculate entropy using the CPU implementation directly
    let cpu_local_entropy = kernel.box_kernel_local_values();
    let cpu_global_entropy = cpu_local_entropy.mean().unwrap();

    // Calculate entropy using the local_values method which will use GPU when available
    // The LocalValues trait implementation will use GPU when the feature is enabled
    let gpu_local_entropy = kernel.local_values();
    let gpu_global_entropy = gpu_local_entropy.mean().unwrap();

    // Print results
    println!("{test_name} - CPU global entropy: {cpu_global_entropy}");
    println!("{test_name} - GPU global entropy: {gpu_global_entropy}");

    // Assert that the global entropy values are approximately equal
    let epsilon = 1e-6;
    let max_relative = 1e-6;
    assert_relative_eq!(
        cpu_global_entropy,
        gpu_global_entropy,
        epsilon = epsilon,
        max_relative = max_relative
    );

    // Assert that the local entropy values are approximately equal
    // Sample a subset of values to keep the test fast
    let sample_size = cpu_local_entropy.len().min(10);
    let step = cpu_local_entropy.len() / sample_size.max(1);

    for i in (0..cpu_local_entropy.len()).step_by(step.max(1)) {
        if i < cpu_local_entropy.len() && i < gpu_local_entropy.len() {
            let cpu_val = cpu_local_entropy[i];
            let gpu_val = gpu_local_entropy[i];

            if cpu_val.abs() > 1e-6 && gpu_val.abs() > 1e-6 {
                let epsilon = 1e-6;
                let max_relative = 1e-5;
                assert_relative_eq!(
                    cpu_val,
                    gpu_val,
                    epsilon = epsilon,
                    max_relative = max_relative
                );
            }
        }
    }

    println!("{test_name} - CPU and GPU implementations match");
}

/// Test that verifies the GPU fallback mechanism works correctly
///
/// This test is only run when the `gpu_support` feature is enabled.
/// It tests that the GPU implementation falls back to the CPU implementation
/// under certain conditions and produces the same results.
#[test]
#[cfg(feature = "gpu_support")]
fn test_box_kernel_gpu_fallback() {
    // Set up test parameters
    let seed = 42;

    // Test small dataset (should trigger fallback due to small size)
    let small_size = 50; // Less than 100 should trigger fallback
    let bandwidth = 1.0;
    let dimensions = [1, 2, 3, 4, 8];

    println!("Testing box kernel GPU fallback mechanism for small datasets");

    for &dim in &dimensions {
        // Generate small random data
        let small_data = generate_random_nd_data(small_size, dim, seed);

        // Create test name
        let test_name =
            format!("Box Kernel Fallback (small dataset, dim={dim}, size={small_size})");
        println!("Testing {test_name}");

        // Test fallback based on dimension
        match dim {
            1 => test_box_kernel_fallback::<1>(small_data, bandwidth, &test_name, "small_dataset"),
            2 => test_box_kernel_fallback::<2>(small_data, bandwidth, &test_name, "small_dataset"),
            3 => test_box_kernel_fallback::<3>(small_data, bandwidth, &test_name, "small_dataset"),
            4 => test_box_kernel_fallback::<4>(small_data, bandwidth, &test_name, "small_dataset"),
            8 => test_box_kernel_fallback::<8>(small_data, bandwidth, &test_name, "small_dataset"),
            _ => panic!("Unsupported dimension: {dim}"),
        };
    }

    // Test large dimensions (should trigger fallback due to dimension limit)
    // Note: We can't actually test dimensions > 32 due to const generic limitations,
    // but we can verify that the results are consistent for the dimensions we can test
    println!("Testing box kernel GPU fallback mechanism for large dimensions");

    let large_size = 1000;
    let large_dim = 8; // Use the largest dimension we can test
    let large_data = generate_random_nd_data(large_size, large_dim, seed);

    let test_name =
        format!("Box Kernel Fallback (large dimension, dim={large_dim}, size={large_size})");
    println!("Testing {test_name}");

    test_box_kernel_fallback::<8>(large_data, bandwidth, &test_name, "large_dimension");
}

/// Helper function to test the GPU fallback mechanism
fn test_box_kernel_fallback<const K: usize>(
    data: Array2<f64>,
    bandwidth: f64,
    test_name: &str,
    fallback_type: &str,
) {
    // Create a KernelEntropy instance
    let kernel = Entropy::nd_kernel_with_type::<K>(data.clone(), "box".to_string(), bandwidth);

    // Calculate entropy using the CPU implementation directly
    let cpu_local_entropy = kernel.box_kernel_local_values();
    let cpu_global_entropy = cpu_local_entropy.mean().unwrap();

    // Calculate entropy using the local_values method which will use GPU when available
    // The LocalValues trait implementation will use GPU when the feature is enabled
    // and will automatically fall back to CPU when needed
    let gpu_local_entropy = kernel.local_values();
    let gpu_global_entropy = gpu_local_entropy.mean().unwrap();

    // Print results
    println!("{test_name} - CPU global entropy: {cpu_global_entropy}");
    println!("{test_name} - GPU with fallback global entropy: {gpu_global_entropy}");

    // For small datasets, the GPU implementation should fall back to CPU and produce exactly the same results
    if fallback_type == "small_dataset" {
        // Assert that the global entropy values are exactly equal (since it should use the same implementation)
        assert_eq!(cpu_global_entropy, gpu_global_entropy);

        // Assert that all local entropy values are exactly equal
        assert_eq!(cpu_local_entropy.len(), gpu_local_entropy.len());
        for i in 0..cpu_local_entropy.len() {
            assert_eq!(cpu_local_entropy[i], gpu_local_entropy[i]);
        }

        println!("{test_name} - CPU and GPU with fallback implementations match exactly");
    } else {
        // For large dimensions, the GPU implementation might not fall back to CPU
        // but should still produce very similar results
        assert_relative_eq!(
            cpu_global_entropy,
            gpu_global_entropy,
            epsilon = 1e-6,
            max_relative = 1e-6
        );

        // Assert that the local entropy values are approximately equal
        // Sample a subset of values to keep the test fast
        let sample_size = cpu_local_entropy.len().min(10);
        let step = cpu_local_entropy.len() / sample_size.max(1);

        for i in (0..cpu_local_entropy.len()).step_by(step.max(1)) {
            if i < cpu_local_entropy.len() && i < gpu_local_entropy.len() {
                let cpu_val = cpu_local_entropy[i];
                let gpu_val = gpu_local_entropy[i];

                if cpu_val.abs() > 1e-6 && gpu_val.abs() > 1e-6 {
                    assert_relative_eq!(cpu_val, gpu_val, epsilon = 1e-6, max_relative = 1e-5);
                }
            }
        }

        println!("{test_name} - CPU and GPU implementations match approximately");
    }
}

/// Test that measures the performance of the box kernel CPU and GPU implementations
///
/// This test is only run when the `gpu_support` feature is enabled.
/// It measures the execution time of both implementations and prints the speedup.
#[test]
#[cfg(feature = "gpu_support")]
#[ignore]
fn test_box_kernel_performance() {
    // Set up test parameters
    let seed = 42;
    let sizes = [1000, 10000, 100000];
    let bandwidth = 1.0;
    let dimensions = [1, 2, 3, 4, 8];
    let num_runs = 3; // Number of runs to average over

    println!("Measuring box kernel CPU vs GPU performance");
    println!("| Dimensions | Size | CPU Time (ms) | GPU Time (ms) | Speedup |");
    println!("|------------|------|---------------|---------------|---------|");

    for &dim in &dimensions {
        for &size in &sizes {
            // Skip very large combinations that might be too slow
            if size >= 100000 && dim >= 4 {
                continue;
            }

            // Generate random data
            let data = generate_random_nd_data(size, dim, seed);

            // Measure performance based on dimension
            match dim {
                1 => measure_box_kernel_performance::<1>(data, bandwidth, num_runs),
                2 => measure_box_kernel_performance::<2>(data, bandwidth, num_runs),
                3 => measure_box_kernel_performance::<3>(data, bandwidth, num_runs),
                4 => measure_box_kernel_performance::<4>(data, bandwidth, num_runs),
                8 => measure_box_kernel_performance::<8>(data, bandwidth, num_runs),
                _ => panic!("Unsupported dimension: {dim}"),
            };
        }
    }
}

/// Helper function to measure the performance of CPU and GPU implementations of the box kernel
fn measure_box_kernel_performance<const K: usize>(
    data: Array2<f64>,
    bandwidth: f64,
    num_runs: usize,
) {
    // Create a KernelEntropy instance
    let kernel = Entropy::nd_kernel_with_type::<K>(data.clone(), "box".to_string(), bandwidth);

    // Measure CPU time
    let mut cpu_total_time = 0.0;
    for _ in 0..num_runs {
        let duration = measure_execution_time(|| {
            let _ = kernel.box_kernel_local_values();
        });
        cpu_total_time += duration.as_secs_f64() * 1000.0; // Convert to milliseconds
    }
    let cpu_avg_time = cpu_total_time / num_runs as f64;

    // Measure GPU time
    let mut gpu_total_time = 0.0;
    for _ in 0..num_runs {
        let duration = measure_execution_time(|| {
            let _ = kernel.local_values();
        });
        gpu_total_time += duration.as_secs_f64() * 1000.0; // Convert to milliseconds
    }
    let gpu_avg_time = gpu_total_time / num_runs as f64;

    // Calculate speedup
    let speedup = if gpu_avg_time > 0.0 {
        cpu_avg_time / gpu_avg_time
    } else {
        0.0
    };

    // Print results in table format
    println!(
        "| {K} | {} | {cpu_avg_time:.2} | {gpu_avg_time:.2} | {speedup:.2}x |",
        data.nrows()
    );
}

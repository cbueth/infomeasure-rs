// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_relative_eq;
use infomeasure::estimators::entropy::{Entropy, GlobalValue, LocalValues};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use validation::python;

/// Test to verify that the Python interface works correctly by comparing
/// the Python implementation with the Rust implementation.
#[test]
fn test_python_interface_works() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;

    // Generate random integers between 0 and 10
    let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..10) as i32).collect();

    // Convert data to Array1 for Rust implementation
    let data_array = Array1::from(data.clone());

    // Calculate entropy using the Rust implementation
    let discrete_entropy = Entropy::new_discrete(data_array);
    let rust_entropy = discrete_entropy.global_value();

    // Calculate entropy using the Python implementation with discrete approach
    let python_entropy = python::calculate_entropy(&data, "discrete", &[]).unwrap();

    // Print the results to verify both implementations
    println!("Rust entropy: {}", rust_entropy);
    println!("Python entropy: {}", python_entropy);

    // Assert that the entropy values from both implementations are approximately equal
    assert_relative_eq!(
        rust_entropy,
        python_entropy,
        epsilon = 1e-10,
        max_relative = 1e-6
    );
}

/// Test to verify that the calculate_entropy_float function works correctly.
#[test]
fn test_calculate_entropy_float() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;

    // Generate random floats between 0.0 and 10.0
    let data: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();

    // Convert data to Array1 for Rust implementation
    let data_array = Array1::from(data.clone());

    // Calculate entropy using the Rust implementation with kernel approach
    let bandwidth = 1.0;
    let kernel_entropy = Entropy::new_kernel(data_array, bandwidth);
    let rust_entropy = kernel_entropy.global_value();

    // Calculate entropy using the Python implementation with kernel approach
    let kernel_kwargs = [
        ("kernel".to_string(), "\"box\"".to_string()),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let python_entropy = python::calculate_entropy_float(&data, "kernel", &kernel_kwargs).unwrap();

    // Print the results to verify both implementations
    println!("Rust kernel entropy: {}", rust_entropy);
    println!("Python kernel entropy: {}", python_entropy);

    // Assert that the entropy values from both implementations are approximately equal
    assert_relative_eq!(
        rust_entropy,
        python_entropy,
        epsilon = 1e-6,
        max_relative = 1e-3
    );
}

/// Test to verify that the calculate_local_entropy function works correctly.
#[test]
fn test_calculate_local_entropy() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;

    // Generate random integers between 0 and 10
    let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..10) as i32).collect();

    // Convert data to Array1 for Rust implementation
    let data_array = Array1::from(data.clone());

    // Calculate local entropy using the Rust implementation
    let discrete_entropy = Entropy::new_discrete(data_array);
    let rust_local_entropy = discrete_entropy.local_values();

    // Calculate local entropy using the Python implementation with discrete approach
    let python_local_entropy = python::calculate_local_entropy(&data, "discrete", &[]).unwrap();

    // Print the results to verify both implementations
    println!("Rust local entropy: {:?}", rust_local_entropy);
    println!("Python local entropy: {:?}", python_local_entropy);

    // Assert that the local entropy values from both implementations are approximately equal
    for (rust_val, python_val) in rust_local_entropy.iter().zip(python_local_entropy.iter()) {
        assert_relative_eq!(*rust_val, *python_val, epsilon = 1e-10, max_relative = 1e-6);
    }
}

/// Test to verify that the calculate_local_entropy_float function works correctly.
#[test]
fn test_calculate_local_entropy_float() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;

    // Generate random floats between 0.0 and 10.0
    let data: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();

    // Convert data to Array1 for Rust implementation
    let data_array = Array1::from(data.clone());

    // Calculate local entropy using the Rust implementation with kernel approach
    let bandwidth = 1.0;
    let kernel_entropy = Entropy::new_kernel(data_array, bandwidth);
    let rust_local_entropy = kernel_entropy.local_values();

    // Calculate local entropy using the Python implementation with kernel approach
    let kernel_kwargs = [
        ("kernel".to_string(), "\"box\"".to_string()),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let python_local_entropy =
        python::calculate_local_entropy_float(&data, "kernel", &kernel_kwargs).unwrap();

    // Print the results to verify both implementations
    println!("Rust local kernel entropy: {:?}", rust_local_entropy);
    println!("Python local kernel entropy: {:?}", python_local_entropy);

    // Assert that the local entropy values from both implementations are approximately equal
    for (rust_val, python_val) in rust_local_entropy.iter().zip(python_local_entropy.iter()) {
        assert_relative_eq!(*rust_val, *python_val, epsilon = 1e-6, max_relative = 1e-3);
    }
}

/// Test to verify that the calculate_entropy_float_nd function works correctly.
#[test]
fn test_calculate_entropy_float_nd() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;
    let dims = 2;

    // Generate random 2D data
    let mut data = Array2::zeros((size, dims));
    for i in 0..size {
        for j in 0..dims {
            data[[i, j]] = rng.gen_range(0.0..10.0);
        }
    }

    // Calculate entropy using the Rust implementation with kernel approach
    let bandwidth = 1.0;
    let kernel_entropy = Entropy::nd_kernel::<2>(data.clone(), bandwidth);
    let rust_entropy = kernel_entropy.global_value();

    // Convert the 2D array to a flat array for Python
    let flat_data: Vec<f64> = data.iter().cloned().collect();

    // Calculate entropy using the Python implementation with n-dimensional functions
    let kernel_kwargs = [
        ("kernel".to_string(), "\"box\"".to_string()),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let python_entropy =
        python::calculate_entropy_float_nd(&flat_data, dims, "kernel", &kernel_kwargs).unwrap();

    // Print the results to verify both implementations
    println!("Rust 2D kernel entropy: {}", rust_entropy);
    println!("Python 2D kernel entropy: {}", python_entropy);

    // Assert that the entropy values from both implementations are approximately equal
    assert_relative_eq!(
        rust_entropy,
        python_entropy,
        epsilon = 1e-6,
        max_relative = 0.1
    );
}

/// Test to verify that the calculate_local_entropy_float_nd function works correctly.
#[test]
fn test_calculate_local_entropy_float_nd() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;
    let dims = 2;

    // Generate random 2D data
    let mut data = Array2::zeros((size, dims));
    for i in 0..size {
        for j in 0..dims {
            data[[i, j]] = rng.gen_range(0.0..10.0);
        }
    }

    // Calculate local entropy using the Rust implementation with kernel approach
    let bandwidth = 1.0;
    let kernel_entropy = Entropy::nd_kernel::<2>(data.clone(), bandwidth);
    let rust_local_entropy = kernel_entropy.local_values();

    // Convert the 2D array to a flat array for Python
    let flat_data: Vec<f64> = data.iter().cloned().collect();

    // Calculate local entropy using the Python implementation with n-dimensional functions
    let kernel_kwargs = [
        ("kernel".to_string(), "\"box\"".to_string()),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let python_local_entropy =
        python::calculate_local_entropy_float_nd(&flat_data, dims, "kernel", &kernel_kwargs)
            .unwrap();

    // Print the results to verify both implementations
    println!("Rust 2D local kernel entropy: {:?}", rust_local_entropy);
    println!("Python 2D local kernel entropy: {:?}", python_local_entropy);

    // Assert that the local entropy values from both implementations are approximately equal
    let sample_size = rust_local_entropy.len().min(10);
    let step = rust_local_entropy.len() / sample_size.max(1);

    for i in (0..rust_local_entropy.len()).step_by(step.max(1)) {
        if i < rust_local_entropy.len() && i < python_local_entropy.len() {
            let rust_val = rust_local_entropy[i];
            let python_val = python_local_entropy[i];

            if rust_val.abs() > 1e-6 && python_val.abs() > 1e-6 {
                assert_relative_eq!(rust_val, python_val, epsilon = 1e-6, max_relative = 0.3);
            }
        }
    }
}

/// Test to verify that the benchmark_entropy function works correctly.
#[test]
fn test_benchmark_entropy() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;

    // Generate random integers between 0 and 10
    let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..10) as i32).collect();

    // Benchmark the Python implementation
    let num_runs = 5;
    let python_time = python::benchmark_entropy(&data, num_runs).unwrap();

    // Verify that the benchmark returns a positive time value
    println!("Python benchmark time: {} seconds", python_time);
    assert!(python_time > 0.0, "Benchmark time should be positive");
}

/// Test to verify that the benchmark_entropy_float_nd function works correctly.
#[test]
fn test_benchmark_entropy_float_nd() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;
    let dims = 2;

    // Generate random 2D data as a flat array
    let flat_data: Vec<f64> = (0..size * dims).map(|_| rng.gen_range(0.0..10.0)).collect();

    // Benchmark the Python implementation
    let num_runs = 5;
    let kernel_kwargs = [
        ("kernel".to_string(), "\"box\"".to_string()),
        ("bandwidth".to_string(), "1.0".to_string()),
    ];
    let python_time =
        python::benchmark_entropy_float_nd(&flat_data, dims, "kernel", &kernel_kwargs, num_runs)
            .unwrap();

    // Verify that the benchmark returns a positive time value
    println!(
        "Python benchmark time for float_nd: {} seconds",
        python_time
    );
    assert!(python_time > 0.0, "Benchmark time should be positive");
}

/// Test to verify that the benchmark_entropy_generic function works correctly.
#[test]
fn test_benchmark_entropy_generic() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;

    // Generate random floats between 0.0 and 10.0
    let data: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();

    // Benchmark the Python implementation
    let num_runs = 5;
    let kernel_kwargs = [
        ("kernel".to_string(), "\"box\"".to_string()),
        ("bandwidth".to_string(), "1.0".to_string()),
    ];
    let python_time =
        python::benchmark_entropy_generic(&data, "kernel", &kernel_kwargs, num_runs).unwrap();

    // Verify that the benchmark returns a positive time value
    println!("Python benchmark time for generic: {} seconds", python_time);
    assert!(python_time > 0.0, "Benchmark time should be positive");
}

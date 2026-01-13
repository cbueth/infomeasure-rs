use ndarray::Array1;
use infomeasure::estimators::entropy::{Entropy, GlobalValue, LocalValues};
use validation::python;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

// Import test helper functions
use crate::test_helpers::assert_entropy_values_close;

/// Helper function to compare discrete entropy between Rust and Python implementations
///
/// This function:
/// 1. Takes a vector of integers as input data
/// 2. Calculates entropy using both Rust and Python implementations
/// 3. Compares the results and asserts they are approximately equal
/// 4. Returns the calculated entropy values for potential further analysis
fn compare_discrete_entropy(data: Vec<i32>, test_name: &str) -> (f64, Vec<f64>) {
    let data_array = Array1::from(data.clone());

    // Calculate entropy using the Rust implementation
    let discrete_entropy = Entropy::new_discrete(data_array);
    let rust_global_entropy = discrete_entropy.global_value();
    let rust_local_entropy = discrete_entropy.local_values();

    // Calculate entropy using the Python implementation with discrete approach
    let python_global_entropy = python::calculate_entropy(&data, "discrete", &[]).unwrap();
    let python_local_entropy = python::calculate_local_entropy(&data, "discrete", &[]).unwrap();

    // Compare the results
    println!("{} - Rust global entropy (base e): {}", test_name, rust_global_entropy);
    println!("{} - Python global entropy (base e): {}", test_name, python_global_entropy);

    // Assert that the global entropy values are approximately equal
    assert_entropy_values_close(
        rust_global_entropy, 
        python_global_entropy, 
        1e-10,
        1e-6,
        test_name
    );

    // Assert that the local entropy values are approximately equal
    for (rust_val, python_val) in rust_local_entropy.iter()
        .zip(python_local_entropy.iter()) {
        assert_entropy_values_close(
            *rust_val, 
            *python_val, 
            1e-10,
            1e-6,
            &format!("{} (local)", test_name)
        );
    }

    (rust_global_entropy, rust_local_entropy.to_vec())
}

/// Test that compares the discrete entropy implementation in Rust with the Python infomeasure package
/// using random uniform data with different numbers of possible states.
#[test]
fn test_discrete_entropy_different_states() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 100;

    // Test with different numbers of possible states: [2, 3, 4, 10, 15, 20]
    let states = [2, 3, 4, 10, 15, 20];

    for &num_states in &states {
        // Generate random integers between 0 and (num_states-1)
        let data: Vec<i32> = (0..size)
            .map(|_| rng.gen_range(0..num_states) as i32)
            .collect();

        compare_discrete_entropy(data, &format!("Uniform Random ({} states)", num_states));
    }
}

/// Test with a short input (only a few elements)
#[test]
fn test_discrete_entropy_short_input() {
    // Short input with only 5 elements
    let data = vec![1, 2, 3, 2, 1];
    compare_discrete_entropy(data, "Short Input");
}

/// Test with an input containing only one unique value
#[test]
fn test_discrete_entropy_single_value() {
    // Input with only one unique value (should have zero entropy)
    let data = vec![5, 5, 5, 5, 5, 5, 5, 5, 5, 5];
    compare_discrete_entropy(data, "Single Value");
}

/// Test with data from Gaussian distributions with different parameters
#[test]
fn test_discrete_entropy_gaussian() {
    let seed = 123;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 100;

    // Define different Gaussian parameters to test: (mean, std_dev)
    let gaussian_params = [
        (0.0, 1.0),   // Standard normal distribution
        (10.0, 2.0),  // Mean 10, std 2
    ];

    for &(mean, std_dev) in &gaussian_params {
        // Create a normal distribution with the specified parameters
        let normal = Normal::new(mean, std_dev).unwrap();

        // Generate samples from the normal distribution and convert to integers
        let data: Vec<i32> = (0..size)
            .map(|_| (normal.sample(&mut rng) as f64).round() as i32)
            .collect();

        compare_discrete_entropy(data, &format!("Gaussian (mean={}, std={})", mean, std_dev));
    }
}

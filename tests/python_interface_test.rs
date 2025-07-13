use validation::python;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use infomeasure::estimators::entropy::{Entropy, LocalValues};
use ndarray::Array1;
use approx::assert_relative_eq;

/// Test to verify that the Python interface works correctly by comparing
/// the Python implementation with the Rust implementation.
#[test]
fn test_python_interface_works() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 10;

    // Generate random integers between 0 and 10
    let data: Vec<i32> = (0..size)
        .map(|_| rng.gen_range(0..10) as i32)
        .collect();

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

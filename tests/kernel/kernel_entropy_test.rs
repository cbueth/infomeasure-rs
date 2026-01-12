use approx::assert_relative_eq;
use infomeasure::estimators::entropy::{Entropy, LocalValues};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rstest::rstest;
use validation::python;

// Import test helper functions
use crate::test_helpers::generate_gaussian_data;

/// Helper function to compare kernel entropy between Rust and Python implementations
///
/// This function:
/// 1. Takes a vector of floats as input data
/// 2. Calculates entropy using both Rust and Python implementations with kernel approach
/// 3. Compares the results and asserts they are approximately equal
/// 4. Returns the calculated entropy values for potential further analysis
fn compare_kernel_entropy(
    data: Vec<f64>,
    bandwidth: f64,
    kernel_type: &str,
    test_name: &str,
) -> (f64, Vec<f64>) {
    let data_array = Array1::from(data.clone());

    // Calculate entropy using the Rust implementation
    let kernel_entropy =
        Entropy::new_kernel_with_type(data_array, kernel_type.to_string(), bandwidth);
    let rust_global_entropy = kernel_entropy.global_value();
    let rust_local_entropy = kernel_entropy.local_values();

    // Calculate entropy using the Python implementation with kernel approach
    let kernel_kwargs = [
        ("kernel".to_string(), format!("\"{}\"", kernel_type)),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let python_global_entropy =
        python::calculate_entropy_float(&data, "kernel", &kernel_kwargs).unwrap();
    let python_local_entropy =
        python::calculate_local_entropy_float(&data, "kernel", &kernel_kwargs).unwrap();

    // Compare the results
    println!(
        "{} - Rust global entropy (base e): {}",
        test_name, rust_global_entropy
    );
    println!(
        "{} - Python global entropy (base e): {}",
        test_name, python_global_entropy
    );

    // Assert that the global entropy values are approximately equal
    // Note: Kernel entropy calculations should now match Python closely with full covariance
    // We use a looser tolerance for Gaussian when GPU support might be active (float32 precision)
    let (epsilon, max_relative) = match kernel_type {
        "box" => (1e-12, 1e-12),
        "gaussian" => {
            #[cfg(feature = "gpu_support")]
            {
                (1e-6, 1e-6)  // lower precision due to float32 precision
            }
            #[cfg(not(feature = "gpu_support"))]
            {
                (1e-12, 1e-12)
            }
        }
        _ => (1e-12, 1e-12),
    };

    assert_relative_eq!(
        rust_global_entropy,
        python_global_entropy,
        epsilon = epsilon,
        max_relative = max_relative
    );

    // Assert that the local entropy values are approximately equal
    for (rust_val, python_val) in rust_local_entropy.iter().zip(python_local_entropy.iter()) {
        assert_relative_eq!(
            *rust_val,
            *python_val,
            epsilon = epsilon,
            max_relative = max_relative
        );
    }

    (rust_global_entropy, rust_local_entropy.to_vec())
}

/// Test that compares the kernel entropy implementation in Rust with the Python infomeasure package
/// using 2D data with different dimensions (1-8) and different kernel types.
#[test]
fn test_kernel_entropy_2d_different_dimensions() {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 100;

    // Test with a fixed bandwidth and different kernel types
    let bandwidth = 1.0;
    let kernel_types = ["box", "gaussian"];

    // Test dimensions 1-8 with different kernel types
    macro_rules! test_dimension {
        ($dim:expr, $kernel_type:expr) => {{
            let mut data = Array2::zeros((size, $dim));
            for i in 0..size {
                for j in 0..$dim {
                    data[[i, j]] = rng.gen_range(0.0..20.0);
                }
            }

            let test_name = format!("Kernel Entropy 2D (dim={}, kernel={})", $dim, $kernel_type);
            match $dim {
                1 => compare_kernel_entropy_2d_generic::<1>(
                    data,
                    bandwidth,
                    $kernel_type,
                    &test_name,
                ),
                2 => compare_kernel_entropy_2d_generic::<2>(
                    data,
                    bandwidth,
                    $kernel_type,
                    &test_name,
                ),
                3 => compare_kernel_entropy_2d_generic::<3>(
                    data,
                    bandwidth,
                    $kernel_type,
                    &test_name,
                ),
                4 => compare_kernel_entropy_2d_generic::<4>(
                    data,
                    bandwidth,
                    $kernel_type,
                    &test_name,
                ),
                5 => compare_kernel_entropy_2d_generic::<5>(
                    data,
                    bandwidth,
                    $kernel_type,
                    &test_name,
                ),
                6 => compare_kernel_entropy_2d_generic::<6>(
                    data,
                    bandwidth,
                    $kernel_type,
                    &test_name,
                ),
                7 => compare_kernel_entropy_2d_generic::<7>(
                    data,
                    bandwidth,
                    $kernel_type,
                    &test_name,
                ),
                8 => compare_kernel_entropy_2d_generic::<8>(
                    data,
                    bandwidth,
                    $kernel_type,
                    &test_name,
                ),
                _ => panic!("Unsupported dimension: {}", $dim),
            };
        }};
    }

    for &kernel_type in &kernel_types {
        for dim in 1..=8 {
            test_dimension!(dim, kernel_type);
        }
    }
}

/// Generic helper function to compare kernel entropy between Rust and Python implementations
fn compare_kernel_entropy_2d_generic<const K: usize>(
    data: Array2<f64>,
    bandwidth: f64,
    kernel_type: &str,
    test_name: &str,
) -> (f64, Vec<f64>) {
    // Calculate entropy using the Rust implementation
    let kernel_entropy =
        Entropy::nd_kernel_with_type::<K>(data.clone(), kernel_type.to_string(), bandwidth);
    let rust_global_entropy = kernel_entropy.global_value();
    let rust_local_entropy = kernel_entropy.local_values();

    // Convert the 2D array to a flat array for Python
    let flat_data: Vec<f64> = data.iter().cloned().collect();
    let dims = data.ncols();

    // Calculate entropy using the Python implementation with n-dimensional functions
    let kernel_kwargs = [
        ("kernel".to_string(), format!("\"{}\"", kernel_type)),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let python_global_entropy =
        python::calculate_entropy_float_nd(&flat_data, dims, "kernel", &kernel_kwargs).unwrap();
    let python_local_entropy =
        python::calculate_local_entropy_float_nd(&flat_data, dims, "kernel", &kernel_kwargs)
            .unwrap();

    // Compare results
    println!(
        "{} - Rust global entropy (base e): {}",
        test_name, rust_global_entropy
    );
    println!(
        "{} - Python global entropy (base e): {}",
        test_name, python_global_entropy
    );

    let (epsilon, max_relative) = match kernel_type {
        "box" => (1e-12, 1e-12),
        "gaussian" => {
            #[cfg(feature = "gpu_support")]
            {
                (1e-6, 1e-6)  // lower precision due to float32 precision
            }
            #[cfg(not(feature = "gpu_support"))]
            {
                (1e-12, 1e-12)
            }
        }
        _ => (1e-12, 1e-12),
    };

    assert_relative_eq!(
        rust_global_entropy,
        python_global_entropy,
        epsilon = epsilon,
        max_relative = max_relative
    );

    let sample_size = rust_local_entropy.len().min(10);
    let step = rust_local_entropy.len() / sample_size.max(1);

    for i in (0..rust_local_entropy.len()).step_by(step.max(1)) {
        if i < rust_local_entropy.len() && i < python_local_entropy.len() {
            let rust_val = rust_local_entropy[i];
            let python_val = python_local_entropy[i];

            assert_relative_eq!(
                rust_val,
                python_val,
                epsilon = epsilon,
                max_relative = max_relative
            );
        }
    }

    (rust_global_entropy, rust_local_entropy.to_vec())
}

/// Test that compares the kernel entropy implementation in Rust with the Python infomeasure package
/// using random uniform data with different bandwidths and kernel types.
#[rstest]
fn test_kernel_entropy_different_bandwidths(
    #[values("box", "gaussian")] kernel_type: &str,
    #[values(0.5, 1.0, 2.0)] bandwidth: f64,
) {
    // Set up random test data with a fixed seed
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 100;

    // Generate random floats between 0.0 and 20.0
    let data: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..20.0)).collect();

    let test_name = format!(
        "Kernel Entropy (kernel={}, bandwidth={})",
        kernel_type, bandwidth
    );
    compare_kernel_entropy(data, bandwidth, kernel_type, &test_name);
}

/// Test with data from Gaussian distributions
#[rstest]
fn test_kernel_entropy_gaussian(#[values("box", "gaussian")] kernel_type: &str) {
    let seed = 123;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 100;

    // Create a normal distribution
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate samples from the normal distribution directly as floats
    let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();

    // Test with a fixed bandwidth
    let bandwidth = 1.0;

    let test_name = format!("Kernel Entropy (Gaussian data, kernel={})", kernel_type);
    compare_kernel_entropy(data, bandwidth, kernel_type, &test_name);
}

#[rstest]
fn test_kernel_entropy_gaussian_parameter_combinations(
    #[values(1, 2, 3, 4)] dim: usize,
    #[values(0.4, 1.0, 2.1)] bandwidth: f64,
    #[values(1.0, 1.5)] mean: f64,
    #[values(100, 1000)] size: usize,
) {
    let seed = 42;
    let std_dev = 1.0; // Fixed standard deviation
    let kernel_type = "gaussian";

    // Generate Gaussian data
    let data = generate_gaussian_data(size, dim, mean, std_dev, seed);

    // Create test name
    let test_name = format!(
        "Kernel Entropy (dim={}, bandwidth={}, mean={}, size={})",
        dim, bandwidth, mean, size
    );

    // Compare implementations based on dimension
    match dim {
        1 => compare_kernel_entropy_2d_generic::<1>(data, bandwidth, kernel_type, &test_name),
        2 => compare_kernel_entropy_2d_generic::<2>(data, bandwidth, kernel_type, &test_name),
        3 => compare_kernel_entropy_2d_generic::<3>(data, bandwidth, kernel_type, &test_name),
        4 => compare_kernel_entropy_2d_generic::<4>(data, bandwidth, kernel_type, &test_name),
        _ => panic!("Unsupported dimension: {}", dim),
    };
}

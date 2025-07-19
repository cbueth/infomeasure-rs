use infomeasure::estimators::entropy::{Entropy, LocalValues};
use validation::python;
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use approx::assert_relative_eq;

/// Helper function to generate Gaussian data with specified parameters
fn generate_gaussian_data(size: usize, dims: usize, mean: f64, std_dev: f64, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(mean, std_dev).unwrap();

    // Generate data for each dimension
    let data: Vec<f64> = (0..size * dims)
        .map(|_| normal.sample(&mut rng))
        .collect();

    // Reshape into a 2D array with shape (size, dims)
    Array2::from_shape_vec((size, dims), data)
        .expect("Failed to reshape data")
}

/// Debug function to compare kernel entropy between Rust and Python implementations
fn debug_kernel_entropy_2d_generic<const K: usize>(
    data: Array2<f64>, 
    bandwidth: f64, 
    kernel_type: &str, 
    test_name: &str
) -> (f64, Vec<f64>) {
    // Calculate entropy using the Rust implementation
    let kernel_entropy = Entropy::nd_kernel_with_type::<K>(data.clone(), kernel_type.to_string(), bandwidth);
    let rust_global_entropy = kernel_entropy.global_value();
    let rust_local_entropy = kernel_entropy.local_values();

    // Convert the 2D array to a flat array for Python
    let flat_data: Vec<f64> = data.iter().cloned().collect();
    let dims = data.ncols();

    // Calculate entropy using the Python implementation with n-dimensional functions
    let kernel_kwargs = [
        ("kernel".to_string(), format!("\"{}\"", kernel_type)),
        ("bandwidth".to_string(), bandwidth.to_string())
    ];
    let python_global_entropy = python::calculate_entropy_float_nd(&flat_data, dims, "kernel", &kernel_kwargs).unwrap();
    let python_local_entropy = python::calculate_local_entropy_float_nd(&flat_data, dims, "kernel", &kernel_kwargs).unwrap();

    // Compare results
    println!("{} - Rust global entropy (base e): {}", test_name, rust_global_entropy);
    println!("{} - Python global entropy (base e): {}", test_name, python_global_entropy);
    
    // Calculate global relative error
    let global_rel_error = (rust_global_entropy - python_global_entropy).abs() / python_global_entropy.abs();
    println!("{} - Global relative error: {:.6}", test_name, global_rel_error);

    // Check local entropy values
    let local_max_relative = 0.3;
    let sample_size = rust_local_entropy.len().min(10);
    let step = rust_local_entropy.len() / sample_size.max(1);
    
    let mut max_local_error = 0.0;
    let mut max_error_index = 0;
    let mut max_rust_val = 0.0;
    let mut max_python_val = 0.0;
    
    for i in (0..rust_local_entropy.len()).step_by(step.max(1)) {
        if i < rust_local_entropy.len() && i < python_local_entropy.len() {
            let rust_val = rust_local_entropy[i];
            let python_val = python_local_entropy[i];
            
            if rust_val.abs() > 1e-6 && python_val.abs() > 1e-6 {
                let rel_error = (rust_val - python_val).abs() / python_val.abs();
                
                if rel_error > max_local_error {
                    max_local_error = rel_error;
                    max_error_index = i;
                    max_rust_val = rust_val;
                    max_python_val = python_val;
                }
                
                if rel_error > local_max_relative {
                    println!("{} - Large discrepancy at index {}: Rust={}, Python={}, RelError={:.6}", 
                        test_name, i, rust_val, python_val, rel_error);
                }
            }
        }
    }
    
    println!("{} - Max local error: {:.6} at index {} (Rust={}, Python={})", 
        test_name, max_local_error, max_error_index, max_rust_val, max_python_val);

    (rust_global_entropy, rust_local_entropy.to_vec())
}

#[test]
fn debug_kernel_entropy_gaussian_parameter_combinations() {
    // Define parameter ranges
    let dimensions = [1, 2, 3, 4];
    let bandwidths = [0.4, 1.0, 2.1];
    let means = [1.0, 1.5];
    let sizes = [100, 1000];
    let seed = 42;
    let std_dev = 1.0; // Fixed standard deviation
    let kernel_type = "gaussian";

    println!("Testing {} parameter combinations", dimensions.len() * bandwidths.len() * means.len() * sizes.len());

    // Test each combination
    for &dim in &dimensions {
        for &bandwidth in &bandwidths {
            for &mean in &means {
                for &size in &sizes {
                    // Generate Gaussian data
                    let data = generate_gaussian_data(size, dim, mean, std_dev, seed);

                    // Create test name
                    let test_name = format!(
                        "Kernel Entropy (dim={}, bandwidth={}, mean={}, size={})",
                        dim, bandwidth, mean, size
                    );

                    // Compare implementations based on dimension
                    match dim {
                        1 => debug_kernel_entropy_2d_generic::<1>(data, bandwidth, kernel_type, &test_name),
                        2 => debug_kernel_entropy_2d_generic::<2>(data, bandwidth, kernel_type, &test_name),
                        3 => debug_kernel_entropy_2d_generic::<3>(data, bandwidth, kernel_type, &test_name),
                        4 => debug_kernel_entropy_2d_generic::<4>(data, bandwidth, kernel_type, &test_name),
                        _ => panic!("Unsupported dimension: {}", dim),
                    };
                }
            }
        }
    }
}
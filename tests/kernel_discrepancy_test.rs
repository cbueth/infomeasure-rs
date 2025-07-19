use infomeasure::estimators::entropy::{Entropy, LocalValues};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use std::fs::File;
use std::io::Write;

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

/// Compare GPU and CPU implementations and collect statistics on discrepancies
fn compare_implementations<const K: usize>(
    data: Array2<f64>,
    bandwidth: f64,
    test_name: &str,
) -> (f64, f64, Vec<(f64, f64, f64)>) {
    // Calculate entropy using the CPU implementation
    #[cfg(feature = "gpu_support")]
    let cpu_entropy = {
        // Temporarily disable GPU support
        let kernel_entropy = Entropy::nd_kernel_with_type::<K>(data.clone(), "gaussian".to_string(), bandwidth);
        // Force CPU implementation by directly calling the method
        let cpu_local_entropy = kernel_entropy.gaussian_kernel_local_values();
        let cpu_global_entropy = cpu_local_entropy.mean().unwrap();
        (cpu_global_entropy, cpu_local_entropy)
    };
    
    #[cfg(not(feature = "gpu_support"))]
    let cpu_entropy = {
        let kernel_entropy = Entropy::nd_kernel_with_type::<K>(data.clone(), "gaussian".to_string(), bandwidth);
        let cpu_local_entropy = kernel_entropy.local_values();
        let cpu_global_entropy = kernel_entropy.global_value();
        (cpu_global_entropy, cpu_local_entropy)
    };

    // Calculate entropy using the GPU implementation
    #[cfg(feature = "gpu_support")]
    let gpu_entropy = {
        let kernel_entropy = Entropy::nd_kernel_with_type::<K>(data.clone(), "gaussian".to_string(), bandwidth);
        // Force GPU implementation by directly calling the method
        let gpu_local_entropy = kernel_entropy.gaussian_kernel_local_values_gpu();
        let gpu_global_entropy = gpu_local_entropy.mean().unwrap();
        (gpu_global_entropy, gpu_local_entropy)
    };
    
    #[cfg(not(feature = "gpu_support"))]
    let gpu_entropy = cpu_entropy.clone(); // No GPU support, use CPU results

    // Compare results
    println!("{} - CPU global entropy: {}", test_name, cpu_entropy.0);
    println!("{} - GPU global entropy: {}", test_name, gpu_entropy.0);
    
    // Calculate global relative error
    let global_rel_error = if cpu_entropy.0.abs() > 1e-6 {
        (gpu_entropy.0 - cpu_entropy.0).abs() / cpu_entropy.0.abs()
    } else {
        0.0
    };
    
    println!("{} - Global relative error: {:.6}", test_name, global_rel_error);
    
    // Calculate local relative errors
    let mut local_errors = Vec::new();
    let sample_size = cpu_entropy.1.len().min(20);
    let step = cpu_entropy.1.len() / sample_size.max(1);
    
    for i in (0..cpu_entropy.1.len()).step_by(step.max(1)) {
        if i < cpu_entropy.1.len() && i < gpu_entropy.1.len() {
            let cpu_val = cpu_entropy.1[i];
            let gpu_val = gpu_entropy.1[i];
            
            let rel_error = if cpu_val.abs() > 1e-6 {
                (gpu_val - cpu_val).abs() / cpu_val.abs()
            } else {
                0.0
            };
            
            local_errors.push((cpu_val, gpu_val, rel_error));
            
            if rel_error > 0.3 { // 30% threshold used in the original test
                println!("{} - Large discrepancy at index {}: CPU={}, GPU={}, RelError={:.6}", 
                    test_name, i, cpu_val, gpu_val, rel_error);
            }
        }
    }
    
    (cpu_entropy.0, gpu_entropy.0, local_errors)
}

#[test]
fn test_kernel_entropy_discrepancies() {
    // Define parameter ranges
    let dimensions = [1, 2, 3, 4];
    let bandwidths = [0.4, 1.0, 2.1];
    let means = [1.0, 1.5];
    let sizes = [100, 1000];
    let seed = 42;
    let std_dev = 1.0; // Fixed standard deviation
    
    // Create a CSV file to store the results
    let mut file = File::create("kernel_discrepancies.csv").unwrap();
    writeln!(file, "Dimensions,Bandwidth,Mean,Size,CPU Global,GPU Global,Global Rel Error,Max Local Rel Error,Avg Local Rel Error,Median Local Rel Error").unwrap();
    
    println!("Testing parameter combinations for discrepancies...");
    
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
                    let (cpu_global, gpu_global, local_errors) = match dim {
                        1 => compare_implementations::<1>(data, bandwidth, &test_name),
                        2 => compare_implementations::<2>(data, bandwidth, &test_name),
                        3 => compare_implementations::<3>(data, bandwidth, &test_name),
                        4 => compare_implementations::<4>(data, bandwidth, &test_name),
                        _ => panic!("Unsupported dimension: {}", dim),
                    };
                    
                    // Calculate statistics
                    let global_rel_error = if cpu_global.abs() > 1e-6 {
                        (gpu_global - cpu_global).abs() / cpu_global.abs()
                    } else {
                        0.0
                    };
                    
                    let local_rel_errors: Vec<f64> = local_errors.iter()
                        .map(|&(_, _, rel_error)| rel_error)
                        .collect();
                    
                    let max_local_rel_error = local_rel_errors.iter()
                        .fold(0.0, |max_val: f64, &val| max_val.max(val));
                    
                    let avg_local_rel_error = if !local_rel_errors.is_empty() {
                        local_rel_errors.iter().sum::<f64>() / local_rel_errors.len() as f64
                    } else {
                        0.0
                    };
                    
                    // Calculate median (sort and take middle value)
                    let mut sorted_errors = local_rel_errors.clone();
                    sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median_local_rel_error = if !sorted_errors.is_empty() {
                        sorted_errors[sorted_errors.len() / 2]
                    } else {
                        0.0
                    };
                    
                    // Write to CSV
                    writeln!(file, "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
                        dim, bandwidth, mean, size, cpu_global, gpu_global, 
                        global_rel_error, max_local_rel_error, avg_local_rel_error, median_local_rel_error
                    ).unwrap();
                    
                    // Print summary
                    println!("{} - Summary: Global RelErr={:.6}, Max Local RelErr={:.6}, Avg Local RelErr={:.6}, Median Local RelErr={:.6}",
                        test_name, global_rel_error, max_local_rel_error, avg_local_rel_error, median_local_rel_error);
                }
            }
        }
    }
    
    println!("Discrepancy analysis complete. Results saved to kernel_discrepancies.csv");
}
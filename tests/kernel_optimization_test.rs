use infomeasure::estimators::entropy::{Entropy, LocalValues};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;

/// Generate random multi-dimensional data with specified size and dimensions from a normal distribution
fn generate_random_nd_data(size: usize, dims: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let data: Vec<f64> = (0..size * dims)
        .map(|_| normal.sample(&mut rng))
        .collect();

    Array2::from_shape_vec((size, dims), data).unwrap()
}

/// Measure the performance of the Gaussian kernel entropy calculation
fn measure_gaussian_kernel_performance(size: usize, dims: usize, bandwidth: f64, seed: u64) -> Duration {
    // Generate random data
    let data = generate_random_nd_data(size, dims, seed);

    // Measure performance
    let start = Instant::now();

    match dims {
        1 => {
            Entropy::nd_kernel_with_type::<1>(
                data.clone(),
                "gaussian".to_string(),
                bandwidth
            ).global_value()
        },
        2 => {
            Entropy::nd_kernel_with_type::<2>(
                data.clone(),
                "gaussian".to_string(),
                bandwidth
            ).global_value()
        },
        3 => {
            Entropy::nd_kernel_with_type::<3>(
                data.clone(),
                "gaussian".to_string(),
                bandwidth
            ).global_value()
        },
        4 => {
            Entropy::nd_kernel_with_type::<4>(
                data.clone(),
                "gaussian".to_string(),
                bandwidth
            ).global_value()
        },
        8 => {
            Entropy::nd_kernel_with_type::<8>(
                data.clone(),
                "gaussian".to_string(),
                bandwidth
            ).global_value()
        },
        16 => {
            Entropy::nd_kernel_with_type::<16>(
                data.clone(),
                "gaussian".to_string(),
                bandwidth
            ).global_value()
        },
        32 => {
            Entropy::nd_kernel_with_type::<32>(
                data.clone(),
                "gaussian".to_string(),
                bandwidth
            ).global_value()
        },
        _ => panic!("Unsupported number of dimensions: {}", dims)
    };

    let duration = start.elapsed();

    duration
}

/// Measure the performance of the Box kernel entropy calculation
fn measure_box_kernel_performance(size: usize, dims: usize, bandwidth: f64, seed: u64) -> Duration {
    // Generate random data
    let data = generate_random_nd_data(size, dims, seed);

    // Measure performance
    let start = Instant::now();

    match dims {
        1 => {
            Entropy::nd_kernel_with_type::<1>(
                data.clone(),
                "box".to_string(),
                bandwidth
            ).global_value()
        },
        2 => {
            Entropy::nd_kernel_with_type::<2>(
                data.clone(),
                "box".to_string(),
                bandwidth
            ).global_value()
        },
        3 => {
            Entropy::nd_kernel_with_type::<3>(
                data.clone(),
                "box".to_string(),
                bandwidth
            ).global_value()
        },
        4 => {
            Entropy::nd_kernel_with_type::<4>(
                data.clone(),
                "box".to_string(),
                bandwidth
            ).global_value()
        },
        8 => {
            Entropy::nd_kernel_with_type::<8>(
                data.clone(),
                "box".to_string(),
                bandwidth
            ).global_value()
        },
        16 => {
            Entropy::nd_kernel_with_type::<16>(
                data.clone(),
                "box".to_string(),
                bandwidth
            ).global_value()
        },
        32 => {
            Entropy::nd_kernel_with_type::<32>(
                data.clone(),
                "box".to_string(),
                bandwidth
            ).global_value()
        },
        _ => panic!("Unsupported number of dimensions: {}", dims)
    };

    let duration = start.elapsed();

    duration
}

#[test]
fn test_gaussian_kernel_performance() {
    // Define test parameters
    let sizes = [100, 500, 1000, 5000, 10000];
    let dimensions = [1, 2, 4, 8, 16, 32];
    let bandwidth = 0.5;
    let seed = 42;
    let num_runs = 3; // Reduced number of runs to speed up testing

    // Create a file to store the results
    #[cfg(all(feature = "gpu_support", feature = "fast_exp"))]
    let mut file = File::create("gaussian_gpu_fast_exp_performance.md").unwrap();
    #[cfg(all(feature = "gpu_support", not(feature = "fast_exp")))]
    let mut file = File::create("gaussian_gpu_performance.md").unwrap();
    #[cfg(all(not(feature = "gpu_support"), feature = "fast_exp"))]
    let mut file = File::create("gaussian_fast_exp_performance.md").unwrap();
    #[cfg(all(not(feature = "gpu_support"), not(feature = "fast_exp")))]
    let mut file = File::create("gaussian_baseline_performance.md").unwrap();

    #[cfg(all(feature = "gpu_support", feature = "fast_exp"))]
    writeln!(file, "# Gaussian Kernel GPU with Fast Exp Performance\n").unwrap();
    #[cfg(all(feature = "gpu_support", not(feature = "fast_exp")))]
    writeln!(file, "# Gaussian Kernel GPU Performance\n").unwrap();
    #[cfg(all(not(feature = "gpu_support"), feature = "fast_exp"))]
    writeln!(file, "# Gaussian Kernel Fast Exp Performance\n").unwrap();
    #[cfg(all(not(feature = "gpu_support"), not(feature = "fast_exp")))]
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

            println!("Gaussian - Size: {}, Dims: {}, Time: {} ms", size, dims, avg_ms);
            writeln!(file, "| {} | {} | {} |", size, dims, avg_ms).unwrap();
        }
    }

    #[cfg(all(feature = "gpu_support", feature = "fast_exp"))]
    println!("Gaussian kernel GPU with Fast Exp performance results have been saved to gaussian_gpu_fast_exp_performance.md");
    #[cfg(all(feature = "gpu_support", not(feature = "fast_exp")))]
    println!("Gaussian kernel GPU performance results have been saved to gaussian_gpu_performance.md");
    #[cfg(all(not(feature = "gpu_support"), feature = "fast_exp"))]
    println!("Gaussian kernel Fast Exp performance results have been saved to gaussian_fast_exp_performance.md");
    #[cfg(all(not(feature = "gpu_support"), not(feature = "fast_exp")))]
    println!("Gaussian kernel baseline performance results have been saved to gaussian_baseline_performance.md");
}

#[test]
fn test_box_kernel_performance() {
    // Define test parameters
    let sizes = [100, 500, 1000, 5000, 10000];
    let dimensions = [1, 2, 4, 8, 16, 32];
    let bandwidth = 0.5;
    let seed = 42;
    let num_runs = 3; // Reduced number of runs to speed up testing

    // Create a file to store the results
    #[cfg(all(feature = "gpu_support", feature = "fast_exp"))]
    let mut file = File::create("box_gpu_fast_exp_performance.md").unwrap();
    #[cfg(all(feature = "gpu_support", not(feature = "fast_exp")))]
    let mut file = File::create("box_gpu_performance.md").unwrap();
    #[cfg(all(not(feature = "gpu_support"), feature = "fast_exp"))]
    let mut file = File::create("box_fast_exp_performance.md").unwrap();
    #[cfg(all(not(feature = "gpu_support"), not(feature = "fast_exp")))]
    let mut file = File::create("box_baseline_performance.md").unwrap();

    #[cfg(all(feature = "gpu_support", feature = "fast_exp"))]
    writeln!(file, "# Box Kernel GPU with Fast Exp Performance\n").unwrap();
    #[cfg(all(feature = "gpu_support", not(feature = "fast_exp")))]
    writeln!(file, "# Box Kernel GPU Performance\n").unwrap();
    #[cfg(all(not(feature = "gpu_support"), feature = "fast_exp"))]
    writeln!(file, "# Box Kernel Fast Exp Performance\n").unwrap();
    #[cfg(all(not(feature = "gpu_support"), not(feature = "fast_exp")))]
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

            println!("Box - Size: {}, Dims: {}, Time: {} ms", size, dims, avg_ms);
            writeln!(file, "| {} | {} | {} |", size, dims, avg_ms).unwrap();
        }
    }

    #[cfg(all(feature = "gpu_support", feature = "fast_exp"))]
    println!("Box kernel GPU with Fast Exp performance results have been saved to box_gpu_fast_exp_performance.md");
    #[cfg(all(feature = "gpu_support", not(feature = "fast_exp")))]
    println!("Box kernel GPU performance results have been saved to box_gpu_performance.md");
    #[cfg(all(not(feature = "gpu_support"), feature = "fast_exp"))]
    println!("Box kernel Fast Exp performance results have been saved to box_fast_exp_performance.md");
    #[cfg(all(not(feature = "gpu_support"), not(feature = "fast_exp")))]
    println!("Box kernel baseline performance results have been saved to box_baseline_performance.md");
}

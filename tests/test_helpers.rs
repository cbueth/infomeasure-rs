use std::time::{Duration, Instant};

// Import and re-export commonly used items
pub use approx::assert_relative_eq;
pub use ndarray::Array2;
pub use rand::rngs::StdRng;
pub use rand::{Rng, SeedableRng};
pub use rand_distr::{Distribution, Normal};

/// Generate random n-dimensional data (used in multiple files)
pub fn generate_random_nd_data(size: usize, dims: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Array2::zeros((size, dims));
    for i in 0..size {
        for j in 0..dims {
            data[[i, j]] = rng.gen_range(0.0..20.0);
        }
    }
    data
}

/// Generate Gaussian distributed data (duplicated in 4+ files)
pub fn generate_gaussian_data(
    size: usize,
    dims: usize,
    mean: f64,
    std_dev: f64,
    seed: u64,
) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(mean, std_dev).unwrap();
    let data: Vec<f64> = (0..size * dims).map(|_| normal.sample(&mut rng)).collect();
    Array2::from_shape_vec((size, dims), data).expect("Failed to reshape data")
}

/// Common tolerance values for different kernel types
pub fn get_tolerance_for_kernel(kernel_type: &str) -> (f64, f64) {
    match kernel_type {
        "box" => (1e-6, 1e-3),
        "gaussian" => (1e-2, 1e-1),
        _ => (1e-6, 1e-3),
    }
}

/// Common assertion helper for entropy comparisons
pub fn assert_entropy_values_close(
    rust_val: f64,
    python_val: f64,
    epsilon: f64,
    max_relative: f64,
    test_name: &str,
) {
    // Print comparison info before assertion
    println!(
        "Comparing in {}: Rust={}, Python={}",
        test_name, rust_val, python_val
    );

    assert_relative_eq!(
        rust_val,
        python_val,
        epsilon = epsilon,
        max_relative = max_relative
    );
}

/// Common performance measurement utility
pub fn measure_execution_time<F>(f: F) -> Duration
where
    F: FnOnce(),
{
    let start = Instant::now();
    f();
    start.elapsed()
}

/// Standard test parameters used across multiple test files
pub struct TestConfig {
    pub sizes: &'static [usize],
    pub dimensions: &'static [usize],
    pub bandwidths: &'static [f64],
    pub kernel_types: &'static [&'static str],
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            sizes: &[100, 1000],
            dimensions: &[1, 2, 3, 4],
            bandwidths: &[0.4, 1.0, 2.1],
            kernel_types: &["box", "gaussian"],
        }
    }
}

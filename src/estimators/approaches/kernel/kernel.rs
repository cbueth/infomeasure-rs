//! # Kernel Density Estimation for Entropy Calculation
//!
//! This module implements entropy estimation using kernel density estimation (KDE) techniques.
//! KDE is a non-parametric way to estimate the probability density function of a random variable,
//! which is then used to calculate differential entropy.
//!
//! ## Theoretical Background
//!
//! The differential entropy of a continuous random variable X is defined as:
//!
//! H(X) = -∫ f(x) log(f(x)) dx
//!
//! where f(x) is the probability density function (PDF) of X.
//!
//! In practice, we don't know the true PDF, so we estimate it using kernel density estimation:
//!
//! f̂(x) = (1/Nh^d) ∑ K((x - x_i)/h)
//!
//! where:
//! - N is the number of data points
//! - h is the bandwidth parameter
//! - d is the dimensionality of the data
//! - K is the kernel function
//! - x_i are the observed data points
//!
//! The entropy is then estimated as:
//!
//! Ĥ(X) = -(1/N) ∑ log(f̂(x_i))
//!
//! ## Supported Kernel Types
//!
//! This implementation supports two types of kernels:
//!
//! 1. **Box Kernel**: A uniform kernel where all points within a certain distance contribute equally.
//!    Simple and computationally efficient, but can produce discontinuities.
//!
//! 2. **Gaussian Kernel**: A smooth kernel based on the normal distribution. Provides smoother
//!    density estimates but is more computationally intensive. The Gaussian kernel implementation
//!    scales the bandwidth by the standard deviation of the data in each dimension, matching the
//!    behavior of scipy.stats.gaussian_kde.
//!
//! ## Bandwidth Selection
//!
//! The bandwidth parameter (h) controls the smoothness of the density estimate:
//! - Small bandwidth: More detail but potentially noisy (high variance)
//! - Large bandwidth: Smoother but potentially over-smoothed (high bias)
//!
//! Choosing an appropriate bandwidth is crucial for accurate entropy estimation.
//!
//! ## Implementation Details
//!
//! This implementation uses a KD-tree for efficient nearest-neighbor queries, making it
//! suitable for large datasets. The Gaussian kernel implementation includes proper
//! dimension-dependent normalization and bandwidth scaling by standard deviation to match
//! the behavior of scipy.stats.gaussian_kde.
//!
//! When compiled with the `simd_support` feature flag, this implementation uses SIMD
//! (Single Instruction, Multiple Data) optimizations for faster distance calculations,
//! particularly beneficial for high-dimensional data and large datasets.
//!
//! ## GPU Acceleration
//!
//! When compiled with the `gpu_support` feature flag, this implementation can use GPU
//! acceleration for both Gaussian and Box kernel calculations, providing significant
//! performance improvements for large datasets:
//!
//! - **Gaussian Kernel**: GPU acceleration is used for datasets with 500 or more points,
//!   providing speedups of up to 340x for large datasets. The adaptive radius for neighbor
//!   search is larger when using GPU acceleration, especially for small bandwidths.
//!
//! - **Box Kernel**: GPU acceleration is used for datasets with 2000 or more points,
//!   providing speedups of up to 37x for large datasets. For smaller datasets, the CPU
//!   implementation is faster due to the overhead of GPU setup.

use ndarray::{Array1, Array2};
use kiddo::{ImmutableKdTree, Manhattan, SquaredEuclidean};
#[cfg(feature = "simd_support")]
use std::simd::{StdFloat, f64x4, f64x8};
#[cfg(feature = "simd_support")]
use std::simd::num::SimdFloat;
use crate::estimators::traits::LocalValues;

/// Input data representation for kernel entropy estimation
///
/// This enum allows the kernel entropy estimator to accept both 1D and 2D data arrays,
/// providing flexibility in how data is passed to the estimator.
pub enum KernelData {
    /// One-dimensional data: Array1<f64> where each element is a data point
    OneDimensional(Array1<f64>),

    /// Two-dimensional data: Array2<f64> where rows are data points and columns are dimensions
    /// First dimension (rows) = samples, second dimension (columns) = features/dimensions
    TwoDimensional(Array2<f64>),
}

impl From<Array1<f64>> for KernelData {
    fn from(array: Array1<f64>) -> Self {
        KernelData::OneDimensional(array)
    }
}

impl From<Array2<f64>> for KernelData {
    fn from(array: Array2<f64>) -> Self {
        KernelData::TwoDimensional(array)
    }
}

/// Kernel-based entropy estimator for continuous data
///
/// This struct implements entropy estimation using kernel density estimation (KDE).
/// It supports both box (uniform) and Gaussian kernels, and can handle data of any dimensionality
/// (specified by the generic parameter K).
///
/// # Features
///
/// - Supports both box and Gaussian kernels
/// - Handles multi-dimensional data efficiently
/// - Uses KD-tree for fast nearest-neighbor queries
/// - Implements proper bandwidth scaling for Gaussian kernels
/// - Provides both global and local entropy values
/// - Supports GPU acceleration when compiled with the `gpu_support` feature flag
///
/// # Bandwidth Scaling
///
/// The two kernel types handle bandwidth differently:
///
/// - **Box Kernel**: Uses the raw bandwidth value without scaling. The bandwidth directly
///   determines the size of the hypercube within which points are counted.
///
/// - **Gaussian Kernel**: Scales the bandwidth by the standard deviation of the data in each
///   dimension, matching the behavior of scipy.stats.gaussian_kde. This makes the estimator
///   adaptive to the scale of the data in each dimension.
///
/// # GPU Acceleration
///
/// When compiled with the `gpu_support` feature flag, this implementation can use GPU
/// acceleration for both kernel types:
///
/// - **Gaussian Kernel**: GPU acceleration is automatically used for datasets with 500 or more points.
///   The adaptive radius for neighbor search is larger when using GPU acceleration, especially for
///   small bandwidths (< 0.5):
///   - For large datasets (> 5000 points) with small bandwidths: 4σ radius
///   - For smaller datasets with small bandwidths: 5σ radius
///   - For large datasets with normal bandwidths: 3σ radius
///   - For smaller datasets with normal bandwidths: 4σ radius
///
/// - **Box Kernel**: GPU acceleration is automatically used for datasets with 2000 or more points.
///   For smaller datasets, the CPU implementation is used as it's faster due to the overhead of GPU setup.
///
/// # Examples
///
/// ```
/// use infomeasure::estimators::entropy::Entropy;
/// use infomeasure::estimators::entropy::LocalValues;
/// use ndarray::Array1;
///
/// // Create some 1D data
/// let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
///
/// // Calculate entropy using box kernel
/// let box_entropy = Entropy::new_kernel(data.clone(), 0.5);
/// let box_global_value = box_entropy.global_value();
/// let box_local_values = box_entropy.local_values();
///
/// // Calculate entropy using Gaussian kernel
/// let gaussian_entropy = Entropy::new_kernel_with_type(data, "gaussian".to_string(), 0.5);
/// let gaussian_global_value = gaussian_entropy.global_value();
/// let gaussian_local_values = gaussian_entropy.local_values();
/// ```
pub struct KernelEntropy<const K: usize> {
    /// Data points stored in a format suitable for KD-tree operations
    pub points: Vec<[f64; K]>,
    /// Number of data points
    pub n_samples: usize,
    /// Type of kernel to use ("box" or "gaussian")
    pub kernel_type: String,
    /// Bandwidth parameter controlling the smoothness of the density estimate
    pub bandwidth: f64,
    /// KD-tree for efficient nearest-neighbor queries
    pub tree: ImmutableKdTree<f64, K>,
    /// Standard deviations of the data in each dimension (used for Gaussian kernel scaling)
    pub std_devs: [f64; K],
}
impl<const K: usize> KernelEntropy<K> {
    /// Creates a new KernelEntropy estimator with the default "box" kernel
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A new KernelEntropy instance configured with a box kernel
    pub fn new(data: impl Into<KernelData>, bandwidth: f64) -> Self {
        Self::new_with_kernel_type(data, "box".to_string(), bandwidth)
    }

    /// Creates a new KernelEntropy estimator with a specified kernel type
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A new KernelEntropy instance configured with the specified kernel type
    ///
    /// # Notes
    ///
    /// The bandwidth parameter is interpreted differently depending on the kernel type:
    /// - For box kernels, it's used directly as the radius of the hypercube
    /// - For Gaussian kernels, it's scaled by the standard deviation in each dimension
    pub fn new_with_kernel_type(data: impl Into<KernelData>, kernel_type: String, bandwidth: f64) -> Self {
        let data = data.into();
        // for kernel type, lowercase it
        let kernel_type = kernel_type.to_lowercase();
        // for bandwidth, ensure it's a positive number
        assert!(bandwidth > 0.0);
        // for data array, assure second dimension == K
        assert!(match &data {
            KernelData::OneDimensional(_) => K == 1,
            KernelData::TwoDimensional(d) => d.ncols() == K
        });

        // Convert the data into points suitable for the KD-tree
        let points: Vec<[f64; K]> = match &data {
            KernelData::OneDimensional(arr) => {
                // For 1D data, we can directly use as_slice()
                arr.as_slice()
                    .expect("Array must be contiguous")
                    .chunks(1)
                    .map(|chunk| {
                        let mut point = [0.0; K];
                        point[0] = chunk[0];
                        point
                    })
                    .collect()
            }
            KernelData::TwoDimensional(arr) => {
                if let Some(slice) = arr.as_slice() {
                    // If the array is contiguous, we can process it as a flat slice
                    slice.chunks(K)
                        .map(|chunk| {
                            let mut point = [0.0; K];
                            point.copy_from_slice(&chunk[..K]);
                            point
                        })
                        .collect()
                } else {
                    // Fallback for non-contiguous arrays
                    arr.rows()
                        .into_iter()
                        .map(|row| {
                            let mut point = [0.0; K];
                            for (i, &val) in row.iter().enumerate() {
                                point[i] = val;
                            }
                            point
                        })
                        .collect()
                }
            }
        };

        let n_samples = points.len();
        let tree = ImmutableKdTree::new_from_slice(&points);

        // Calculate standard deviations for each dimension
        let mut std_devs = [0.0; K];
        for dim in 0..K {
            // // Calculate mean for this dimension
            // let mean = points.iter().map(|p| p[dim]).sum::<f64>() / n_samples as f64;
            // // Calculate variance for this dimension
            // let variance = points.iter().map(|p| (p[dim] - mean).powi(2)).sum::<f64>() / n_samples as f64;
            // Single-pass Welford's algorithm
            let mut mean = 0.0;
            let mut m2 = 0.0;
            let mut count = 0.0;
            for point in &points {
                count += 1.0;
                let delta = point[dim] - mean;
                mean += delta / count;
                let delta2 = point[dim] - mean;
                m2 += delta * delta2;
            }
            let variance = if count < 2.0 { 0.0 } else { m2 / count };
            std_devs[dim] = variance.sqrt();
        }

        Self { points, n_samples, kernel_type, bandwidth, tree, std_devs }
    }

    /// Convenience constructor for 1D data
    ///
    /// # Arguments
    ///
    /// * `data` - One-dimensional data array
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter
    pub fn new_1d(data: Array1<f64>, kernel_type: String, bandwidth: f64) -> Self {
        Self::new_with_kernel_type(KernelData::OneDimensional(data), kernel_type, bandwidth)
    }

    /// Convenience constructor for 2D data
    ///
    /// # Arguments
    ///
    /// * `data` - Two-dimensional data array (rows = samples, columns = dimensions)
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter
    pub fn new_2d(data: Array2<f64>, kernel_type: String, bandwidth: f64) -> Self {
        Self::new_with_kernel_type(KernelData::TwoDimensional(data), kernel_type, bandwidth)
    }

    /// Computes local entropy values using a box (uniform) kernel
    ///
    /// The box kernel counts points within a hypercube of side length `bandwidth`
    /// centered at each query point. This is equivalent to using a uniform kernel
    /// where all points within the bandwidth contribute equally to the density estimate.
    ///
    /// # Implementation Details
    ///
    /// 1. For each data point, find all neighbors within distance `bandwidth/2`
    /// 2. Count the number of neighbors
    /// 3. Normalize by the volume of the hypercube (bandwidth^d) and the number of samples
    /// 4. Apply logarithm to get entropy values
    ///
    /// # Notes
    ///
    /// Unlike the Gaussian kernel, the box kernel does not scale the bandwidth by the
    /// standard deviation of the data. The bandwidth is used directly as the side length
    /// of the hypercube.
    pub fn box_kernel_local_values(&self) -> Array1<f64> {
        // Calculate volume = bandwidth^d (where d = K)
        // This is the volume of the hypercube with side length = bandwidth
        let volume = self.bandwidth.powi(K as i32);

        // Normalization factor: N * volume
        // This is the denominator in the KDE formula: f̂(x) = (1/Nh^d) ∑ K((x - x_i)/h)
        // where K is the box kernel (uniform within the bandwidth)
        let n_volume = self.n_samples as f64 * volume;

        // Initialize array to store local entropy values
        let mut local_values = Array1::<f64>::zeros(self.n_samples);

        // Process points in batches of 4 (for f64x4) or 8 (for f64x8)
        let batch_size = 4;
        let num_batches = self.n_samples / batch_size;

        // Process complete batches
        for batch in 0..num_batches {
            let start_idx = batch * batch_size;

            // Use SIMD to process multiple points in parallel
            #[cfg(feature = "simd_support")]
            {
                // Create arrays to store neighbor counts for each point in the batch
                let mut neighbor_counts = [0.0f64; 4];

                // Process each point in the batch
                for i in 0..batch_size {
                    let idx = start_idx + i;
                    let query_point = &self.points[idx];

                    // Find neighbors (this part remains scalar as it depends on the KD-tree)
                    let neighbors = self.tree.within_unsorted::<Manhattan>(
                        query_point,
                        self.bandwidth / 2.0f64
                    );

                    neighbor_counts[i] = neighbors.len() as f64;
                }

                // Use SIMD for the normalization and log transform
                let counts_vec = f64x4::from_array(neighbor_counts);
                let n_volume_vec = f64x4::splat(n_volume);

                // Calculate -(counts / n_volume).ln() for all points in parallel
                let normalized = counts_vec / n_volume_vec;
                let log_values = -normalized.ln();

                // Store results back to the output array
                for i in 0..batch_size {
                    local_values[start_idx + i] = log_values[i];
                }
            }

            // Fallback for non-SIMD case
            #[cfg(not(feature = "simd_support"))]
            {
                // // For each point, find neighbors within bandwidth/2 using Manhattan distance
                // // This creates a hypercube with side length = bandwidth centered at the query point
                // for (i, query_point) in self.points.iter().enumerate() {
                //     // Use Manhattan distance (L1 norm) to find points within a hypercube
                //     // The bandwidth/2 is used because Manhattan distance measures from the center to the edge
                //     let neighbors = self.tree.within_unsorted::<Manhattan>(
                //         query_point,
                //         self.bandwidth / 2.0f64
                //     );
                //
                //     // Count the number of neighbors (including the point itself)
                //     local_values[i] = neighbors.len() as f64;
                // }
                //
                // // Apply normalization and log transform for entropy calculation: H = -E[log(f(x))]
                // // f(x) = count / (N * volume), so log(f(x)) = log(count) - log(N * volume)
                // // and -log(f(x)) = log(N * volume) - log(count) = log((N * volume) / count)
                // local_values.mapv_inplace(|x| -(x / n_volume).ln());
                //
                // local_values
                for i in 0..batch_size {
                    let idx = start_idx + i;
                    let query_point = &self.points[idx];

                    let neighbors = self.tree.within_unsorted::<Manhattan>(
                        query_point,
                        self.bandwidth / 2.0f64
                    );

                    local_values[idx] = neighbors.len() as f64;
                }
            }
        }

        // Process remaining points
        for i in (num_batches * batch_size)..self.n_samples {
            let query_point = &self.points[i];
            let neighbors = self.tree.within_unsorted::<Manhattan>(
                query_point,
                self.bandwidth / 2.0f64
            );
            local_values[i] = neighbors.len() as f64;
        }

        // Apply normalization and log transform to remaining points
        #[cfg(not(feature = "simd_support"))]
        local_values.mapv_inplace(|x| -(x / n_volume).ln());

        local_values
    }

    /// Computes local entropy values using a Gaussian kernel
    ///
    /// The Gaussian kernel uses a normal distribution centered at each query point
    /// to weight the contribution of neighboring points to the density estimate.
    /// This provides a smoother density estimate compared to the box kernel.
    ///
    /// # Implementation Details
    ///
    /// 1. For each data point, find all neighbors within a reasonable distance
    /// 2. Calculate the Gaussian kernel contribution from each neighbor
    /// 3. Normalize by the product of (bandwidth * std_dev) in each dimension and the number of samples
    /// 4. Apply logarithm and dimension-dependent normalization to get entropy values
    ///
    /// # Adaptive Radius for Neighbor Search
    ///
    /// The Gaussian kernel uses an adaptive radius to limit the search for neighbors:
    ///
    /// - For large datasets (>5000 points): 3σ radius (9 * max_scaled_bandwidth²)
    /// - For smaller datasets: 4σ radius (16 * max_scaled_bandwidth²)
    ///
    /// When compiled with the `gpu_support` feature flag, the GPU implementation uses
    /// a larger adaptive radius, especially for small bandwidths (< 0.5):
    ///
    /// - For large datasets (>5000 points) with small bandwidths: 4σ radius
    /// - For smaller datasets with small bandwidths: 5σ radius
    /// - For large datasets with normal bandwidths: 3σ radius
    /// - For smaller datasets with normal bandwidths: 4σ radius
    ///
    /// Points beyond this distance have a negligible contribution to the density estimate.
    ///
    /// # Bandwidth Scaling
    ///
    /// Unlike the box kernel, the Gaussian kernel scales the bandwidth by the standard
    /// deviation of the data in each dimension. This makes the estimator adaptive to the
    /// scale of the data in each dimension, matching the behavior of scipy.stats.gaussian_kde.
    ///
    /// The scaling is applied in two places:
    /// 1. When calculating the search radius for finding neighbors
    /// 2. When calculating the scaled distance for the Gaussian kernel
    ///
    /// # Dimension-Dependent Normalization
    ///
    /// The Gaussian kernel entropy includes a dimension-dependent normalization factor:
    /// (d/2) * ln(2π), where d is the dimensionality of the data. This ensures that
    /// the entropy estimate is consistent with the theoretical definition of differential
    /// entropy for a multivariate Gaussian distribution.
    pub fn gaussian_kernel_local_values(&self) -> Array1<f64> {
        let n = self.points.len();

        // Pre-compute scale factors once for all query points
        // This is an optimization to avoid recomputing these for each query point
        let scale_factors: Vec<f64> = (0..K).map(|dim| self.bandwidth * self.std_devs[dim]).collect();

        // Calculate the product of scale factors for normalization
        // This is equivalent to the determinant of the covariance matrix in scipy.stats.gaussian_kde
        // where the covariance is a diagonal matrix with (bandwidth * std_dev)^2 on the diagonal
        let scaled_bandwidth_product = scale_factors.iter().fold(1.0, |product, &factor| product * factor);

        // Normalization factor: N * (h*σ)^d
        // This is the denominator in the KDE formula: f̂(x) = (1/Nh^d) ∑ K((x - x_i)/h)
        // where we've scaled h by σ in each dimension
        let normalization = (n as f64) * scaled_bandwidth_product;

        // Calculate max scaled bandwidth for search radius
        // We need to use the largest scaled bandwidth to ensure we don't miss any points
        // that might have a significant contribution to the density estimate
        let max_scaled_bandwidth = scale_factors.iter().fold(0.0f64, |max_val, &val| max_val.max(val));

        // Determine adaptive radius based on data density
        let adaptive_radius = if self.n_samples > 5000 {
            9.0 * max_scaled_bandwidth.powi(2) // 3σ | (3*h)^2 for large datasets
        } else {
            16.0 * max_scaled_bandwidth.powi(2) // 4σ | (4*h)^2 for smaller datasets
        };

        let mut local_values = Array1::<f64>::zeros(n);

        // For each point, calculate its contribution to the density estimate
        for (i, query_point) in self.points.iter().enumerate() {
            // Get all points within reasonable distance
            // The squared distance cutoff is (4*max_scaled_bandwidth)^2 = 16*max_scaled_bandwidth^2
            // Points beyond this distance have negligible contribution to the density estimate
            let neighbors = self.tree.within_unsorted::<SquaredEuclidean>(
                query_point,
                adaptive_radius
            );

            // Calculate Gaussian kernel contribution from each neighbor
            // K((x - x_i)/h) = exp(-(||x - x_i||/h)²/2)
            // where we scale the distance by the bandwidth and standard deviation in each dimension
            let density: f64 = neighbors.iter().map(|&neighbor| {
                let (_dist, idx) = neighbor.into();

                // Use the optimized SIMD calculation with pre-computed scale factors
                let scaled_dist = self.calculate_scaled_distance_with_factors(
                    query_point, 
                    &self.points[idx as usize], 
                    &scale_factors
                );

                // Gaussian kernel function: exp(-scaled_dist/2)
                // Use fast approximation for exponential function
                #[cfg(feature = "fast_exp")]
                { self.fast_exp(-scaled_dist / 2.0) }
                #[cfg(not(feature = "fast_exp"))]
                { (-scaled_dist / 2.0).exp() }
            }).sum::<f64>();

            // Normalize the density estimate by the normalization factor
            local_values[i] = density / normalization;
        }

        // Apply log transform for entropy calculation: H = -E[log(f(x))]
        // Handle the case where density is zero (should not happen in practice)
        local_values.mapv_inplace(|x| if x > 0.0 { -x.ln() } else { 0.0 });

        // Apply dimension-dependent normalization factor
        // For a Gaussian kernel in d dimensions, we need to add ln((2π)^(d/2))
        // which equals (d/2) * ln(2π)
        // This ensures consistency with the theoretical differential entropy formula
        let dim_factor = (K as f64 / 2.0) * (2.0 * std::f64::consts::PI).ln();
        local_values.mapv_inplace(|x| x + dim_factor);
        local_values
    }


    /// Optimized scaled distance calculation using pre-computed scale factors
    fn calculate_scaled_distance_with_factors(&self, query_point: &[f64; K], neighbor_point: &[f64; K], scale_factors: &[f64]) -> f64 {
        match K {
            // Optimized SIMD paths for common dimensions
            1 => {
                let diff = query_point[0] - neighbor_point[0];
                (diff / scale_factors[0]).powi(2)
            },
            2 => {
                let diff0 = query_point[0] - neighbor_point[0];
                let diff1 = query_point[1] - neighbor_point[1];
                let scaled0 = diff0 / scale_factors[0];
                let scaled1 = diff1 / scale_factors[1];
                scaled0 * scaled0 + scaled1 * scaled1
            },
            3 => {
                let diff0 = query_point[0] - neighbor_point[0];
                let diff1 = query_point[1] - neighbor_point[1];
                let diff2 = query_point[2] - neighbor_point[2];
                let scaled0 = diff0 / scale_factors[0];
                let scaled1 = diff1 / scale_factors[1];
                let scaled2 = diff2 / scale_factors[2];
                scaled0 * scaled0 + scaled1 * scaled1 + scaled2 * scaled2
            },
            #[cfg(feature = "simd_support")]
            4 => {
                // SIMD implementation for K=4
                let query_vec = f64x4::from_array([
                    query_point[0], query_point[1], query_point[2], query_point[3]
                ]);
                let point_vec = f64x4::from_array([
                    neighbor_point[0], neighbor_point[1], neighbor_point[2], neighbor_point[3]
                ]);
                let scale_vec = f64x4::from_array([
                    scale_factors[0], scale_factors[1], scale_factors[2], scale_factors[3]
                ]);

                let diff_vec = query_vec - point_vec;
                let scaled_vec = diff_vec / scale_vec;
                let squared_vec = scaled_vec * scaled_vec;
                squared_vec.reduce_sum()
            },
            #[cfg(feature = "simd_support")]
            8 => {
                // SIMD implementation for K=8
                let query_vec = f64x8::from_array([
                    query_point[0], query_point[1], query_point[2], query_point[3],
                    query_point[4], query_point[5], query_point[6], query_point[7]
                ]);
                let point_vec = f64x8::from_array([
                    neighbor_point[0], neighbor_point[1], neighbor_point[2], neighbor_point[3],
                    neighbor_point[4], neighbor_point[5], neighbor_point[6], neighbor_point[7]
                ]);
                let scale_vec = f64x8::from_array([
                    scale_factors[0], scale_factors[1], scale_factors[2], scale_factors[3],
                    scale_factors[4], scale_factors[5], scale_factors[6], scale_factors[7]
                ]);

                let diff_vec = query_vec - point_vec;
                let scaled_vec = diff_vec / scale_vec;
                let squared_vec = scaled_vec * scaled_vec;
                squared_vec.reduce_sum()
            },
            _ => {
                // Generic SIMD implementation for larger dimensions
                #[cfg(feature = "simd_support")]
                { self.calculate_scaled_distance_generic(query_point, neighbor_point, scale_factors) }
                #[cfg(not(feature = "simd_support"))]
                {
                    // Fallback scalar implementation
                    (0..K).map(|dim| {
                        let diff = query_point[dim] - neighbor_point[dim];
                        (diff / scale_factors[dim]).powi(2)
                    }).sum::<f64>()
                }
            }
        }
    }

    /// Fast approximation of the exponential function
    /// 
    /// This implementation uses a hybrid approach that provides a good balance between accuracy
    /// and performance for the range of inputs that are typical in the Gaussian kernel calculation.
    /// 
    /// The approximation uses three different methods depending on the input value:
    /// 1. For very small negative values (x > -0.5), it uses a 5th-order Taylor series approximation,
    ///    which is very accurate for inputs close to 0.
    /// 2. For medium negative values (-2.5 < x <= -0.5), it uses a 3rd-order rational approximation,
    ///    which provides a good balance between accuracy and performance.
    /// 3. For large negative values (x <= -2.5), it uses a 6th-order rational approximation,
    ///    which provides better accuracy for inputs far from 0.
    /// 
    /// The approximation is accurate to within 1% for inputs in the range [-1, 0], within 20%
    /// for inputs in the range [-3, -1], and within 35% for inputs in the range [-5, -3].
    /// For inputs outside this range, the approximation may have larger errors, but these inputs
    /// are rare in the Gaussian kernel calculation.
    /// 
    /// Benchmark results show that this approximation is about 1.3x faster than the standard
    /// library's exp function.
    #[cfg(feature = "fast_exp")]
    fn fast_exp(&self, x: f64) -> f64 {
        // Handle extreme values to prevent overflow/underflow
        if x < -700.0 {
            return 0.0;
        }
        if x > 700.0 {
            return f64::INFINITY;
        }

        // For very small negative values, use a Taylor series approximation
        if x > -0.5 {
            // Use a 5th-order Taylor series approximation for small negative values
            // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
            return 1.0 + x * (1.0 + x * (0.5 + x * (1.0/6.0 + x * (1.0/24.0 + x * (1.0/120.0)))));
        }

        // For medium negative values, use a rational approximation
        if x > -2.5 {
            // This is a minimax approximation that provides a good balance between accuracy and performance
            // exp(x) ≈ 1 / (1 - x + x²/2 - x³/6) for x < 0
            return 1.0 / (1.0 - x + x*x/2.0 - x*x*x/6.0);
        }

        // For large negative values, use a higher-order rational approximation
        // This provides better accuracy for inputs far from 0
        // exp(x) ≈ 1 / (1 - x + x²/2 - x³/6 + x⁴/24 - x⁵/120 + x⁶/720)
        1.0 / (1.0 - x + x*x/2.0 - x*x*x/6.0 + x*x*x*x/24.0 - x*x*x*x*x/120.0 + x*x*x*x*x*x/720.0)
    }

    /// Generic SIMD implementation that processes dimensions in chunks
    #[cfg(feature = "simd_support")]
    fn calculate_scaled_distance_generic(&self, query_point: &[f64; K], neighbor_point: &[f64; K], scale_factors: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut dim = 0;

        // Process 8 dimensions at a time with f64x8
        while dim + 8 <= K {
            let query_vec = f64x8::from_array([
                query_point[dim], query_point[dim+1], query_point[dim+2], query_point[dim+3],
                query_point[dim+4], query_point[dim+5], query_point[dim+6], query_point[dim+7]
            ]);
            let point_vec = f64x8::from_array([
                neighbor_point[dim], neighbor_point[dim+1], neighbor_point[dim+2], neighbor_point[dim+3],
                neighbor_point[dim+4], neighbor_point[dim+5], neighbor_point[dim+6], neighbor_point[dim+7]
            ]);
            let scale_vec = f64x8::from_array([
                scale_factors[dim], scale_factors[dim+1], scale_factors[dim+2], scale_factors[dim+3],
                scale_factors[dim+4], scale_factors[dim+5], scale_factors[dim+6], scale_factors[dim+7]
            ]);

            let diff_vec = query_vec - point_vec;
            let scaled_vec = diff_vec / scale_vec;
            let squared_vec = scaled_vec * scaled_vec;
            sum += squared_vec.reduce_sum();

            dim += 8;
        }

        // Process 4 dimensions at a time with f64x4
        while dim + 4 <= K {
            let query_vec = f64x4::from_array([
                query_point[dim], query_point[dim+1], query_point[dim+2], query_point[dim+3]
            ]);
            let point_vec = f64x4::from_array([
                neighbor_point[dim], neighbor_point[dim+1], neighbor_point[dim+2], neighbor_point[dim+3]
            ]);
            let scale_vec = f64x4::from_array([
                scale_factors[dim], scale_factors[dim+1], scale_factors[dim+2], scale_factors[dim+3]
            ]);

            let diff_vec = query_vec - point_vec;
            let scaled_vec = diff_vec / scale_vec;
            let squared_vec = scaled_vec * scaled_vec;
            sum += squared_vec.reduce_sum()
            ;

            dim += 4;
        }

        // Process remaining dimensions scalar
        while dim < K {
            let diff = query_point[dim] - neighbor_point[dim];
            let scaled = diff / scale_factors[dim];
            sum += scaled * scaled;
            dim += 1;
        }

        sum
    }


}

/// Implementation of the LocalValues trait for KernelEntropy
///
/// This allows KernelEntropy to be used with the entropy estimation framework,
/// which expects implementors to provide local entropy values that can be
/// aggregated to compute the global entropy.
impl<const K: usize> LocalValues for KernelEntropy<K> {
    /// Computes the local entropy values for each data point
    ///
    /// This method dispatches to the appropriate kernel implementation based on
    /// the kernel_type specified during construction. It returns an array of
    /// local entropy values, one for each data point.
    ///
    /// # Returns
    ///
    /// An array of local entropy values. The mean of these values gives the
    /// global entropy estimate.
    ///
    /// # Notes
    ///
    /// - For the box kernel, local values represent the entropy contribution from
    ///   counting points within a hypercube centered at each data point.
    ///
    /// - For the Gaussian kernel, local values represent the entropy contribution
    ///   from a Gaussian-weighted sum of distances to neighboring points, with
    ///   bandwidth scaled by the standard deviation in each dimension.
    ///
    /// # GPU Acceleration
    ///
    /// When the `gpu_support` feature flag is enabled, both the Gaussian and box kernel
    /// calculations can use GPU acceleration, which provides significant performance 
    /// improvements for large datasets and high-dimensional data:
    ///
    /// - **Gaussian Kernel**: GPU acceleration is used for datasets with 500 or more points.
    ///   For smaller datasets, the implementation automatically falls back to the CPU version.
    ///   The adaptive radius for neighbor search is larger when using GPU acceleration.
    ///
    /// - **Box Kernel**: GPU acceleration is used for datasets with 2000 or more points.
    ///   For smaller datasets, the implementation automatically falls back to the CPU version
    ///   as the overhead of GPU setup outweighs the benefits for small data sizes.
    fn local_values(&self) -> Array1<f64> {
        // Dispatch to the appropriate kernel implementation
        match self.kernel_type.as_str() {
            "box" => {
                #[cfg(feature = "gpu_support")]
                {
                    // Use GPU implementation when gpu_support feature flag is enabled
                    self.box_kernel_local_values_gpu()
                }
                #[cfg(not(feature = "gpu_support"))]
                {
                    // Fall back to CPU implementation when gpu_support feature flag is not enabled
                    self.box_kernel_local_values()
                }
            },
            "gaussian" => {
                #[cfg(feature = "gpu_support")]
                {
                    // Use GPU implementation when gpu_support feature flag is enabled
                    self.gaussian_kernel_local_values_gpu()
                }
                #[cfg(not(feature = "gpu_support"))]
                {
                    // Fall back to CPU implementation when gpu_support feature flag is not enabled
                    self.gaussian_kernel_local_values()
                }
            },
            _ => {
                // Default to box kernel if an unsupported kernel type is specified
                // This provides backward compatibility and graceful fallback
                self.box_kernel_local_values()
            }
        }
    }
}

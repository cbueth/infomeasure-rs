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

use ndarray::{Array1, Array2};
use kiddo::{ImmutableKdTree, Manhattan, SquaredEuclidean};
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
/// # Examples
///
/// ```
/// use infomeasure::estimators::entropy::Entropy;
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
    points: Vec<[f64; K]>,
    /// Number of data points
    n_samples: usize,
    /// Type of kernel to use ("box" or "gaussian")
    kernel_type: String,
    /// Bandwidth parameter controlling the smoothness of the density estimate
    bandwidth: f64,
    /// KD-tree for efficient nearest-neighbor queries
    tree: ImmutableKdTree<f64, K>,
    /// Standard deviations of the data in each dimension (used for Gaussian kernel scaling)
    std_devs: [f64; K],
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
            // Calculate mean for this dimension
            let mean = points.iter().map(|p| p[dim]).sum::<f64>() / n_samples as f64;
            // Calculate variance for this dimension
            let variance = points.iter().map(|p| (p[dim] - mean).powi(2)).sum::<f64>() / n_samples as f64;
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
    fn box_kernel_local_values(&self) -> Array1<f64> {
        // Calculate volume = bandwidth^d (where d = K)
        // This is the volume of the hypercube with side length = bandwidth
        let volume = self.bandwidth.powi(K as i32);

        // Normalization factor: N * volume
        // This is the denominator in the KDE formula: f̂(x) = (1/Nh^d) ∑ K((x - x_i)/h)
        // where K is the box kernel (uniform within the bandwidth)
        let n_volume = self.n_samples as f64 * volume;

        // Initialize array to store local entropy values
        let mut local_values = Array1::<f64>::zeros(self.n_samples);

        // For each point, find neighbors within bandwidth/2 using Manhattan distance
        // This creates a hypercube with side length = bandwidth centered at the query point
        for (i, query_point) in self.points.iter().enumerate() {
            // Use Manhattan distance (L1 norm) to find points within a hypercube
            // The bandwidth/2 is used because Manhattan distance measures from the center to the edge
            let neighbors = self.tree.within_unsorted::<Manhattan>(
                query_point,
                self.bandwidth / 2.0f64
            );

            // Count the number of neighbors (including the point itself)
            local_values[i] = neighbors.len() as f64;
        }

        // Apply normalization and log transform for entropy calculation: H = -E[log(f(x))]
        // f(x) = count / (N * volume), so log(f(x)) = log(count) - log(N * volume)
        // and -log(f(x)) = log(N * volume) - log(count) = log((N * volume) / count)
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
    /// Note: The Gaussian kernel uses a cutoff of 4 times the maximum scaled bandwidth to limit
    /// the search radius for neighbors. Points beyond this distance have negligible contribution
    /// to the density estimate.
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
    fn gaussian_kernel_local_values(&self) -> Array1<f64> {
        let n = self.points.len();

        // Calculate the product of (bandwidth * std_dev) for each dimension
        // This is equivalent to the determinant of the covariance matrix in scipy.stats.gaussian_kde
        // where the covariance is a diagonal matrix with (bandwidth * std_dev)^2 on the diagonal
        let scaled_bandwidth_product = (0..K).fold(1.0, |product, dim| {
            product * (self.bandwidth * self.std_devs[dim])
        });

        // Normalization factor: N * (h*σ)^d
        // This is the denominator in the KDE formula: f̂(x) = (1/Nh^d) ∑ K((x - x_i)/h)
        // where we've scaled h by σ in each dimension
        let normalization = (n as f64) * scaled_bandwidth_product;

        let mut local_values = Array1::<f64>::zeros(n);

        // For each point, calculate its contribution to the density estimate
        for (i, query_point) in self.points.iter().enumerate() {
            // Calculate max scaled bandwidth for search radius
            // We need to use the largest scaled bandwidth to ensure we don't miss any points
            // that might have a significant contribution to the density estimate
            let max_scaled_bandwidth = self.std_devs.iter().fold(self.bandwidth, |max_bw, &std_dev| {
                max_bw.max(self.bandwidth * std_dev)
            });

            // Get all points within reasonable distance (4*max_scaled_bandwidth is a common choice)
            // The squared distance cutoff is (4*max_scaled_bandwidth)^2 = 16*max_scaled_bandwidth^2
            // Points beyond this distance have negligible contribution to the density estimate
            let neighbors = self.tree.within_unsorted::<SquaredEuclidean>(
                query_point,
                16.0 * max_scaled_bandwidth.powi(2) // (4*h)^2 as cutoff
            );

            // Calculate Gaussian kernel contribution from each neighbor
            // K((x - x_i)/h) = exp(-(||x - x_i||/h)²/2)
            // where we scale the distance by the bandwidth and standard deviation in each dimension
            let density: f64 = neighbors.iter().map(|&neighbor| {
                let (_dist, idx) = neighbor.into();

                // Calculate scaled distance using standard deviations
                // This is equivalent to the Mahalanobis distance with a diagonal covariance matrix
                // where the diagonal elements are (bandwidth * std_dev)^2
                let scaled_dist = (0..K).map(|dim| {
                    let diff = query_point[dim] - self.points[idx as usize][dim];
                    (diff / (self.bandwidth * self.std_devs[dim])).powi(2)
                }).sum::<f64>();

                // Gaussian kernel function: exp(-scaled_dist/2)
                (-scaled_dist / 2.0).exp()
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
    fn local_values(&self) -> Array1<f64> {
        // Dispatch to the appropriate kernel implementation
        match self.kernel_type.as_str() {
            "box" => self.box_kernel_local_values(),
            "gaussian" => self.gaussian_kernel_local_values(),
            _ => {
                // Default to box kernel if an unsupported kernel type is specified
                // This provides backward compatibility and graceful fallback
                self.box_kernel_local_values()
            }
        }
    }
}

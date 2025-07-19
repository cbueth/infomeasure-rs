use ndarray::Array1;
use crate::estimators::approaches::{discrete, kernel};
pub use crate::estimators::traits::LocalValues;

/// Entropy estimation methods for various data types
///
/// This struct provides static methods for creating entropy estimators
/// for different types of data and estimation approaches.
pub struct Entropy;

// Non-generic implementation (1D default case)
impl Entropy {
    /// Creates a new discrete entropy estimator for 1D integer data
    ///
    /// # Arguments
    ///
    /// * `data` - One-dimensional array of integer data
    ///
    /// # Returns
    ///
    /// A discrete entropy estimator configured for the provided data
    pub fn new_discrete(data: Array1<i32>) -> discrete::DiscreteEntropy {
        discrete::DiscreteEntropy::new(data)
    }

    /// Creates a new kernel entropy estimator for 1D data using the default box kernel
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A kernel entropy estimator configured with a box kernel
    ///
    /// # GPU Acceleration
    ///
    /// When compiled with the `gpu_support` feature flag, this method will use GPU
    /// acceleration for datasets with 2000 or more points, providing significant
    /// performance improvements for large datasets.
    pub fn new_kernel(data: impl Into<kernel::KernelData>, bandwidth: f64) -> kernel::KernelEntropy<1> {
        kernel::KernelEntropy::new(data, bandwidth)
    }

    /// Creates a new kernel entropy estimator for 1D data with a specified kernel type
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A kernel entropy estimator configured with the specified kernel type
    ///
    /// # GPU Acceleration
    ///
    /// When compiled with the `gpu_support` feature flag, this method will use GPU
    /// acceleration based on the kernel type and dataset size:
    ///
    /// - For Gaussian kernel: GPU is used for datasets with 500 or more points
    /// - For Box kernel: GPU is used for datasets with 2000 or more points
    ///
    /// The Gaussian kernel with GPU acceleration uses an enhanced adaptive radius
    /// calculation, especially for small bandwidths (< 0.5).
    pub fn new_kernel_with_type(
        data: impl Into<kernel::KernelData>,
        kernel_type: String,
        bandwidth: f64
    ) -> kernel::KernelEntropy<1> {
        kernel::KernelEntropy::new_with_kernel_type(data, kernel_type, bandwidth)
    }

    /// Creates a new kernel entropy estimator for N-dimensional data using the default box kernel
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A kernel entropy estimator configured with a box kernel for N-dimensional data
    ///
    /// # GPU Acceleration
    ///
    /// When compiled with the `gpu_support` feature flag, this method will use GPU
    /// acceleration for datasets with 2000 or more points, providing significant
    /// performance improvements for large datasets and high-dimensional data.
    pub fn nd_kernel<const K: usize>(
        data: impl Into<kernel::KernelData>,
        bandwidth: f64
    ) -> kernel::KernelEntropy<K> {
        kernel::KernelEntropy::new(data, bandwidth)
    }

    /// Creates a new kernel entropy estimator for N-dimensional data with a specified kernel type
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A kernel entropy estimator configured with the specified kernel type for N-dimensional data
    ///
    /// # GPU Acceleration
    ///
    /// When compiled with the `gpu_support` feature flag, this method will use GPU
    /// acceleration based on the kernel type and dataset size:
    ///
    /// - For Gaussian kernel: GPU is used for datasets with 500 or more points, providing
    ///   speedups of up to 340x for large datasets
    /// - For Box kernel: GPU is used for datasets with 2000 or more points, providing
    ///   speedups of up to 37x for large datasets
    ///
    /// The GPU implementation is particularly beneficial for high-dimensional data,
    /// where it can complete calculations that would timeout on the CPU.
    pub fn nd_kernel_with_type<const K: usize>(
        data: impl Into<kernel::KernelData>,
        kernel_type: String,
        bandwidth: f64
    ) -> kernel::KernelEntropy<K> {
        kernel::KernelEntropy::new_with_kernel_type(data, kernel_type, bandwidth)
    }
}

use ndarray::Array1;
use crate::estimators::approaches::{discrete, kernel};
pub use crate::estimators::traits::LocalValues;

pub struct Entropy;

// Non-generic implementation (1D default case)
impl Entropy {
    pub fn new_discrete(data: Array1<i32>) -> discrete::DiscreteEntropy {
        discrete::DiscreteEntropy::new(data)
    }

    pub fn new_kernel(data: impl Into<kernel::KernelData>, bandwidth: f64) -> kernel::KernelEntropy<1> {
        kernel::KernelEntropy::new(data, bandwidth)
    }

    pub fn new_kernel_with_type(
        data: impl Into<kernel::KernelData>,
        kernel_type: String,
        bandwidth: f64
    ) -> kernel::KernelEntropy<1> {
        kernel::KernelEntropy::new_with_kernel_type(data, kernel_type, bandwidth)
    }

    // Static methods for N-dimensional cases
    pub fn nd_kernel<const K: usize>(
        data: impl Into<kernel::KernelData>,
        bandwidth: f64
    ) -> kernel::KernelEntropy<K> {
        kernel::KernelEntropy::new(data, bandwidth)
    }

    pub fn nd_kernel_with_type<const K: usize>(
        data: impl Into<kernel::KernelData>,
        kernel_type: String,
        bandwidth: f64
    ) -> kernel::KernelEntropy<K> {
        kernel::KernelEntropy::new_with_kernel_type(data, kernel_type, bandwidth)
    }
}

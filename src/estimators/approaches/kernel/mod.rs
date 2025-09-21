mod kernel; // core kernel entropy implementation
pub use kernel::*; // re-export KernelEntropy, KernelData, etc.

// Include the GPU implementations when the gpu_support feature flag is enabled
#[cfg(feature = "gpu_support")]
pub mod kernel_gpu;
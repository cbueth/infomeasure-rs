pub mod discrete;
pub mod kernel;
// pub mod ordinal;
// pub mod renyi;
// pub mod tsallis;

// Include the GPU implementation when the gpu_support feature flag is enabled
#[cfg(feature = "gpu_support")]
pub mod kernel_gpu;

pub use discrete::DiscreteEntropy;
pub use kernel::KernelEntropy;
// pub use ordinal::OrdinalEntropy;
// pub use renyi::RenyiEntropy;
// pub use tsallis::TsallisEntropy;

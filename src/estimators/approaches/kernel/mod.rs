// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

mod kernel_estimator; // core kernel entropy implementation
pub use kernel_estimator::*; // re-export KernelEntropy, KernelData, etc.

// Include the GPU implementations when the gpu feature flag is enabled
#[cfg(feature = "gpu")]
pub mod kernel_gpu;

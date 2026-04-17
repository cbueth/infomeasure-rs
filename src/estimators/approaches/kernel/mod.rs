// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Kernel Density Estimation (KDE) Estimators
//!
//! This module implements estimators for entropy and derived measures using
//! Kernel Density Estimation (KDE) for continuous variables.
//!
//! ## Theory
//!
//! The kernel entropy estimator computes differential Shannon entropy by first
//! estimating the probability density function (PDF) using kernels:
//!
//! $$\hat{f}(x) = \frac{1}{N h^d} \sum_{i=1}^{N} K\left(\frac{x - x_i}{h}\right)$$
//!
//! where:
//! - $K(\cdot)$ is the kernel function (e.g., Gaussian or Box).
//! - $h$ is the bandwidth parameter.
//! - $d$ is the dimensionality of the data.
//! - $N$ is the number of data points.
//!
//! The entropy is then approximated as the average log-density at the sample points:
//!
//! $$\hat{H}(X) \approx -\frac{1}{N} \sum_{i=1}^{N} \log \hat{f}(x_i)$$
//!
//! ## Parameters
//!
//! - **Bandwidth ($h$)**: Controls the smoothness of the density estimate. Small values
//!   may lead to under-smoothing (overfitting), while large values may over-smooth
//!   important features.
//! - **Kernel Type**:
//!   - `gaussian`: Provides smooth density estimates.
//!   - `box`: Computationally efficient uniform kernel.
//!
//! ## Measures Implemented
//!
//! - **Entropy**: $\hat{H}(X)$
//! - **Mutual Information**: $I(X; Y) = H(X) + H(Y) - H(X, Y)$
//! - **Conditional MI**: $I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)$
//! - **Transfer Entropy**: $T(X \to Y)$ estimated via the CMI entropy-summation formula.
//!
//! ## GPU Acceleration
//!
//! This module provides a GPU-accelerated implementation of the kernel density
//! estimation when the `gpu` feature is enabled, significantly speeding up
//! calculations for large datasets.
//!
//! ## See Also
//! - [Estimator Usage Guide](crate::guide::estimator_usage) — Examples and quick start
//! - [Estimator Approaches](super) — Overview of all estimation techniques
//! - [KSG Estimators](super::expfam::ksg) — kNN-based alternative for MI/TE
//!
//! ## References
//!
//! - [Silverman, 1986](../../../../guide/references/index.html#silverman1986)
//! - [García-Portugués, 2025](../../../../guide/references/index.html#garcia_portugues2025)

mod kernel_estimator; // core kernel entropy implementation
pub use kernel_estimator::*; // re-export KernelEntropy, KernelData, etc.

// Include the GPU implementations when the gpu feature flag is enabled
#[cfg(feature = "gpu")]
pub mod kernel_gpu;

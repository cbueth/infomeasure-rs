// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Estimator Approaches
//!
//! This module provides various algorithmic approaches for estimating
//! information-theoretic measures. These approaches are categorized based on the
//! type of data they handle and the underlying mathematical techniques.
//!
//! ## Taxonomy of Approaches
//!
//! ### 1. Discrete Estimators
//! Used for categorical, integer, or symbolic data. These estimators analyze
//! frequency counts (histograms) and often include bias-correction terms.
//! - **MLE**: Simple plug-in estimator.
//! - **NSB / Bayes**: Bayesian approaches for undersampled data.
//! - **Shrinkage**: Regularization toward a uniform distribution.
//! - [More details in Discrete Module](discrete)
//!
//! ### 2. Exponential Family (kNN-based)
//! Used for continuous, real-valued data. These non-parametric estimators use
//! k-nearest neighbor (kNN) distances to estimate local density.
//! - **KSG**: Specifically optimized for Mutual Information and Transfer Entropy.
//! - **Kozachenko-Leonenko (KL)**: Asymptotically unbiased differential entropy.
//! - **Rényi / Tsallis**: Generalized entropies.
//! - [More details in ExpFam Module](expfam)
//!
//! ### 3. Kernel Density Estimation (KDE)
//! Used for continuous data. KDE approximates the probability density function
//! by placing a "kernel" (e.g., Gaussian) on each data point.
//! - [More details in Kernel Module](kernel)
//!
//! ### 4. Ordinal (Permutation)
//! Used for time series. Analyzes the relative order of values in sliding
//! windows (permutation patterns) rather than their absolute values.
//! - [More details in Ordinal Module](ordinal)
//!
//! ## Which approach to use?
//!
//! See the [Estimator Selection Guide](crate::guide::estimator_selection) for
//! detailed recommendations based on your data size and characteristics.
//!
//! ## Implementation Note
//!
//! Approaches in this module are implemented as structs that provide the
//! mathematical core. They are often used by the high-level facade types in the
//! parent [`estimators`](crate::estimators) module.

pub mod common_nd;
pub mod discrete;
pub mod expfam;
pub mod kernel;
pub mod ordinal;

// Unified re-exports for common estimators so tests and users can import
// infomeasure::estimators::approaches::* ergonomically.
// Discrete estimators
pub use discrete::ansb::AnsbEntropy;
pub use discrete::bayes::BayesEntropy;
pub use discrete::bonachela::BonachelaEntropy;
pub use discrete::chao_shen::ChaoShenEntropy;
pub use discrete::chao_wang_jost::ChaoWangJostEntropy;
pub use discrete::grassberger::GrassbergerEntropy;
pub use discrete::miller_madow::MillerMadowEntropy;
pub use discrete::mle::DiscreteEntropy;
pub use discrete::nsb::NsbEntropy;
pub use discrete::shrink::ShrinkEntropy;
pub use discrete::zhang::ZhangEntropy;

// Exponential family estimators
pub use expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
pub use expfam::ksg::{
    KsgConditionalMutualInformation, KsgConditionalTransferEntropy, KsgMutualInformation2,
    KsgMutualInformation3, KsgMutualInformation4, KsgMutualInformation5, KsgMutualInformation6,
    KsgTransferEntropy,
};
pub use expfam::renyi::RenyiEntropy;
pub use expfam::tsallis::TsallisEntropy;
pub use expfam::utils::KsgType;

// Kernel and Ordinal
pub use kernel::KernelEntropy;
pub use ordinal::ordinal_estimator::OrdinalEntropy;

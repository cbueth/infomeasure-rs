// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/cbueth/infomeasure/refs/heads/main/docs/_static/im_icon_transparent-200x200.png"
)]

//! High-performance Rust library for information-theoretic measures including entropy,
//! mutual information, and transfer entropy with multiple estimation approaches.
//!
//! # Quick Start
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // Discrete entropy
//! let data = array![1, 2, 1, 3, 2, 1];
//! let entropy = Entropy::new_discrete(data).global_value();
//!
//! // Kernel entropy for continuous data
//! let continuous = array![[1.0, 1.5], [2.0, 3.0], [4.0, 5.0]];
//! let kernel_entropy = Entropy::nd_kernel::<2>(continuous, 1.0).global_value();
//! ```
//!
//! # Features
//!
//! | Measure | Discrete | Kernel | Ordinal | Exp. Family | Notes |
//! |---------|----------|--------|---------|-------------|-------|
//! | Entropy | ✅ | ✅ | ✅ | ✅ | All variants |
//! | Joint Entropy | ✅ | ✅ | ✅ | ✅ | Via multi-variable estimators |
//! | Conditional Entropy | ✅ | ✅ | ✅ | ✅ | |
//! | Cross-Entropy | ✅[^1] | ✅ | ✅ | ✅ | All approaches |
//! | KLD | ❌ | ❌ | ❌ | ❌ | Planned |
//! | JSD | ❌ | ❌ | ❌ | ❌ | Planned |
//! | MI | ✅ | ✅ | ✅ | ✅ | All variants |
//! | CMI | ✅ | ✅ | ✅ | ✅ | Conditional MI |
//! | TE | ✅ | ✅ | ✅ | ✅ | Transfer Entropy |
//! | CTE | ✅ | ✅ | ✅ | ✅ | Conditional TE |
//!
//! ✅ = Implemented | ❌ = Not implemented
//!
//! [^1]: For discrete estimators, cross-entropy is only available for MLE, Miller-Madow, and Bayesian estimators. NSB, Chao-Shen, and Chao-Wang-Jost do not support cross-entropy due to theoretical inconsistencies in applying bias corrections to cross-entropy.
//!
//! # Estimation Approaches
//!
//! ## Discrete Estimation
//! Histogram-based probability estimation for categorical or binned data.
//! Supports 11+ bias-corrected estimators (MLE, Miller-Madow, NSB, etc.).
//!
//! ## Kernel Estimation
//! Non-parametric density estimation for continuous data using Box and Gaussian kernels.
//! Optional GPU acceleration for large datasets.
//!
//! ## Ordinal Pattern Analysis
//! Permutation pattern encoding for time series data, robust to amplitude variations.
//!
//! ## Exponential Family (k-NN)
//! Distance-based estimation using k-nearest neighbours for differential entropy.
//! Supports Rényi and Tsallis generalized entropies.
//!
//! # Architecture
//!
//! The library follows a four-layer architecture:
//!
//! 1. **Public API Layer**: Factory types (`Entropy`, `MutualInformation`, `TransferEntropy`)
//! 2. **Estimation Approaches**: Four independent strategies for different data types
//! 3. **Core Infrastructure**: Shared traits and data structures
//! 4. **Performance Layer**: Optional GPU acceleration and mathematical optimisations
//!
//! # Feature Flags
//!
//! - `gpu`: Enable GPU acceleration for kernel estimators
//! - `fast_exp`: Use fast exponential approximations (trades accuracy for speed)
//!
//! # Python Compatibility
//!
//! This crate is designed to be a high-performance Rust backend for the
//! [infomeasure](https://github.com/cbueth/infomeasure) Python package, maintaining
//! API parity while providing significant performance improvements.
//!
//! # Guides
//!
//! - **[Estimator Usage Guide](guide/estimator_usage)** - How to use this crate
//! - **[Estimator Selection Guide](guide/estimator_selection)** - Choosing the right estimator
//!
//! For more details on the theory behind these measures, see the Python package documentation.

pub mod estimators;
pub mod guide;

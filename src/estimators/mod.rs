// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Information-theoretic estimators for entropy, mutual information, and transfer entropy.
//!
//! This module provides the main entry points for creating and using different types
//! of estimators. All estimators implement common traits for extracting results:
//!
//! - [`GlobalValue`] – Provides [`.global_value()`](GlobalValue::global_value) method for scalar results
//! - [`LocalValues`] – Provides [`.local_values()`](LocalValues::local_values) method for per-sample contributions
//! - [`OptionalLocalValues`] – Fallible [local value extraction](OptionalLocalValues::local_values_opt)
//!
//! ## Overview
//!
//! The `infomeasure` crate provides a comprehensive set of estimators for fundamental
//! information-theoretic measures. These are organized into several high-level
//! facade modules:
//!
//! ### 1. [Entropy](entropy)
//! Measures the uncertainty or information content of a random variable.
//! - [Shannon Entropy](entropy) ($H(X)$)
//! - [Joint Entropy](entropy) ($H(X, Y)$)
//! - [Conditional Entropy](entropy) ($H(X|Y)$)
//! - [Cross-Entropy](traits::CrossEntropy) ($H_Q(P)$)
//!
//! ### 2. [Mutual Information](mutual_information)
//! Quantifies the statistical dependence between random variables.
//! - [Mutual Information](mutual_information) ($I(X; Y)$)
//! - [Conditional Mutual Information](mutual_information) ($I(X; Y | Z)$)
//!
//! ### 3. [Transfer Entropy](transfer_entropy)
//! A directional, model-free measure of information flow between time series.
//! - [Transfer Entropy](transfer_entropy) ($T_{X \to Y}$)
//! - [Conditional Transfer Entropy](transfer_entropy) ($T_{X \to Y | Z}$)
//!
//! ## Estimation Approaches
//!
//! Each measure can be estimated using different algorithmic approaches depending
//! on the data type (discrete vs. continuous) and desired properties:
//!
//! - **[Discrete](approaches::discrete)**: Histogram-based with bias correction (MLE, NSB, etc.)
//! - **[Exponential Family](approaches::expfam)**: kNN-based non-parametric (KSG, KL, Rényi, Tsallis)
//! - **[Kernel](approaches::kernel)**: Kernel Density Estimation (KDE)
//! - **[Ordinal](approaches::ordinal)**: Permutation pattern analysis
//!
//! For detailed guidance on choosing the right estimator, see the
//! [Estimator Selection Guide](crate::guide::estimator_selection).

pub mod approaches;
#[macro_use]
pub(crate) mod doc_macros;
pub mod entropy;
pub mod mutual_information;
pub mod traits;
pub mod transfer_entropy;
pub mod utils;

// Re-export commonly used types for external access
pub use approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
pub use approaches::expfam::ksg::KsgType;
pub use entropy::Entropy;
pub use traits::{CrossEntropy, GlobalValue, JointEntropy, LocalValues, OptionalLocalValues};

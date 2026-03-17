// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Introduction to Information-Theoretic Measures
//!
//! This crate provides high-performance implementations of information-theoretic measures
//! including entropy, mutual information, and transfer entropy.
//!
//! ## What Are Information-Theoretic Measures?
//!
//! Information theory, founded by Claude Shannon in 1948, provides mathematical tools
//! for quantifying information. Key measures include:
//!
//! - **Entropy $H(X)$**: Uncertainty or information content of a random variable
//! - **Mutual Information $I(X;Y)$**: Shared information between two variables
//! - **Transfer Entropy $T_{X \\to Y}$**: Directed information flow from X to Y
//! - **Conditional variants**: $H(X|Y)$, $I(X;Y|Z)$, $T_{X \\to Y|Z}$
//!
//! ## Why Use This Crate?
//!
//! - **Performance**: Written in Rust for maximum performance
//! - **GPU Support**: Optional GPU acceleration for kernel estimators
//! - **Type Safety**: Compile-time checked estimators
//! - **Multiple Approaches**: Discrete, kernel, ordinal, and k-NN based estimators
//!
//! ## Getting Started
//!
//! See the [Estimator Usage Guide](super::estimator_usage) for code examples
//! and the [Estimator Selection Guide](super::estimator_selection) to choose the right estimator.
//!
//! ## Features Implemented
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

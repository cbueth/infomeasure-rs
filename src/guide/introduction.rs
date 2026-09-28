// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Introduction to Information-Theoretic Measures
//!
//! This crate provides high-performance implementations of information-theoretic measures
//! including entropy, mutual information, transfer entropy, and the Kullback–Leibler and
//! Jensen–Shannon divergences.
//!
//! ## What Are Information-Theoretic Measures?
//!
//! Information theory, founded by Claude Shannon in 1948, provides mathematical tools
//! for quantifying information. Key measures include:
//!
//! - **Entropy $H(X)$**: Uncertainty or information content of a random variable
//! - **Mutual Information $I(X;Y)$**: Shared information between two variables
//! - **Transfer Entropy $T_{X \\to Y}$**: Directed information flow from X to Y
//! - **Kullback–Leibler Divergence $D_{\\mathrm{KL}}(P \\parallel Q)$**: Extra cost of encoding P with a code built for Q
//! - **Jensen–Shannon Divergence $JSD(P \\parallel Q)$**: Symmetric, bounded divergence between distributions
//! - **Conditional variants**: $H(X|Y)$, $I(X;Y|Z)$, $T_{X \\to Y|Z}$
//!
//! ## Why Use This Crate?
//!
//! - **Performance**: Written in Rust for maximum performance
//! - **Acceleration**: Optional GPU (wgpu) and CPU-parallel (rayon) backends for
//!   the kernel and k-NN (exponential-family) estimators
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
//! | **KLD** | ✅ | ✅ | ✅ | ✅ | Via cross-entropy |
//! | **JSD** | ✅[^2] | ✅ | ✅ | ❌ | Via pmf mixture / pooled KDE |
//! | MI | ✅ | ✅ | ✅ | ✅ | All variants |
//! | CMI | ✅ | ✅ | ✅ | ✅ | Conditional MI |
//! | TE | ✅ | ✅ | ✅ | ✅ | Transfer Entropy |
//! | CTE | ✅ | ✅ | ✅ | ✅ | Conditional TE |
//!
//! ✅ = Implemented | ⚠️ = Available via trait | ❌ = Not implemented
//!
//! [^1]: For discrete estimators, cross-entropy is only available for MLE, Miller-Madow, and Bayesian estimators. NSB, Chao-Shen, and Chao-Wang-Jost do not support cross-entropy due to theoretical inconsistencies in applying bias corrections to cross-entropy.
//!
//! [^2]: JSD requires a mixture distribution: it is available for estimators exposing a normalized pmf (discrete MLE, Bayes, shrinkage, ordinal) and for kernel via pooling the samples. Differential/generalized-entropy estimators (Kozachenko-Leonenko, Rényi, Tsallis) are mathematically excluded.

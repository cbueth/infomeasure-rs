// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Information-Theoretic Measures Guide
//!
//! This module provides comprehensive guides for using the `infomeasure` Rust crate.
//!
//! ## Philosophy: Rust vs Python
//!
//! Unlike the Python `infomeasure` package which uses a flexible string-based approach
//! selection (e.g., `approach="nsb"`), this Rust crate uses **Rust's type system** to
//! provide compile-time type safety and optimization opportunities.
//!
//! **In Python**, you might write:
//! ```python
//! im.entropy(data, approach="nsb")
//! ```
//!
//! **In Rust**, you specify the estimator type at compile time:
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let data = array![0, 1, 0, 1, 0, 1, 0, 1];
//! let entropy = Entropy::new_discrete(data).global_value();
//! ```
//!
//! This approach:
//! - Catches errors at compile time rather than runtime
//! - Enables compiler optimizations for each specific estimator type
//! - Provides better IDE support and documentation
//! - Makes the API more explicit and self-documenting
//!
//! ## Guide Contents
//!
//! ### Getting Started
//! - [Introduction](introduction) - Overview of information theory
//! - [Estimator Usage](estimator_usage) - How to use this crate
//! - [Estimator Selection](estimator_selection) - Choosing the right estimator
//! - [Macros](macros) - Convenience macros for estimators
//!
//! ### Entropy
//! - [Entropy Overview](entropy) - All entropy estimators
//!   - [Discrete](entropy::discrete) - Discrete entropy estimators
//!
//! ### Information Measures
//! - [Conditional Entropy](cond_entropy) - $H(X|Y)$
//! - [Cross-Entropy](cross_entropy) - $H_Q(P)$
//
//! ### Mutual Information
//! - [Conditional MI](cond_mi) - $I(X;Y|Z)$
//!
//! ### Transfer Entropy
//! - [Conditional TE](cond_te) - $T_{X \\to Y|Z}$
//!
//! ### Composite Measures (Planned)
//! - [KLD Guide](kld) - Kullback-Leibler Divergence
//! - [JSD Guide](jsd) - Jensen-Shannon Divergence
//!
//! ### Configuration
//! - [Settings](settings) - Configuration options
//! - [Statistical Tests](statistical_tests) - Hypothesis testing
//!
//! ## Quick Links
//!
//! - **[Crate Root](../index.html)** - Main crate documentation
//! - **[GitHub Repository](https://codeberg.org/cbueth/infomeasure-rs)** - Source code
//! - **[Python Package](https://github.com/cbueth/infomeasure)** - For when you need runtime flexibility

pub mod cond_entropy;
pub mod cond_mi;
pub mod cond_te;
pub mod cross_entropy;
pub mod entropy;
pub mod estimator_selection;
pub mod estimator_usage;
pub mod introduction;
pub mod jsd;
pub mod kld;
pub mod macros;
pub mod settings;
pub mod statistical_tests;

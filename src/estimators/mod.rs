// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Information-theoretic estimators for entropy, mutual information, and transfer entropy.
//!
//! This module provides the main entry points for creating and using different types
//! of estimators. All estimators implement common traits for extracting results:
//!
//! - [`GlobalValue`] - Provides `.global_value()` method for scalar results
//! - [`LocalValues`] - Provides `.local_values()` method for per-sample contributions
//! - [`OptionalLocalValues`] - Fallible local value extraction
//!
//! # Usage Pattern
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::{GlobalValue, LocalValues};
//! use ndarray::array;
//!
//! // Create estimator
//! let data = array!(1, 2, 1, 3, 2, 1);
//! let estimator = Entropy::new_discrete(data);
//!
//! // Extract results
//! let global = estimator.global_value();
//! let local = estimator.local_values();
//! ```

pub mod approaches;
pub mod entropy;
pub mod mutual_information;
pub mod traits;
pub mod transfer_entropy;
pub mod utils;

pub use entropy::Entropy;
pub use traits::{CrossEntropy, GlobalValue, JointEntropy, LocalValues, OptionalLocalValues};

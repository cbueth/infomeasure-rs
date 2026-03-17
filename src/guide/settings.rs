// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Settings and Configuration
//!
//! **Note**: This is a placeholder page. Global configuration is not yet implemented.
//!
//! ## Current Status
//!
//! The Rust crate does not yet have a global configuration system like the Python package.
//! Each estimator is configured through its constructor parameters.
//!
//! ## Planned Features
//!
//! Future versions will include:
//! - Global logarithmic base setting
//! - Default statistical test parameters
//! - Logging configuration
//!
//! ## Current Workarounds
//!
//! For now, configure each estimator individually through its constructor:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // Each estimator is configured through its constructor
//! let data = array![0, 1, 0, 1, 0, 1];
//! let h = Entropy::new_discrete(data).global_value();
//! ```
//!
//! ## Python Equivalent
//!
//! In the Python package, you can use:
//! ```python
//! import infomeasure as im
//! im.Config.set_logarithmic_unit("bits")
//! im.entropy(data, approach="discrete")  # Uses bits
//! ```
//!
//! This functionality will be added to the Rust crate in a future release.

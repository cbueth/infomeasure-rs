// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Statistical Tests
//! **Note**: This is a placeholder page. Built-in statistical testing is not yet implemented.
//! ## Overview
//! Statistical tests help assess the significance of observed information-theoretic measures.
//! They answer: "Is the observed value significantly different from random?"
//! ## Current Status
//! The Rust crate does not yet include built-in statistical testing. To perform hypothesis
//! testing, you can:
//! 1. **Implement your own**: Use permutation or bootstrap tests
//! 2. **Use the Python package**: For full statistical testing support
//! ## Planned Implementation
//! Future versions will include:
//! - Permutation tests
//! - Bootstrap tests
//! - p-value computation
//! - Confidence intervals
//! - Effective Transfer Entropy (eTE)
//! ## Manual Implementation Example
//! For now, you can implement permutation testing manually:
//! ```text
//! 1. Compute observed MI/TE value
//! 2. Shuffle one variable multiple times
//! 3. Compute MI/TE for each shuffle
//! 4. Calculate p-value as proportion of shuffled values >= observed
//! ```
//! ## Python Equivalent
//! In the Python package:
//! ```python
//! est = im.estimator(x, y, measure="mi", approach="discrete")
//! result = est.statistical_test(n_tests=100, method="permutation_test")
//! print(result.p_value)  # p-value for significance
//! ```
//! ## References
//! - [Schreiber, 2000](../../guide/references/index.html#schreiber2000)
//! - [Lizier, 2014](../../guide/references/index.html#lizier2014)

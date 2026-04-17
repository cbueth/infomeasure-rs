// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Cross-Entropy $H_Q(P)$
//!
//! Cross-entropy measures the information required when using an encoding based on
//! distribution $Q$ to represent data that actually follows distribution $P$.
//!
//! ## Definition
//!
//! For discrete random variables $P$ and $Q$ over the same set $\mathcal{X}$:
//!
//! $$H_Q(P) = -\sum_{x \in \mathcal{X}} P(x) \log Q(x)$$
//!
//! For continuous variables, it is defined using the differential entropy form:
//!
//! $$H_Q(P) = -\int P(x) \log Q(x) \, dx$$
//!
//! ## Relationship to Other Measures
//!
//! - **Entropy**: $H(P) = H_P(P)$ (using P to encode itself)
//! - **KLD**: $D_{KL}(P \parallel Q) = H_Q(P) - H(P)$
//!
//! Cross-entropy always satisfies $H_Q(P) \ge H(P)$, with equality if and only if $P = Q$.
//!
//! ## Implemented in This Crate
//!
//! Cross-entropy is available via the [`CrossEntropy`](crate::estimators::traits::CrossEntropy) trait.
//! It is supported by:
//!
//! - [`RenyiEntropy`](crate::estimators::approaches::expfam::renyi::RenyiEntropy)
//! - [`TsallisEntropy`](crate::estimators::approaches::expfam::tsallis::TsallisEntropy)
//! - [`OrdinalEntropy`](crate::estimators::approaches::ordinal::ordinal_estimator::OrdinalEntropy)
//!
//! ## Example
//!
//! ```rust
//! use infomeasure::estimators::approaches::expfam::RenyiEntropy;
//! use infomeasure::estimators::traits::CrossEntropy;
//! use ndarray::array;
//!
//! let p = RenyiEntropy::new_1d(array![0.1, 0.2, 0.3], 3, 1.0, 0.0);
//! let q = RenyiEntropy::new_1d(array![0.15, 0.25, 0.35], 3, 1.0, 0.0);
//!
//! let ce = p.cross_entropy(&q);
//! assert!(ce >= 0.0);
//! ```
//!
//! ## See Also
//!
//! - [Entropy Guide](super::entropy) — Base entropy
//! - [KLD Guide](super::kld) - $D_{KL}(P \parallel Q) = H_Q(P) - H(P)$
//! - [JSD Guide](super::jsd) - Symmetric divergence measure

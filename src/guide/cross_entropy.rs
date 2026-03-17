// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Cross-Entropy $H_Q(P)$
//!
//! **Note**: This is a placeholder page.
//!
//! ## Overview
//!
//! Cross-entropy measures the information when using distribution Q to encode distribution P:
//!
//! $$H_Q(P) = -\\sum_x P(x) \\log Q(x)$$
//!
//! ## Relationship to Other Measures
//!
//! - **Entropy**: $H(P) = H_P(P)$ (using P to encode itself)
//! - **KLD**: $D_{KL}(P||Q) = H_Q(P) - H(P)$
//!
//! ## Current Status
//!
//! Cross-entropy is partially available through the [`OrdinalEntropy::cross_entropy`](crate::estimators::approaches::ordinal::ordinal_estimator::OrdinalEntropy::cross_entropy) method
//! for ordinal estimators.
//!
//! ## Planned Implementation
//!
//! Cross-entropy support will be extended to more estimators in future versions.
//!
//! ## See Also
//!
//! - [Entropy Guide](super::entropy) - Base entropy
//! - [KLD Guide](super::kld) - Kullback-Leibler Divergence

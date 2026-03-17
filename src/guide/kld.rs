// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Kullback-Leibler Divergence (KLD)
//!
//! **Note**: This is a placeholder page. KLD support is planned for a future release.
//!
//! ## Overview
//!
//! The Kullback-Leibler divergence (also known as relative entropy) measures
//! the information lost when distribution Q is used to approximate distribution P:
//!
//! $$D_{KL}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = H_Q(P) - H(P)$$
//!
//! where $H_Q(P)$ is the cross-entropy and $H(P)$ is the entropy of P.
//!
//! ## Current Status
//!
//! KLD computation is not yet directly exposed in the Rust API. However, you can
//! compute it manually using cross-entropy and entropy.
//!
//! ```text
//! KLD(P||Q) = H_Q(P) - H(P)
//! Compute H(P) and H_Q(P) separately, then subtract
//! ```
//!
//! ## Planned Implementation
//!
//! Future versions will include:
//! - Direct KLD computation via facade types
//! - Support for all entropy estimators (discrete, kernel, ordinal, KL, Rényi, Tsallis)
//! - Weighted KLD for multiple distributions
//!
//! ## Related Measures
//!
//! - [Entropy Guide](super::entropy) - Base entropy computation
//! - [Cross-Entropy Guide](super::cross_entropy) - $H_Q(P)$
//! - [JSD Guide](super::jsd) - Jensen-Shannon Divergence (symmetric)
//! - [Mutual Information](super::mutual_information) - $I(X;Y) = D_{KL}(p(x,y) || p(x)p(y))$
//!
//! ## References
//!
//! - Cover, T. M., & Thomas, J. A. (2012). Elements of Information Theory
//! - Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency

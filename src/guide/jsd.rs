// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Jensen-Shannon Divergence (JSD)
//!
//! **Note**: This is a placeholder page. JSD support is planned for a future release.
//!
//! ## Overview
//!
//! The Jensen-Shannon divergence is a symmetric measure of similarity between
//! two probability distributions:
//!
//! $$JSD(P || Q) = \frac{1}{2} D_{KL}(P || M) + \frac{1}{2} D_{KL}(Q || M)$$
//!
//! where $M = \frac{1}{2}(P + Q)$ is the average (mixture) distribution.
//!
//! Equivalently:
//!
//! $$JSD(P || Q) = H\left(\frac{P + Q}{2}\right) - \frac{1}{2}H(P) - \frac{1}{2}H(Q)$$
//!
//! ## Properties
//!
//! - **Symmetric**: $JSD(P||Q) = JSD(Q||P)$
//! - **Bounded**: $0 \leq JSD(P||Q) \leq \log_2(2) = 1$ for base-2 logarithm
//! - **Square root is a metric**: $\sqrt{JSD}$ satisfies the triangle inequality
//!
//! ## Current Status
//!
//! JSD computation is not yet directly exposed in the Rust API. However, you can
//! compute it manually using entropy.
//!
//! ## Planned Implementation
//!
//! Future versions will include:
//! - Direct JSD computation via facade types
//! - Generalized JSD for multiple distributions with custom weights
//! - Support for all entropy estimators
//!
//! ## Related Measures
//!
//! - [Entropy Guide](super::entropy) - Base entropy computation
//! - [KLD Guide](super::kld) - Kullback-Leibler Divergence (asymmetric)
//! - [Cross-Entropy Guide](super::cross_entropy) - Cross-entropy $H_Q(P)$
//!
//! ## References
//!
//! - Lin, J. (1991). Divergence measures based on the Shannon entropy
//! - Endres, D. M., & Schindelin, J. E. (2003). A new metric for probability distributions

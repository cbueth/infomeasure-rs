// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Kullback-Leibler Divergence (KLD)
//!
//! The Kullback-Leibler divergence (also known as relative entropy) measures
//! the difference between two probability distributions $P$ and $Q$. It
//! represents the information lost when distribution $Q$ is used to approximate $P$.
//!
//! ## Theory
//!
//! For discrete random variables $P$ and $Q$:
//!
//! $$D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}$$
//!
//! This can be expressed in terms of cross-entropy $H_Q(P)$ and Shannon entropy $H(P)$:
//!
//! $$D_{KL}(P \parallel Q) = H_Q(P) - H(P)$$
//!
//! For continuous variables, it is defined as:
//!
//! $$D_{KL}(P \parallel Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx$$
//!
//! ## Interpretation
//!
//! KLD represents the "extra effort" or "surprise" when using an encoding based
//! on $Q$ instead of the true distribution $P$. Unlike distance metrics, KLD is
//! **asymmetric**: $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$.
//!
//! ## Implementation Status
//!
//! KLD is not yet directly implemented in this crate. However, it can be computed
//! using the relationship $D_{KL}(P \parallel Q) = H_Q(P) - H(P)$ for estimators
//! that support cross-entropy via the [`CrossEntropy`](crate::estimators::traits::CrossEntropy) trait.
//!
//! ## See Also
//!
//! - [Entropy Guide](super::entropy) — Base entropy
//! - [Cross-Entropy Guide](super::cross_entropy) — Total encoding cost
//! - [JSD Guide](super::jsd) - Symmetric divergence measure
//!
//! ## References
//!
//! - [Kullback & Leibler, 1951](../../guide/references/index.html#kullback1951)
//! - [Cover & Thomas, 2012](../../guide/references/index.html#cover2012elements)

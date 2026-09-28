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
//! $$D_{\mathrm{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}$$
//!
//! This can be expressed in terms of cross-entropy $H_Q(P)$ and Shannon entropy $H(P)$:
//!
//! $$D_{\mathrm{KL}}(P \parallel Q) = H_Q(P) - H(P)$$
//!
//! For continuous variables, it is defined as:
//!
//! $$D_{\mathrm{KL}}(P \parallel Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx$$
//!
//! ## Interpretation
//!
//! KLD represents the "extra effort" or "surprise" when using an encoding based
//! on $Q$ instead of the true distribution $P$. Unlike distance metrics, KLD is
//! **asymmetric**: $D_{\mathrm{KL}}(P \parallel Q) \neq D_{\mathrm{KL}}(Q \parallel P)$.
//!
//! ## Implementation
//!
//! KLD is available for every estimator that supports cross-entropy via the
//! [`CrossEntropy`](crate::estimators::traits::CrossEntropy) trait, through the
//! [`Kld`](crate::estimators::composite_measures::Kld) extension trait and the
//! [`kld`](crate::estimators::composite_measures::kld) function. This covers the
//! discrete MLE, Miller–Madow, Bayes, kernel, ordinal, Kozachenko–Leonenko,
//! Rényi and Tsallis approaches — exactly the estimators where cross-entropy is
//! mathematically sound. Bias-corrected discrete variants (NSB, Chao–Shen,
//! Chao–Wang–Jost, Grassberger, ANSB, Zhang, Bonachela) and shrinkage do not
//! implement cross-entropy and are therefore not available.
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::composite_measures::Kld;
//! use ndarray::array;
//!
//! let p = Entropy::new_discrete(array![1, 1, 1, 2, 2, 3, 4, 5]);
//! let q = Entropy::new_discrete(array![1, 1, 2, 2, 2, 3, 3, 5]);
//! let d_kl = p.kld(&q); // asymmetric: H_Q(P) - H(P)
//! assert!(d_kl.is_finite());
//! ```
//!
//! ## See Also
//!
//! - [Entropy Guide](super::entropy) — Base entropy
//! - [Cross-Entropy Guide](super::cross_entropy) — Total encoding cost
//! - [JSD Guide](super::jsd) — Symmetric divergence measure
//!
//! ## References
//!
//! - [Kullback & Leibler, 1951](super::references#kullback1951)
//! - [Cover & Thomas, 2012](super::references#cover2012elements)

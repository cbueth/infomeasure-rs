// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Jensen-Shannon Divergence (JSD)
//! The Jensen-Shannon divergence (JSD) is a symmetric measure of similarity between
//! probability distributions. It is based on the Kullback-Leibler divergence but
//! addresses its lack of symmetry and potential for infinite values.
//! ## Theory
//! The JSD between two distributions $P$ and $Q$ is defined as the average KLD
//! between each distribution and their mixture $M = \frac{1}{2}(P + Q)$:
//! $$JSD(P \parallel Q) = \frac{1}{2} D_{\mathrm{KL}}(P \parallel M) + \frac{1}{2} D_{\mathrm{KL}}(Q \parallel M)$$
//! This can be expressed more simply in terms of Shannon entropy $H$:
//! $$JSD(P \parallel Q) = H\left(\frac{P + Q}{2}\right) - \frac{1}{2}H(P) - \frac{1}{2}H(Q)$$
//! ### Generalized JSD
//! For $n$ probability distributions $P_1, \ldots, P_n$ with weights $\pi_1, \ldots, \pi_n$
//! ($\sum \pi_i = 1$), the generalized JSD is:
//! $$JS_{\pi}(P_1, \ldots, P_n) = H\left( \sum_{i=1}^{n} \pi_i P_i \right) - \sum_{i=1}^{n} \pi_i H(P_i)$$
//! ## Properties
//! - **Symmetry**: $JSD(P \parallel Q) = JSD(Q \parallel P)$
//! - **Boundedness**: $0 \leq JSD \leq \log(n)$ (for $n$ distributions)
//! - **Metric Property**: The square root $\sqrt{JSD}$ is a true distance metric
//!   satisfying the triangle inequality [Endres & Schindelin, 2003](super::references#endres2003).
//! ## Implementation
//!
//! JSD is available for estimators that expose a normalized probability mass
//! function via [`ProbabilityMass`](crate::estimators::traits::ProbabilityMass)
//! — the discrete MLE, Bayes, shrinkage and ordinal approaches — through the
//! [`jsd`](crate::estimators::composite_measures::jsd) function. For continuous
//! data, [`jsd_kernel_1d`](crate::estimators::composite_measures::jsd_kernel_1d)
//! and [`jsd_kernel_nd`](crate::estimators::composite_measures::jsd_kernel_nd)
//! pool the samples into a single kernel density estimate.
//!
//! JSD is **not** a cross-entropy wrapper: it requires a well-defined mixture
//! distribution. Differential and generalized-entropy estimators
//! (Kozachenko–Leonenko, Rényi, Tsallis) therefore do not support it.
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::composite_measures::{jsd, jsd_kernel_1d};
//! use ndarray::array;
//!
//! // Discrete: symmetric, 0 <= JSD <= ln(n)
//! let p = Entropy::new_discrete(array![0, 0, 0, 1, 1, 2, 3, 4]);
//! let q = Entropy::new_discrete(array![0, 0, 1, 1, 1, 2, 2, 4]);
//! let d = jsd(&[p, q], None);
//! assert!(d >= 0.0);
//!
//! // Kernel: continuous data of the same dimension
//! let x = array![1.0, 2.0, 2.5, 4.0, 5.0];
//! let y = array![1.5, 2.2, 3.0, 4.4, 5.5];
//! assert!(jsd_kernel_1d(&[x, y], None, 1.0, "gaussian").is_finite());
//! ```
//!
//! ## See Also
//! - [Entropy Guide](super::entropy) — Base entropy computation
//! - [KLD Guide](super::kld) — Asymmetric divergence measure
//! - [Cross-Entropy Guide](super::cross_entropy) — Total encoding cost
//!
//! ## References
//! - [Cover & Thomas, 2012](super::references#cover2012elements)
//! - [Endres & Schindelin, 2003](super::references#endres2003)

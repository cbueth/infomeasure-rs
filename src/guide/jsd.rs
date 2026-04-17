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
//! $$JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$
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
//!   satisfying the triangle inequality [Endres & Schindelin, 2003](../../guide/references/index.html#endres2003).
//! ## Implementation Status
//! JSD is not yet directly implemented in this crate. It is planned for a future release.
//! ## See Also
//! - [Entropy Guide](super::entropy) — Base entropy computation
//! - [KLD Guide](super::kld) — Asymmetric divergence measure
//! - [Cross-Entropy Guide](super::cross_entropy) — Total encoding cost
//! ## References
//! - [Cover & Thomas, 2012](../../guide/references/index.html#cover2012elements)

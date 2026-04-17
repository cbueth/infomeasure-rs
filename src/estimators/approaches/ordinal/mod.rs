// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// Ordinal (permutation) estimators module
// This module contains the ordinal entropy estimator and its utilities.

//! # Ordinal (Permutation) Estimators
//!
//! This module implements ordinal entropy and derived measures, which quantify
//! the complexity of time series by analyzing permutation patterns.
//!
//! ## Theory
//!
//! Ordinal entropy (also known as Permutation Entropy) [Bandt & Pompe, 2002](../../../../guide/references/index.html#bandt2002)
//! transforms a continuous time series into a sequence of discrete symbols based
//! on the relative order of values in sliding windows.
//!
//! ### Symbolization
//!
//! The permutation pattern is determined by the ordinal comparison of values in the
//! time series [Laisant, 1888](../../../../guide/references/index.html#laisant1888).
//! The rank ordering approach was systematized for combinatorial applications
//! [Lehmer, 1960](../../../../guide/references/index.html#lehmer1960).
//!
//! Given a window of length $d$ (the `embedding_dim`), the values are replaced
//! by their rank order (permutation). For example, with $d=3$:
//! - $(1.2, 5.4, 2.1) \to (0, 2, 1)$
//! - $(9.1, 1.0, 4.5) \to (2, 0, 1)$
//!
//! There are $d!$ possible permutation patterns.
//!
//! ### Entropy Calculation
//!
//! After symbolization, the entropy is calculated using the standard Shannon
//! formula on the relative frequencies of the permutation patterns:
//!
//! $$H(X) = -\sum_{p \in \mathcal{S}_d} \hat{P}(p) \log \hat{P}(p)$$
//!
//! where $\mathcal{S}_d$ is the set of all $d!$ permutations and $\hat{P}(p)$ is
//! the empirical probability of permutation $p$.
//!
//! ## Advantages
//!
//! - **Robustness**: Invariant to any monotonic transformation of the data.
//! - **Simplicity**: No parameters to tune other than the embedding dimension $d$.
//! - **Dynamics**: Captures temporal structure that is lost in standard histogram methods.
//!
//! ## Measures Implemented
//!
//! - **Ordinal Entropy**: $H_{ord}(X)$
//! - **Ordinal Mutual Information**: $I_{ord}(X; Y) = H_{ord}(X) + H_{ord}(Y) - H_{ord}(X, Y)$
//! - **Ordinal Transfer Entropy**: $T_{ord}(X \to Y)$
//!
//! ## See Also
//! - [Entropy Guide](crate::guide::entropy) — Conceptual background
//! - [Estimator Approaches](super) — Overview of all estimation techniques
//!
//! ## References
//!
//! - [Bandt & Pompe, 2002](../../../../guide/references/index.html#bandt2002)
//! - [Staniek & Lehnertz, 2008](../../../../guide/references/index.html#staniek2008)

pub mod ordinal_estimator;
pub mod ordinal_utils;

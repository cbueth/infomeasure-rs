// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Conditional Transfer Entropy $T_{X \\to Y|Z}$
//!
//! Conditional Transfer Entropy (CTE) measures directed information flow from X to Y
//! while controlling for Z. It corresponds to the amount of uncertainty reduced in the
//! future values of target $Y$ by knowing the past values of source $X$, $Z$ and also
//! after considering the past values of target $Y$ itself.
//!
//! CTE is useful to eliminate the influence of other possible information sources $Z$
//! from being mistaken as that of the source $X$.
//!
//! ## Definition
//!
//! $$TE(X \\to Y \\mid Z) = -\\sum_{y_{n+1}, \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}, \\mathbf{z}_n^{(m)}}
//! p(y_{n+1}, \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}, \\mathbf{z}_n^{(m)})
//! \\log \\left( \\frac{p(y_{n+1} \\mid \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}, \\mathbf{z}_n^{(m)})}
//! {p(y_{n+1} \\mid \\mathbf{y}_n^{(l)}, \\mathbf{z}_n^{(m)})} \\right)$$
//!
//! where:
//! - $p(\\cdot)$ represents the probability distribution,
//! - $\\mathbf{y}_n^{(l)}$ represents the past history of $Y$ with embedding length $l$,
//! - $\\mathbf{x}_n^{(k)}$ represents the past history of $X$ with embedding length $k$,
//! - $\\mathbf{z}_n^{(m)}$ represents the past history of $Z$ with embedding length $m$,
//! - $y_{n+1}$ is the future state of $Y$.
//!
//! ## Local Conditional Transfer Entropy
//!
//! Similar to local TE and local CMI measures, we can extract the **local or point-wise conditional TE**:
//!
//! $$t_{X \\to Y \\mid Z}(n+1, k, l) = -\\log \\left( \\frac{p(y_{n+1} \\mid \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}, \\mathbf{z}_n)}
//! {p(y_{n+1} \\mid \\mathbf{y}_n^{(l)}, \\mathbf{z}_n)} \\right)$$
//!
//! The CTE can be written as the average of local CTE:
//!
//! $$T_{X \\to Y \\mid Z}(k, l) = \\langle t_{X \\to Y}(n + 1, k, l) \\rangle$$
//!
//! In this crate, local CTE can be accessed via the [`LocalValues`](crate::estimators::traits::LocalValues) trait
//! on CTE estimators that support it.
//!
//! ## Entropy Combination Form
//!
//! The CTE expression can be written as the combination of entropies and joint entropies:
//!
//! $$TE(X \\to Y \\mid Z) =$$
//! $$H(y_{n+1}, \\mathbf{y}_n^{(l)}, \\mathbf{z}_n^{(m)}) - H(\\mathbf{y}_n^{(l)}, \\mathbf{z}_n^{(m)})$$
//! $$- H(y_{n+1}, \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}, \\mathbf{z}_n^{(m)}) + H(\\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}, \\mathbf{z}_n^{(m)})$$
//!
//! This form is used internally for Rényi and Tsallis CTE estimators.
//!
//! ## Implemented in This Crate
//!
//! CTE is available through the [`TransferEntropy`](crate::estimators::transfer_entropy::TransferEntropy) facade type:
//!
//! - [`TransferEntropy::new_cte_discrete_mle`](crate::estimators::transfer_entropy::TransferEntropy::new_cte_discrete_mle)
//! - [`TransferEntropy::new_cte_kernel`](crate::estimators::transfer_entropy::TransferEntropy::new_cte_kernel)
//! - And more estimators
//!
//! ## Example
//!
//! ```rust
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let source = array![0, 1, 0, 1, 0, 1, 0, 1];
//! let dest = array![0, 0, 1, 0, 1, 0, 1, 0];
//! let cond = array![0, 0, 0, 1, 0, 1, 1, 1];
//!
//! let cte = TransferEntropy::new_cte_discrete_mle(
//!     &source, &dest, &cond,
//!     1, // source history
//!     1, // dest history
//!     1, // condition history
//!     1  // step size
//! ).global_value();
//! ```
//!
//! ## See Also
//!
//! - [Estimator Usage Guide](super::estimator_usage) - Base TE
//! - [Transfer Entropy Guide](super::transfer_entropy) - Unconditional TE
//! - [Conditional MI Guide](super::cond_mi) - CMI with conditioning

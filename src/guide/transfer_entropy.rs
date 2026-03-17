// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Transfer Entropy $T_{X \\to Y}$
//!
//! Transfer entropy (TE) from the source process $X$ to the target process $Y$ is the amount of
//! uncertainty reduced in the future values of target $Y$ by knowing the past values of source $X$,
//! after considering the past values of the target $Y$.
//! It is the model-free and directional measure between two processes.
//!
//! ## Definition
//!
//! Let $X(x_n)$ and $Y(y_n)$ be two time series processes as source and target variables.
//! Then $T_{X \\to Y}$ from source to target is written as:
//!
//! $$T_{x \\to y}(k, l) = -\\sum_{y_{n+1}, \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}}
//! p(y_{n+1}, \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)})
//! \\log \\left( \\frac{p(y_{n+1} \\mid \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)})}
//! {p(y_{n+1} \\mid \\mathbf{y}_n^{(l)})} \\right)$$
//!
//! where:
//! - $y_{n+1}$ is the next state of $Y$ at time $n$,
//! - $\\mathbf{y}_n^{(l)} = \\{y_n, \\dots, y_{n-l+1}\\}$ is the embedding vector of $Y$ with history length $l$,
//! - $\\mathbf{x}_n^{(k)} = \\{x_n, \\dots, x_{n-k+1}\\}$ is the embedding vector of $X$ with history length $k$.
//!
//! ## Transfer Entropy with Lag
//!
//! One can add a source-to-destination time propagation or lag $u$:
//!
//! $$T_{x \\to y}(k, l, u) = -\\sum_{y_{n+1+u}, \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}}
//! p(y_{n+1+u}, \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)})
//! \\log \\left( \\frac{p(y_{n+1+u} \\mid \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)})}
//! {p(y_{n+1+u} \\mid \\mathbf{y}_n^{(l)})} \\right)$$
//!
//! ## Local Transfer Entropy
//!
//! Similar to local entropy and MI, we can extract the **local or point-wise transfer entropy**.
//! It is the amount of information transfer attributed to the specific realisation
//! $(x_{n+1}, \\mathbf{X}_n^{(k)}, \\mathbf{Y}_n^{(l)})$ at time step $n+1$:
//!
//! $$t_{X \\to Y}(n+1, k, l) = -\\log \\left(\\frac{p(y_{n+1} \\mid \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)})}
//! {p(y_{n+1} \\mid \\mathbf{y}_n^{(l)})} \\right)$$
//!
//! The TE can be written as the global average of the local TE:
//!
//! $$T_{X \\to Y}(k, l, u) = \\langle t_{X \\to Y}(n + 1, k, l) \\rangle$$
//!
//! Local TE values can be negative, unlike its global counterpart; this means the source
//! is misleading about the prediction of target's next step.
//!
//! In this crate, local TE can be accessed via the [`LocalValues`](crate::estimators::traits::LocalValues) trait
//! on TE estimators that support it.
//!
//! ## Effective Transfer Entropy (eTE)
//!
//! Real world time series data are usually biased due to finite size effects.
//! Effective TE is defined as the difference between the original TE and TE calculated on
//! surrogate (randomly shuffled) data:
//!
//! $$\\operatorname{eTE} = \\operatorname{T}_{X \\to Y} - \\operatorname{T}_{X_{\\text{shuffled}} \\to Y}$$
//!
//! This helps correct for bias from finite sample sizes.
//!
//! ## Implemented in This Crate
//!
//! TE is available through the [`TransferEntropy`](crate::estimators::transfer_entropy::TransferEntropy) facade type:
//!
//! - [`TransferEntropy::new_discrete_mle`](crate::estimators::transfer_entropy::TransferEntropy::new_discrete_mle)
//! - [`TransferEntropy::new_kernel`](crate::estimators::transfer_entropy::TransferEntropy::new_kernel)
//! - [`TransferEntropy::new_ksg`](crate::estimators::transfer_entropy::TransferEntropy::new_ksg)
//! - [`TransferEntropy::new_ordinal`](crate::estimators::transfer_entropy::TransferEntropy::new_ordinal)
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
//!
//! let te = TransferEntropy::new_discrete_mle(
//!     &source, &dest,
//!     1, // source history length
//!     1, // dest history length
//!     1  // step size
//! ).global_value();
//! ```
//!
//! ## See Also
//!
//! - [Estimator Usage Guide](super::estimator_usage) - Detailed usage examples
//! - [Conditional TE Guide](super::cond_te) - TE with conditioning
//! - [Macros Guide](super::macros) - Convenience macros for TE estimators

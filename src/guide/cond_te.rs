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
//! ## Why Condition on Z?
//!
//! Consider a scenario where you observe two time series $X$ and $Y$ and want to determine
//! if $X$ causes $Y$. Without conditioning, you might measure $T_{X \to Y}$ and find a
//! positive value—but this could be spurious if both $X$ and $Y$ are driven by a common
//! source $Z$:
//!
//! ```text
//!     Z (common driver)
//!    /   \
//!   X     Y
//! ```
//!
//! In this configuration:
//! - $T_{X \to Y}$ could be non-zero due to the common driver $Z$
//! - $T_{X \to Y \mid Z}$ removes this spurious influence
//!
//! This makes CTE essential for:
//! - **Granger causality** testing with multiple variables
//! - **Causal discovery** in networks with confounders
//! - **Brain connectivity** analysis where common drives must be accounted for
//!
//! ## CTE as Conditional MI
//!
//! ## Definition
//!
//! $$TE(X \\to Y \\mid Z) = -\\sum_{y_{n+1}, \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}, \\mathbf{z}_n^{(m)}} p(y_{n+1}, \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}, \\mathbf{z}_n^{(m)}) \\log \\left( \\frac{p(y_{n+1} \\mid \\mathbf{y}_n^{(l)}, \\mathbf{x}_n^{(k)}, \\mathbf{z}_n^{(m)})} {p(y_{n+1} \\mid \\mathbf{y}_n^{(l)}, \\mathbf{z}_n^{(m)})} \\right)$$
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
//! ### Continuous CTE: Kraskov-Stögbauer-Grassberger (KSG)
//! The KSG method [Kraskov et al., 2004](../../guide/references/index.html#ksg2004) can be extended to conditional transfer entropy
//! [Baboukani et al., 2020](../../guide/references/index.html#baboukani2020):
//! $$TE(X \to Y \mid Z) = \psi(k) + \langle \psi(n_{Y_{past}, Z_{past}} + 1) - \psi(n_{Y_{future}, Y_{past}, Z_{past}} + 1) - \psi(n_{X_{past}, Y_{past}, Z_{past}} + 1) \rangle$$
//! where $n$ refers to neighbor counts in the respective joint subspaces.
//! See the [KSG Approach Module](crate::estimators::approaches::expfam::ksg) for technical details.
//! ```rust
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! let x = array![0.1, 0.2, 0.3];
//! let y = array![0.15, 0.25, 0.35];
//! let z = array![0.1, 0.2, 0.3];
//! let cte = TransferEntropy::new_cte_ksg(&x, &y, &z, 3, 1e-10).global_value();
//! assert!(cte >= 0.0);
//! ```
//! ### Other Estimators
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
//! assert!(cte >= 0.0);
//! ```
//!
//! ### Practical Example: Conditioning on Common Driver
//!
//! In this example, $Z$ drives both $X$ and $Y$. We measure TE from $X$ to $Y$ (which may
//! appear causal due to $Z$) and then condition on $Z$ to remove this spurious influence:
//!
//! ```rust
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // Z drives both X and Y, but X and Y are conditionally independent given Z
//! // Z: 0,1,0,1,0,1,0,1,0,1
//! // X = Z (copy)
//! // Y = 1-Z (inverse - still driven by Z)
//! let z = array![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
//! let x = array![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
//! let y = array![1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
//!
//! // Unconditional TE from X to Y appears non-zero (spurious!)
//! let te_spurious = TransferEntropy::new_discrete_mle(&x, &y, 1, 1, 1).global_value();
//! assert!(te_spurious >= 0.0);
//!
//! // Conditional TE conditioning on Z removes the spurious link
//! let te_conditional = TransferEntropy::new_cte_discrete_mle(
//!     &x, &y, &z,
//!     1, 1, 1, 1
//! ).global_value();
//! assert!(te_conditional >= 0.0);
//! // After conditioning on common driver, CTE should be smaller
//! assert!(te_conditional <= te_spurious);
//! ```
//!
//! ## See Also
//!
//! - [Mutual Information](super::mutual_information) - Base MI
//! - [Conditional MI](super::cond_mi) - CMI (general case)
//! - [Transfer Entropy Guide](super::transfer_entropy) - Unconditional TE
//! - [Estimator Usage Guide](super::estimator_usage) - Detailed usage examples

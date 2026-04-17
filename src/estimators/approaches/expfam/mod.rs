// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// Exponential-family (kNN-based) estimators and shared utilities
// This module groups shared utilities and specific estimators like Rényi and Tsallis.

//! # Exponential-Family (kNN-based) Estimators
//!
//! This module provides non-parametric estimators for information measures based on
//! k-nearest neighbor (kNN) distances. These estimators are part of the
//! exponential-family formulation of information measures [Leonenko et al., 2008](../../../../guide/references/index.html#leonenko2008).
//!
//! ## Overview
//!
//! The kNN-based approaches are particularly effective for continuous variables in
//! higher dimensions. They exploit the relationship between the distance to the
//! $k$-th neighbor and the local probability density:
//!
//! $$f(x_i) \approx \frac{k}{N V_m \rho_{i,k}^m}$$
//!
//! By substituting this local density estimate into the standard entropy and
//! mutual information formulas, we obtain estimators that are non-parametric
//! and often asymptotically unbiased.
//!
//! ## Estimators
//!
//! - [**KSG**](crate::estimators::approaches::expfam::ksg): Specifically designed to cancel out bias when computing Mutual Information
//!   and Transfer Entropy. It uses different neighbour counting rules in marginal spaces.
//! - [**Kozachenko-Leonenko (KL)**](crate::estimators::approaches::expfam::kozachenko_leonenko): The standard kNN-based differential entropy estimator.
//! - [**Rényi**](crate::estimators::approaches::expfam::renyi) / [**Tsallis**](crate::estimators::approaches::expfam::tsallis): Generalized entropies that can capture different aspects
//!   of the distribution (e.g., tail behavior) via the $\alpha$ parameter.
//!
//! ## See Also
//! - [Estimator Approaches](super) — Overview of all estimation techniques
//!
//! ## Shared Utilities
//!
//! This module also contains shared utilities in [`utils`](crate::estimators::approaches::expfam::utils) for handling
//! kNN radii, volume of the unit ball calculation, and noise addition to avoid
//! identical values in datasets.
//!
//! ## References
//!
//! - [Leonenko et al., 2008](../../../../guide/references/index.html#leonenko2008)
//! - [Kraskov et al., 2004](../../../../guide/references/index.html#ksg2004)

pub mod kozachenko_leonenko;
pub mod ksg;
pub mod renyi;
pub mod tsallis;
pub mod utils;

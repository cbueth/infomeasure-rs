// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Discrete Entropy Estimators
//!
//! This page documents the discrete entropy estimators available in the crate.
//!
//! ## Available Estimators
//!
//! | Estimator | Constructor | Bias Correction | Local Values |
//! |-----------|-------------|-----------------|--------------|
//! | MLE | [`Entropy::new_discrete`](crate::estimators::entropy::Entropy::new_discrete) | None | ✅ |
//! | Miller-Madow | [`Entropy::new_miller_madow`](crate::estimators::entropy::Entropy::new_miller_madow) | $(K-1)/(2N)$ | ❌ |
//! | Shrinkage | [`Entropy::new_shrink`](crate::estimators::entropy::Entropy::new_shrink) | James-Stein | ✅ |
//! | Grassberger | [`Entropy::new_grassberger`](crate::estimators::entropy::Entropy::new_grassberger) | Digamma-based | ✅ |
//! | Zhang | [`Entropy::new_zhang`](crate::estimators::entropy::Entropy::new_zhang) | Cumulative product | ❌ |
//! | Bayes | [`Entropy::new_bayes`](crate::estimators::entropy::Entropy::new_bayes) | Dirichlet prior | ❌ |
//! | Bonachela | [`Entropy::new_bonachela`](crate::estimators::entropy::Entropy::new_bonachela) | Harmonic sum | ❌ |
//! | Chao-Shen | [`Entropy::new_chao_shen`](crate::estimators::entropy::Entropy::new_chao_shen) | Coverage-based | ❌ |
//! | Chao-Wang-Jost | [`Entropy::new_chao_wang_jost`](crate::estimators::entropy::Entropy::new_chao_wang_jost) | Coverage-based | ❌ |
//! | NSB | [`Entropy::new_nsb`](crate::estimators::entropy::Entropy::new_nsb) | Bayesian mixture | ❌ |
//! | ANSB | [`Entropy::new_ansb`](crate::estimators::entropy::Entropy::new_ansb) | Asymptotic NSB | ❌ |
//!
//! ## Quick Start
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let data = array![0, 1, 0, 1, 0, 1, 0, 1];
//!
//! // MLE (fastest)
//! let h_mle = Entropy::new_discrete(data.clone()).global_value();
//!
//! // Miller-Madow (simple bias correction)
//! let h_mm = Entropy::new_miller_madow(data.clone()).global_value();
//!
//! // Shrinkage (good for small samples)
//! let h_shrink = Entropy::new_shrink(data).global_value();
//! ```
//!
//! ## When to Use Which?
//!
//! - **Large samples (N > 1000)**: MLE or Miller-Madow
//! - **Small samples**: Shrinkage, NSB, or Chao-Shen
//! - **Correlated data**: NSB
//! - **Undersampled**: Chao-Shen, Chao-Wang-Jost, ANSB
//!
//! See the [Estimator Selection Guide](crate::guide::estimator_selection) for detailed recommendations.

// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Discrete Entropy Estimators
//!
//! This page documents the discrete entropy estimators available in the crate.
//!
//! ## Available Estimators
//!
//! ### Maximum Likelihood (MLE)
//! The standard plug-in estimator:
//! $$\hat{H}_{MLE} = -\sum \hat{p}_i \log \hat{p}_i$$
//! Fast but negatively biased for small samples.
//!
//! ### Miller-Madow
//! Corrects MLE by adding a term related to the number of bins $K$:
//! $$\hat{H}_{MM} = \hat{H}_{MLE} + \frac{K-1}{2N}$$
//!
//! ### Shrinkage (James-Stein)
//! Regularizes probabilities toward uniform distribution ($1/K$):
//! $$\hat{p}_i^{SHR} = \lambda (1/K) + (1-\lambda) \hat{p}_i^{ML}$$
//!
//! ### Grassberger
//! Uses the property that the expectation of the digamma function $\psi$ is related to entropy:
//! $$\hat{H}_{G} = \frac{1}{N} \sum n_i [\psi(N) - \psi(n_i)]$$
//!
//! ### Bonachela
//! Designed for very small samples:
//! $$\hat{H}_{B} = \frac{1}{N+2} \sum_{i=1}^{K} \left( (n_i + 1) \sum_{j=n_i + 2}^{N+2} \frac{1}{j} \right)$$
//!
//! ### Chao-Shen
//! Accounts for unobserved states using Horvitz-Thompson adjustment and coverage $C$:
//! $$\hat{H}_{CS} = - \sum \frac{C \hat{p}_i^{ML} \log (C \hat{p}_i^{ML})}{1 - (1 - C \hat{p}_i^{ML})^N}$$
//!
//! ### NSB (Nemenman-Shafee-Bialek)
//! Bayesian estimator that averages over Dirichlet priors to provide a flat prior over entropy.
//! Particularly robust for extremely undersampled data.
//!
//! ## Summary Table
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
//! use approx::assert_abs_diff_eq;
//!
//! let data = array![0, 1, 0, 1, 0, 1, 0, 1];
//!
//! // MLE (fastest): uniform binary has H = log(2) ≈ 0.693147 nats
//! let h_mle = Entropy::new_discrete(data.clone()).global_value();
//! assert_abs_diff_eq!(h_mle, 0.693147, epsilon = 1e-4);
//!
//! // Miller-Madow (simple bias correction): slightly higher than MLE
//! let h_mm = Entropy::new_miller_madow(data.clone()).global_value();
//! assert!(h_mm > h_mle);
//!
//! // Shrinkage (good for small samples): regularized estimate
//! let h_shrink = Entropy::new_shrink(data).global_value();
//! assert!(h_shrink >= 0.0);
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
//!
//! ## See Also
//!
//! - [Entropy Overview](super) - All entropy estimators
//! - [Mutual Information](super::super::mutual_information) - Uses entropy
//! - [Conditional Entropy](super::super::cond_entropy) - Uses joint entropy

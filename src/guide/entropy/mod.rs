// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Entropy Guide
//!
//! This module contains detailed guides for entropy estimation.
//!
//! ## Overview
//!
//! Entropy measures the uncertainty or information content of a random variable.
//! On the flip side, this uncertainty is nothing but the lack of information.
//! The larger the information required to accurately predict the state of the random variable,
//! the higher is the uncertainty we initially had about it.
//!
//! ### Hartley Entropy (Equiprobable Case)
//!
//! For an unknown outcome $x$ from a set of $N$ equiprobable elements, the information is:
//!
//! $$H(x) = \\log_2(N)$$
//!
//! ### Shannon Entropy
//!
//! Claude Shannon developed a mathematical measure to quantify the amount of information
//! produced by a source variable $X$. For a discrete random variable:
//!
//! $$H(X) = -\\sum_{x \\in X} p(x) \\log_b p(x)$$
//!
//! where:
//! - $X$: The set of possible values of the random variable.
//! - $p(x)$: The probability of the value $x$ occurring.
//! - $b$: The base of the logarithm.
//!   - If $b = 2$, the unit of information is "bit".
//!   - If $b = e$ (natural logarithm), the unit is "nat".
//!
//! This crate uses nats (base $e$) by default.
//!
//! ### Differential Entropy (Continuous Variable)
//!
//! For a continuous random variable $X$ with probability density function $p(x)$, the differential entropy is:
//!
//! $$H(X) = -\\int_{X} p(x) \\log_b p(x) \\, dx$$
//!
//! The differential entropy is closely related to Shannon entropy.
//!
//! ## Local Entropy
//!
//! The **local information** measure, also referred to as **point-wise** information,
//! characterizes the information associated with individual value points.
//! Applied to time series data, local information can uncover dynamic structures
//! that averaged measures overlook.
//!
//! The **local entropy** of an outcome $x$ is given by:
//!
//! $$h(x) = -\\log_b p(x)$$
//!
//! $h(x)$ is the information content attributed to the individual measurement $x$.
//! The global entropy $H(X)$ is the **average** or **expectation value** of the local information:
//!
//! $$H(X) = \\langle h(x) \\rangle$$
//!
//! In this crate, you can obtain local entropy values using the [`LocalValues`](crate::estimators::traits::LocalValues) trait.
//!
//! ## Generalized Entropies
//!
//! While Shannon entropy is the most common, there are generalizations useful for
//! complex systems, particularly where additivity does not hold.
//!
//! ### Rényi $\\alpha$-Entropy
//!
//! Rényi entropy is a generalized family of one-parameter entropy that preserves
//! additivity for independent systems. For $\\alpha > 0$:
//!
//! $$H_\\alpha[\\mathcal{P}] := \\frac{1}{1-\\alpha} \\log \\left( \\sum_{i=1}^{n} p_i^\\alpha \\right)$$
//! $$H_\\alpha[\\mathcal{P}] := \\frac{1}{1-\\alpha} \\log \\left( \\sum_{i=1}^{n} p_i^\\alpha \\right)$$
//!
//! where $\\mathcal{P} = (p_1, ..., p_n)$ is a probability distribution.
//! For $\\alpha = 1$, Rényi entropy reduces to Shannon entropy.
//! Small values of probabilities are emphasized for $\\alpha < 1$ and larger probabilities for $\\alpha > 1$.
//!
//! Use [`Entropy::new_renyi_1d`](crate::estimators::entropy::Entropy::new_renyi_1d) to create a Rényi entropy estimator.
//!
//! ### Tsallis Entropy
//!
//! Tsallis entropy (q-order entropy) is another generalization that modifies the additivity law:
//!
//! $$S_q = \\frac{1}{1 - q} \\left[ \\sum_{k=1}^{n} (p_k)^q - 1 \\right]$$
//!
//! where $q$ is a real parameter. In the $q \\to 1$ limit, Tsallis entropy reduces to Shannon entropy.
//! This class of entropy is particularly useful for studying long-range correlated systems
//! and non-equilibrium phenomena.
//!
//! Use [`Entropy::new_tsallis_1d`](crate::estimators::entropy::Entropy::new_tsallis_1d) to create a Tsallis entropy estimator.
//!
//! ## Estimator Types
//!
//! ### Discrete Estimators
//! Histogram-based estimation for categorical/integer data.
//! - [Discrete Estimators](discrete) - All discrete entropy estimators
//!
//! ### Continuous Estimators
//! Non-parametric estimation for real-valued data.
//! - Use [`Entropy::new_kernel`](crate::estimators::entropy::Entropy::new_kernel) for kernel estimation
//! - Use [`Entropy::new_kl_1d`](crate::estimators::entropy::Entropy::new_kl_1d) for Kozachenko-Leonenko
//! - Use [`Entropy::new_ordinal`](crate::estimators::entropy::Entropy::new_ordinal) for ordinal
//! - Use [`Entropy::new_renyi_1d`](crate::estimators::entropy::Entropy::new_renyi_1d) for Rényi
//! - Use [`Entropy::new_tsallis_1d`](crate::estimators::entropy::Entropy::new_tsallis_1d) for Tsallis
//!
//! ## Quick Example
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // Discrete entropy
//! let discrete_data = array![0, 1, 0, 1, 0, 1];
//! let h = Entropy::new_discrete(discrete_data).global_value();
//!
//! // Kernel entropy
//! let continuous_data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//! let h_kernel = Entropy::nd_kernel::<2>(continuous_data, 1.0).global_value();
//! ```
//!
//! ## See Also
//!
//! - [Discrete Entropy](discrete) - Histogram-based estimators
//! - [Mutual Information](super::mutual_information) - $I(X;Y) = H(X) + H(Y) - H(X,Y)$
//! - [Conditional Entropy](super::cond_entropy) - $H(X|Y) = H(X,Y) - H(Y)$
//! - [KLD](super::kld) - $D_{KL}(P||Q) = H_Q(P) - H(P)$
//! - [JSD](super::jsd) - $JSD = H((P+Q)/2) - 1/2H(P) - 1/2H(Q)$
//! - [Estimator Selection](super::estimator_selection) - Choosing estimators
//!
//! ## References
//!
//! - Shannon, C. E. (1948). A Mathematical Theory of Communication
//! - Cover, T. M., & Thomas, J. A. (2012). Elements of Information Theory
//! - Rényi, A. (1961). On measures of entropy and information
//! - Tsallis, C. (1988). Possible generalization of Boltzmann-Gibbs statistics

pub mod discrete;

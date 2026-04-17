// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Mutual Information $I(X;Y)$
//! Mutual Information (MI) quantifies the statistical dependence between two random
//! variables. It measures the average reduction in uncertainty about $X$ obtained by
//! knowing $Y$, or equivalently, the amount of information shared between the variables.
//! ## Motivation
//! Correlation measures linear relationships, but mutual information captures *any*
//! statistical dependency—whether linear or nonlinear. If $X$ and $Y$ are independent,
//! $I(X;Y) = 0$. If they are deterministically related, $I(X;Y) = H(X) = H(Y)$.
//! ## Definitions
//! ### Discrete Mutual Information
//! For discrete random variables $X$ and $Y$ with joint distribution $p(x,y)$ and
//! marginals $p(x), p(y)$:
//! $$I(X;Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log_b \frac{p(x,y)}{p(x)p(y)}$$
//! This crate uses natural logarithms (base $e$, "nats") by default.
//! ### Entropy Form
//! MI can be expressed in terms of entropies:
//! $$I(X;Y) = H(X) + H(Y) - H(X,Y)$$
//! $$= H(X) - H(X|Y)$$
//! $$= H(Y) - H(Y|X)$$
//! where $H(\cdot)$ is entropy and $H(\cdot|\cdot)$ is conditional entropy.
//! ### Differential Entropy (Continuous Variables)
//! For continuous variables, replace sums with integrals:
//! $$I(X;Y) = \int \int p(x,y) \log_b \frac{p(x,y)}{p(x)p(y)} \, dx \, dy$$
//! **Important**: Unlike discrete MI, differential MI is not invariant to
//! reparameterization. If you apply a nonlinear transformation to $X$ or $Y$,
//! the MI value changes. This is a fundamental property of differential entropy
//! that carries over to continuous MI.
//! ### Local (Pointwise) Mutual Information
//! Just as entropy has a local (pointwise) interpretation, so does MI:
//! $$i(x;y) = \log_b \frac{p(x,y)}{p(x)p(y)}$$
//! The global MI is the expectation (average) of local MI:
//! $$I(X;Y) = \langle i(x;y) \rangle$$
//! **Key property**: Unlike global MI (which is always non-negative),
//! local MI values can be **positive or negative**:
//! - $i(x;y) > 0$: $x$ and $y$ are more likely to co-occur than expected under independence
//! - $i(x;y) < 0$: $x$ and $y$ are less likely to co-occur than expected (misinformation)
//! - $i(x;y) = 0$: $x$ and $y$ are independent at this point
//!
//! When averaging over all samples, positive and negative values can cancel out,
//! which is why global MI remains non-negative despite local sign variations.
//!
//! In this crate, local MI is available via the [`LocalValues`](crate::estimators::traits::LocalValues)
//! trait on estimators that support it.
//!
//! ## Time-Lagged Mutual Information
//! For time series analysis, we often want to know if past values of one series
//! contain information about future values of another:
//! $$I(X_{t-u}; Y_t) = \sum_{x_{t-u}, y_t} p(x_{t-u}, y_t) \log_b \frac{p(x_{t-u}, y_t)}{p(x_{t-u})p(y_t)}$$
//! where $u$ is the **lag** (propagation time). This is closely related to
//! [Transfer Entropy](super::transfer_entropy), but without the conditioning on
//! the destination's own past.
//! - $u = 1$: immediate next-step influence
//! - $u > 1$: delayed influence
//! ## Multivariate Mutual Information

//! MI naturally extends to more than two variables. For $n$ variables:
//! $$I(X_1; X_2; \ldots; X_n) = \sum_{x_1,\ldots,x_n} p(x_1,\ldots,x_n) \log_b \frac{p(x_1,\ldots,x_n)}{\prod_{i=1}^n p(x_i)}$$
//! This measures the total mutual dependence among all variables.
//! ### Pairwise vs Multivariate
//! For more than two variables, pairwise MI values can be misleading because they
//! don't account for shared information already captured by other variables. This
//! leads to the concept of **redundancy** (shared information) and **synergy**
//! (information only present when variables are combined).
//! See [Conditional MI](super::cond_mi) and [Interaction Information](https://en.wikipedia.org/wiki/Interaction_information)
//! for more on multivariate dependence.
//! ## Relationship to Other Measures
//! | Measure | Formula | Relationship to MI |
//! |---------|---------|-------------------|
//! | Entropy | $H(X)$ | $I(X;X) = H(X)$ |
//! | Conditional Entropy | $H(X|Y)$ | $I(X;Y) = H(X) - H(X|Y)$ |
//! | Conditional MI | $I(X;Y|Z)$ | $I(X;Y) = I(X;Y|Z) + I(X;Y;Z)$ |
//! | Transfer Entropy | $T_{X \to Y}$ | $T_{X \to Y} = I(X^{(k)}; Y_{t+1} | Y^{(l)})$ |
//! ## Estimators in This Crate
//! The [`MutualInformation`](crate::estimators::mutual_information::MutualInformation) facade
//! provides multiple estimator types:
//! ### Discrete Estimators
//! - [`new_discrete_mle`](crate::estimators::mutual_information::MutualInformation::new_discrete_mle) - Maximum likelihood
//! - [`new_discrete_miller_madow`](crate::estimators::mutual_information::MutualInformation::new_discrete_miller_madow) - Bias corrected
//! - [`new_discrete_shrink`](crate::estimators::mutual_information::MutualInformation::new_discrete_shrink) - Shrinkage estimator
//! - [`new_discrete_grassberger`](crate::estimators::mutual_information::MutualInformation::new_discrete_grassberger) - Digamma-based
//! - [`new_discrete_chao_shen`](crate::estimators::mutual_information::MutualInformation::new_discrete_chao_shen) - Chao-Shen
//! - [`new_discrete_nsb`](crate::estimators::mutual_information::MutualInformation::new_discrete_nsb) - NSB
//! - [`new_discrete_bayes`](crate::estimators::mutual_information::MutualInformation::new_discrete_bayes) - Bayesian
//! ### Continuous Estimators
//! - [`new_kernel`](crate::estimators::mutual_information::MutualInformation::new_kernel) - Kernel density estimation
//! - [`new_ksg`](crate::estimators::mutual_information::MutualInformation::new_ksg) - Kraskov-Stögbauer-Grassberger
//! - [`new_renyi`](crate::estimators::mutual_information::MutualInformation::new_renyi) - Rényi entropy-based
//! ### Ordinal Estimators
//! - [`new_ordinal`](crate::estimators::mutual_information::MutualInformation::new_ordinal) - Permutation patterns
//! ## Examples
//! ### Discrete MI
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//! // Perfectly correlated variables: MI = log(2) ≈ 0.693147 nats
//! let x = array![0, 0, 0, 0, 1, 1, 1, 1];
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//! let mi = MutualInformation::new_discrete_mle(&[x, y]).global_value();
//! assert_abs_diff_eq!(mi, 0.693147, epsilon = 1e-4); // ≈ log(2)
//! ```

//! ### Continuous MI: Kraskov-Stögbauer-Grassberger (KSG)

//! The Kraskov-Stögbauer-Grassberger (KSG) method [Kraskov et al., 2004](../../guide/references/index.html#ksg2004) is a popular
//! kNN-based estimator that avoids explicit density estimation. It is designed to
//! cancel out bias by using a single k-th neighbor distance from the joint space
//! to define the search range in marginal spaces.

//! The estimator supports two variants:
//! - **Type I**: Uses strict inequality for neighbor counting (distance $< \epsilon$).
//!   $$I(X; Y) = \psi(k) + \psi(N) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle$$
//! - **Type II**: Uses non-strict inequality (distance $\le \epsilon$).
//!   $$I(X; Y) = \psi(k) - 1/k + \psi(N) - \langle \psi(n_x) + \psi(n_y) \rangle$$
//!
//! For more than two variables, the formula extends to:
//!   $$I(X_1; \ldots; X_m) = \psi(k) + (m-1)\psi(N) - \left\langle \sum_{j=1}^m \psi(n_j + 1) \right\rangle$$
//!
//! See the [KSG Approach Module](crate::estimators::approaches::expfam::ksg) for technical details.

//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;

//! // Two correlated continuous variables
//! let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
//! let y = array![0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1];

//! let mi = MutualInformation::new_ksg(&[x, y], 3, 1e-10).global_value();
//! assert!(mi > 0.0); // Positive for correlated data
//! assert!((mi - 1.0).abs() < 0.5); // Should be close to 1 for y = x + 0.1
//! ```

//! ### Local MI

//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::{GlobalValue, LocalValues};
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;

//! let x = array![0, 0, 0, 0, 1, 1, 1, 1];
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];

//! let mi_global = MutualInformation::new_discrete_mle(&[x.clone(), y.clone()]).global_value();
//! let mi_estimator = MutualInformation::new_discrete_mle(&[x, y]);
//! let mi_local = mi_estimator.local_values();

//! // Global MI should be close to mean of local values
//! let local_mean = mi_local.mean().unwrap();
//! assert_abs_diff_eq!(mi_global, local_mean, epsilon = 1e-10);
//! ```
//! ## Choosing an Estimator
//! - **Discrete data** (categorical, counts): Use discrete estimators. For large samples
//!   ($N > 1000$), MLE is often sufficient. For small samples, use bias-corrected
//!   estimators like NSB or Chao-Shen.
//! - **Continuous data** (real-valued): Use kernel or KSG estimators. KSG is generally
//!   preferred for moderate dimensions. Kernel estimators are more flexible but can
//!   struggle in high dimensions.
//! - **Time series with nonlinear dynamics**: Consider ordinal estimators, which are
//!   robust to amplitude variations.
//! - **Multivariate dependencies**: Use conditional MI to isolate direct dependencies.
//!
//! See [Estimator Selection](super::estimator_selection) for more guidance.
//! ## See Also
//! - [Estimator Usage](super::estimator_usage) - Quick start examples
//! - [Conditional MI](super::cond_mi) - MI conditioned on a third variable
//! - [Transfer Entropy](super::transfer_entropy) - Directed information flow
//! - [Entropy](super::entropy) - Foundation for all information measures
//! - [KLD](super::kld) - $I(X;Y) = D_{KL}(p(x,y) || p(x)p(y))$
//! ## References
//! - [Cover & Thomas, 2012](../../guide/references/index.html#cover2012elements)
//! - [Kraskov et al., 2004](../../guide/references/index.html#ksg2004)

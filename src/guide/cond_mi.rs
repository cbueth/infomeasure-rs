// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Conditional Mutual Information $I(X;Y|Z)$
//! Conditional Mutual Information measures the dependence between X and Y while controlling for Z.
//! It provides the shared information between X and Y when considering the knowledge of the conditional variable Z.
//! ## Definition
//! $$I(X;Y \\mid Z) = -\\sum_{x, y, z} p(z)p(x,y\\mid z) \\log \\frac{p(x, y \\mid z)}{p(x \\mid z)p(y \\mid z)}$$
//! This can also be expressed as:
//! $$I(X;Y \\mid Z) = -\\sum_{x, y, z} p(x,y,z) \\log \\frac{p(x,y,z)p(z)}{p(x,z)p(y,z)}$$
//! And equivalently using conditional entropies:
//! $$I(X;Y \\mid Z) = H(X \\mid Z) - H(X \\mid Y,Z)$$
//! ## Multivariate Conditional Mutual Information
//! For more than two variables (without conditioning):
//! $$I(X_1; X_2; \\ldots; X_n) = \\sum_{x_1,\\dots,x_n} p(x_1,\\dots,x_n) \\log \\frac{p(x_1,\\dots,x_n)}{\\prod_{i=1}^n p(x_i)}$$
//! With conditioning on Z:
//! $$I(X_1; X_2; \\ldots; X_n \\mid Z) = -\\sum_{x_1, \\dots, x_n, z} p(z)p(x_1,\\dots,x_n \\mid z) \\log \\frac{p(x_1,\\dots,x_n \\mid z)}{\\prod p(x_i \\mid z)}$$
//! $$= - H(X_1, X_2, \\ldots, X_n, Z) - H(Z) + \\sum_{i=1}^n H(X_i, Z)$$
//! ## Local Conditional Mutual Information
//! Similar to local entropy and MI, **local or point-wise conditional MI** can be defined:
//! $$i(x; y \\mid z) = -\\log_b \\frac{p(x \\mid y, z)}{p(x \\mid z)}$$
//! $$= h(x \\mid z) - h(x \\mid y, z)$$
//! The conditional MI can be calculated as the expected value of its local counterparts:
//! $$I(X; Y \\mid Z) = \\langle i(x; y \\mid z) \\rangle$$
//! Note: The conditional MI $I(X;Y \\mid Z)$ can be either larger or smaller than its
//! non-conditional counterpart $I(X; Y)$. This leads to the concept of **synergy** and **redundancy**.
//! CMI is symmetric under the same condition $Z$: $I(X;Y \\mid Z) = I(Y;X \\mid Z)$.
//! In this crate, local CMI can be accessed via the [`LocalValues`](crate::estimators::traits::LocalValues) trait
//! on CMI estimators that support it.
//! ## Entropy Combination Form
//! The CMI expression can be expressed in terms of entropies and joint entropies:
//! $$I(X;Y \\mid Z) = - H(X,Z,Y) + H(X,Z) + H(Z,Y) - H(Z)$$
//! This formula is used internally for Rényi and Tsallis CMI estimators.
//! ## Implemented in This Crate
//! ### Continuous CMI: Kraskov-Stögbauer-Grassberger (KSG)
//! The KSG method [Kraskov et al., 2004](../../guide/references/index.html#ksg2004) can be extended to conditional mutual information
//! [Frenzel & Pompe, 2007](../../guide/references/index.html#frenzel2007):
//! $$I(X; Y \mid Z) = \psi(k) + \langle \psi(n_z + 1) - \psi(n_{xz} + 1) - \psi(n_{yz} + 1) \rangle$$
//! where $n_z, n_{xz}, n_{yz}$ are neighbor counts in the respective subspaces defined
//! by the distance to the $k$-th neighbor in the joint $(X, Y, Z)$ space.
//! See the [KSG Approach Module](crate::estimators::approaches::expfam::ksg) for technical details.
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! let x = array![0.1, 1.1, 2.1, 3.1, 4.1];
//! let y = array![0.2, 1.2, 2.2, 3.2, 4.2];
//! let z = array![0.15, 1.15, 2.15, 3.15, 4.15];
//! let cmi = MutualInformation::new_cmi_ksg(&[x, y], &z, 3, 1e-10).global_value();
//! assert!(cmi >= 0.0);
//! ```
//! ### Other Estimators
//! CMI is available through the [`MutualInformation`](crate::estimators::mutual_information::MutualInformation) facade.
//! It provides methods for all supported estimation approaches (Discrete, Kernel, KSG, etc.).
//! For a complete list of constructors and convenience macros, see:
//! - [**Mutual Information Facade**](crate::estimators::mutual_information::MutualInformation) — Unified technical interface
//! - [**Macros Guide**](crate::guide::macros) — For automatic dimension calculation
//! - [**Estimator Selection Guide**](crate::guide::estimator_selection) — Choosing the right estimator
//! ## Example
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! // X and Y are conditionally independent given Z
//! let x = array![0, 1, 0, 1, 0, 1];
//! let y = array![0, 0, 1, 1, 0, 1];
//! let z = array![0, 0, 0, 1, 1, 1];
//! let cmi = MutualInformation::new_cmi_discrete_mle(&[x, y], &z).global_value();
//! assert!(cmi >= 0.0); // Should be 0 or very small if truly independent
//! ```
//! ## See Also
//! - [Estimator Usage Guide](super::estimator_usage) - Base MI
//! - [Conditional Transfer Entropy Guide](super::cond_te) - TE with conditioning

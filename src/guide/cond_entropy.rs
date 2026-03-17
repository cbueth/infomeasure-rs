// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Conditional Entropy $H(X|Y)$
//!
//! Conditional entropy is defined as the remaining uncertainty of a variable after considering
//! the information from another variable. In other words, it is the remaining unique information
//! of the first variable after having the knowledge of the conditional variable.
//!
//! ## Direct Definition
//!
//! Let $X$ and $Y$ be two random variables with marginal probability $p(x)$ and $p(y)$,
//! and conditional probability distribution of $X$ conditioned on $Y$ denoted as $p(x|y)$.
//! The conditional entropy is calculated by:
//!
//! $$H(X \\mid Y) = - \\sum_{x,y} p(x, y) \\log p(x \\mid y)$$
//!
//! ## Chain Rule
//!
//! Using the chain rule, the conditional entropy can be expressed in terms of
//! **joint entropy** $H(X,Y)$ and marginal entropy $H(Y)$:
//!
//! $$H(X \\mid Y) = H(X,Y) - H(Y)$$
//!
//! This relationship is useful because this crate provides methods to compute
//! joint entropy via [`Entropy::joint_discrete`](crate::estimators::entropy::Entropy::joint_discrete).
//!
//! ## Joint Entropy
//!
//! The joint entropy represents the amount of information gained by jointly observing
//! two random variables:
//!
//! $$H(X, Y) = - \\sum_{x,y} p(x, y) \\log p(x, y)$$
//!
//! In this crate, joint entropy can be computed by passing multiple variables as a tuple
//! to the entropy estimator. See the [Estimator Usage Guide](super::estimator_usage) for examples.
//!
//! ## Local Conditional Entropy
//!
//! Similar to local entropy, one can define **local or point-wise conditional entropy**:
//!
//! $$h(x \\mid y) = - \\log p(x \\mid y)$$
//!
//! This local conditional entropy also satisfies the chain rule as its average counterparts:
//!
//! $$h(x \\mid y) = h(x,y) - h(y)$$
//!
//! In this crate, local values can be accessed via the [`LocalValues`](crate::estimators::traits::LocalValues) trait
//! on estimators that support it.
//!
//! ## Current Status
//!
//! Conditional entropy can be computed using the chain rule relationship above with joint entropy.
//! The [`Entropy::joint_discrete`](crate::estimators::entropy::Entropy::joint_discrete) method
//! can be used for joint entropy calculation.
//!
//! Future versions may include direct conditional entropy constructors.
//!
//! ## See Also
//!
//! - [Entropy Guide](super::entropy) - Base entropy
//! - [Estimator Usage Guide](super::estimator_usage) - $I(X;Y) = H(X) - H(X|Y)$
//! - [Mutual Information Guide](super::estimator_usage) - MI uses conditional entropy

// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Kullback–Leibler divergence (KLD)
//!
//! The Kullback–Leibler divergence (relative entropy) quantifies the extra cost
//! of encoding data from a distribution $P$ with a code built for $Q$:
//!
//! $$D_{\mathrm{KL}}(P \parallel Q) = H_Q(P) - H(P)$$
//!
//! where $H_Q(P)$ is the cross-entropy and $H(P)$ the entropy of $P$. It is
//! asymmetric, $D_{\mathrm{KL}}(P \parallel Q) \neq D_{\mathrm{KL}}(Q \parallel P)$,
//! and vanishes iff $P = Q$.
//!
//! Because KLD is derived directly from cross-entropy, it is available for
//! **exactly** the approaches for which cross-entropy is mathematically sound
//! (discrete/MLE, Miller–Madow, Bayes, kernel, ordinal, Kozachenko–Leonenko,
//! Rényi, Tsallis). Bias-corrected discrete variants that have no joint
//! $(P, Q)$ form — NSB, Chao–Shen, Chao–Wang–Jost, Grassberger, ANSB, Zhang,
//! Bonachela, and shrinkage — intentionally do not implement
//! [`CrossEntropy`](crate::estimators::traits::CrossEntropy) and therefore
//! cannot be used here.
//!
//! See the [KLD guide](crate::guide::kld) for the theory.

use crate::estimators::traits::{CrossEntropy, GlobalValue};

/// Extension trait computing the Kullback–Leibler divergence from an estimator's
/// cross-entropy and entropy.
///
/// It is implemented for every estimator that provides both
/// [`CrossEntropy`] and [`GlobalValue`], so new approaches inherit KLD for free.
///
/// # Example
///
/// ```rust
/// use infomeasure::estimators::entropy::Entropy;
/// use infomeasure::estimators::composite_measures::Kld;
/// use ndarray::array;
///
/// let p = Entropy::new_discrete(array![1, 1, 1, 2, 2, 3, 4, 5]);
/// let q = Entropy::new_discrete(array![1, 1, 2, 2, 2, 3, 3, 5]);
/// let d = p.kld(&q);
/// assert!(d.is_finite());
/// ```
pub trait Kld: CrossEntropy + GlobalValue + Sized {
    /// Kullback–Leibler divergence $D_{\mathrm{KL}}(P \parallel Q) = H_Q(P) - H(P)$.
    fn kld(&self, other: &Self) -> f64
    where
        Self: Sized,
    {
        self.cross_entropy(other) - self.global_value()
    }
}

impl<T: CrossEntropy + GlobalValue> Kld for T {}

/// Functional form of [`Kld::kld`]: $D_{\mathrm{KL}}(P \parallel Q)$.
///
/// # Example
///
/// ```rust
/// use infomeasure::estimators::entropy::Entropy;
/// use infomeasure::estimators::composite_measures::kld;
/// use ndarray::array;
///
/// let p = Entropy::new_discrete(array![1, 1, 2, 3, 3, 3]);
/// let q = Entropy::new_discrete(array![1, 2, 2, 2, 3, 3]);
/// assert!(kld(&p, &q).is_finite());
/// ```
pub fn kld<P: CrossEntropy + GlobalValue>(p: &P, q: &P) -> f64 {
    p.kld(q)
}

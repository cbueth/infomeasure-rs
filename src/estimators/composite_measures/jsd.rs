// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Jensen–Shannon divergence (JSD)
//!
//! JSD is the symmetric, bounded divergence obtained by comparing each
//! distribution to their mixture $M = \sum_i \pi_i P_i$:
//!
//! $$JS_{\pi}(P_1, \dots, P_n) = H\!\left(\sum_i \pi_i P_i\right)
//!     - \sum_i \pi_i H(P_i), \qquad \sum_i \pi_i = 1,$$
//!
//! with $0 \le JS_{\pi} \le \log_b(n)$ in base-$b$ units.
//!
//! Unlike [`kld`](super::kld), JSD is **not** a cross-entropy wrapper: it
//! requires a *mixture distribution* to be well defined. It is therefore
//! available only for estimators that expose a normalized
//! [`ProbabilityMass`](crate::estimators::traits::ProbabilityMass) (discrete,
//! Bayes, shrinkage, ordinal), or for continuous data through a pooled kernel
//! estimate ([`jsd_kernel_1d`] / [`jsd_kernel_nd`]). Differential and
//! generalized-entropy estimators (Kozachenko–Leonenko, Rényi, Tsallis) do not
//! admit a mixture in this sense and are excluded by construction.
//!
//! Results are in **nats**; divide by $\ln(b)$ for base-$b$ units.
//!
//! See the [JSD guide](crate::guide::jsd) for the theory.

use crate::estimators::approaches::kernel::KernelEntropy;
use crate::estimators::traits::{GlobalValue, ProbabilityMass};
use ndarray::{Array1, Array2, Axis, concatenate};
use rustc_hash::FxHashMap;

/// Jensen–Shannon divergence between two or more distributions (nats).
///
/// `weights` are normalized to sum to one; passing `None` selects uniform
/// weights. Returns `0.0` for fewer than two inputs. The mixture pmf is built in
/// a single pass over the union of the supports — no intermediate arrays.
///
/// # Example
///
/// ```rust
/// use infomeasure::estimators::entropy::Entropy;
/// use infomeasure::estimators::composite_measures::jsd;
/// use ndarray::array;
///
/// let p = Entropy::new_discrete(array![0, 0, 0, 1, 1, 2, 3, 4]);
/// let q = Entropy::new_discrete(array![0, 0, 1, 1, 1, 2, 2, 4]);
/// // Symmetric and bounded by ln(2) for two distributions.
/// let d = jsd(&[p, q], None);
/// assert!(d >= -1e-12 && d <= 2.0_f64.ln() + 1e-12);
/// ```
pub fn jsd<P: ProbabilityMass + GlobalValue>(dists: &[P], weights: Option<&[f64]>) -> f64 {
    if dists.len() < 2 {
        return 0.0;
    }
    let weights = normalized_weights(weights, dists.len());

    let mut marginal = 0.0_f64;
    let mut mixture: FxHashMap<P::Key, f64> = FxHashMap::default();
    for (estimator, &w) in dists.iter().zip(&weights) {
        marginal += w * estimator.global_value();
        for (symbol, p) in estimator.pmf() {
            *mixture.entry(symbol).or_insert(0.0) += w * p;
        }
    }

    let h_mixture = -mixture
        .values()
        .filter(|&&m| m > 0.0)
        .map(|&m| m * m.ln())
        .sum::<f64>();

    h_mixture - marginal
}

/// Jensen–Shannon divergence for 1D continuous data via a pooled kernel estimate.
///
/// The mixture distribution is the union of the samples, so a single
/// [`KernelEntropy`] is built over the concatenated data; this inherits the
/// crate's KD-tree, GPU and `parallel` acceleration paths.
///
/// `weights` are normalized to sum to one; `None` selects uniform weights.
pub fn jsd_kernel_1d(
    dists: &[Array1<f64>],
    weights: Option<&[f64]>,
    bandwidth: f64,
    kernel_type: &str,
) -> f64 {
    if dists.len() < 2 {
        return 0.0;
    }
    let weights = normalized_weights(weights, dists.len());
    let pooled = concatenate(Axis(0), &dists.iter().map(|d| d.view()).collect::<Vec<_>>())
        .expect("cannot pool empty data");
    let h_pooled =
        KernelEntropy::<1>::new_with_kernel_type(pooled, kernel_type.to_string(), bandwidth)
            .global_value();
    let marginal = dists
        .iter()
        .zip(&weights)
        .map(|(d, &w)| {
            w * KernelEntropy::<1>::new_with_kernel_type(
                d.clone(),
                kernel_type.to_string(),
                bandwidth,
            )
            .global_value()
        })
        .sum::<f64>();
    h_pooled - marginal
}

/// [`jsd_kernel_1d`] for `D`-dimensional data (`Array2` with rows = samples).
///
/// `weights` are normalized to sum to one; `None` selects uniform weights.
pub fn jsd_kernel_nd<const D: usize>(
    dists: &[Array2<f64>],
    weights: Option<&[f64]>,
    bandwidth: f64,
    kernel_type: &str,
) -> f64 {
    if dists.len() < 2 {
        return 0.0;
    }
    let weights = normalized_weights(weights, dists.len());
    let pooled = concatenate(Axis(0), &dists.iter().map(|d| d.view()).collect::<Vec<_>>())
        .expect("cannot pool empty data");
    let h_pooled =
        KernelEntropy::<D>::new_with_kernel_type(pooled, kernel_type.to_string(), bandwidth)
            .global_value();
    let marginal = dists
        .iter()
        .zip(&weights)
        .map(|(d, &w)| {
            w * KernelEntropy::<D>::new_with_kernel_type(
                d.clone(),
                kernel_type.to_string(),
                bandwidth,
            )
            .global_value()
        })
        .sum::<f64>();
    h_pooled - marginal
}

/// Normalizes user weights to sum to one, falling back to uniform weights.
fn normalized_weights(weights: Option<&[f64]>, n: usize) -> Vec<f64> {
    if let Some(w) = weights
        && w.len() == n
    {
        let sum: f64 = w.iter().sum();
        if sum.is_finite() && sum > 0.0 {
            return w.iter().map(|x| x / sum).collect();
        }
    }
    vec![1.0 / n as f64; n]
}

use crate::estimators::doc_macros::doc_snippets;
// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array1, Array2};
use statrs::function::gamma::digamma;

use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{GlobalValue, JointEntropy, LocalValues, OptionalLocalValues};

/// ANSB (asymptotic NSB) entropy estimator for discrete data (natural log base).
///
/// Coincidence-based approximation to NSB that is appropriate in undersampled regimes.
/// Requires observed sample size N, observed K (or override), and uses coincidences Δ=N−K.
/// Returns NaN if inputs indicate inapplicability. Global-only.
///
/// ANSB is derived for the strongly undersampled regime where K ∼ N. When the data
/// appear well-sampled (N/K above [`AnsbEntropy::DEFAULT_UNDERSAMPLED_THRESHOLD`]
/// or the configured threshold), the estimate diverges like 2·ln N and a bias-
/// corrected estimator for the well-sampled regime should be preferred instead.
/// Both the threshold warning and the no-coincidences warning are emitted to
/// stderr at most once per process.
///
/// Cross-entropy is not implemented for ANSB estimator.
/// The ANSB estimator is designed for single distribution entropy estimation
/// and cross-entropy creates a theoretical inconsistency.
///
/// Joint entropy is supported by reducing the joint space of multiple variables to a single
/// discrete representation before estimation.
///
/// Local values are not supported for the ANSB estimator.
/// The ANSB estimator is based on global statistics (coincidences) and
/// local values cannot be meaningfully extracted.
///
#[doc = doc_snippets!(discrete_guide_ref)]
pub struct AnsbEntropy {
    dataset: DiscreteDataset,
    /// Optional override for support size K
    k_override: Option<usize>,
    /// Maximum allowed ratio N/K before the data are considered too well-sampled
    /// for ANSB (which requires K ∼ N); a warning is emitted once per process
    /// when exceeded
    undersampled_threshold: f64,
}

impl AnsbEntropy {
    /// Default maximum allowed ratio N/K for the undersampled-regime check.
    ///
    /// With N/K ≤ 2 at least half of the observations are coincidences
    /// (Δ = N − K ≥ N/2), keeping the digamma expansion behind ANSB well-behaved;
    /// beyond that the estimate grows without bound as the data become
    /// well-sampled. This is a heuristic validity bound, not a sharp cutoff —
    /// pass a custom threshold via `new_ansb_with_threshold` (or `f64::INFINITY`
    /// to disable the check).
    pub const DEFAULT_UNDERSAMPLED_THRESHOLD: f64 = 2.0;

    pub fn new(data: Array1<i32>, k_override: Option<usize>, undersampled_threshold: f64) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self {
            dataset,
            k_override,
            undersampled_threshold,
        }
    }

    /// Build a vector of AnsbEntropy estimators, one per row of a 2D array.
    pub fn from_rows(
        data: Array2<i32>,
        k_override: Option<usize>,
        undersampled_threshold: f64,
    ) -> Vec<Self> {
        rows_as_vec(data)
            .into_iter()
            .map(|row| Self::new(row, k_override, undersampled_threshold))
            .collect()
    }
}

/// Emits the too-well-sampled warning at most once per process.
static UNDERSAMPLED_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);
/// Emits the no-coincidences warning at most once per process.
static NO_COINCIDENCES_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

impl GlobalValue for AnsbEntropy {
    fn global_value(&self) -> f64 {
        let n = self.dataset.n;
        if n == 0 {
            return f64::NAN;
        }
        let k_obs = self.dataset.k;
        let k = self.k_override.unwrap_or(k_obs);
        if k == 0 {
            return f64::NAN;
        }

        // ANSB assumes the strongly undersampled regime K ~ N; data that look
        // well-sampled make the estimate diverge like 2·ln N. Emitted at most
        // once per process to stay quiet in hot loops.
        let ratio = n as f64 / k as f64;
        if ratio > self.undersampled_threshold
            && !UNDERSAMPLED_WARNING_EMITTED.swap(true, Ordering::Relaxed)
        {
            eprintln!(
                "Warning: Data is not sufficiently undersampled (N/K = {:.3} > {:.3}), so calculation may diverge...",
                ratio, self.undersampled_threshold
            );
        }

        let coincidences = (n as i64) - (k as i64);
        if coincidences <= 0 {
            if !NO_COINCIDENCES_WARNING_EMITTED.swap(true, Ordering::Relaxed) {
                eprintln!("Warning: No coincidences in data - ANSB estimator is undefined");
            }
            return f64::NAN;
        }

        // (γ - ln 2) + 2 ln N - ψ(Δ)
        const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

        (EULER_GAMMA - 2.0_f64.ln()) + 2.0 * (n as f64).ln() - digamma(coincidences as f64)
    }
}

impl LocalValues for AnsbEntropy {
    fn local_values(&self) -> Array1<f64> {
        Array1::zeros(0)
    }
}

impl JointEntropy for AnsbEntropy {
    type Source = Array1<i32>;
    type Params = (Option<usize>, f64); // k_override, undersampled_threshold

    fn joint_entropy(series: &[Self::Source], params: Self::Params) -> f64 {
        if series.is_empty() {
            return 0.0;
        }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = AnsbEntropy::new(joint_codes, params.0, params.1);
        disc.global_value()
    }
}

impl OptionalLocalValues for AnsbEntropy {
    fn supports_local(&self) -> bool {
        false
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err(
            "Local values are not supported for ANSB estimator as it averages over Dirichlet priors.",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No other unit test in this binary touches `AnsbEntropy::global_value`,
    /// so resetting the process-wide flags here is race-free.
    fn reset_flags() {
        UNDERSAMPLED_WARNING_EMITTED.store(false, Ordering::Relaxed);
        NO_COINCIDENCES_WARNING_EMITTED.store(false, Ordering::Relaxed);
    }

    #[test]
    fn undersampled_warning_fires_once_per_process() {
        reset_flags();
        // Well-sampled: alphabet 2, N=40 → N/K = 20 > default threshold 2.0
        let data: Array1<i32> = (0..40).map(|i| i % 2).collect();
        let est = AnsbEntropy::new(data, None, AnsbEntropy::DEFAULT_UNDERSAMPLED_THRESHOLD);

        let h = est.global_value();
        assert!(h.is_finite());
        assert!(UNDERSAMPLED_WARNING_EMITTED.load(Ordering::Relaxed));

        // Second call stays quiet (flag latched), value unchanged
        let h2 = est.global_value();
        assert_eq!(h, h2);

        // Quiet path: strongly undersampled data never touch the flag
        reset_flags();
        let undersampled = Array1::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 9]); // N/K = 10/9 ≈ 1.11
        let est_q = AnsbEntropy::new(
            undersampled,
            None,
            AnsbEntropy::DEFAULT_UNDERSAMPLED_THRESHOLD,
        );
        assert!(est_q.global_value().is_finite());
        assert!(!UNDERSAMPLED_WARNING_EMITTED.load(Ordering::Relaxed));
    }

    #[test]
    fn no_coincidences_warns_and_returns_nan() {
        reset_flags();
        // All-unique data → Δ = N − K = 0; threshold disabled to isolate the path
        let data = Array1::from(vec![1, 2, 3, 4, 5]);
        let est = AnsbEntropy::new(data, None, f64::INFINITY);

        assert!(est.global_value().is_nan());
        assert!(NO_COINCIDENCES_WARNING_EMITTED.load(Ordering::Relaxed));
        assert!(!UNDERSAMPLED_WARNING_EMITTED.load(Ordering::Relaxed));
    }

    #[test]
    fn custom_threshold_can_silence_the_check() {
        reset_flags();
        let data: Array1<i32> = (0..40).map(|i| i % 2).collect();
        let est = AnsbEntropy::new(data, None, f64::INFINITY);

        assert!(est.global_value().is_finite());
        assert!(!UNDERSAMPLED_WARNING_EMITTED.load(Ordering::Relaxed));
    }
}

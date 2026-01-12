use ndarray::{Array1, Array2};
use statrs::function::gamma::digamma;

use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::traits::{GlobalValue, OptionalLocalValues, JointEntropy};

/// ANSB (asymptotic NSB) entropy estimator for discrete data (natural log base).
///
/// Coincidence-based approximation to NSB that is appropriate in undersampled regimes.
/// Requires observed sample size N, observed K (or override), and uses coincidences Δ=N−K.
/// Returns NaN if inputs indicate inapplicability. Global-only.
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
pub struct AnsbEntropy {
    dataset: DiscreteDataset,
    /// Optional override for support size K
    k_override: Option<usize>,
    /// Threshold for considering data sufficiently undersampled (ratio N/K)
    undersampled_threshold: f64,
}

impl AnsbEntropy {
    pub fn new(
        data: Array1<i32>,
        k_override: Option<usize>,
        undersampled_threshold: f64,
    ) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset, k_override, undersampled_threshold }
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

impl GlobalValue for AnsbEntropy {
    fn global_value(&self) -> f64 {
        let n = self.dataset.n as usize;
        if n == 0 { return f64::NAN; }
        let k_obs = self.dataset.k;
        let k = self.k_override.unwrap_or(k_obs);
        if k == 0 { return f64::NAN; }

        let coincidences = (n as i64) - (k as i64);
        if coincidences <= 0 { return f64::NAN; }

        // (γ - ln 2) + 2 ln N - ψ(Δ)
        const EULER_GAMMA: f64 = 0.577215_664_901_532_9;
        let entropy = (EULER_GAMMA - 2.0_f64.ln()) + 2.0 * (n as f64).ln() - digamma(coincidences as f64);
        entropy
    }
}

impl OptionalLocalValues for AnsbEntropy {
    fn supports_local(&self) -> bool { false }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not supported for ANSB estimator as it averages over Dirichlet priors.")
    }
}

impl JointEntropy for AnsbEntropy {
    type Source = Array1<i32>;
    type Params = (Option<usize>, f64); // k_override, undersampled_threshold

    fn joint_entropy(series: &[Self::Source], params: Self::Params) -> f64 {
        if series.is_empty() { return 0.0; }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = AnsbEntropy::new(joint_codes, params.0, params.1);
        disc.global_value()
    }
}

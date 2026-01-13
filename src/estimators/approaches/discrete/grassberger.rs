use ndarray::{Array1, Array2};
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::traits::{GlobalValue, LocalValues, OptionalLocalValues, JointEntropy};
use statrs::function::gamma::digamma;

/// Grassberger (Gr88) entropy estimator for discrete data.
///
/// Per-count correction using digamma functions with an alternating term:
/// for count $n_i$, local contribution is $\ln N - \psi(n_i) - (-1)^{n_i}/(n_i+1)$.
/// Supports local values. Suitable for moderate undersampling.
///
/// Cross-entropy is not implemented for Grassberger estimator.
/// The Grassberger correction is designed for bias correction in entropy
/// estimation, and cross-entropy mixes probabilities from one distribution
/// with corrections from another, creating a theoretical inconsistency.
pub struct GrassbergerEntropy {
    dataset: DiscreteDataset,
}

impl GrassbergerEntropy {
    pub fn new(data: Array1<i32>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset }
    }

    /// Build a vector of GrassbergerEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>) -> Vec<Self> {
        rows_as_vec(data).into_iter().map(Self::new).collect()
    }
}

impl GlobalValue for GrassbergerEntropy {
    fn global_value(&self) -> f64 {
        let n_total_ln = (self.dataset.n as f64).ln();
        let mut h = 0.0_f64;
        let n_f = self.dataset.n as f64;
        for &cnt in self.dataset.counts.values() {
            let n_i = cnt as i64;
            let n_if = cnt as f64;
            let sign = if n_i % 2 == 0 { 1.0 } else { -1.0 };
            let p_i = n_if / n_f;
            h += p_i * (n_total_ln - digamma(n_if) - sign / (n_if + 1.0));
        }
        h
    }
}

impl LocalValues for GrassbergerEntropy {
    fn local_values(&self) -> Array1<f64> {
        let n_total_ln = (self.dataset.n as f64).ln();
        self.dataset.data.mapv(|v| {
            let n_i = self.dataset.counts[&v] as i64; // integer count for (-1)^n
            let n_if = n_i as f64;
            let sign = if n_i % 2 == 0 { 1.0 } else { -1.0 };
            n_total_ln - digamma(n_if) - sign / (n_if + 1.0)
        })
    }
}

impl JointEntropy for GrassbergerEntropy {
    type Source = Array1<i32>;
    type Params = ();

    fn joint_entropy(series: &[Self::Source], _params: Self::Params) -> f64 {
        if series.is_empty() { return 0.0; }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = GrassbergerEntropy::new(joint_codes);
        disc.global_value()
    }
}

impl OptionalLocalValues for GrassbergerEntropy {
    fn supports_local(&self) -> bool { true }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> { Ok(self.local_values()) }
}

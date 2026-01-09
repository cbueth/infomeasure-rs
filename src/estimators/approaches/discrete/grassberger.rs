use ndarray::{Array1, Array2};
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{LocalValues, OptionalLocalValues};
use statrs::function::gamma::digamma;

/// Grassberger (Gr88) entropy estimator for discrete data.
///
/// Per-count correction using digamma functions with an alternating term:
/// for count $n_i$, local contribution is $\ln N - \psi(n_i) - (-1)^{n_i}/(n_i+1)$.
/// Supports local values. Suitable for moderate undersampling.
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

impl OptionalLocalValues for GrassbergerEntropy {
    fn supports_local(&self) -> bool { true }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> { Ok(self.local_values()) }
}

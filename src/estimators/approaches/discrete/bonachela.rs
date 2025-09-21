use ndarray::{Array1, Array2};

use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{GlobalValue, OptionalLocalValues};

/// Bonachela entropy estimator for discrete data (natural log base).
///
/// De-biases the MLE via a harmonic-sum correction using counts (n_i + 1) Σ_{j=n_i+2}^{N+2} 1/j,
/// then normalizes by (N+2). Recommended when the distribution is undersampled; global-only.
pub struct BonachelaEntropy {
    dataset: DiscreteDataset,
}

impl BonachelaEntropy {
    pub fn new(data: Array1<i32>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset }
    }

    /// Build a vector of BonachelaEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>) -> Vec<Self> {
        rows_as_vec(data).into_iter().map(Self::new).collect()
    }
}

impl GlobalValue for BonachelaEntropy {
    fn global_value(&self) -> f64 {
        let n = self.dataset.n as usize;
        if n == 0 { return 0.0; }
        let mut acc = 0.0_f64;
        // For each count n_i, compute (n_i + 1) * sum_{j=n_i+2}^{N+2} 1/j
        for &cnt in self.dataset.counts.values() {
            let ni_plus1 = (cnt + 1) as f64;
            let start_j = (cnt + 2) as usize;
            let end_j = n + 2;
            if start_j <= end_j {
                let mut inner = 0.0_f64;
                for j in start_j..=end_j { inner += 1.0 / (j as f64); }
                acc += ni_plus1 * inner;
            }
        }
        acc / ((n + 2) as f64)
    }
}

impl OptionalLocalValues for BonachelaEntropy {
    fn supports_local(&self) -> bool { false }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not supported for Bonachela estimator")
    }
}

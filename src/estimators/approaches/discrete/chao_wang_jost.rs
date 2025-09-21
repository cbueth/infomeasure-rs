use ndarray::{Array1, Array2};
use statrs::function::gamma::digamma;

use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{GlobalValue, OptionalLocalValues};

/// Chao–Wang–Jost entropy estimator for discrete data (natural log base).
///
/// Combines digamma-based terms with coverage corrections using the singleton (f1)
/// and doubleton (f2) counts via parameter a. Effective with moderate undersampling;
/// global-only.
pub struct ChaoWangJostEntropy {
    dataset: DiscreteDataset,
}

impl ChaoWangJostEntropy {
    pub fn new(data: Array1<i32>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset }
    }

    /// Build a vector of ChaoWangJostEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>) -> Vec<Self> {
        rows_as_vec(data).into_iter().map(Self::new).collect()
    }
}

impl GlobalValue for ChaoWangJostEntropy {
    fn global_value(&self) -> f64 {
        let n = self.dataset.n as usize;
        if n == 0 { return 0.0; }
        // f1, f2
        let mut f1 = 0usize;
        let mut f2 = 0usize;
        for &cnt in self.dataset.counts.values() {
            if cnt == 1 { f1 += 1; }
            if cnt == 2 { f2 += 1; }
        }
        // A parameter
        let a = if f2 > 0 {
            (2.0 * f2 as f64) / (((n - 1) as f64) * (f1 as f64) + 2.0 * (f2 as f64))
        } else if f1 > 0 {
            2.0 / (((n - 1) as f64) * ((f1 - 1) as f64) + 2.0)
        } else { 1.0 };

        // First part: sum_{1<=n_i<=N-1} n_i * (digamma(N) - digamma(n_i)) / N
        let dg_n = digamma(n as f64);
        let mut cwj = 0.0_f64;
        for &cnt in self.dataset.counts.values() {
            if cnt >= 1 && cnt < n { cwj += (cnt as f64) * (dg_n - digamma(cnt as f64)); }
        }
        cwj /= n as f64;

        // Correction term
        if a != 1.0 && f1 > 0 {
            let one_minus_a = 1.0 - a;
            let mut p2 = 0.0_f64;
            for r in 1..n { p2 += (one_minus_a.powi(r as i32)) / (r as f64); }
            let correction = (f1 as f64) / (n as f64)
                * one_minus_a.powi(1 - n as i32)
                * ( -a.ln() - p2 );
            cwj += correction;
        }

        cwj
    }
}

impl OptionalLocalValues for ChaoWangJostEntropy {
    fn supports_local(&self) -> bool { false }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not supported for Chao-Wang-Jost estimator")
    }
}

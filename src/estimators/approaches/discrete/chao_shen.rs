use ndarray::{Array1, Array2};

use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{GlobalValue, OptionalLocalValues};

/// Chao–Shen coverage-adjusted entropy estimator for discrete data (natural log base).
///
/// Adjusts empirical probabilities by sample coverage C = 1 - f1/N and compensates for
/// unseen mass via the leave-one-out estimator λ = 1 - (1 - C p)^N in the denominator.
/// Recommended for undersampled data with many singletons; global-only.
pub struct ChaoShenEntropy {
    dataset: DiscreteDataset,
}

impl ChaoShenEntropy {
    pub fn new(data: Array1<i32>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset }
    }

    /// Build a vector of ChaoShenEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>) -> Vec<Self> {
        rows_as_vec(data).into_iter().map(Self::new).collect()
    }
}

impl GlobalValue for ChaoShenEntropy {
    fn global_value(&self) -> f64 {
        let n = self.dataset.n as f64;
        if n == 0.0 { return 0.0; }

        // Number of singletons f1
        let mut f1: usize = 0;
        for &cnt in self.dataset.counts.values() {
            if cnt == 1 { f1 += 1; }
        }
        if (f1 as f64) == n { // avoid C=0
            if f1 > 0 { f1 -= 1; }
        }

        let c_cov = 1.0 - (f1 as f64) / n; // coverage C

        // Sum over bins: - sum( pa * ln(pa) / la ) where pa = C * p_ml, la = 1 - (1 - pa)^N
        let mut h = 0.0_f64;
        for (&_val, &p_ml) in &self.dataset.dist {
            let pa = c_cov * p_ml;
            if pa <= 0.0 { continue; }
            let la = 1.0 - (1.0 - pa).powf(n);
            if la <= 0.0 { continue; }
            h -= pa * pa.ln() / la;
        }
        h
    }
}

impl OptionalLocalValues for ChaoShenEntropy {
    fn supports_local(&self) -> bool { false }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not supported for Chao-Shen estimator")
    }
}

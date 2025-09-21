use ndarray::{Array1, Array2};
use std::collections::HashMap;
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{LocalValues, OptionalLocalValues};

/// Shrinkage (James–Stein) entropy estimator for discrete data (natural log base).
///
/// Forms a convex combination between the empirical distribution and the uniform target
/// with a data-driven shrinkage intensity λ ∈ [0,1]. This reduces variance and bias
/// in undersampled regimes. Supports local values via -ln p_shrink(x).
pub struct ShrinkEntropy {
    dataset: DiscreteDataset,
}

impl ShrinkEntropy {
    pub fn new(data: Array1<i32>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset }
    }

    /// Build a vector of ShrinkEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>) -> Vec<Self> {
        rows_as_vec(data).into_iter().map(Self::new).collect()
    }

    fn shrink_probs(&self) -> HashMap<i32, f64> {
        let n = self.dataset.n as f64;
        let k = self.dataset.k as f64;
        let t = 1.0 / k; // uniform target

        // MLE probabilities per symbol
        // u(x) = count/N
        // Precompute var(u) and msp
        let mut var_sum = 0.0_f64;
        let mut msp = 0.0_f64;
        for (&_val, &cnt) in self.dataset.counts.iter() {
            let u = (cnt as f64) / n;
            // variance term
            if self.dataset.n > 1 { var_sum += u * (1.0 - u) / (n - 1.0); }
            // mean squared difference to target
            msp += (u - t) * (u - t);
        }

        // lambda in [0,1]
        let lambda = if self.dataset.n <= 1 {
            1.0
        } else if msp == 0.0 {
            1.0
        } else {
            let mut l = var_sum / msp;
            if l < 0.0 { l = 0.0; }
            if l > 1.0 { l = 1.0; }
            l
        };

        let mut dist_shrink = HashMap::with_capacity(self.dataset.k);
        for (&val, &cnt) in self.dataset.counts.iter() {
            let u = (cnt as f64) / n;
            let p = lambda * t + (1.0 - lambda) * u;
            dist_shrink.insert(val, p);
        }
        dist_shrink
    }
}

impl LocalValues for ShrinkEntropy {
    fn local_values(&self) -> Array1<f64> {
        let dist_shrink = self.shrink_probs();
        // Local = -ln p_shrink(x)
        self.dataset.data.mapv(|v| -dist_shrink[&v].ln())
    }

    fn global_value(&self) -> f64 {
        // H = -sum p_shrink ln p_shrink over unique support
        let dist_shrink = self.shrink_probs();
        let mut h = 0.0_f64;
        for &p in dist_shrink.values() {
            if p > 0.0 { h -= p * p.ln(); }
        }
        h
    }
}

impl OptionalLocalValues for ShrinkEntropy {
    fn supports_local(&self) -> bool { true }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> { Ok(self.local_values()) }
}

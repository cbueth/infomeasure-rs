use ndarray::{Array1, Array2};
use crate::estimators::traits::{LocalValues, OptionalLocalValues};
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};

/// Standard Shannon entropy estimator for discrete data using maximum likelihood (natural log base).
///
/// This baseline estimator computes H = -Σ p_i ln p_i from empirical probabilities p_i = n_i/N.
/// It supports local values via LocalValues, where each sample contributes -ln p(x).
///
/// Suitable as a reference and for well-sampled regimes; for small N or large K consider
/// bias-corrected estimators (Miller–Madow, Grassberger, Zhang) or coverage-based ones (Chao–Shen).
///
/// GPU acceleration (optional): When compiled with the `gpu_support` feature, `from_rows` may
/// use a WGSL compute shader to accelerate dense per-row histogramming for 2D inputs with a small
/// global value range. If the GPU path is unavailable or inapplicable, it transparently falls back
/// to the CPU implementation.
pub struct DiscreteEntropy {
    dataset: DiscreteDataset,
}

impl DiscreteEntropy {
    pub fn new(data: Array1<i32>) -> Self {
        let dataset = DiscreteDataset::from_data(data);
        Self { dataset }
    }

    /// Build a vector of DiscreteEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<i32>) -> Vec<Self> {
        #[cfg(feature = "gpu_support")]
        {
            if let Some(counts_per_row) = crate::estimators::approaches::discrete::discrete_gpu::gpu_histogram_rows_dense(&data) {
                // Build using precomputed counts to avoid CPU histogram work
                let rows = rows_as_vec(data.clone());
                return rows.into_iter().zip(counts_per_row.into_iter()).map(|(row, counts)| {
                    let dataset = DiscreteDataset::from_counts_and_data(row, counts);
                    Self { dataset }
                }).collect();
            }
        }
        // Fallback to CPU path
        rows_as_vec(data).into_iter().map(Self::new).collect()
    }
}

impl LocalValues for DiscreteEntropy {
    /// Calculate local entropy values for each element in the dataset.
    fn local_values(&self) -> Array1<f64> {
        // Map each value to its probability: local = -ln p(x)
        let p_local = self.dataset.map_probs();
        -p_local.mapv(f64::ln)
    }

    /// Calculate global entropy for the data set.
    /// Separate implementation, not inferred from local_values.
    fn global_value(&self) -> f64 {
        let n_f = self.dataset.n as f64;
        // -sum(p * ln p). Order of iteration doesn't matter for sum.
        let mut h = 0.0_f64;
        for &cnt in self.dataset.counts.values() {
            let p = (cnt as f64) / n_f;
            h -= if p > 0.0 { p * p.ln() } else { 0.0 };
        }
        h
    }
}

impl OptionalLocalValues for DiscreteEntropy {
    fn supports_local(&self) -> bool { true }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Ok(self.local_values())
    }
}

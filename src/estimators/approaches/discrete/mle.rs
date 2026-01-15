// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{
    CrossEntropy, GlobalValue, JointEntropy, LocalValues, OptionalLocalValues,
};
use ndarray::{Array1, Array2};

/// Standard Shannon entropy estimator for discrete data using maximum likelihood (natural log base).
///
/// This baseline estimator computes H = -Σ p_i ln p_i from empirical probabilities p_i = n_i/N.
/// It supports local values via LocalValues, where each sample contributes -ln p(x).
///
/// Cross-entropy is supported between two distributions.
///
/// Joint entropy is supported by reducing the joint space of multiple variables to a single
/// discrete representation before estimation.
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
            if let Some(counts_per_row) =
                crate::estimators::approaches::discrete::mle_gpu::gpu_histogram_rows_dense(&data)
            {
                // Build using precomputed counts to avoid CPU histogram work
                let rows = rows_as_vec(data.clone());
                return rows
                    .into_iter()
                    .zip(counts_per_row)
                    .map(|(row, counts)| {
                        let dataset = DiscreteDataset::from_counts_and_data(row, counts);
                        Self { dataset }
                    })
                    .collect();
            }
        }
        // Fallback to CPU path
        rows_as_vec(data).into_iter().map(Self::new).collect()
    }
}

impl GlobalValue for DiscreteEntropy {
    /// Calculate global entropy for the data set.
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

impl LocalValues for DiscreteEntropy {
    /// Calculate local entropy values for each element in the dataset.
    fn local_values(&self) -> Array1<f64> {
        // Map each value to its probability: local = -ln p(x)
        let p_local = self.dataset.map_probs();
        -p_local.mapv(f64::ln)
    }
}

impl CrossEntropy for DiscreteEntropy {
    /// Cross-entropy H(P||Q) = -Σ_x p(x) ln q(x)
    fn cross_entropy(&self, other: &DiscreteEntropy) -> f64 {
        use std::collections::HashSet;
        // Build sets of supports
        let supp_p: HashSet<i32> = self.dataset.counts.keys().cloned().collect();
        let supp_q: HashSet<i32> = other.dataset.counts.keys().cloned().collect();
        let inter: HashSet<i32> = supp_p.intersection(&supp_q).cloned().collect();
        if inter.is_empty() {
            return 0.0;
        }

        let p_map = &self.dataset.dist;
        let q_map = &other.dataset.dist;
        let mut h = 0.0_f64;
        for v in inter {
            if let (Some(&p), Some(&q)) = (p_map.get(&v), q_map.get(&v))
                && p > 0.0
                && q > 0.0
            {
                h -= p * q.ln();
            }
        }
        h
    }
}

impl JointEntropy for DiscreteEntropy {
    type Source = Array1<i32>;
    type Params = ();

    fn joint_entropy(series: &[Self::Source], _params: Self::Params) -> f64 {
        if series.is_empty() {
            return 0.0;
        }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = DiscreteEntropy::new(joint_codes);
        GlobalValue::global_value(&disc)
    }
}

impl OptionalLocalValues for DiscreteEntropy {
    fn supports_local(&self) -> bool {
        true
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Ok(self.local_values())
    }
}

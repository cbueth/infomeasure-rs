// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::approaches::discrete::discrete_utils::{DiscreteDataset, rows_as_vec};
use crate::estimators::traits::{GlobalValue, JointEntropy, LocalValues, OptionalLocalValues};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Shrinkage (James–Stein) entropy estimator for discrete data (natural log base).
///
/// Forms a convex combination between the empirical distribution and the uniform target
/// with a data-driven shrinkage intensity $λ ∈ \[0,1\]$. This reduces variance and bias
/// in undersampled regimes. Supports local values via $-\ln p_\mathrm{shrink}(x)$.
///
/// Cross-entropy is not implemented for shrinkage estimator.
/// The shrinkage correction is designed for bias correction in entropy
/// estimation using a specific shrinkage target, and cross-entropy mixes
/// probabilities from one distribution with corrections from another,
/// creating a theoretical inconsistency.
///
/// Joint entropy is supported by reducing the joint space of multiple variables to a single
/// discrete representation before estimation.
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
            if self.dataset.n > 1 {
                var_sum += u * (1.0 - u) / (n - 1.0);
            }
            // mean squared difference to target
            msp += (u - t) * (u - t);
        }

        // lambda in [0,1]
        let lambda = if self.dataset.n <= 1 || msp == 0.0 {
            1.0
        } else {
            let l = var_sum / msp;
            l.clamp(0.0, 1.0)
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

impl GlobalValue for ShrinkEntropy {
    fn global_value(&self) -> f64 {
        // H = -sum p_shrink ln p_shrink over unique support
        let dist_shrink = self.shrink_probs();
        let mut h = 0.0_f64;
        for &p in dist_shrink.values() {
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    }
}

impl LocalValues for ShrinkEntropy {
    fn local_values(&self) -> Array1<f64> {
        let dist_shrink = self.shrink_probs();
        // Local = -ln p_shrink(x)
        self.dataset.data.mapv(|v| -dist_shrink[&v].ln())
    }
}

impl JointEntropy for ShrinkEntropy {
    type Source = Array1<i32>;
    type Params = ();

    fn joint_entropy(series: &[Self::Source], _params: Self::Params) -> f64 {
        if series.is_empty() {
            return 0.0;
        }
        let joint_codes = reduce_joint_space_compact(series);
        let disc = ShrinkEntropy::new(joint_codes);
        GlobalValue::global_value(&disc)
    }
}

impl OptionalLocalValues for ShrinkEntropy {
    fn supports_local(&self) -> bool {
        true
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Ok(self.local_values())
    }
}

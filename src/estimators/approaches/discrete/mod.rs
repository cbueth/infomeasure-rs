// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// Discrete estimators module: groups all discrete-related submodules
// and exposes them to the parent approaches module.

#[cfg(feature = "gpu_support")]
pub mod discrete_gpu;
pub mod discrete_utils;

pub mod ansb;
pub mod bayes;
pub mod bonachela;
pub mod chao_shen;
pub mod chao_wang_jost;
pub mod grassberger;
pub mod miller_madow;
pub mod mle;
pub mod nsb;
pub mod shrink;
pub mod zhang;

// Additional helpers
pub mod discrete_batch;

use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::traits::{
    ConditionalMutualInformationEstimator, GlobalValue, LocalValues, MutualInformationEstimator,
    OptionalLocalValues,
};
use ndarray::Array1;

/// Discrete Mutual Information estimator using the entropy-summation formula.
///
/// This estimator can wrap any discrete entropy estimator.
pub struct DiscreteMutualInformation<E> {
    marginals: Vec<E>,
    joint: E,
}

impl<E> DiscreteMutualInformation<E> {
    pub fn new<F>(series: &[Array1<i32>], constructor: F) -> Self
    where
        F: Fn(Array1<i32>) -> E + Clone,
    {
        let marginals = series.iter().cloned().map(constructor.clone()).collect();
        let joint_codes = reduce_joint_space_compact(series);
        let joint = constructor(joint_codes);
        Self { marginals, joint }
    }
}

impl<E: GlobalValue> GlobalValue for DiscreteMutualInformation<E> {
    fn global_value(&self) -> f64 {
        let h_marginals: f64 = self.marginals.iter().map(|m| m.global_value()).sum();
        let h_joint = self.joint.global_value();
        // I(X1; ...; Xn) = sum H(Xi) - H(X1, ..., Xn)
        h_marginals - h_joint
    }
}

impl<E: LocalValues> LocalValues for DiscreteMutualInformation<E> {
    fn local_values(&self) -> Array1<f64> {
        let mut res = Array1::zeros(self.joint.local_values().len());
        for m in &self.marginals {
            res += &m.local_values();
        }
        res -= &self.joint.local_values();
        res
    }
}

impl<E: OptionalLocalValues> OptionalLocalValues for DiscreteMutualInformation<E> {
    fn supports_local(&self) -> bool {
        self.joint.supports_local() && self.marginals.iter().all(|m| m.supports_local())
    }

    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        if !self.supports_local() {
            return Err("One or more underlying entropy estimators do not support local values.");
        }

        let mut res = self.marginals[0].local_values_opt()?;
        for m in &self.marginals[1..] {
            res += &m.local_values_opt()?;
        }
        res -= &self.joint.local_values_opt()?;
        // i(x,y) = h(x) + h(y) - h(x,y)
        Ok(res)
    }
}

impl<E: GlobalValue + OptionalLocalValues> MutualInformationEstimator
    for DiscreteMutualInformation<E>
{
}

/// Discrete Conditional Mutual Information estimator using the entropy-summation formula.
pub struct DiscreteConditionalMutualInformation<E> {
    marginal_conds: Vec<E>,
    joint_cond: E,
    cond_only: E,
}

impl<E> DiscreteConditionalMutualInformation<E> {
    pub fn new<F>(series: &[Array1<i32>], cond: &Array1<i32>, constructor: F) -> Self
    where
        F: Fn(Array1<i32>) -> E + Clone,
    {
        // I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
        // General: I(X1; ...; Xn | Z) = sum H(Xi, Z) - H(X1, ..., Xn, Z) - (n-1)H(Z)

        let marginal_conds = series
            .iter()
            .map(|s| {
                let joint_xz = reduce_joint_space_compact(&[s.clone(), cond.clone()]);
                constructor.clone()(joint_xz)
            })
            .collect();

        let mut joint_all_vec = series.to_vec();
        joint_all_vec.push(cond.clone());
        let joint_all_codes = reduce_joint_space_compact(&joint_all_vec);
        let joint_cond = constructor.clone()(joint_all_codes);

        let cond_only = constructor(cond.clone());

        Self {
            marginal_conds,
            joint_cond,
            cond_only,
        }
    }
}

impl<E: GlobalValue> GlobalValue for DiscreteConditionalMutualInformation<E> {
    fn global_value(&self) -> f64 {
        let n = self.marginal_conds.len() as f64;
        let sum_h_xz: f64 = self.marginal_conds.iter().map(|m| m.global_value()).sum();
        let h_xyz = self.joint_cond.global_value();
        let h_z = self.cond_only.global_value();
        sum_h_xz - h_xyz - (n - 1.0) * h_z
    }
}

impl<E: LocalValues> LocalValues for DiscreteConditionalMutualInformation<E> {
    fn local_values(&self) -> Array1<f64> {
        let n = self.marginal_conds.len() as f64;
        let mut res = Array1::zeros(self.joint_cond.local_values().len());
        for m in &self.marginal_conds {
            res += &m.local_values();
        }
        res -= &self.joint_cond.local_values();
        res -= &((n - 1.0) * self.cond_only.local_values());
        res
    }
}

impl<E: OptionalLocalValues> OptionalLocalValues for DiscreteConditionalMutualInformation<E> {
    fn supports_local(&self) -> bool {
        self.joint_cond.supports_local()
            && self.cond_only.supports_local()
            && self.marginal_conds.iter().all(|m| m.supports_local())
    }

    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        if !self.supports_local() {
            return Err("One or more underlying entropy estimators do not support local values.");
        }
        let n = self.marginal_conds.len() as f64;
        let mut res = self.marginal_conds[0].local_values_opt()?;
        for m in &self.marginal_conds[1..] {
            res += &m.local_values_opt()?;
        }
        res -= &self.joint_cond.local_values_opt()?;
        res -= &((n - 1.0) * self.cond_only.local_values_opt()?);
        Ok(res)
    }
}

impl<E: GlobalValue + OptionalLocalValues> ConditionalMutualInformationEstimator
    for DiscreteConditionalMutualInformation<E>
{
}

/// Discrete Transfer Entropy estimator using the entropy-summation formula (via CMI).
pub struct DiscreteTransferEntropy<E> {
    inner: DiscreteConditionalMutualInformation<E>,
}

impl<E> DiscreteTransferEntropy<E> {
    pub fn new<F>(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
        constructor: F,
    ) -> Self
    where
        F: Fn(Array1<i32>) -> E + Clone,
    {
        use crate::estimators::approaches::discrete::discrete_utils::reduce_array2_compact;
        use crate::estimators::utils::te_slicing::te_slices;

        let (dest_future, dest_history, src_history) =
            te_slices(source, destination, src_hist_len, dest_hist_len, step_size);

        let src_past_codes = reduce_array2_compact(&src_history);
        let dest_past_codes = reduce_array2_compact(&dest_history);
        let dest_future_flat = dest_future.column(0).to_owned();

        // TE(X -> Y) = I(X_past; Y_next | Y_past)
        let inner = DiscreteConditionalMutualInformation::new(
            &[src_past_codes, dest_future_flat],
            &dest_past_codes,
            constructor,
        );
        Self { inner }
    }
}

impl<E: GlobalValue> GlobalValue for DiscreteTransferEntropy<E> {
    fn global_value(&self) -> f64 {
        self.inner.global_value()
    }
}

impl<E: OptionalLocalValues> OptionalLocalValues for DiscreteTransferEntropy<E> {
    fn supports_local(&self) -> bool {
        self.inner.supports_local()
    }

    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        self.inner.local_values_opt()
    }
}

/// Discrete Conditional Transfer Entropy estimator using the entropy-summation formula (via CMI).
pub struct DiscreteConditionalTransferEntropy<E> {
    inner: DiscreteConditionalMutualInformation<E>,
}

impl<E> DiscreteConditionalTransferEntropy<E> {
    pub fn new<F>(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
        constructor: F,
    ) -> Self
    where
        F: Fn(Array1<i32>) -> E + Clone,
    {
        use crate::estimators::approaches::discrete::discrete_utils::{
            reduce_array2_compact, reduce_joint_space_compact,
        };
        use crate::estimators::utils::te_slicing::cte_slices;

        let (dest_future, dest_history, src_history, cond_history) = cte_slices(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
        );

        let src_past_codes = reduce_array2_compact(&src_history);
        let dest_past_codes = reduce_array2_compact(&dest_history);
        let cond_past_codes = reduce_array2_compact(&cond_history);
        let dest_future_flat = dest_future.column(0).to_owned();

        // CTE(X -> Y | Z) = I(X_past; Y_next | Y_past, Z_past)
        let joint_cond_codes = reduce_joint_space_compact(&[dest_past_codes, cond_past_codes]);

        let inner = DiscreteConditionalMutualInformation::new(
            &[src_past_codes, dest_future_flat],
            &joint_cond_codes,
            constructor,
        );
        Self { inner }
    }
}

impl<E: GlobalValue> GlobalValue for DiscreteConditionalTransferEntropy<E> {
    fn global_value(&self) -> f64 {
        self.inner.global_value()
    }
}

impl<E: OptionalLocalValues> OptionalLocalValues for DiscreteConditionalTransferEntropy<E> {
    fn supports_local(&self) -> bool {
        self.inner.supports_local()
    }

    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        self.inner.local_values_opt()
    }
}

// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Discrete Estimators
//!
//! This module implements estimators for Shannon entropy and derived measures
//! (Mutual Information, Transfer Entropy) for discrete/categorical data.
//!
//! ## Theory
//!
//! Discrete estimators use frequency counts (histograms) to estimate probabilities
//! and then compute information measures. The most basic is the Maximum Likelihood
//! Estimator (MLE):
//!
//! $$\hat{H} = -\sum_{i=1}^{K} \hat{p}_i \log \hat{p}_i, \quad \hat{p}_i = \frac{n_i}{N}$$
//!
//! where $n_i$ are counts of unique values, $N$ is total samples, and $K$ is number
//! of unique values (bins).
//!
//! ## Bias Correction
//!
//! Discrete estimators are notoriously biased for small sample sizes. This module
//! provides several bias-corrected variants:
//!
//! - **Miller-Madow**: Adds a simple correction term $(K-1)/(2N)$.
//! - **Grassberger**: Uses digamma functions to reduce bias in small samples.
//! - **Shrinkage (James-Stein)**: Regularizes probability estimates toward a uniform distribution.
//! - **Chao-Shen**: Uses coverage estimation to account for unobserved states.
//! - **NSB (Nemenman-Shafee-Bialek)**: A Bayesian estimator using a mixture of Dirichlet priors,
//!   designed for extremely undersampled data.
//!
//! ## Measures Implemented
//!
//! This module provides generic wrappers that can take *any* of the discrete entropy
//! estimators listed above and use them to compute:
//!
//! - **Mutual Information**: $I(X; Y) = H(X) + H(Y) - H(X, Y)$
//! - **Conditional MI**: $I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)$
//! - **Transfer Entropy**: $T(X \to Y) = I(X_{\mathrm{past}}; Y_{\mathrm{future}} | Y_{\mathrm{past}})$
//!
//! ## See Also
//! - [Discrete Entropy Guide](crate::guide::entropy::discrete) — Detailed comparison of estimators
//! - [Estimator Approaches](super) — Overview of all estimation techniques
//!
//! ## References
//!
//! - [Grassberger, 1988](crate::guide::references#grassberger1988)
//! - [Hausser & Strimmer, 2009](crate::guide::references#hausser2009)
//! - [Nemenman et al., 2002](crate::guide::references#nsb2002)

pub mod discrete_utils;
#[cfg(feature = "gpu")]
pub mod mle_gpu;

mod dense_cmi;
mod dense_mi;

pub use dense_cmi::{DenseCmiBuilder, DenseCmiGlobal};
pub use dense_mi::{DenseMiBuilder, DenseMiGlobal};

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

use crate::estimators::approaches::discrete::dense_cmi::DenseCmi;
use crate::estimators::approaches::discrete::dense_mi::DenseMi;
use crate::estimators::approaches::discrete::discrete_utils::{
    reduce_joint_space_compact, reduce_views_compact,
};
use crate::estimators::doc_macros::doc_snippets;
use crate::estimators::traits::{
    ConditionalMutualInformationEstimator, GlobalValue, LocalValues, MutualInformationEstimator,
    OptionalLocalValues,
};
use ndarray::{Array1, ArrayView1};

/// Discrete Mutual Information estimator using the entropy-summation formula.
///
/// ## Theory
///
#[doc = doc_snippets!(mi_formula "Discrete", "", "")]
///
/// This estimator can wrap any discrete entropy estimator.
pub struct DiscreteMutualInformation<E> {
    inner: MiInner<E>,
}

/// Backing representation: generic per-space entropy estimators, or the dense
/// direct counts used by the MLE fast path.
enum MiInner<E> {
    Spaces { marginals: Vec<E>, joint: E },
    Dense(Box<DenseMi>),
}

impl<E> DiscreteMutualInformation<E> {
    pub fn new<F>(series: &[Array1<i32>], constructor: F) -> Self
    where
        F: Fn(Array1<i32>) -> E + Clone,
    {
        let marginals = series.iter().cloned().map(constructor.clone()).collect();
        let joint_codes = reduce_joint_space_compact(series);
        let joint = constructor(joint_codes);
        Self {
            inner: MiInner::Spaces { marginals, joint },
        }
    }

    /// Wrap a dense direct state as the estimator.
    pub(crate) fn from_dense(dense: DenseMi) -> Self {
        Self {
            inner: MiInner::Dense(Box::new(dense)),
        }
    }
}

impl DiscreteMutualInformation<crate::estimators::approaches::discrete::mle::DiscreteEntropy> {
    /// Fused MLE construction: dense direct counts when the joint alphabet is
    /// small, else the generic entropy-summation estimator.
    pub(crate) fn new_mle(series: &[Array1<i32>]) -> Self {
        let views: Vec<ArrayView1<i32>> = series.iter().map(|s| s.view()).collect();
        if let Some(dense) = DenseMi::new(&views) {
            return Self::from_dense(dense);
        }
        Self::new(
            series,
            crate::estimators::approaches::discrete::mle::DiscreteEntropy::new,
        )
    }
}

impl<E: GlobalValue> GlobalValue for DiscreteMutualInformation<E> {
    fn global_value(&self) -> f64 {
        match &self.inner {
            MiInner::Dense(dense) => dense.global_value(),
            MiInner::Spaces { marginals, joint } => {
                let h_marginals: f64 = marginals.iter().map(|m| m.global_value()).sum();
                // I(X1; ...; Xn) = sum H(Xi) - H(X1, ..., Xn)
                h_marginals - joint.global_value()
            }
        }
    }
}

impl<E: LocalValues> LocalValues for DiscreteMutualInformation<E> {
    fn local_values(&self) -> Array1<f64> {
        match &self.inner {
            MiInner::Dense(dense) => dense.local_values(),
            MiInner::Spaces { marginals, joint } => {
                let mut res = Array1::zeros(joint.local_values().len());
                for m in marginals {
                    res += &m.local_values();
                }
                res -= &joint.local_values();
                res
            }
        }
    }
}

impl<E: OptionalLocalValues> OptionalLocalValues for DiscreteMutualInformation<E> {
    fn supports_local(&self) -> bool {
        match &self.inner {
            MiInner::Dense(_) => true,
            MiInner::Spaces { marginals, joint } => {
                joint.supports_local() && marginals.iter().all(|m| m.supports_local())
            }
        }
    }

    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        match &self.inner {
            MiInner::Dense(dense) => Ok(dense.local_values()),
            MiInner::Spaces { marginals, joint } => {
                if !self.supports_local() {
                    return Err(
                        "One or more underlying entropy estimators do not support local values.",
                    );
                }
                let mut res = marginals[0].local_values_opt()?;
                for m in &marginals[1..] {
                    res += &m.local_values_opt()?;
                }
                res -= &joint.local_values_opt()?;
                // i(x,y) = h(x) + h(y) - h(x,y)
                Ok(res)
            }
        }
    }
}

impl<E: GlobalValue + OptionalLocalValues> MutualInformationEstimator
    for DiscreteMutualInformation<E>
{
}

/// Discrete Conditional Mutual Information estimator using the entropy-summation formula.
///
/// ## Theory
///
#[doc = doc_snippets!(cmi_formula "Discrete", "", "")]
pub struct DiscreteConditionalMutualInformation<E> {
    inner: CmiInner<E>,
}

/// Backing representation: generic per-space entropy estimators, or the dense
/// direct counts used by the MLE fast path.
enum CmiInner<E> {
    Spaces {
        marginal_conds: Vec<E>,
        joint_cond: E,
        cond_only: E,
    },
    Dense(Box<DenseCmi>),
}

impl<E> DiscreteConditionalMutualInformation<E> {
    pub fn new<F>(series: &[Array1<i32>], cond: &Array1<i32>, constructor: F) -> Self
    where
        F: Fn(Array1<i32>) -> E + Clone,
    {
        // I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
        // General: I(X1; ...; Xn | Z) = sum H(Xi, Z) - H(X1, ..., Xn, Z) - (n-1)H(Z)

        // Reduce straight over borrowed views: cloning each series per
        // marginal (and the whole stack for the joint) showed up as ~19% of
        // discrete TE wall time in profiler scans.
        let mut cond_view_stack: Vec<ArrayView1<i32>> = series.iter().map(|s| s.view()).collect();
        cond_view_stack.push(cond.view());

        let marginal_conds = (0..series.len())
            .map(|i| {
                let pair = [series[i].view(), cond.view()];
                let joint_xz = reduce_views_compact(&pair);
                constructor.clone()(joint_xz)
            })
            .collect();

        let joint_all_codes = reduce_views_compact(&cond_view_stack);
        let joint_cond = constructor.clone()(joint_all_codes);

        let cond_only = constructor(cond.clone());

        Self {
            inner: CmiInner::Spaces {
                marginal_conds,
                joint_cond,
                cond_only,
            },
        }
    }
}

impl
    DiscreteConditionalMutualInformation<
        crate::estimators::approaches::discrete::mle::DiscreteEntropy,
    >
{
    /// Fused MLE construction from already-reduced code columns.
    ///
    /// Uses the dense direct path when the joint alphabet is small (one
    /// counting pass, no per-space entropy/dataset maps); otherwise falls back
    /// to the generic entropy-summation estimator.
    pub(crate) fn new_mle(series: &[Array1<i32>], cond: &Array1<i32>) -> Self {
        let views: Vec<ArrayView1<i32>> = series.iter().map(|s| s.view()).collect();
        Self::new_mle_views(&views, cond.view())
    }

    pub(crate) fn new_mle_views(series: &[ArrayView1<i32>], cond: ArrayView1<i32>) -> Self {
        use crate::estimators::approaches::discrete::mle::DiscreteEntropy;

        let groups: Vec<&[ArrayView1<i32>]> = series.iter().map(std::slice::from_ref).collect();
        let cond_group = std::slice::from_ref(&cond);
        if let Some(dense) = DenseCmi::new(&groups, cond_group) {
            return Self::from_dense(dense);
        }
        // Large alphabet: generic engine. Own the views so the estimator is
        // self-contained.
        let owned: Vec<Array1<i32>> = series.iter().map(|v| v.to_owned()).collect();
        Self::new(&owned, &cond.to_owned(), DiscreteEntropy::new)
    }
}

impl<E> DiscreteConditionalMutualInformation<E> {
    /// Wrap a dense direct state as the estimator.
    pub(crate) fn from_dense(dense: DenseCmi) -> Self {
        Self {
            inner: CmiInner::Dense(Box::new(dense)),
        }
    }
}

impl<E: GlobalValue> GlobalValue for DiscreteConditionalMutualInformation<E> {
    fn global_value(&self) -> f64 {
        match &self.inner {
            CmiInner::Dense(dense) => dense.global_value(),
            CmiInner::Spaces {
                marginal_conds,
                joint_cond,
                cond_only,
            } => {
                let n = marginal_conds.len() as f64;
                let sum_h_xz: f64 = marginal_conds.iter().map(|m| m.global_value()).sum();
                sum_h_xz - joint_cond.global_value() - (n - 1.0) * cond_only.global_value()
            }
        }
    }
}

impl<E: LocalValues> LocalValues for DiscreteConditionalMutualInformation<E> {
    fn local_values(&self) -> Array1<f64> {
        match &self.inner {
            CmiInner::Dense(dense) => dense.local_values(),
            CmiInner::Spaces {
                marginal_conds,
                joint_cond,
                cond_only,
            } => {
                let n = marginal_conds.len() as f64;
                let mut res = Array1::zeros(joint_cond.local_values().len());
                for m in marginal_conds {
                    res += &m.local_values();
                }
                res -= &joint_cond.local_values();
                res -= &((n - 1.0) * cond_only.local_values());
                res
            }
        }
    }
}

impl<E: OptionalLocalValues> OptionalLocalValues for DiscreteConditionalMutualInformation<E> {
    fn supports_local(&self) -> bool {
        match &self.inner {
            CmiInner::Dense(_) => true,
            CmiInner::Spaces {
                marginal_conds,
                joint_cond,
                cond_only,
            } => {
                joint_cond.supports_local()
                    && cond_only.supports_local()
                    && marginal_conds.iter().all(|m| m.supports_local())
            }
        }
    }

    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        match &self.inner {
            CmiInner::Dense(dense) => Ok(dense.local_values()),
            CmiInner::Spaces {
                marginal_conds,
                joint_cond,
                cond_only,
            } => {
                if !joint_cond.supports_local()
                    || !cond_only.supports_local()
                    || !marginal_conds.iter().all(|m| m.supports_local())
                {
                    return Err(
                        "One or more underlying entropy estimators do not support local values.",
                    );
                }
                let n = marginal_conds.len() as f64;
                let mut res = marginal_conds[0].local_values_opt()?;
                for m in &marginal_conds[1..] {
                    res += &m.local_values_opt()?;
                }
                res -= &joint_cond.local_values_opt()?;
                res -= &((n - 1.0) * cond_only.local_values_opt()?);
                Ok(res)
            }
        }
    }
}

impl<E: GlobalValue + OptionalLocalValues> ConditionalMutualInformationEstimator
    for DiscreteConditionalMutualInformation<E>
{
}

/// Discrete Transfer Entropy estimator using the entropy-summation formula (via CMI).
///
/// ## Theory
///
#[doc = doc_snippets!(te_formula "Discrete", "", "")]
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
        use crate::estimators::approaches::discrete::discrete_utils::reduce_hist_columns_compact;
        use crate::estimators::utils::te_slicing::te_embedding_views;

        // Zero-copy strided views instead of materialising three intermediate
        // arrays that were only ever reduced column-wise.
        let views = te_embedding_views(source, destination, src_hist_len, dest_hist_len, step_size);

        let src_past_codes = reduce_hist_columns_compact(views.src_past_cols.iter().copied());
        let dest_past_codes = reduce_hist_columns_compact(views.dest_past_cols.iter().copied());
        let dest_future_flat = views.dest_future.to_owned();

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

impl DiscreteTransferEntropy<crate::estimators::approaches::discrete::mle::DiscreteEntropy> {
    /// Fused MLE construction: one counting pass per information space, no
    /// intermediate dense-code recounting. Numerics identical to
    /// [`DiscreteTransferEntropy::new`] with an MLE constructor up to
    /// floating-point summation order.
    pub(crate) fn new_mle(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> Self {
        use crate::estimators::approaches::discrete::discrete_utils::reduce_hist_columns_compact;
        use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
        use crate::estimators::utils::te_slicing::te_embedding_views;

        let views = te_embedding_views(source, destination, src_hist_len, dest_hist_len, step_size);

        // One fused pass over the raw columns: the source history (a column
        // group) and Y_t are the variables, Y_past the condition.
        let series: [&[ArrayView1<i32>]; 2] = [
            &views.src_past_cols,
            std::slice::from_ref(&views.dest_future),
        ];
        if let Some(dense) = DenseCmi::new(&series, &views.dest_past_cols) {
            return Self {
                inner: DiscreteConditionalMutualInformation::from_dense(dense),
            };
        }
        // Large joint: reduce histories to codes and use the generic engine.
        let src_past = reduce_hist_columns_compact(views.src_past_cols.iter().copied());
        let dest_past = reduce_hist_columns_compact(views.dest_past_cols.iter().copied());
        let inner = DiscreteConditionalMutualInformation::new(
            &[src_past, views.dest_future.to_owned()],
            &dest_past,
            DiscreteEntropy::new,
        );
        Self { inner }
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
///
/// ## Theory
///
#[doc = doc_snippets!(cte_formula "Discrete", "", "")]
pub struct DiscreteConditionalTransferEntropy<E> {
    inner: DiscreteConditionalMutualInformation<E>,
}

impl<E> DiscreteConditionalTransferEntropy<E> {
    #[allow(clippy::too_many_arguments)]
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
            reduce_hist_columns_compact, reduce_joint_space_compact,
        };
        use crate::estimators::utils::te_slicing::cte_embedding_views;

        let views = cte_embedding_views(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
        );

        let src_past_codes = reduce_hist_columns_compact(views.src_past_cols.iter().copied());
        let dest_past_codes = reduce_hist_columns_compact(views.dest_past_cols.iter().copied());
        let cond_past_codes = reduce_hist_columns_compact(views.cond_past_cols.iter().copied());
        let dest_future_flat = views.dest_future.to_owned();

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

impl
    DiscreteConditionalTransferEntropy<
        crate::estimators::approaches::discrete::mle::DiscreteEntropy,
    >
{
    /// Fused MLE construction for conditional transfer entropy; see
    /// [`DiscreteTransferEntropy::new_mle`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_mle(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> Self {
        use crate::estimators::approaches::discrete::discrete_utils::{
            reduce_hist_columns_compact, reduce_joint_space_compact,
        };
        use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
        use crate::estimators::utils::te_slicing::cte_embedding_views;

        let views = cte_embedding_views(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
        );

        // Fused pass: X_past and Y_t are the variables; the condition group is
        // (Y_past, Z_past). No separate history reduction or Z' packing.
        let mut cond_cols: Vec<ArrayView1<i32>> = views.dest_past_cols.clone();
        cond_cols.extend(views.cond_past_cols.iter().copied());
        let series: [&[ArrayView1<i32>]; 2] = [
            &views.src_past_cols,
            std::slice::from_ref(&views.dest_future),
        ];
        if let Some(dense) = DenseCmi::new(&series, &cond_cols) {
            return Self {
                inner: DiscreteConditionalMutualInformation::from_dense(dense),
            };
        }
        // Large joint: reduce histories to codes and use the generic engine.
        let src_past = reduce_hist_columns_compact(views.src_past_cols.iter().copied());
        let dest_past = reduce_hist_columns_compact(views.dest_past_cols.iter().copied());
        let cond_past = reduce_hist_columns_compact(views.cond_past_cols.iter().copied());
        let joint_cond = reduce_joint_space_compact(&[dest_past, cond_past]);
        let inner = DiscreteConditionalMutualInformation::new(
            &[src_past, views.dest_future.to_owned()],
            &joint_cond,
            DiscreteEntropy::new,
        );
        Self { inner }
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

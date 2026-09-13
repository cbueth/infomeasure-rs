// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dense direct discrete mutual information for small alphabets.
//!
//! The generic [`DiscreteMutualInformation`] reduces every marginal and the
//! joint into separate entropy estimators (hash maps). When the joint alphabet
//! is small the whole measure can instead be computed from dense count arrays:
//! `I(X1; …; Xn) = Σ H(Xi) − H(X1, …, Xn)`.
//!
//! See [`DenseCmiBuilder`](super::DenseCmiBuilder) for the timing-optimised
//! builder pattern shared by both measures: [`DenseMiBuilder::with_alphabet`]
//! skips the min/max scan and [`DenseMiBuilder::global_only`] retains no inputs.
//!
//! [`DiscreteMutualInformation`]: super::DiscreteMutualInformation

use crate::estimators::approaches::discrete::DiscreteMutualInformation;
use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
use crate::estimators::traits::{GlobalValue, MutualInformationEstimator, OptionalLocalValues};
use ndarray::{Array1, ArrayView1};
use std::borrow::Cow;

/// Largest dense joint table built for the direct path. Above this the caller
/// falls back to the generic entropy-summation estimator.
const DENSE_MI_CAP: u128 = 1 << 20;

/// Number of variables above which the direct path bails out.
const MAX_DENSE_VARS: usize = 16;

/// Layout of a dense MI over contiguous code columns (one column per variable).
struct DenseMiPlan {
    n: usize,
    n_vars: usize,
    col_min: Vec<i32>,
    var_range: Vec<usize>,
    var_joint_stride: Vec<usize>,
    joint_len: usize,
    marginal_off: Vec<usize>,
    marginal_len: usize,
}

impl DenseMiPlan {
    fn new(cols: &[&[i32]], alphabet: Option<usize>) -> Option<Self> {
        if cols.is_empty() || cols.len() > MAX_DENSE_VARS {
            return None;
        }
        let n = cols[0].len();
        if n == 0 || cols.iter().any(|c| c.len() != n) {
            return None;
        }
        let n_vars = cols.len();
        let mut col_min = vec![0i32; n_vars];
        let mut var_range = vec![0usize; n_vars];
        for (i, col) in cols.iter().enumerate() {
            let (mn, rng) = match alphabet {
                Some(k) => (0i32, k),
                None => {
                    let (mn, mx) = col
                        .iter()
                        .fold((i32::MAX, i32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                    if mn == i32::MAX {
                        return None;
                    }
                    (mn, (mx as i64 - mn as i64 + 1) as usize)
                }
            };
            col_min[i] = mn;
            var_range[i] = rng;
        }

        let mut cap: u128 = 1;
        for &r in &var_range {
            cap = cap.checked_mul(r as u128)?;
            if cap > DENSE_MI_CAP {
                return None;
            }
        }
        let joint_len = cap as usize;

        let mut var_joint_stride = vec![0usize; n_vars];
        let mut acc = 1usize;
        for i in 0..n_vars {
            var_joint_stride[i] = acc;
            acc = acc.saturating_mul(var_range[i]);
        }
        let mut marginal_off = vec![0usize; n_vars];
        let mut marginal_len = 0usize;
        for i in 0..n_vars {
            marginal_off[i] = marginal_len;
            marginal_len += var_range[i];
        }

        Some(Self {
            n,
            n_vars,
            col_min,
            var_range,
            var_joint_stride,
            joint_len,
            marginal_off,
            marginal_len,
        })
    }
}

struct DenseMiCounts {
    joint: Vec<u32>,
    marginal: Vec<u32>,
    global: f64,
}

/// One fused pass: mixed-radix joint code and per-variable marginal counts,
/// then `Σ H(Xi) − H(X1, …, Xn)` over the dense cells.
#[allow(clippy::needless_range_loop)]
fn count_mi(cols: &[&[i32]], plan: &DenseMiPlan) -> DenseMiCounts {
    let mut joint = vec![0u32; plan.joint_len];
    let mut marginal = vec![0u32; plan.marginal_len];
    for t in 0..plan.n {
        let mut jidx = 0usize;
        for i in 0..plan.n_vars {
            let code = (cols[i][t] - plan.col_min[i]) as usize;
            jidx += code * plan.var_joint_stride[i];
            marginal[plan.marginal_off[i] + code] += 1;
        }
        joint[jidx] += 1;
    }

    let n = plan.n as f64;
    let entropy = |counts: &[u32]| -> f64 {
        let mut h = 0.0;
        for &c in counts {
            if c > 0 {
                let p = c as f64 / n;
                h -= p * p.ln();
            }
        }
        h
    };
    let mut h_marginals = 0.0;
    for i in 0..plan.n_vars {
        let off = plan.marginal_off[i];
        h_marginals += entropy(&marginal[off..off + plan.var_range[i]]);
    }
    let global = h_marginals - entropy(&joint);
    DenseMiCounts {
        joint,
        marginal,
        global,
    }
}

/// Dense direct discrete mutual information state (retains inputs for local
/// values).
pub(crate) struct DenseMi {
    plan: DenseMiPlan,
    /// All input columns, concatenated; variable `i` is column `i * n`.
    vals: Vec<i32>,
    joint_counts: Vec<u32>,
    marginal_counts: Vec<u32>,
    global: f64,
}

impl DenseMi {
    /// Build from `ndarray` columns (one variable each). Non-contiguous views
    /// are materialised.
    pub(crate) fn new(series: &[ArrayView1<i32>]) -> Option<Self> {
        let n = series.first()?.len();
        if n == 0 {
            return None;
        }
        let mut cols: Vec<Cow<[i32]>> = Vec::with_capacity(series.len());
        for v in series {
            if v.len() != n {
                return None;
            }
            match v.as_slice() {
                Some(s) => cols.push(Cow::Borrowed(s)),
                None => cols.push(Cow::Owned(v.iter().copied().collect())),
            }
        }
        let refs: Vec<&[i32]> = cols.iter().map(|c| &**c).collect();
        Self::from_columns(&refs, None)
    }

    pub(crate) fn from_columns(cols: &[&[i32]], alphabet: Option<usize>) -> Option<Self> {
        let plan = DenseMiPlan::new(cols, alphabet)?;
        let counts = count_mi(cols, &plan);
        let mut vals = Vec::with_capacity(cols.len() * plan.n);
        for c in cols {
            vals.extend_from_slice(c);
        }
        Some(Self {
            plan,
            vals,
            joint_counts: counts.joint,
            marginal_counts: counts.marginal,
            global: counts.global,
        })
    }

    pub(crate) fn global_value(&self) -> f64 {
        self.global
    }

    fn local_at(&self, t: usize) -> f64 {
        let mut jidx = 0usize;
        let mut denom = 0.0;
        for i in 0..self.plan.n_vars {
            let code = (self.vals[i * self.plan.n + t] - self.plan.col_min[i]) as usize;
            jidx += code * self.plan.var_joint_stride[i];
            denom += (self.marginal_counts[self.plan.marginal_off[i] + code] as f64).ln();
        }
        let n = self.plan.n as f64;
        (self.joint_counts[jidx] as f64).ln() - denom + (self.plan.n_vars as f64 - 1.0) * n.ln()
    }

    pub(crate) fn local_values(&self) -> Array1<f64> {
        let mut out = Array1::zeros(self.plan.n);
        for t in 0..self.plan.n {
            out[t] = self.local_at(t);
        }
        out
    }
}

/// Dense direct MI average without retaining inputs.
pub(crate) fn dense_mi_global(cols: &[&[i32]], alphabet: Option<usize>) -> Option<f64> {
    let plan = DenseMiPlan::new(cols, alphabet)?;
    Some(count_mi(cols, &plan).global)
}

/// Builder for the dense direct discrete-MLE mutual information.
///
/// Obtain one from [`MutualInformation::mi_discrete_mle`]. Borrows the raw
/// integer code columns, so construction never clones the inputs. Terminate
/// with [`build`](Self::build) (local-value-capable) or
/// [`global_only`](Self::global_only) (value-only, retains no inputs and does
/// not implement [`LocalValues`](crate::estimators::traits::LocalValues)).
///
/// [`MutualInformation::mi_discrete_mle`]: crate::estimators::mutual_information::MutualInformation::mi_discrete_mle
pub struct DenseMiBuilder<'a> {
    cols: Vec<Cow<'a, [i32]>>,
    alphabet: Option<usize>,
}

impl<'a> DenseMiBuilder<'a> {
    /// Build from flat raw code columns (one variable each).
    pub fn new(series: &[&'a [i32]]) -> Self {
        Self {
            cols: series.iter().map(|s| Cow::Borrowed(*s)).collect(),
            alphabet: None,
        }
    }

    /// Declare a known per-column alphabet (`states`), skipping the min/max
    /// scan. Every column must hold codes in `0..alphabet`.
    pub fn with_alphabet(mut self, alphabet: usize) -> Self {
        self.alphabet = Some(alphabet);
        self
    }

    /// Finish as a global-only estimator (inputs are consumed by the counting
    /// pass and nothing is retained).
    pub fn global_only(self) -> DenseMiGlobal {
        let refs: Vec<&[i32]> = self.cols.iter().map(|c| &**c).collect();
        let global = match dense_mi_global(&refs, self.alphabet) {
            Some(g) => g,
            None => self.generic_estimate().global_value(),
        };
        DenseMiGlobal { global }
    }

    /// Finish as the local-value-capable estimator.
    pub fn build(self) -> DiscreteMutualInformation<DiscreteEntropy> {
        let refs: Vec<&[i32]> = self.cols.iter().map(|c| &**c).collect();
        match DenseMi::from_columns(&refs, self.alphabet) {
            Some(dense) => DiscreteMutualInformation::from_dense(dense),
            None => self.generic_estimate(),
        }
    }

    fn generic_estimate(self) -> DiscreteMutualInformation<DiscreteEntropy> {
        let series: Vec<Array1<i32>> = self.cols.iter().map(|c| Array1::from(c.to_vec())).collect();
        DiscreteMutualInformation::new(&series, DiscreteEntropy::new)
    }
}

/// Global-only dense MI result.
///
/// Deliberately does **not** implement [`LocalValues`](crate::estimators::traits::LocalValues):
/// a global-only construction retains no per-sample inputs.
pub struct DenseMiGlobal {
    global: f64,
}

impl GlobalValue for DenseMiGlobal {
    fn global_value(&self) -> f64 {
        self.global
    }
}

impl OptionalLocalValues for DenseMiGlobal {
    fn supports_local(&self) -> bool {
        false
    }

    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("global-only estimator does not retain local values")
    }
}

impl MutualInformationEstimator for DenseMiGlobal {}

// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dense direct discrete CMI for small alphabets.
//!
//! The generic [`DiscreteConditionalMutualInformation`] builds one entropy
//! estimator (with its count/distribution maps) per information space — for
//! CMI that is four spaces and eight hash maps per call. When the joint
//! alphabet is small (the common case: ≤10 states, short histories) the whole
//! measure can instead be computed directly from dense count arrays in a single
//! pass, exactly as JIDT's `ConditionalMutualInformationCalculatorDiscrete`
//! does.
//!
//! Each *variable* is a **group of raw integer columns** (a history embed when
//! the history length is > 1); the group is mapped to a single mixed-radix code
//! inline, so the embedding, the joint/marginal counting and the sum all happen
//! in one pass — no separate history reductions or joint re-packing.
//!
//! ## Timing-optimised builder
//!
//! The generic dense path infers each column's alphabet with a min/max scan,
//! copies every column into owned storage (needed for local values) and indexes
//! `ndarray` views in the hot loop. The [`DenseCmiBuilder`] offers the same
//! values with a leaner setup for callers that know the alphabet and only need
//! the average:
//!
//! - [`DenseCmiBuilder::with_alphabet`] skips the scan (codes must be 0-based),
//! - [`DenseCmiBuilder::global_only`] retains no inputs and returns
//!   [`DenseCmiGlobal`], which does **not** implement [`LocalValues`], so a
//!   local request cannot silently produce garbage.
//!
//! Both terminals fall back to the generic entropy-summation estimator when the
//! joint alphabet exceeds the dense cap.
//!
//! [`DiscreteConditionalMutualInformation`]: super::DiscreteConditionalMutualInformation
//! [`LocalValues`]: crate::estimators::traits::LocalValues

use crate::estimators::approaches::discrete::DiscreteConditionalMutualInformation;
use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
use crate::estimators::traits::{
    ConditionalMutualInformationEstimator, GlobalValue, OptionalLocalValues,
};
use ndarray::{Array1, ArrayView1};
use std::borrow::Cow;

/// Largest dense joint table built for the direct path. Above this the caller
/// falls back to the generic entropy-summation estimator.
const DENSE_CMI_CAP: u128 = 1 << 20;

/// Size of the small stack buffer used to build variable codes without a heap
/// allocation. Series with more variables than this are rare; they simply fall
/// back to the generic engine (the plan refuses them).
const MAX_DENSE_VARS: usize = 16;

/// Number of observation rows for an embedding (mirrors `te_observations`).
fn te_row_count(n: usize, max_delay: usize, step: usize) -> usize {
    if max_delay >= n {
        0
    } else {
        (n - max_delay).div_ceil(step)
    }
}

/// History column `col` (0 = oldest) as a borrowed/owned code column.
fn hist_col_cow(
    arr: &[i32],
    col: usize,
    hist_len: usize,
    max_delay: usize,
    step: usize,
    n: usize,
) -> Cow<'_, [i32]> {
    if n == 0 {
        return Cow::Borrowed(&[]);
    }
    let start = max_delay - (hist_len - col) * step;
    if step == 1 {
        Cow::Borrowed(&arr[start..start + n])
    } else {
        Cow::Owned((0..n).map(|i| arr[start + i * step]).collect())
    }
}

/// Future column (`Y_t`) as a borrowed/owned code column.
fn future_col_cow(arr: &[i32], max_delay: usize, step: usize, n: usize) -> Cow<'_, [i32]> {
    if n == 0 {
        return Cow::Borrowed(&[]);
    }
    if step == 1 {
        Cow::Borrowed(&arr[max_delay..max_delay + n])
    } else {
        Cow::Owned((0..n).map(|i| arr[max_delay + i * step]).collect())
    }
}

/// Layout of a dense CMI over contiguous code columns.
///
/// `var_cols` holds one group of column indices per variable, with the
/// conditioning group as the **last** entry (so `var_cols.len() - 1` is the
/// number of `series` variables).
struct DensePlan {
    n: usize,
    n_series: usize,
    col_min: Vec<i32>,
    col_stride: Vec<usize>,
    var_range: Vec<usize>,
    var_joint_stride: Vec<usize>,
    cond_range: usize,
    marginal_off: Vec<usize>,
    joint_len: usize,
    marginal_len: usize,
}

impl DensePlan {
    fn new(cols: &[&[i32]], var_cols: &[Vec<usize>], alphabet: Option<usize>) -> Option<Self> {
        if var_cols.len() < 2 || cols.is_empty() {
            return None;
        }
        let n_vars = var_cols.len();
        if n_vars > MAX_DENSE_VARS {
            return None;
        }
        let n = cols[0].len();
        if n == 0 || cols.iter().any(|c| c.len() != n) {
            return None;
        }
        let n_series = n_vars - 1;

        let mut col_min = vec![0i32; cols.len()];
        let mut col_stride = vec![0usize; cols.len()];
        let mut var_range = vec![0usize; n_vars];
        for (v, idxs) in var_cols.iter().enumerate() {
            if idxs.is_empty() {
                return None;
            }
            let mut range_total = 1usize;
            for &c in idxs {
                let (mn, rng) = match alphabet {
                    Some(k) => (0i32, k),
                    None => {
                        let (mn, mx) = cols[c]
                            .iter()
                            .fold((i32::MAX, i32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                        if mn == i32::MAX {
                            return None;
                        }
                        (mn, (mx as i64 - mn as i64 + 1) as usize)
                    }
                };
                col_min[c] = mn;
                col_stride[c] = range_total;
                range_total = range_total.checked_mul(rng)?;
            }
            var_range[v] = range_total;
        }

        let mut cap: u128 = 1;
        for &r in &var_range {
            cap = cap.checked_mul(r as u128)?;
            if cap > DENSE_CMI_CAP {
                return None;
            }
        }
        let joint_len = cap as usize;

        let mut var_joint_stride = vec![0usize; n_vars];
        let mut acc = 1usize;
        for v in 0..n_vars {
            var_joint_stride[v] = acc;
            acc = acc.saturating_mul(var_range[v]);
        }
        let cond_range = var_range[n_series];
        let mut marginal_off = Vec::with_capacity(n_series);
        let mut marginal_len = 0usize;
        for &r in var_range.iter().take(n_series) {
            marginal_off.push(marginal_len);
            marginal_len += r * cond_range;
        }

        Some(Self {
            n,
            n_series,
            col_min,
            col_stride,
            var_range,
            var_joint_stride,
            cond_range,
            marginal_off,
            joint_len,
            marginal_len,
        })
    }
}

/// Counts and average produced by [`count_dense`].
struct DenseCounts {
    joint: Vec<u32>,
    marginal: Vec<u32>,
    cond: Vec<u32>,
    global: f64,
}

/// Single fused pass over the observations: mixed-radix codes are computed
/// inline and the joint, per-marginal and conditioning counts incremented
/// directly (no hashing, no datasets), then the measure is summed over the
/// dense joint cells.
///
/// The common case — every variable a single raw column, the condition one or
/// more — uses a tighter loop that skips the per-group indirection.
fn count_dense(cols: &[&[i32]], var_cols: &[Vec<usize>], plan: &DensePlan) -> DenseCounts {
    let single_column_series = var_cols[..plan.n_series].iter().all(|g| g.len() == 1);
    if single_column_series {
        count_dense_flat(cols, var_cols, plan)
    } else {
        count_dense_grouped(cols, var_cols, plan)
    }
}

fn count_dense_flat(cols: &[&[i32]], var_cols: &[Vec<usize>], plan: &DensePlan) -> DenseCounts {
    let mut joint = vec![0u32; plan.joint_len];
    let mut marginal = vec![0u32; plan.marginal_len];
    let mut cond = vec![0u32; plan.cond_range];

    let series: Vec<(&[i32], i32)> = (0..plan.n_series)
        .map(|i| {
            let c = var_cols[i][0];
            (cols[c], plan.col_min[c])
        })
        .collect();
    let cond_cols: Vec<(&[i32], i32, usize)> = var_cols[plan.n_series]
        .iter()
        .map(|&c| (cols[c], plan.col_min[c], plan.col_stride[c]))
        .collect();
    let var_stride = &plan.var_joint_stride;
    let marginal_off = &plan.marginal_off;
    let cond_range = plan.cond_range;
    let cond_joint_stride = plan.var_joint_stride[plan.n_series];

    // Two-variable CMI/TE/CTE: the dominant shape. Hoist the two series slices
    // and unroll their loop; the condition group is short (1-2 columns).
    if plan.n_series == 2 {
        let (c0, m0) = series[0];
        let (c1, m1) = series[1];
        let (s0, s1) = (var_stride[0], var_stride[1]);
        let (o0, o1) = (marginal_off[0], marginal_off[1]);
        for t in 0..plan.n {
            let mut cond_code = 0usize;
            for &(col, mn, st) in &cond_cols {
                cond_code += (col[t] - mn) as usize * st;
            }
            let x = (c0[t] - m0) as usize;
            let y = (c1[t] - m1) as usize;
            joint[cond_code * cond_joint_stride + x * s0 + y * s1] += 1;
            marginal[o0 + x * cond_range + cond_code] += 1;
            marginal[o1 + y * cond_range + cond_code] += 1;
            cond[cond_code] += 1;
        }
        return finish_counts(joint, marginal, cond, plan);
    }

    for t in 0..plan.n {
        let mut cond_code = 0usize;
        for &(col, mn, st) in &cond_cols {
            cond_code += (col[t] - mn) as usize * st;
        }
        let mut jidx = cond_code * cond_joint_stride;
        for (i, &(col, mn)) in series.iter().enumerate() {
            let code = (col[t] - mn) as usize;
            jidx += code * var_stride[i];
            marginal[marginal_off[i] + code * cond_range + cond_code] += 1;
        }
        joint[jidx] += 1;
        cond[cond_code] += 1;
    }

    finish_counts(joint, marginal, cond, plan)
}

#[allow(clippy::needless_range_loop)]
fn count_dense_grouped(cols: &[&[i32]], var_cols: &[Vec<usize>], plan: &DensePlan) -> DenseCounts {
    let mut joint = vec![0u32; plan.joint_len];
    let mut marginal = vec![0u32; plan.marginal_len];
    let mut cond = vec![0u32; plan.cond_range];

    let cond_cols = &var_cols[plan.n_series];
    for t in 0..plan.n {
        let mut cond_code = 0usize;
        for &c in cond_cols {
            cond_code += (cols[c][t] - plan.col_min[c]) as usize * plan.col_stride[c];
        }
        let mut jidx = cond_code * plan.var_joint_stride[plan.n_series];
        for i in 0..plan.n_series {
            let mut code = 0usize;
            for &c in &var_cols[i] {
                code += (cols[c][t] - plan.col_min[c]) as usize * plan.col_stride[c];
            }
            jidx += code * plan.var_joint_stride[i];
            marginal[plan.marginal_off[i] + code * plan.cond_range + cond_code] += 1;
        }
        joint[jidx] += 1;
        cond[cond_code] += 1;
    }

    finish_counts(joint, marginal, cond, plan)
}

/// Sum the measure over the dense joint cells.
fn finish_counts(
    joint: Vec<u32>,
    marginal: Vec<u32>,
    cond: Vec<u32>,
    plan: &DensePlan,
) -> DenseCounts {
    let n_minus_1 = (plan.n_series as f64) - 1.0;
    // Logs of the count tables are reused across every joint cell that touches
    // them; computing them inline dominated the sum for small alphabets.
    let marginal_ln: Vec<f64> = marginal
        .iter()
        .map(|&c| if c > 0 { (c as f64).ln() } else { 0.0 })
        .collect();
    let cond_ln: Vec<f64> = cond
        .iter()
        .map(|&c| if c > 0 { (c as f64).ln() } else { 0.0 })
        .collect();
    let mut sum = 0.0_f64;

    if plan.n_series == 2 {
        // Decode the joint layout with nested loops instead of per-cell
        // integer division/modulo.
        let (r0, r1) = (plan.var_range[0], plan.var_range[1]);
        let (o0, o1) = (plan.marginal_off[0], plan.marginal_off[1]);
        let cond_range = plan.cond_range;
        let cond_joint_stride = plan.var_joint_stride[2];
        for cc in 0..cond_range {
            if cond[cc] == 0 {
                continue;
            }
            let cln = cond_ln[cc];
            let jbase = cc * cond_joint_stride;
            for x in 0..r0 {
                let mxl = marginal_ln[o0 + x * cond_range + cc];
                for y in 0..r1 {
                    let jc = joint[jbase + x + y * r0];
                    if jc == 0 {
                        continue;
                    }
                    let local = (jc as f64).ln() + n_minus_1 * cln
                        - mxl
                        - marginal_ln[o1 + y * cond_range + cc];
                    sum += jc as f64 * local;
                }
            }
        }
    } else {
        for (jidx, &jc_u) in joint.iter().enumerate() {
            if jc_u == 0 {
                continue;
            }
            let cond_code = (jidx / plan.var_joint_stride[plan.n_series]) % plan.cond_range;
            let mut denom = 0.0;
            for i in 0..plan.n_series {
                let code = (jidx / plan.var_joint_stride[i]) % plan.var_range[i];
                denom += marginal_ln[plan.marginal_off[i] + code * plan.cond_range + cond_code];
            }
            sum += jc_u as f64 * ((jc_u as f64).ln() + n_minus_1 * cond_ln[cond_code] - denom);
        }
    }

    DenseCounts {
        joint,
        marginal,
        cond,
        global: sum / plan.n as f64,
    }
}

/// Dense direct discrete conditional MI state (retains the inputs for local
/// values).
pub(crate) struct DenseCmi {
    plan: DensePlan,
    /// All input columns, concatenated: column `c` is `vals[c*n .. (c+1)*n]`.
    vals: Vec<i32>,
    /// Column indices forming each variable (series 0.., then the condition).
    var_cols: Vec<Vec<usize>>,
    joint_counts: Vec<u32>,
    marginal_counts: Vec<u32>,
    cond_counts: Vec<u32>,
    global: f64,
}

impl DenseCmi {
    /// Build from `ndarray` column groups: `series[i]` is the i-th variable (one
    /// or more columns forming its embed), `cond` the condition group. Returns
    /// `None` when the joint alphabet is too large. Non-contiguous (strided)
    /// views are materialised.
    pub(crate) fn new(series: &[&[ArrayView1<i32>]], cond: &[ArrayView1<i32>]) -> Option<Self> {
        if series.is_empty() || cond.is_empty() {
            return None;
        }
        let n = cond[0].len();
        if n == 0 {
            return None;
        }
        let mut cols: Vec<Cow<[i32]>> = Vec::new();
        let mut var_cols: Vec<Vec<usize>> = Vec::new();
        for group in series.iter().chain(std::iter::once(&cond)) {
            let mut idxs = Vec::with_capacity(group.len());
            for v in group.iter() {
                if v.len() != n {
                    return None;
                }
                idxs.push(cols.len());
                match v.as_slice() {
                    Some(s) => cols.push(Cow::Borrowed(s)),
                    None => cols.push(Cow::Owned(v.iter().copied().collect())),
                }
            }
            var_cols.push(idxs);
        }
        let refs: Vec<&[i32]> = cols.iter().map(|c| &**c).collect();
        Self::from_columns(&refs, &var_cols, None)
    }

    /// Build from contiguous code columns and explicit variable groups. The last
    /// group of `var_cols` is the condition. When `alphabet` is given, every
    /// column is assumed to hold values in `0..alphabet` and the min/max scan is
    /// skipped.
    pub(crate) fn from_columns(
        cols: &[&[i32]],
        var_cols: &[Vec<usize>],
        alphabet: Option<usize>,
    ) -> Option<Self> {
        let plan = DensePlan::new(cols, var_cols, alphabet)?;
        let counts = count_dense(cols, var_cols, &plan);
        let mut vals = Vec::with_capacity(cols.len() * plan.n);
        for c in cols {
            vals.extend_from_slice(c);
        }
        Some(Self {
            plan,
            vals,
            var_cols: var_cols.to_vec(),
            joint_counts: counts.joint,
            marginal_counts: counts.marginal,
            cond_counts: counts.cond,
            global: counts.global,
        })
    }

    pub(crate) fn global_value(&self) -> f64 {
        self.global
    }

    fn code_at(&self, v: usize, t: usize) -> usize {
        let mut code = 0usize;
        for &c in &self.var_cols[v] {
            code += (self.vals[c * self.plan.n + t] - self.plan.col_min[c]) as usize
                * self.plan.col_stride[c];
        }
        code
    }

    fn local_at(&self, t: usize) -> f64 {
        let cond_code = self.code_at(self.plan.n_series, t);
        let mut jidx = cond_code * self.plan.var_joint_stride[self.plan.n_series];
        let mut denom = 0.0;
        for i in 0..self.plan.n_series {
            let code = self.code_at(i, t);
            jidx += code * self.plan.var_joint_stride[i];
            denom += (self.marginal_counts
                [self.plan.marginal_off[i] + code * self.plan.cond_range + cond_code]
                as f64)
                .ln();
        }
        let n_minus_1 = (self.plan.n_series as f64) - 1.0;
        (self.joint_counts[jidx] as f64).ln()
            + n_minus_1 * (self.cond_counts[cond_code] as f64).ln()
            - denom
    }

    pub(crate) fn local_values(&self) -> Array1<f64> {
        let mut out = Array1::zeros(self.plan.n);
        for t in 0..self.plan.n {
            out[t] = self.local_at(t);
        }
        out
    }
}

/// Compute the dense direct CMI average without retaining any inputs.
///
/// Returns `None` when the joint alphabet exceeds [`DENSE_CMI_CAP`]; the caller
/// then falls back to the generic estimator.
pub(crate) fn dense_cmi_global(
    cols: &[&[i32]],
    var_cols: &[Vec<usize>],
    alphabet: Option<usize>,
) -> Option<f64> {
    let plan = DensePlan::new(cols, var_cols, alphabet)?;
    Some(count_dense(cols, var_cols, &plan).global)
}

/// Builder for the dense direct discrete-MLE conditional mutual information.
///
/// Obtain one from [`MutualInformation::cmi_discrete_mle`],
/// [`TransferEntropy::te_discrete_mle`] or
/// [`TransferEntropy::cte_discrete_mle`]. The builder borrows the raw integer
/// code columns, so construction never clones the inputs.
///
/// Terminate with:
/// - [`build`](Self::build) — the local-value-capable estimator
///   ([`DiscreteConditionalMutualInformation`]); or
/// - [`global_only`](Self::global_only) — a value-only [`DenseCmiGlobal`] that
///   retains no inputs and does not implement [`LocalValues`].
///
/// Both fall back to the generic entropy-summation estimator when the joint
/// alphabet is too large.
///
/// [`LocalValues`]: crate::estimators::traits::LocalValues
/// [`MutualInformation::cmi_discrete_mle`]: crate::estimators::mutual_information::MutualInformation::cmi_discrete_mle
/// [`TransferEntropy::te_discrete_mle`]: crate::estimators::transfer_entropy::TransferEntropy::te_discrete_mle
/// [`TransferEntropy::cte_discrete_mle`]: crate::estimators::transfer_entropy::TransferEntropy::cte_discrete_mle
pub struct DenseCmiBuilder<'a> {
    cols: Vec<Cow<'a, [i32]>>,
    /// Series variable groups; the conditioning group is the last entry.
    var_cols: Vec<Vec<usize>>,
    alphabet: Option<usize>,
}

impl<'a> DenseCmiBuilder<'a> {
    /// Build from flat raw code columns: each entry of `series` is its own
    /// variable and `cond` is the single conditioning variable.
    pub fn new(series: &[&'a [i32]], cond: &'a [i32]) -> Self {
        let mut cols: Vec<Cow<'a, [i32]>> = series.iter().map(|s| Cow::Borrowed(*s)).collect();
        let mut var_cols: Vec<Vec<usize>> = (0..series.len()).map(|i| vec![i]).collect();
        cols.push(Cow::Borrowed(cond));
        var_cols.push(vec![cols.len() - 1]);
        Self {
            cols,
            var_cols,
            alphabet: None,
        }
    }

    /// Build from owned/borrowed columns and explicit variable groups. The
    /// conditioning group is appended to `var_groups`.
    pub(crate) fn from_groups(
        cols: Vec<Cow<'a, [i32]>>,
        var_groups: Vec<Vec<usize>>,
        cond_cols: Vec<usize>,
    ) -> Self {
        let mut var_cols = var_groups;
        var_cols.push(cond_cols);
        Self {
            cols,
            var_cols,
            alphabet: None,
        }
    }

    /// Transfer-entropy embedding: `I(X_past; Y_t | Y_past)`.
    pub(crate) fn from_te(
        source: &'a [i32],
        destination: &'a [i32],
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> Self {
        let max_delay = src_hist_len.max(dest_hist_len) * step_size;
        let n = te_row_count(destination.len(), max_delay, step_size);
        let mut cols: Vec<Cow<'a, [i32]>> = Vec::new();
        let mut src_past = Vec::with_capacity(src_hist_len);
        for c in 0..src_hist_len {
            src_past.push(cols.len());
            cols.push(hist_col_cow(
                source,
                c,
                src_hist_len,
                max_delay,
                step_size,
                n,
            ));
        }
        let dest_future = cols.len();
        cols.push(future_col_cow(destination, max_delay, step_size, n));
        let mut dest_past = Vec::with_capacity(dest_hist_len);
        for c in 0..dest_hist_len {
            dest_past.push(cols.len());
            cols.push(hist_col_cow(
                destination,
                c,
                dest_hist_len,
                max_delay,
                step_size,
                n,
            ));
        }
        Self::from_groups(cols, vec![src_past, vec![dest_future]], dest_past)
    }

    /// Conditional transfer-entropy embedding:
    /// `I(X_past; Y_t | Y_past, Z_past)`.
    pub(crate) fn from_cte(
        source: &'a [i32],
        destination: &'a [i32],
        condition: &'a [i32],
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> Self {
        let max_delay = src_hist_len.max(dest_hist_len).max(cond_hist_len) * step_size;
        let n = te_row_count(destination.len(), max_delay, step_size);
        let mut cols: Vec<Cow<'a, [i32]>> = Vec::new();
        let mut src_past = Vec::with_capacity(src_hist_len);
        for c in 0..src_hist_len {
            src_past.push(cols.len());
            cols.push(hist_col_cow(
                source,
                c,
                src_hist_len,
                max_delay,
                step_size,
                n,
            ));
        }
        let dest_future = cols.len();
        cols.push(future_col_cow(destination, max_delay, step_size, n));
        let mut dest_past = Vec::with_capacity(dest_hist_len);
        for c in 0..dest_hist_len {
            dest_past.push(cols.len());
            cols.push(hist_col_cow(
                destination,
                c,
                dest_hist_len,
                max_delay,
                step_size,
                n,
            ));
        }
        let mut cond_past = Vec::with_capacity(cond_hist_len);
        for c in 0..cond_hist_len {
            cond_past.push(cols.len());
            cols.push(hist_col_cow(
                condition,
                c,
                cond_hist_len,
                max_delay,
                step_size,
                n,
            ));
        }
        dest_past.extend(cond_past);
        Self::from_groups(cols, vec![src_past, vec![dest_future]], dest_past)
    }

    /// Declare a known per-column alphabet (`states`), skipping the min/max
    /// scan. Every column must hold codes in `0..alphabet`; otherwise the
    /// result is unspecified.
    pub fn with_alphabet(mut self, alphabet: usize) -> Self {
        self.alphabet = Some(alphabet);
        self
    }

    /// Finish as a global-only estimator: the inputs are consumed by the
    /// counting pass and nothing is retained.
    pub fn global_only(self) -> DenseCmiGlobal {
        let global = match self.dense_global() {
            Some(g) => g,
            None => self.generic_estimate().global_value(),
        };
        DenseCmiGlobal { global }
    }

    /// Finish as the local-value-capable estimator.
    pub fn build(self) -> DiscreteConditionalMutualInformation<DiscreteEntropy> {
        let refs: Vec<&[i32]> = self.cols.iter().map(|c| &**c).collect();
        match DenseCmi::from_columns(&refs, &self.var_cols, self.alphabet) {
            Some(dense) => DiscreteConditionalMutualInformation::from_dense(dense),
            None => self.generic_estimate(),
        }
    }

    fn dense_global(&self) -> Option<f64> {
        let refs: Vec<&[i32]> = self.cols.iter().map(|c| &**c).collect();
        dense_cmi_global(&refs, &self.var_cols, self.alphabet)
    }

    /// Fallback: reduce each variable group to one mixed-radix code column and
    /// use the generic entropy-summation estimator.
    fn generic_estimate(self) -> DiscreteConditionalMutualInformation<DiscreteEntropy> {
        let n_series = self.var_cols.len() - 1;
        let series: Vec<Array1<i32>> = (0..n_series)
            .map(|i| self.reduce_group(&self.var_cols[i]))
            .collect();
        let cond = self.reduce_group(&self.var_cols[n_series]);
        DiscreteConditionalMutualInformation::new(&series, &cond, DiscreteEntropy::new)
    }

    fn reduce_group(&self, group: &[usize]) -> Array1<i32> {
        let n = self.cols[group[0]].len();
        let mut strides = Vec::with_capacity(group.len());
        let mut mins = Vec::with_capacity(group.len());
        let mut acc = 1i32;
        for &c in group {
            let col = &self.cols[c];
            let (mn, mx) = match self.alphabet {
                Some(k) => (0, k as i32 - 1),
                None => (
                    *col.iter().min().unwrap_or(&0),
                    *col.iter().max().unwrap_or(&0),
                ),
            };
            strides.push(acc);
            mins.push(mn);
            acc = acc.saturating_mul(mx - mn + 1);
        }
        let mut out = Array1::zeros(n);
        for t in 0..n {
            let mut code = 0i32;
            for (j, &c) in group.iter().enumerate() {
                code += (self.cols[c][t] - mins[j]).saturating_mul(strides[j]);
            }
            out[t] = code;
        }
        out
    }
}

/// Global-only dense CMI result.
///
/// Deliberately does **not** implement [`LocalValues`](crate::estimators::traits::LocalValues):
/// a global-only construction retains no per-sample inputs, so requesting local
/// values is a compile-time error rather than a wrong answer.
pub struct DenseCmiGlobal {
    global: f64,
}

impl GlobalValue for DenseCmiGlobal {
    fn global_value(&self) -> f64 {
        self.global
    }
}

impl OptionalLocalValues for DenseCmiGlobal {
    fn supports_local(&self) -> bool {
        false
    }

    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("global-only estimator does not retain local values")
    }
}

impl ConditionalMutualInformationEstimator for DenseCmiGlobal {}

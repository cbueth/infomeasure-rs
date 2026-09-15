// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dense direct discrete CMI.
//!
//! The generic [`DiscreteConditionalMutualInformation`] builds one entropy
//! estimator (with its count/distribution maps) per information space — for
//! CMI that is four spaces and eight hash maps per call. This module instead
//! counts the joint, per-marginal and conditioning distributions in **one
//! fused pass** and sums the measure directly.
//!
//! Each *variable* is a **group of raw integer columns** (a history embed when
//! the history length is > 1); the group is mapped to a single mixed-radix code
//! inline, so the embedding, the counting and the sum all happen in one pass —
//! no separate history reductions or joint re-packing.
//!
//! ## Joint storage: dense table vs. hash map
//!
//! The marginal and conditioning tables are always dense (they are small). The
//! **joint** table is the product of all variable ranges (`base^d`) and can be
//! far larger. Two representations are available; they produce **identical
//! values** and differ only in cost:
//!
//! - **Dense** `Vec<u32>` — direct indexing, but time and memory grow with the
//!   number of joint cells.
//! - **Sparse** `FxHashMap<u64, u32>` — only occupied cells are stored, so cost
//!   grows with the number of observations `N` rather than the alphabet, at the
//!   price of hashing every sample.
//!
//! This is a **performance-only** heuristic; it never changes the result. The
//! joint is counted densely while it is small relative to the data — at most
//! [`DENSE_CMI_CELLS_PER_OBS`] cells *per observation*, and never more than
//! [`DENSE_CMI_CAP`] cells in absolute terms — and otherwise as a hash map,
//! which keeps the algorithm `O(N)`: a *dense* joint-count table instead scales
//! with `base^d` (one cell per alphabet combination) and dominates for wide
//! alphabets or long histories.
//! The per-observation constant is an approximation measured on the project's
//! benchmark machine; near the crossover the two paths cost about the same, so
//! a different machine only shifts a decision whose either choice is within a
//! small factor. The choice is **deterministic** (no runtime timing/autotuning),
//! which keeps results reproducible. The direct path is abandoned for the
//! generic estimator only when the *marginal* tables would be too large
//! ([`MARGINAL_CAP`]).
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
//! [`DiscreteConditionalMutualInformation`]: super::DiscreteConditionalMutualInformation
//! [`LocalValues`]: crate::estimators::traits::LocalValues

use crate::estimators::approaches::discrete::DiscreteConditionalMutualInformation;
use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
use crate::estimators::traits::{
    ConditionalMutualInformationEstimator, GlobalValue, OptionalLocalValues,
};
use ndarray::{Array1, ArrayView1};
use rustc_hash::FxHashMap;
use std::borrow::Cow;

/// Joint table size up to which a dense `Vec<u32>` is used; above it the joint
/// is a hash map. The marginal/conditioning tables are always dense.
const DENSE_CMI_CAP: u128 = 1 << 20;

/// A dense joint is only scanned cell-by-cell while it is small relative to the
/// number of observations. Measured crossover is ~50 joint cells per observation
/// for both CMI and CTE, so above this the sparse hash joint is faster (and
/// scales with `N` rather than the alphabet). Keep the hard `DENSE_CMI_CAP` as a
/// memory bound too.
const DENSE_CMI_CELLS_PER_OBS: u128 = 50;

/// Total dense marginal + conditioning cells above which the direct path is
/// abandoned for the generic estimator (bounds memory for huge alphabets).
const MARGINAL_CAP: u128 = 1 << 25;

/// Number of variables above which the direct path bails out.
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

/// Layout of a direct CMI over contiguous code columns.
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
    /// Joint cells when dense; 0 when the joint is a hash map.
    joint_len: usize,
    dense_joint: bool,
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

        // Joint size (product of all variable ranges); decide storage.
        let mut joint_cells: u128 = 1;
        for &r in &var_range {
            joint_cells = joint_cells.checked_mul(r as u128)?;
        }
        let dense_joint =
            joint_cells <= DENSE_CMI_CAP && joint_cells <= DENSE_CMI_CELLS_PER_OBS * n as u128;
        let joint_len = if dense_joint { joint_cells as usize } else { 0 };
        // The sparse joint key packs the mixed-radix code into a u64.
        if !dense_joint && joint_cells > u64::MAX as u128 {
            return None;
        }

        let mut var_joint_stride = vec![0usize; n_vars];
        let mut acc: usize = 1;
        for v in 0..n_vars {
            var_joint_stride[v] = acc;
            acc = acc.saturating_mul(var_range[v]);
        }
        let cond_range = var_range[n_series];

        // Marginals `(variable, condition)` stay dense; bound their memory.
        let mut marginal_off = Vec::with_capacity(n_series);
        let mut marginal_cells: u128 = 0;
        for &r in var_range.iter().take(n_series) {
            marginal_off.push(marginal_cells as usize);
            marginal_cells += r as u128 * cond_range as u128;
        }
        if marginal_cells + cond_range as u128 > MARGINAL_CAP {
            return None;
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
            dense_joint,
            marginal_len: marginal_cells as usize,
        })
    }
}

/// Joint-count storage.
enum Joint {
    Dense(Vec<u32>),
    Sparse(FxHashMap<u64, u32>),
}

impl Joint {
    #[inline]
    fn get(&self, key: u64) -> u32 {
        match self {
            Joint::Dense(v) => v.get(key as usize).copied().unwrap_or(0),
            Joint::Sparse(m) => m.get(&key).copied().unwrap_or(0),
        }
    }
}

/// Fused counting pass: each variable's mixed-radix code is computed inline and
/// the joint, per-marginal and conditioning counts are incremented in one walk.
/// The measure is summed within the same call (as in the pre-existing dense
/// path) so the tables stay hot.
///
/// The joint is accumulated either in a dense table or in a hash map, chosen by
/// [`DensePlan`]. The two are fully separate loops (see [`count_dense_flat`] and
/// [`count_sparse_flat`]) so the dense path keeps its tight indexing.
fn count(
    cols: &[&[i32]],
    var_cols: &[Vec<usize>],
    plan: &DensePlan,
) -> (Joint, Vec<u32>, Vec<u32>, f64) {
    let flat = var_cols[..plan.n_series].iter().all(|g| g.len() == 1);
    if plan.dense_joint {
        let (joint, marginal, cond, global) = if flat {
            count_dense_flat(cols, var_cols, plan)
        } else {
            count_dense_grouped(cols, var_cols, plan)
        };
        (Joint::Dense(joint), marginal, cond, global)
    } else {
        let (joint, marginal, cond, global) = if flat {
            count_sparse_flat(cols, var_cols, plan)
        } else {
            count_sparse_grouped(cols, var_cols, plan)
        };
        (Joint::Sparse(joint), marginal, cond, global)
    }
}

/// Dense single pass over the observations for the dominant shape: every series
/// variable is a single raw column, the condition one or more. The joint,
/// per-marginal and conditioning counts are incremented directly (no hashing,
/// no datasets), then the measure is summed over the dense joint cells.
fn count_dense_flat(
    cols: &[&[i32]],
    var_cols: &[Vec<usize>],
    plan: &DensePlan,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, f64) {
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
        let global = finish_dense(plan, &marginal, &cond, &joint);
        return (joint, marginal, cond, global);
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

    let global = finish_dense(plan, &marginal, &cond, &joint);
    (joint, marginal, cond, global)
}

/// Dense single pass for grouped (embedded) series variables.
#[allow(clippy::needless_range_loop)]
fn count_dense_grouped(
    cols: &[&[i32]],
    var_cols: &[Vec<usize>],
    plan: &DensePlan,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, f64) {
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

    let global = finish_dense(plan, &marginal, &cond, &joint);
    (joint, marginal, cond, global)
}

/// Sparse single pass for the flat shape: the packed mixed-radix key is inserted
/// into a hash map. Used when the joint is large relative to `N`.
fn count_sparse_flat(
    cols: &[&[i32]],
    var_cols: &[Vec<usize>],
    plan: &DensePlan,
) -> (FxHashMap<u64, u32>, Vec<u32>, Vec<u32>, f64) {
    let mut joint: FxHashMap<u64, u32> = FxHashMap::default();
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
    let cond_joint_stride = plan.var_joint_stride[plan.n_series] as u64;

    if plan.n_series == 2 {
        let (c0, m0) = series[0];
        let (c1, m1) = series[1];
        let (s0, s1) = (var_stride[0] as u64, var_stride[1] as u64);
        let (o0, o1) = (marginal_off[0], marginal_off[1]);
        for t in 0..plan.n {
            let mut cond_code = 0usize;
            for &(col, mn, st) in &cond_cols {
                cond_code += (col[t] - mn) as usize * st;
            }
            let x = (c0[t] - m0) as usize;
            let y = (c1[t] - m1) as usize;
            *joint
                .entry(cond_code as u64 * cond_joint_stride + x as u64 * s0 + y as u64 * s1)
                .or_insert(0) += 1;
            marginal[o0 + x * cond_range + cond_code] += 1;
            marginal[o1 + y * cond_range + cond_code] += 1;
            cond[cond_code] += 1;
        }
        let global = finish_sparse(plan, &marginal, &cond, &joint);
        return (joint, marginal, cond, global);
    }

    for t in 0..plan.n {
        let mut cond_code = 0usize;
        for &(col, mn, st) in &cond_cols {
            cond_code += (col[t] - mn) as usize * st;
        }
        let mut key = cond_code as u64 * cond_joint_stride;
        for (i, &(col, mn)) in series.iter().enumerate() {
            let code = (col[t] - mn) as usize;
            key += code as u64 * var_stride[i] as u64;
            marginal[marginal_off[i] + code * cond_range + cond_code] += 1;
        }
        *joint.entry(key).or_insert(0) += 1;
        cond[cond_code] += 1;
    }

    let global = finish_sparse(plan, &marginal, &cond, &joint);
    (joint, marginal, cond, global)
}

/// Sparse single pass for grouped (embedded) series variables.
#[allow(clippy::needless_range_loop)]
fn count_sparse_grouped(
    cols: &[&[i32]],
    var_cols: &[Vec<usize>],
    plan: &DensePlan,
) -> (FxHashMap<u64, u32>, Vec<u32>, Vec<u32>, f64) {
    let mut joint: FxHashMap<u64, u32> = FxHashMap::default();
    let mut marginal = vec![0u32; plan.marginal_len];
    let mut cond = vec![0u32; plan.cond_range];

    let cond_cols = &var_cols[plan.n_series];
    for t in 0..plan.n {
        let mut cond_code = 0usize;
        for &c in cond_cols {
            cond_code += (cols[c][t] - plan.col_min[c]) as usize * plan.col_stride[c];
        }
        let mut key = cond_code as u64 * plan.var_joint_stride[plan.n_series] as u64;
        for i in 0..plan.n_series {
            let mut code = 0usize;
            for &c in &var_cols[i] {
                code += (cols[c][t] - plan.col_min[c]) as usize * plan.col_stride[c];
            }
            key += code as u64 * plan.var_joint_stride[i] as u64;
            marginal[plan.marginal_off[i] + code * plan.cond_range + cond_code] += 1;
        }
        *joint.entry(key).or_insert(0) += 1;
        cond[cond_code] += 1;
    }

    let global = finish_sparse(plan, &marginal, &cond, &joint);
    (joint, marginal, cond, global)
}

/// Dense sum: scan the joint cell-by-cell with nested loops (no per-cell
/// integer division) over precomputed log tables.
fn finish_dense(plan: &DensePlan, marginal: &[u32], cond: &[u32], joint: &[u32]) -> f64 {
    let n_minus_1 = (plan.n_series as f64) - 1.0;
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
        let (r0, r1) = (plan.var_range[0], plan.var_range[1]);
        let (o0, o1) = (plan.marginal_off[0], plan.marginal_off[1]);
        let cond_range = plan.cond_range;
        let cond_joint_stride = plan.var_joint_stride[2];
        for cc in 0..cond_range {
            if cond[cc] == 0 {
                continue;
            }
            let clnc = cond_ln[cc];
            let jbase = cc * cond_joint_stride;
            for x in 0..r0 {
                let mxl = marginal_ln[o0 + x * cond_range + cc];
                for y in 0..r1 {
                    let jc = joint[jbase + x + y * r0];
                    if jc == 0 {
                        continue;
                    }
                    let lc = (jc as f64).ln() + n_minus_1 * clnc
                        - mxl
                        - marginal_ln[o1 + y * cond_range + cc];
                    sum += jc as f64 * lc;
                }
            }
        }
        return sum / plan.n as f64;
    }

    for (jidx, &jc) in joint.iter().enumerate() {
        if jc == 0 {
            continue;
        }
        let cond_code = (jidx / plan.var_joint_stride[plan.n_series]) % plan.cond_range;
        let mut denom = 0.0;
        for i in 0..plan.n_series {
            let code = (jidx / plan.var_joint_stride[i]) % plan.var_range[i];
            denom += marginal_ln[plan.marginal_off[i] + code * plan.cond_range + cond_code];
        }
        sum += jc as f64 * ((jc as f64).ln() + n_minus_1 * cond_ln[cond_code] - denom);
    }
    sum / plan.n as f64
}

/// Sparse sum: only `O(N)` joint cells are occupied, so the logs of the
/// (possibly large) marginal/conditioning tables are precomputed only while
/// those tables are small and otherwise evaluated on demand for the cells that
/// are actually present.
fn finish_sparse(
    plan: &DensePlan,
    marginal: &[u32],
    cond: &[u32],
    joint: &FxHashMap<u64, u32>,
) -> f64 {
    const LOG_PRE_MAX: usize = 8192;
    let n_minus_1 = (plan.n_series as f64) - 1.0;
    let marginal_ln: Option<Vec<f64>> = (marginal.len() <= LOG_PRE_MAX).then(|| {
        marginal
            .iter()
            .map(|&c| if c > 0 { (c as f64).ln() } else { 0.0 })
            .collect()
    });
    let cond_ln: Option<Vec<f64>> = (cond.len() <= LOG_PRE_MAX).then(|| {
        cond.iter()
            .map(|&c| if c > 0 { (c as f64).ln() } else { 0.0 })
            .collect()
    });
    let mln = |i: usize| -> f64 {
        match &marginal_ln {
            Some(v) => v[i],
            None => {
                if marginal[i] > 0 {
                    (marginal[i] as f64).ln()
                } else {
                    0.0
                }
            }
        }
    };
    let cln = |i: usize| -> f64 {
        match &cond_ln {
            Some(v) => v[i],
            None => {
                if cond[i] > 0 {
                    (cond[i] as f64).ln()
                } else {
                    0.0
                }
            }
        }
    };
    let mut sum = 0.0_f64;

    // Two series: `var_joint_stride[0] == 1`, so the packed key decodes with one
    // division plus a remainder, instead of the generic div/mod per series.
    if plan.n_series == 2 {
        let r0 = plan.var_range[0] as u64;
        let s3 = plan.var_joint_stride[2] as u64;
        let (o0, o1) = (plan.marginal_off[0], plan.marginal_off[1]);
        let cond_range = plan.cond_range;
        for (&key, &jc) in joint.iter() {
            let cond_code = (key / s3) as usize;
            let rem = key % s3;
            let x = (rem % r0) as usize;
            let y = (rem / r0) as usize;
            let lc = (jc as f64).ln() + n_minus_1 * cln(cond_code)
                - mln(o0 + x * cond_range + cond_code)
                - mln(o1 + y * cond_range + cond_code);
            sum += jc as f64 * lc;
        }
        return sum / plan.n as f64;
    }

    for (&key, &jc) in joint.iter() {
        let cond_code =
            ((key / plan.var_joint_stride[plan.n_series] as u64) % plan.cond_range as u64) as usize;
        let mut denom = 0.0;
        for i in 0..plan.n_series {
            let code =
                ((key / plan.var_joint_stride[i] as u64) % plan.var_range[i] as u64) as usize;
            denom += mln(plan.marginal_off[i] + code * plan.cond_range + cond_code);
        }
        sum += jc as f64 * ((jc as f64).ln() + n_minus_1 * cln(cond_code) - denom);
    }
    sum / plan.n as f64
}

/// Direct discrete conditional MI state (retains the inputs for local values).
pub(crate) struct DenseCmi {
    plan: DensePlan,
    /// All input columns, concatenated: column `c` is `vals[c*n .. (c+1)*n]`.
    vals: Vec<i32>,
    /// Column indices forming each variable (series 0.., then the condition).
    var_cols: Vec<Vec<usize>>,
    joint: Joint,
    marginal_counts: Vec<u32>,
    cond_counts: Vec<u32>,
    global: f64,
}

impl DenseCmi {
    /// Build from `ndarray` column groups: `series[i]` is the i-th variable (one
    /// or more columns forming its embed), `cond` the condition group. Returns
    /// `None` when the marginal tables would be too large. Non-contiguous
    /// (strided) views are materialised.
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
        let (joint, marginal, cond, global) = count(cols, var_cols, &plan);
        let mut vals = Vec::with_capacity(cols.len() * plan.n);
        for c in cols {
            vals.extend_from_slice(c);
        }
        Some(Self {
            plan,
            vals,
            var_cols: var_cols.to_vec(),
            joint,
            marginal_counts: marginal,
            cond_counts: cond,
            global,
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
        let mut key = cond_code as u64 * self.plan.var_joint_stride[self.plan.n_series] as u64;
        let mut denom = 0.0;
        for i in 0..self.plan.n_series {
            let code = self.code_at(i, t);
            key += code as u64 * self.plan.var_joint_stride[i] as u64;
            denom += (self.marginal_counts
                [self.plan.marginal_off[i] + code * self.plan.cond_range + cond_code]
                as f64)
                .ln();
        }
        let n_minus_1 = (self.plan.n_series as f64) - 1.0;
        (self.joint.get(key) as f64).ln() + n_minus_1 * (self.cond_counts[cond_code] as f64).ln()
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

/// Compute the direct CMI average without retaining any inputs.
///
/// Returns `None` only when the marginal tables would be too large; the caller
/// then falls back to the generic estimator.
pub(crate) fn dense_cmi_global(
    cols: &[&[i32]],
    var_cols: &[Vec<usize>],
    alphabet: Option<usize>,
) -> Option<f64> {
    let plan = DensePlan::new(cols, var_cols, alphabet)?;
    let (_, _, _, global) = count(cols, var_cols, &plan);
    Some(global)
}

/// Builder for the direct discrete-MLE conditional mutual information.
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
/// The joint is counted densely while it is small relative to `N` and as a hash
/// map otherwise; both give the same value, only the speed differs. Only a huge
/// *marginal* table falls back to the generic estimator. See the module-level
/// "Joint storage" notes above.
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
    #[allow(clippy::too_many_arguments)]
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

/// Global-only direct CMI result.
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

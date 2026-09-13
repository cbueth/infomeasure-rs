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
//! [`DiscreteConditionalMutualInformation`]: super::DiscreteConditionalMutualInformation

use ndarray::{Array1, ArrayView1};

/// Largest dense joint table built for the direct path. Above this the caller
/// falls back to the generic entropy-summation estimator.
const DENSE_CMI_CAP: u128 = 1 << 20;

/// Dense direct discrete conditional MI state.
pub(crate) struct DenseCmi {
    n: usize,
    n_series: usize,
    /// All input columns, concatenated: column `c` is `vals[c*n .. (c+1)*n]`.
    vals: Vec<i32>,
    col_min: Vec<i32>,
    col_stride: Vec<usize>,
    /// Column indices forming each variable (series 0.., then the condition).
    var_cols: Vec<Vec<usize>>,
    var_joint_stride: Vec<usize>,
    cond_range: usize,
    joint_counts: Vec<u32>,
    /// Flattened per-marginal counts: block `i` is `var_range[i] * cond_range`.
    marginal_counts: Vec<u32>,
    marginal_off: Vec<usize>,
    cond_counts: Vec<u32>,
    global: f64,
}

impl DenseCmi {
    /// Build from raw column groups: `series[i]` is the i-th variable (one or
    /// more columns forming its embed), `cond` the condition group. Returns
    /// `None` when the joint alphabet is too large.
    pub(crate) fn new(series: &[&[ArrayView1<i32>]], cond: &[ArrayView1<i32>]) -> Option<Self> {
        if series.is_empty() || cond.is_empty() {
            return None;
        }
        let n = cond[0].len();
        if n == 0 {
            return None;
        }
        let mut var_cols: Vec<Vec<usize>> = Vec::with_capacity(series.len() + 1);
        let mut all: Vec<ArrayView1<i32>> = Vec::new();
        for group in series.iter().chain(std::iter::once(&cond)) {
            let mut idxs = Vec::with_capacity(group.len());
            for v in group.iter() {
                if v.len() != n {
                    return None;
                }
                idxs.push(all.len());
                all.push(*v);
            }
            var_cols.push(idxs);
        }
        let n_series = series.len();
        let n_vars = n_series + 1;

        // Per-column range and mixed-radix stride within its variable.
        let mut col_min = vec![0i32; all.len()];
        let mut col_stride = vec![0usize; all.len()];
        let mut var_range = vec![0usize; n_vars];
        for (v, idxs) in var_cols.iter().enumerate() {
            let mut range_total = 1usize;
            for (pos, &c) in idxs.iter().enumerate() {
                let (mn, mx) = all[c]
                    .iter()
                    .fold((i32::MAX, i32::MIN), |(mn, mx), &x| (mn.min(x), mx.max(x)));
                if mn == i32::MAX {
                    return None;
                }
                col_min[c] = mn;
                col_stride[c] = range_total;
                range_total = range_total.checked_mul((mx as i64 - mn as i64 + 1) as usize)?;
                let _ = pos;
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
        let cap = cap as usize;

        let mut var_joint_stride = vec![0usize; n_vars];
        let mut acc = 1usize;
        for v in 0..n_vars {
            var_joint_stride[v] = acc;
            acc = acc.saturating_mul(var_range[v]);
        }
        let cond_range = var_range[n_series];
        let mut marginal_off = Vec::with_capacity(n_series);
        let mut marginal_total = 0usize;
        for &r in var_range.iter().take(n_series) {
            marginal_off.push(marginal_total);
            marginal_total += r * cond_range;
        }

        let mut joint_counts = vec![0u32; cap];
        let mut cond_counts = vec![0u32; cond_range];
        let mut marginal_counts = vec![0u32; marginal_total];

        // Keep the input columns for on-demand local values.
        let mut vals: Vec<i32> = Vec::with_capacity(all.len() * n);
        for v in &all {
            vals.extend(v.iter().copied());
        }

        // Single fused pass: compute each variable's mixed-radix code and count
        // the joint + marginals + condition (no hashing, no datasets).
        for (t, _) in all[0].iter().enumerate() {
            let mut codes = [0usize; 8];
            if n_vars > codes.len() {
                return None;
            }
            for v in 0..n_vars {
                let mut code = 0usize;
                for &c in &var_cols[v] {
                    code += (all[c][t] - col_min[c]) as usize * col_stride[c];
                }
                codes[v] = code;
            }
            let cond_code = codes[n_series];
            let mut jidx = cond_code * var_joint_stride[n_series];
            for i in 0..n_series {
                jidx += codes[i] * var_joint_stride[i];
                marginal_counts[marginal_off[i] + codes[i] * cond_range + cond_code] += 1;
            }
            joint_counts[jidx] += 1;
            cond_counts[cond_code] += 1;
        }

        // Sum over the dense joint cells (not observations).
        let n_minus_1 = (n_series as f64) - 1.0;
        let mut sum = 0.0_f64;
        for (jidx, &jc_u) in joint_counts.iter().enumerate() {
            if jc_u == 0 {
                continue;
            }
            let cond_code = (jidx / var_joint_stride[n_series]) % cond_range;
            let mut denom = 0.0;
            for i in 0..n_series {
                let code = (jidx / var_joint_stride[i]) % var_range[i];
                denom +=
                    (marginal_counts[marginal_off[i] + code * cond_range + cond_code] as f64).ln();
            }
            let local =
                (jc_u as f64).ln() + n_minus_1 * (cond_counts[cond_code] as f64).ln() - denom;
            sum += jc_u as f64 * local;
        }
        let global = sum / n as f64;

        Some(Self {
            n,
            n_series,
            vals,
            col_min,
            col_stride,
            var_cols,
            var_joint_stride,
            cond_range,
            joint_counts,
            marginal_counts,
            marginal_off,
            cond_counts,
            global,
        })
    }

    pub(crate) fn global_value(&self) -> f64 {
        self.global
    }

    fn code_at(&self, v: usize, t: usize) -> usize {
        let mut code = 0usize;
        for &c in &self.var_cols[v] {
            code += (self.vals[c * self.n + t] - self.col_min[c]) as usize * self.col_stride[c];
        }
        code
    }

    fn local_at(&self, t: usize) -> f64 {
        let cond_code = self.code_at(self.n_series, t);
        let mut jidx = cond_code * self.var_joint_stride[self.n_series];
        let mut denom = 0.0;
        for i in 0..self.n_series {
            let code = self.code_at(i, t);
            jidx += code * self.var_joint_stride[i];
            denom += (self.marginal_counts
                [self.marginal_off[i] + code * self.cond_range + cond_code]
                as f64)
                .ln();
        }
        let n_minus_1 = (self.n_series as f64) - 1.0;
        (self.joint_counts[jidx] as f64).ln()
            + n_minus_1 * (self.cond_counts[cond_code] as f64).ln()
            - denom
    }

    pub(crate) fn local_values(&self) -> Array1<f64> {
        let mut out = Array1::zeros(self.n);
        for t in 0..self.n {
            out[t] = self.local_at(t);
        }
        out
    }
}

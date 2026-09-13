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
//! does. This is the fast path for the MLE constructors; larger alphabets fall
//! back to the generic estimator.
//!
//! [`DiscreteConditionalMutualInformation`]: super::DiscreteConditionalMutualInformation

use ndarray::{Array1, ArrayView1};

/// Largest dense joint table built for the direct path. Above this the caller
/// falls back to the generic entropy-summation estimator.
const DENSE_CMI_CAP: u128 = 1 << 20;

/// Dense direct discrete conditional MI state: joint, per-marginal and
/// conditioning counts, plus the input columns (kept so local values can be
/// derived lazily without rescanning during the timed counting pass).
pub(crate) struct DenseCmi {
    n: usize,
    n_series: usize,
    /// One block per variable (series 0.., then the condition), each `n` long.
    vals: Vec<i32>,
    mins: Vec<i32>,
    joint_stride: Vec<usize>,
    cond_range: usize,
    joint_counts: Vec<u32>,
    /// Flattened per-marginal counts: block `i` is `ranges[i] * cond_range`.
    marginal_counts: Vec<u32>,
    marginal_off: Vec<usize>,
    cond_counts: Vec<u32>,
    global: f64,
}

impl DenseCmi {
    /// Build from code columns: `series[i]` is the i-th variable, `cond` the
    /// condition. Returns `None` when the joint alphabet is too large (the
    /// caller falls back to the generic estimator).
    pub(crate) fn new(series: &[ArrayView1<i32>], cond: ArrayView1<i32>) -> Option<Self> {
        let n = cond.len();
        if series.is_empty() || n == 0 {
            return None;
        }
        for s in series {
            if s.len() != n {
                return None;
            }
        }
        let n_series = series.len();

        // Per-variable range (joint is laid out with the condition last).
        let mut mins: Vec<i32> = Vec::with_capacity(n_series + 1);
        let mut ranges: Vec<usize> = Vec::with_capacity(n_series + 1);
        for v in series.iter().chain(std::iter::once(&cond)) {
            let (mn, mx) = v
                .iter()
                .fold((i32::MAX, i32::MIN), |(mn, mx), &x| (mn.min(x), mx.max(x)));
            if mn == i32::MAX {
                return None;
            }
            mins.push(mn);
            ranges.push((mx as i64 - mn as i64 + 1) as usize);
        }

        let mut cap: u128 = 1;
        for &r in &ranges {
            cap = cap.checked_mul(r as u128)?;
            if cap > DENSE_CMI_CAP {
                return None;
            }
        }
        let cap = cap as usize;

        let mut joint_stride = vec![0usize; n_series + 1];
        let mut acc = 1usize;
        for k in 0..=n_series {
            joint_stride[k] = acc;
            acc = acc.saturating_mul(ranges[k]);
        }
        let cond_range = ranges[n_series];

        let mut marginal_off = Vec::with_capacity(n_series);
        let mut marginal_total = 0usize;
        for &r in ranges.iter().take(n_series) {
            marginal_off.push(marginal_total);
            marginal_total += r * cond_range;
        }

        let mut joint_counts = vec![0u32; cap];
        let mut cond_counts = vec![0u32; cond_range];
        let mut marginal_counts = vec![0u32; marginal_total];

        // Keep the input columns so local values can be derived on demand
        // without rescanning during the timed pass.
        let mut vals: Vec<i32> = Vec::with_capacity((n_series + 1) * n);
        for v in series.iter().chain(std::iter::once(&cond)) {
            vals.extend(v.iter().copied());
        }

        // Single counting pass: joint + every marginal + the condition, all by
        // direct index (no hashing, no per-space datasets, no per-observation
        // bookkeeping).
        for t in 0..n {
            let c = (cond[t] - mins[n_series]) as usize;
            let mut jidx = c * joint_stride[n_series];
            for i in 0..n_series {
                let d = (series[i][t] - mins[i]) as usize;
                jidx += d * joint_stride[i];
                marginal_counts[marginal_off[i] + d * cond_range + c] += 1;
            }
            joint_counts[jidx] += 1;
            cond_counts[c] += 1;
        }

        // I(X1;..;Xn|Z) = Σ p(x,z) [ ln c(x,z) + (n-1) ln c(z) - Σ ln c(x_i,z) ].
        // The N factors cancel between joint, condition and marginals. Sum over
        // the (small) dense joint cells, not the observations.
        let n_minus_1 = (n_series as f64) - 1.0;
        let mut sum = 0.0_f64;
        for (jidx, &jc_u) in joint_counts.iter().enumerate() {
            if jc_u == 0 {
                continue;
            }
            let c = (jidx / joint_stride[n_series]) % cond_range;
            let mut denom = 0.0;
            for i in 0..n_series {
                let d = (jidx / joint_stride[i]) % ranges[i];
                denom += (marginal_counts[marginal_off[i] + d * cond_range + c] as f64).ln();
            }
            let local = (jc_u as f64).ln() + n_minus_1 * (cond_counts[c] as f64).ln() - denom;
            sum += jc_u as f64 * local;
        }
        let global = sum / n as f64;

        Some(Self {
            n,
            n_series,
            vals,
            mins,
            joint_stride,
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

    /// Local value at observation `t` from the stored counts + input columns.
    fn local_at(&self, t: usize) -> f64 {
        let c = (self.vals[self.n_series * self.n + t] - self.mins[self.n_series]) as usize;
        let mut jidx = c * self.joint_stride[self.n_series];
        let mut denom = 0.0;
        for i in 0..self.n_series {
            let d = (self.vals[i * self.n + t] - self.mins[i]) as usize;
            jidx += d * self.joint_stride[i];
            denom +=
                (self.marginal_counts[self.marginal_off[i] + d * self.cond_range + c] as f64).ln();
        }
        let n_minus_1 = (self.n_series as f64) - 1.0;
        (self.joint_counts[jidx] as f64).ln() + n_minus_1 * (self.cond_counts[c] as f64).ln()
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

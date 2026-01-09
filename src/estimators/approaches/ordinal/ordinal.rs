use ndarray::Array1;

use crate::estimators::traits::LocalValues;
use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
use crate::estimators::approaches::ordinal::ordinal_utils::{symbolize_series_compact, lehmer_code, reduce_joint_space_compact};
use ndarray::s;
use std::collections::HashMap;

/// Ordinal (permutation) entropy estimator.
///
/// This estimator converts a 1D time series into ordinal patterns of order `m` with
/// step size fixed to 1 (to strictly match Python implementation). It uses the canonical
/// `symbolize_series_u64` function internally to compute Lehmer codes (up to m ≤ 20),
/// remaps them to compact i32 IDs, then computes Shannon entropy using the discrete estimator.
///
/// Local values correspond to -ln p(pattern_t) for each window t.
pub struct OrdinalEntropy {
    inner: DiscreteEntropy,
    pub order: usize,
    pub step_size: usize,
    pub stable: bool,
}

impl OrdinalEntropy {
    /// Build from a single 1D time series.
    ///
    /// Note: step size is fixed to 1 to match the Python estimator's current behavior.
    pub fn new(data: Array1<f64>, order: usize) -> Self {
        Self::new_with_step_and_stable(data, order, 1, true)
    }

    /// Build from a single 1D time series with a configurable step size (delay).
    /// Stable tie-handling is enabled to mirror Python behaviour.
    pub fn new_with_step(data: Array1<f64>, order: usize, step_size: usize) -> Self {
        Self::new_with_step_and_stable(data, order, step_size, true)
    }

    /// Build from a single 1D time series with configurable step size and stability.
    pub fn new_with_step_and_stable(data: Array1<f64>, order: usize, step_size: usize, stable: bool) -> Self {
        // Special-case m=3 to match Python estimator's optimized path and tie semantics
        // Note: currently fast path always uses stable-like matching
        let codes: Array1<i32> = if order == 3 && step_size == 1 {
            let n = data.len();
            if n < 3 { Array1::zeros(0) } else {
                let mut out: Vec<i32> = Vec::with_capacity(n - 2);
                for t in 0..(n - 2) {
                    let x0 = data[t];
                    let x1 = data[t + 1];
                    let x2 = data[t + 2];
                    let gt1 = x0 < x1; // 0 < 1
                    let gt2 = x1 < x2; // 1 < 2
                    let gt3 = x0 < x2; // 0 < 2
                    // Map booleans to permutation as in Python fast path
                    let perm: [usize; 3] = match (gt1, gt2, gt3) {
                        (true,  true,  true)  => [0, 1, 2],
                        (true,  false, true)  => [0, 2, 1],
                        (false, true,  true)  => [1, 0, 2],
                        (true,  false, false) => [1, 2, 0],
                        (false, true,  false) => [2, 0, 1],
                        (false, false, false) => [2, 1, 0],
                        // The remaining combinations (true,true,false) and (false,false,true)
                        // are not realizable with strict < comparisons.
                        _other => {
                            // Should be unreachable due to transitivity of <; fallback to per-window stable argsort
                            let w = [x0, x1, x2];
                            // Use a simple stable sort on indices 0..3
                            let mut idx = [0usize, 1usize, 2usize];
                            idx.sort_by(|&i, &j| {
                                match w[i].partial_cmp(&w[j]) {
                                    Some(core::cmp::Ordering::Less) => core::cmp::Ordering::Less,
                                    Some(core::cmp::Ordering::Greater) => core::cmp::Ordering::Greater,
                                    Some(core::cmp::Ordering::Equal) | None => {
                                        if stable { i.cmp(&j) } else { core::cmp::Ordering::Equal }
                                    },
                                }
                            });
                            idx
                        }
                    };
                    let code = lehmer_code(&perm) as i32;
                    out.push(code);
                }
                Array1::from(out)
            }
        } else {
            // Use canonical symbolization (u64 internally, remapped to i32) for discrete estimator
            symbolize_series_compact(&data, order, step_size, stable)
        };
        let inner = DiscreteEntropy::new(codes);
        Self { inner, order, step_size, stable }
    }

    /// Compute joint ordinal entropy for multiple 1D series.
    /// Aligns windowed codes by truncating to the minimum length across series.
    pub fn joint_entropy(series_list: &[Array1<f64>], order: usize, step_size: usize, stable: bool) -> f64 {
        if series_list.is_empty() { return 0.0; }
        // Symbolize each series
        let mut code_arrays: Vec<Array1<i32>> = Vec::with_capacity(series_list.len());
        let mut min_len = usize::MAX;
        for s in series_list.iter() {
            let codes = symbolize_series_compact(s, order, step_size, stable);
            min_len = min_len.min(codes.len());
            code_arrays.push(codes);
        }
        if min_len == 0 { return 0.0; }
        // Truncate to min length to align windows
        for arr in code_arrays.iter_mut() {
            if arr.len() > min_len {
                let view = arr.slice(s![..min_len]).to_owned();
                *arr = view;
            }
        }
        // Reduce to joint compact space
        let joint_codes = reduce_joint_space_compact(&code_arrays);
        let disc = DiscreteEntropy::new(joint_codes);
        disc.global_value()
    }

    /// Compute ordinal cross-entropy H(p||q) between two series' ordinal pattern distributions.
    /// Uses only the intersection of supports; if disjoint, returns 0.0 (parity with Python semantics).
    pub fn cross_entropy(x: &Array1<f64>, y: &Array1<f64>, order: usize, step_size: usize, stable: bool) -> f64 {
        fn counts_to_probs(codes: &Array1<i32>) -> HashMap<i32, f64> {
            let mut map: HashMap<i32, usize> = HashMap::new();
            for &c in codes.iter() { *map.entry(c).or_insert(0) += 1; }
            let n = codes.len() as f64;
            let mut out: HashMap<i32, f64> = HashMap::with_capacity(map.len());
            for (k, v) in map.into_iter() { out.insert(k, (v as f64) / n); }
            out
        }

        let cx = symbolize_series_compact(x, order, step_size, stable);
        let cy = symbolize_series_compact(y, order, step_size, stable);

        if cx.is_empty() || cy.is_empty() { return 0.0; }

        let px = counts_to_probs(&cx);
        let qy = counts_to_probs(&cy);
        let mut h = 0.0_f64;
        let mut has_overlap = false;
        for (code, p) in px.iter() {
            if let Some(&q) = qy.get(code) {
                if *p > 0.0 && q > 0.0 {
                    h -= p * q.ln();
                    has_overlap = true;
                }
            }
        }
        if has_overlap { h } else { 0.0 }
    }
}

impl LocalValues for OrdinalEntropy {
    fn local_values(&self) -> Array1<f64> {
        self.inner.local_values()
    }
    fn global_value(&self) -> f64 {
        self.inner.global_value()
    }
}

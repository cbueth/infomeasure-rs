use ndarray::Array1;

use crate::estimators::traits::LocalValues;
use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
use crate::estimators::approaches::ordinal::ordinal_utils::{symbolize_series_compact, lehmer_code};

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
}

impl OrdinalEntropy {
    /// Build from a single 1D time series.
    ///
    /// Note: step size is fixed to 1 to match the Python estimator's current behavior.
    pub fn new(data: Array1<f64>, order: usize) -> Self {
        // Special-case m=3 to match Python estimator's optimized path and tie semantics
        let codes: Array1<i32> = if order == 3 {
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
                        other => {
                            // Should be unreachable due to transitivity of <; fallback to per-window stable argsort
                            let mut w = [x0, x1, x2];
                            // Use a simple stable sort on indices 0..3
                            let mut idx = [0usize, 1usize, 2usize];
                            idx.sort_by(|&i, &j| {
                                match w[i].partial_cmp(&w[j]) {
                                    Some(core::cmp::Ordering::Less) => core::cmp::Ordering::Less,
                                    Some(core::cmp::Ordering::Greater) => core::cmp::Ordering::Greater,
                                    Some(core::cmp::Ordering::Equal) | None => i.cmp(&j),
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
            symbolize_series_compact(&data, order, 1, true)
        };
        let inner = DiscreteEntropy::new(codes);
        Self { inner, order }
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

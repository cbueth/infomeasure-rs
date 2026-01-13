use ndarray::Array1;

use crate::estimators::approaches::discrete::discrete_utils::reduce_joint_space_compact;
use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
use crate::estimators::approaches::ordinal::ordinal_utils::{
    lehmer_code, symbolize_series_compact,
};
use crate::estimators::traits::{CrossEntropy, GlobalValue, JointEntropy, LocalValues};
use ndarray::s;

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

impl CrossEntropy for OrdinalEntropy {
    /// Compute ordinal cross-entropy H(p||q) between two series' ordinal pattern distributions.
    /// Uses only the intersection of supports; if disjoint, returns 0.0 (parity with Python semantics).
    fn cross_entropy(&self, other: &OrdinalEntropy) -> f64 {
        // We assume they might have different data but they should ideally have same order/step/stable
        // for the comparison to be meaningful, but the trait just takes &other.
        // We use self's parameters for both if we want strict H(P||Q) where P and Q are same type of estimator.
        // But here we'll just use their respective internal DiscreteEntropy.
        self.inner.cross_entropy(&other.inner)
    }
}

impl JointEntropy for OrdinalEntropy {
    type Source = Array1<f64>;
    type Params = (usize, usize, bool); // order, step_size, stable

    /// Compute joint ordinal entropy for multiple 1D series.
    /// Aligns windowed codes by truncating to the minimum length across series.
    fn joint_entropy(series_list: &[Self::Source], params: Self::Params) -> f64 {
        let (order, step_size, stable) = params;
        if series_list.is_empty() {
            return 0.0;
        }
        // Symbolize each series
        let mut code_arrays: Vec<Array1<i32>> = Vec::with_capacity(series_list.len());
        let mut min_len = usize::MAX;
        for s in series_list.iter() {
            let codes = symbolize_series_compact(s, order, step_size, stable);
            min_len = min_len.min(codes.len());
            code_arrays.push(codes);
        }
        if min_len == 0 {
            return 0.0;
        }
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
        GlobalValue::global_value(&disc)
    }
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
    pub fn new_with_step_and_stable(
        data: Array1<f64>,
        order: usize,
        step_size: usize,
        stable: bool,
    ) -> Self {
        // Special-case m=3 to match Python estimator's optimized path and tie semantics
        // Note: currently fast path always uses stable-like matching
        let codes: Array1<i32> = if order == 3 && step_size == 1 {
            let n = data.len();
            if n < 3 {
                Array1::zeros(0)
            } else {
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
                        (true, true, true) => [0, 1, 2],
                        (true, false, true) => [0, 2, 1],
                        (false, true, true) => [1, 0, 2],
                        (true, false, false) => [1, 2, 0],
                        (false, true, false) => [2, 0, 1],
                        (false, false, false) => [2, 1, 0],
                        // The remaining combinations (true,true,false) and (false,false,true)
                        // are not realizable with strict < comparisons.
                        _other => {
                            // Should be unreachable due to transitivity of <; fallback to per-window stable argsort
                            let w = [x0, x1, x2];
                            use crate::estimators::approaches::ordinal::ordinal_utils::argsort;
                            // Use a simple stable sort on indices 0..3
                            let mut idx = [0usize, 1usize, 2usize];
                            argsort(&w, &mut idx, stable);
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
        Self {
            inner,
            order,
            step_size,
            stable,
        }
    }

    /// Compute joint ordinal entropy for multiple 1D series.
    /// Aligns windowed codes by truncating to the minimum length across series.
    #[deprecated(note = "Use JointEntropy::joint_entropy instead")]
    pub fn joint_entropy(
        series_list: &[Array1<f64>],
        order: usize,
        step_size: usize,
        stable: bool,
    ) -> f64 {
        <Self as JointEntropy>::joint_entropy(series_list, (order, step_size, stable))
    }

    /// Compute ordinal cross-entropy H(p||q) between two series' ordinal pattern distributions.
    /// Uses only the intersection of supports; if disjoint, returns 0.0 (parity with Python semantics).
    #[deprecated(note = "Use CrossEntropy::cross_entropy instead")]
    pub fn cross_entropy(
        x: &Array1<f64>,
        y: &Array1<f64>,
        order: usize,
        step_size: usize,
        stable: bool,
    ) -> f64 {
        let ex = Self::new_with_step_and_stable(x.clone(), order, step_size, stable);
        let ey = Self::new_with_step_and_stable(y.clone(), order, step_size, stable);
        ex.cross_entropy(&ey)
    }
}

impl GlobalValue for OrdinalEntropy {
    fn global_value(&self) -> f64 {
        self.inner.global_value()
    }
}

impl LocalValues for OrdinalEntropy {
    fn local_values(&self) -> Array1<f64> {
        self.inner.local_values()
    }
}

use ndarray::Array1;

use crate::estimators::traits::LocalValues;
use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
use crate::estimators::approaches::ordinal::ordinal_utils::symbolize_series;

/// Ordinal (permutation) entropy estimator.
///
/// This estimator converts a 1D time series into ordinal patterns of order `m` with
/// step size `τ`, encodes each pattern as a Lehmer code (temporarily limited to m ≤ 12),
/// then computes Shannon entropy using the standard discrete estimator on the codes.
///
/// Local values correspond to -ln p(pattern_t) for each window t.
pub struct OrdinalEntropy {
    inner: DiscreteEntropy,
    pub order: usize,
    pub step_size: usize,
}

impl OrdinalEntropy {
    /// Build from a single 1D time series.
    pub fn new(data: Array1<f64>, order: usize, step_size: usize) -> Self {
        let codes: Array1<i32> = symbolize_series(&data, order, step_size, true);
        let inner = DiscreteEntropy::new(codes);
        Self { inner, order, step_size }
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

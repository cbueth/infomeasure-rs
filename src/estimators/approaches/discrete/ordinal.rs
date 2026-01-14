// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use ndarray::{Array1, Array2, Axis};

use crate::estimators::traits::{GlobalValue, LocalValues};
use super::mle::DiscreteEntropy;
use crate::estimators::approaches::ordinal::ordinal_utils::symbolize_series_compact;

/// Ordinal (permutation) entropy estimator.
///
/// This estimator converts a 1D time series into ordinal patterns of order `m` with
/// delay `tau`, using the canonical symbolization approach (up to m ≤ 20),
/// then computes Shannon entropy using the standard discrete estimator on the codes.
///
/// Local values correspond to -ln p(pattern_t) for each window t.
pub struct OrdinalEntropy {
    inner: DiscreteEntropy,
    pub order: usize,
    pub delay: usize,
}

impl OrdinalEntropy {
    /// Build from a single 1D time series.
    pub fn new(data: Array1<f64>, order: usize, delay: usize) -> Self {
        let codes: Array1<i32> = symbolize_series_compact(&data, order, delay, true);
        let inner = DiscreteEntropy::new(codes);
        Self { inner, order, delay }
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

// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use ndarray::{Array1, Array2};

use crate::estimators::approaches::discrete::discrete_utils::count_frequencies_slice;

/// Row-wise batch entropy (MLE) for 2D integer data.
///
/// Given a 2D array of i32 values, computes the discrete Shannon entropy for each row
/// independently using maximum-likelihood probabilities (natural log base).
pub struct DiscreteEntropyBatchRows {
    data: Array2<i32>,
}

impl DiscreteEntropyBatchRows {
    pub fn new(data: Array2<i32>) -> Self {
        Self { data }
    }

    /// Compute global entropy for each row in parallel.
    pub fn global_values(&self) -> Array1<f64> {
        let nrows = self.data.nrows();
        let results: Vec<f64> = (0..nrows)
            .map(|i| {
                let row = self.data.row(i);
                let slice = row.as_slice().expect("Row should be contiguous in memory");
                let n = slice.len() as f64;
                if n == 0.0 {
                    return 0.0;
                }
                let counts = count_frequencies_slice(slice);
                let mut h = 0.0_f64;
                for &cnt in counts.values() {
                    let p = (cnt as f64) / n;
                    if p > 0.0 {
                        h -= p * p.ln();
                    }
                }
                h
            })
            .collect();
        Array1::from(results)
    }
}

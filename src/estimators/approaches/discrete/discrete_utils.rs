use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Shared dataset and utilities for discrete (histogram-based) entropy estimators.
pub struct DiscreteDataset {
    /// Original integer data (1D)
    pub data: Array1<i32>,
    /// Counts per unique symbol
    pub counts: HashMap<i32, usize>,
    /// Total number of observations
    pub n: usize,
    /// Number of unique symbols
    pub k: usize,
    /// Probability dictionary p(x) for each unique symbol
    pub dist: HashMap<i32, f64>,
}

impl DiscreteDataset {
    /// Build a DiscreteDataset from raw 1D integer data
    pub fn from_data(data: Array1<i32>) -> Self {
        let n = data.len();
        let counts = count_frequencies(&data);
        let k = counts.len();
        let n_f = n as f64;
        let mut dist = HashMap::with_capacity(k);
        for (val, cnt) in counts.iter() {
            dist.insert(*val, *cnt as f64 / n_f);
        }
        Self { data, counts, n, k, dist }
    }

    /// Build a DiscreteDataset from precomputed counts (e.g., GPU histogram) and original data.
    pub fn from_counts_and_data(data: Array1<i32>, counts: HashMap<i32, usize>) -> Self {
        let n = data.len();
        let k = counts.len();
        let n_f = n as f64;
        let mut dist = HashMap::with_capacity(k);
        for (val, cnt) in counts.iter() {
            dist.insert(*val, *cnt as f64 / n_f);
        }
        Self { data, counts, n, k, dist }
    }

    /// Map each sample to its probability using the cached distribution dictionary
    pub fn map_probs(&self) -> Array1<f64> {
        self.data.mapv(|v| self.dist[&v])
    }
}

/// Helper function to count the occurrences of each value in an array.
/// Uses a dense vector for small non-negative ranges, otherwise falls back to HashMap.
pub fn count_frequencies(data: &Array1<i32>) -> HashMap<i32, usize> {
    count_frequencies_slice(data.as_slice().expect("ndarray Array1 should be contiguous"))
}

/// Count frequencies from a raw slice of i32 values with an optimized dense mode.
pub fn count_frequencies_slice(data: &[i32]) -> HashMap<i32, usize> {
    let n = data.len();
    if n == 0 {
        return HashMap::new();
    }

    // Determine min and max to decide whether to use dense counting.
    let mut min_v = i32::MAX;
    let mut max_v = i32::MIN;
    for &v in data.iter() {
        if v < min_v { min_v = v; }
        if v > max_v { max_v = v; }
    }

    // Heuristic threshold: use dense mode if values are non-negative and range is small.
    // Range limit chosen to balance memory and speed; can be tuned.
    const MAX_DENSE_RANGE: i32 = 4096;
    if min_v >= 0 {
        let range = max_v - min_v; // since min_v>=0, this won't underflow
        if range <= MAX_DENSE_RANGE {
            let len = (range as usize) + 1;
            let mut dense = vec![0usize; len];
            for &v in data.iter() {
                let idx = (v - min_v) as usize;
                dense[idx] += 1;
            }
            let mut map = HashMap::with_capacity(len);
            for (i, &cnt) in dense.iter().enumerate() {
                if cnt != 0 {
                    map.insert(min_v + (i as i32), cnt);
                }
            }
            return map;
        }
    }

    // Fallback: generic HashMap counting
    let mut frequency_map = HashMap::new();
    for &value in data.iter() {
        *frequency_map.entry(value).or_insert(0) += 1;
    }
    frequency_map
}


/// Split a 2D array into a Vec of owned 1D rows for batch processing.
pub fn rows_as_vec(data: Array2<i32>) -> Vec<Array1<i32>> {
    data.axis_iter(Axis(0)).map(|row| row.to_owned()).collect()
}

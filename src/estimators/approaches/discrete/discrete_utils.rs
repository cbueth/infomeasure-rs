// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rustc_hash::FxHashMap;

/// Shared dataset and utilities for discrete (histogram-based) entropy estimators.
pub struct DiscreteDataset {
    /// Original integer data (1D)
    pub data: Array1<i32>,
    /// Counts per unique symbol
    pub counts: FxHashMap<i32, usize>,
    /// Total number of observations
    pub n: usize,
    /// Number of unique symbols
    pub k: usize,
    /// Probability dictionary p(x) for each unique symbol
    pub dist: FxHashMap<i32, f64>,
}

impl DiscreteDataset {
    /// Build a DiscreteDataset from raw 1D integer data
    pub fn from_data(data: Array1<i32>) -> Self {
        let n = data.len();
        let counts = count_frequencies(&data);
        let k = counts.len();
        let n_f = n as f64;
        let mut dist = FxHashMap::with_capacity_and_hasher(k, Default::default());
        for (val, cnt) in counts.iter() {
            dist.insert(*val, *cnt as f64 / n_f);
        }
        Self {
            data,
            counts,
            n,
            k,
            dist,
        }
    }

    /// Build a DiscreteDataset from precomputed counts (e.g. GPU histogram) and original data.
    pub fn from_counts_and_data(data: Array1<i32>, counts: FxHashMap<i32, usize>) -> Self {
        let n = data.len();
        let k = counts.len();
        let n_f = n as f64;
        let mut dist = FxHashMap::with_capacity_and_hasher(k, Default::default());
        for (val, cnt) in counts.iter() {
            dist.insert(*val, *cnt as f64 / n_f);
        }
        Self {
            data,
            counts,
            n,
            k,
            dist,
        }
    }

    /// Map each sample to its probability using the cached distribution dictionary
    pub fn map_probs(&self) -> Array1<f64> {
        self.data.mapv(|v| self.dist[&v])
    }
}

/// Helper function to count the occurrences of each value in an array.
/// Uses a dense vector for small non-negative ranges, otherwise falls back to HashMap.
pub fn count_frequencies(data: &Array1<i32>) -> FxHashMap<i32, usize> {
    count_frequencies_slice(
        data.as_slice()
            .expect("ndarray Array1 should be contiguous"),
    )
}

/// Count frequencies from a raw slice of i32 values with an optimised dense mode.
pub fn count_frequencies_slice(data: &[i32]) -> FxHashMap<i32, usize> {
    let n = data.len();
    if n == 0 {
        return FxHashMap::default();
    }

    // Determine min and max to decide whether to use dense counting.
    let mut min_v = i32::MAX;
    let mut max_v = i32::MIN;
    for &v in data.iter() {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
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
            let mut map = FxHashMap::with_capacity_and_hasher(len, Default::default());
            for (i, &cnt) in dense.iter().enumerate() {
                if cnt != 0 {
                    map.insert(min_v + (i as i32), cnt);
                }
            }
            return map;
        }
    }

    // Fallback: generic HashMap counting
    let mut frequency_map = FxHashMap::default();
    for &value in data.iter() {
        *frequency_map.entry(value).or_insert(0) += 1;
    }
    frequency_map
}

/// Split a 2D array into a Vec of owned 1D rows for batch processing.
pub fn rows_as_vec(data: Array2<i32>) -> Vec<Array1<i32>> {
    data.axis_iter(Axis(0)).map(|row| row.to_owned()).collect()
}

/// Pack a joint tuple of `codes` (shifted by `min_code` so they are non-negative) into a
/// single `u128` key, giving each dimension a fixed `bits`-wide field. Returns `None` if any
/// shifted code does not fit in `bits` bits (caller should fall back to a `Vec` key).
fn pack_joint_key(codes: impl IntoIterator<Item = i32>, min_code: i32, bits: u32) -> Option<u128> {
    let mut key: u128 = 0;
    for (d, c) in codes.into_iter().enumerate() {
        let v = (c as i64 - min_code as i64) as u128;
        if v >= (1u128 << bits) {
            return None;
        }
        key |= v << (d as u32 * bits);
    }
    Some(key)
}

/// Reduce multiple code arrays (aligned by index) into a single compact joint code space.
///
/// Given k arrays of equal length containing compact i32 codes, this function produces a
/// single `Array1<i32>` where each position's tuple of codes is mapped to a unique compact i32 ID.
/// The mapping preserves first-occurrence order for determinism.
///
/// The joint tuple is packed into a single `u128` key (shifting codes to non-negative and
/// giving each dimension a fixed bit width) to avoid per-entry `Vec<i32>` allocation. If the
/// packing would overflow (`k * bit_width > 128`), a `Vec<i32>`-keyed fallback is used.
pub fn reduce_joint_space_compact(code_arrays: &[Array1<i32>]) -> Array1<i32> {
    let views: Vec<ArrayView1<i32>> = code_arrays.iter().map(|arr| arr.view()).collect();
    reduce_views_compact(&views)
}

/// Core of [`reduce_joint_space_compact`] operating on borrowed column views,
/// so callers holding strided embedding columns can reduce without
/// materialising arrays.
pub(crate) fn reduce_views_compact(cols: &[ArrayView1<i32>]) -> Array1<i32> {
    if cols.is_empty() {
        return Array1::zeros(0);
    }
    let len = cols[0].len();
    for col in cols.iter() {
        assert_eq!(
            col.len(),
            len,
            "All code arrays must have the same length for joint reduction"
        );
    }
    let k = cols.len();

    let mut min_code = i32::MAX;
    let mut max_code = i32::MIN;
    for col in cols.iter() {
        for &c in col.iter() {
            min_code = min_code.min(c);
            max_code = max_code.max(c);
        }
    }

    let mut out: Vec<i32> = Vec::with_capacity(len);
    if min_code == i32::MAX {
        // All code arrays are empty. nothing to reduce.
        return Array1::from(out);
    }

    let range = (max_code as i64 - min_code as i64) as u128;
    let bits = (128 - range.leading_zeros()).max(1);
    if k as u128 * bits as u128 <= 128 {
        // Packed key path: each dimension occupies `bits` bits of a single u128.
        let mut map: FxHashMap<u128, i32> = FxHashMap::default();
        let mut next_id: i32 = 0;
        for i in 0..len {
            let key = pack_joint_key(cols.iter().map(|col| col[i]), min_code, bits)
                .expect("width pre-checked");
            let id = *map.entry(key).or_insert_with(|| {
                let v = next_id;
                next_id = next_id
                    .checked_add(1)
                    .expect("Too many unique joint patterns to fit into i32");
                v
            });
            out.push(id);
        }
    } else {
        // Fallback: keyed by the full tuple Vec (packing width too large).
        let mut map: FxHashMap<Vec<i32>, i32> = FxHashMap::default();
        let mut next_id: i32 = 0;
        for i in 0..len {
            let mut key: Vec<i32> = Vec::with_capacity(k);
            for col in cols.iter() {
                key.push(col[i]);
            }
            let id = *map.entry(key).or_insert_with(|| {
                let v = next_id;
                next_id = next_id
                    .checked_add(1)
                    .expect("Too many unique joint patterns to fit into i32");
                v
            });
            out.push(id);
        }
    }
    Array1::from(out)
}
/// Reduce a 2D array (samples x dimensions) into a single compact 1D code array.
pub fn reduce_array2_compact(data: &Array2<i32>) -> Array1<i32> {
    let columns: Vec<Array1<i32>> = data.axis_iter(Axis(1)).map(|col| col.to_owned()).collect();
    reduce_joint_space_compact(&columns)
}

/// Reduce history *columns* given as (possibly strided) views into one compact
/// 1D code array. Equivalent to [`reduce_array2_compact`] over a matrix whose
/// columns are these views, without materialising that matrix.
pub(crate) fn reduce_hist_columns_compact<'a, I>(cols: I) -> Array1<i32>
where
    I: IntoIterator<Item = ndarray::ArrayView1<'a, i32>>,
{
    let views: Vec<ndarray::ArrayView1<i32>> = cols.into_iter().collect();
    reduce_views_compact(&views)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case([2, 3], 2, 3, Some(0b001_000))]
    #[case([5, 0], 0, 3, Some(0b000_101))]
    #[case([-1, 3], -1, 3, Some(0b100_000))]
    fn pack_joint_key_round_trip(
        #[case] codes: [i32; 2],
        #[case] min_code: i32,
        #[case] bits: u32,
        #[case] expected: Option<u128>,
    ) {
        assert_eq!(pack_joint_key(codes, min_code, bits), expected);
    }

    #[rstest]
    #[case([0, 4], 0, 2)]
    #[case([4, 0], 0, 2)]
    #[case([7, 0], 0, 2)]
    fn pack_joint_key_overflow_returns_none(
        #[case] codes: [i32; 2],
        #[case] min_code: i32,
        #[case] bits: u32,
    ) {
        // bits 2 can hold shifted values 0..3. codes >= 4 do not fit
        assert_eq!(pack_joint_key(codes, min_code, bits), None);
    }

    #[rstest]
    #[case(vec![-2, -2, -1], vec![3, 4, 3], vec![0, 1, 2])]
    #[case(vec![-1, -1], vec![2, 2], vec![0, 0])]
    #[case(vec![-3, -3, -3], vec![0, 1, 2], vec![0, 1, 2])]
    fn reduce_joint_space_compact_negative_codes(
        #[case] a: Vec<i32>,
        #[case] b: Vec<i32>,
        #[case] expected: Vec<i32>,
    ) {
        // Packed path must handle negative codes via the min shift.
        let result = reduce_joint_space_compact(&[Array1::from(a), Array1::from(b)]);
        assert_eq!(result, Array1::from(expected));
    }

    #[test]
    fn reduce_joint_space_compact_wide_range_falls_back() {
        // 5 dimensions spanning the full i32 range give k*bits = 5*32 > 128,
        // which forces the Vec-key fallback path.
        let a = Array1::from(vec![i32::MIN, i32::MIN, i32::MAX]);
        let b = Array1::from(vec![0, i32::MAX, 0]);
        let c = Array1::from(vec![0, 0, i32::MIN]);
        let d = Array1::from(vec![i32::MAX, 0, 0]);
        let e = Array1::from(vec![0, i32::MAX, 0]);
        let result = reduce_joint_space_compact(&[a, b, c, d, e]);
        assert_eq!(result, Array1::from(vec![0, 1, 2]));
    }

    #[test]
    fn reduce_joint_space_compact_empty_arrays() {
        let empty: Array1<i32> = Array1::from(vec![]);
        let expected: Array1<i32> = Array1::from(vec![]);
        assert_eq!(
            reduce_joint_space_compact(&[empty.clone(), empty]),
            expected
        );
    }

    #[rstest]
    #[case(1)]
    #[case(100)]
    #[case(10_000)]
    fn reduce_joint_space_compact_distinct_patterns(#[case] n: usize) {
        // Long input where every sample is a distinct joint pattern: all IDs unique
        // in order, regardless of packed or fallback path.
        let a = Array1::from((0..n as i32).collect::<Vec<_>>());
        let b = Array1::from((0..n as i32).rev().collect::<Vec<_>>());
        let result = reduce_joint_space_compact(&[a, b]);
        let expected = Array1::from((0..n as i32).collect::<Vec<_>>());
        assert_eq!(result, expected);
    }
}

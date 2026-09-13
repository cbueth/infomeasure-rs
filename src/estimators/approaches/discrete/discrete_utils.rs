// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rustc_hash::FxHashMap;

/// Shared dataset and utilities for discrete (histogram-based) entropy estimators.
pub struct DiscreteDataset {
    /// Original integer data (1D). Empty when built from a borrowed slice
    /// (see [`DiscreteDataset::from_borrowed`]).
    pub data: Array1<i32>,
    /// Counts per unique symbol
    pub counts: FxHashMap<i32, usize>,
    /// Total number of observations
    pub n: usize,
    /// Number of unique symbols
    pub k: usize,
    /// Probability dictionary p(x) for each unique symbol
    pub dist: FxHashMap<i32, f64>,
    /// Whether `data` holds the observations (local values available). False
    /// for the borrowed, global-value-only constructor.
    pub has_data: bool,
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
            has_data: true,
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
            has_data: true,
        }
    }

    /// Build a global-value-only dataset from a **borrowed** slice, computing
    /// the frequency counts without taking ownership of the observations.
    ///
    /// `data` is left empty and `has_data` is false, so [`map_probs`] and the
    /// local-value paths are unavailable; this exists so a caller that already
    /// owns the input can `Entropy::new_discrete_from_slice(...)` a pre-loaded
    /// column without an N-copy per call.
    ///
    /// [`map_probs`]: Self::map_probs
    pub fn from_borrowed(data: &[i32]) -> Self {
        let n = data.len();
        let counts = count_frequencies_slice(data);
        let k = counts.len();
        let n_f = n as f64;
        let mut dist = FxHashMap::with_capacity_and_hasher(k, Default::default());
        for (val, cnt) in counts.iter() {
            dist.insert(*val, *cnt as f64 / n_f);
        }
        Self {
            data: Array1::zeros(0),
            counts,
            n,
            k,
            dist,
            has_data: false,
        }
    }

    /// Map each sample to its probability using the cached distribution dictionary
    pub fn map_probs(&self) -> Array1<f64> {
        if !self.has_data {
            return Array1::zeros(0);
        }
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
    reduce_views_compact_counted(cols).0
}

/// Same reduction as [`reduce_views_compact`], additionally emitting the
/// frequency of every dense code in code order (`counts[code]`), so callers
/// can build datasets without recounting.
/// Largest dense joint table we build. Above this the joint space is too large
/// (or too sparse) for direct indexing and the packed-key hash map is used.
const DENSE_JOINT_CAP: u128 = 1 << 20;

pub(crate) fn reduce_views_compact_counted(cols: &[ArrayView1<i32>]) -> (Array1<i32>, Vec<usize>) {
    if cols.is_empty() {
        return (Array1::zeros(0), Vec::new());
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
    if len == 0 {
        return (Array1::zeros(0), Vec::new());
    }

    // Per-column minima/ranges drive the dense mixed-radix path; the global
    // min/max is kept for the hash-map fallbacks.
    let mut mins = vec![i32::MAX; k];
    let mut maxs = vec![i32::MIN; k];
    for (d, col) in cols.iter().enumerate() {
        for &c in col.iter() {
            if c < mins[d] {
                mins[d] = c;
            }
            if c > maxs[d] {
                maxs[d] = c;
            }
        }
    }
    let min_code = *mins.iter().min().expect("non-empty");
    let max_code = *maxs.iter().max().expect("non-empty");

    let mut out: Vec<i32> = Vec::with_capacity(len);
    let mut counts: Vec<usize> = Vec::new();

    // Dense direct-index path: capacity = product of the per-column alphabet
    // sizes. Taken only when the table is small *and* not wildly sparse relative
    // to `len`; high-cardinality columns fall through to the hash map, which
    // handles huge/empty joint spaces without an allocation blowup.
    let mut capacity: u128 = 1;
    for d in 0..k {
        let range = (maxs[d] as i64 - mins[d] as i64 + 1) as u128;
        capacity = match capacity.checked_mul(range) {
            Some(c) if c <= DENSE_JOINT_CAP => c,
            _ => {
                capacity = 0;
                break;
            }
        };
    }
    if capacity > 0 && capacity <= (16 * len as u128).max(4096) {
        let cap = capacity as usize;
        let mut stride = vec![0usize; k];
        let mut acc: usize = 1;
        for d in 0..k {
            stride[d] = acc;
            acc = acc.saturating_mul((maxs[d] as i64 - mins[d] as i64 + 1) as usize);
        }
        let mut slot: Vec<i32> = vec![-1; cap];
        let mut next_id: i32 = 0;
        for (i, _) in cols[0].iter().enumerate() {
            let mut idx: usize = 0;
            for (d, col) in cols.iter().enumerate() {
                idx += (col[i] - mins[d]) as usize * stride[d];
            }
            let s = slot[idx];
            let id = if s < 0 {
                let v = next_id;
                next_id = next_id
                    .checked_add(1)
                    .expect("Too many unique joint patterns to fit into i32");
                slot[idx] = v;
                counts.push(0);
                v
            } else {
                s
            };
            counts[id as usize] += 1;
            out.push(id);
        }
        return (Array1::from(out), counts);
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
            if counts.len() < next_id as usize {
                counts.push(0);
            }
            counts[id as usize] += 1;
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
            if counts.len() < next_id as usize {
                counts.push(0);
            }
            counts[id as usize] += 1;
            out.push(id);
        }
    }
    let codes = Array1::from(out);
    (codes, counts)
}
/// Largest dense full-joint table for single-pass marginalisation. Above this
/// the joint is too large/sparse for the approach (the caller falls back to
/// per-space counting).
pub(crate) const JOINT_MARGINAL_CAP: u128 = 1 << 20;

/// Count a joint over all `cols` in **one** pass and return each requested
/// projection as compact `(codes, counts)`.
///
/// `projections[i]` lists the column indices of space `i` (ascending). Every
/// space is produced by marginalising the same single pass instead of
/// rescanning the data, which is the point of the fused MLE constructors: CMI,
/// TE and CTE otherwise run four independent reduction+count passes per call.
///
/// Returns `None` when the full-joint alphabet exceeds [`JOINT_MARGINAL_CAP`];
/// the caller then falls back to [`reduce_views_compact_counted`] per space.
pub(crate) fn joint_marginal_counts(
    cols: &[ArrayView1<i32>],
    projections: &[Vec<usize>],
) -> Option<Vec<(Array1<i32>, Vec<usize>)>> {
    if cols.is_empty() {
        return Some(Vec::new());
    }
    let len = cols[0].len();
    for col in cols {
        assert_eq!(
            col.len(),
            len,
            "All code arrays must have the same length for joint reduction"
        );
    }
    let m = cols.len();

    let mut mins = vec![i32::MAX; m];
    let mut maxs = vec![i32::MIN; m];
    for (d, col) in cols.iter().enumerate() {
        for &c in col.iter() {
            if c < mins[d] {
                mins[d] = c;
            }
            if c > maxs[d] {
                maxs[d] = c;
            }
        }
    }

    let mut full_cap: u128 = 1;
    for d in 0..m {
        let range = (maxs[d] as i64 - mins[d] as i64 + 1) as u128;
        full_cap = full_cap.checked_mul(range)?;
        if full_cap > JOINT_MARGINAL_CAP {
            return None;
        }
    }

    let np = projections.len();
    let mut pstrides: Vec<Vec<u64>> = Vec::with_capacity(np);
    let mut slots: Vec<Vec<i32>> = Vec::with_capacity(np);
    for p in projections {
        let mut strides = Vec::with_capacity(p.len());
        let mut acc: u64 = 1;
        for &d in p {
            strides.push(acc);
            acc = acc.saturating_mul((maxs[d] as i64 - mins[d] as i64 + 1) as u64);
        }
        slots.push(vec![-1i32; acc as usize]);
        pstrides.push(strides);
    }

    let mut out_codes: Vec<Vec<i32>> = projections
        .iter()
        .map(|_| Vec::with_capacity(len))
        .collect();
    let mut out_counts: Vec<Vec<usize>> = vec![Vec::new(); np];
    let mut next: Vec<i32> = vec![0; np];
    let mut digits = vec![0u64; m];

    for (i, _) in cols[0].iter().enumerate() {
        for d in 0..m {
            digits[d] = (cols[d][i] - mins[d]) as u64;
        }
        for pi in 0..np {
            let mut idx: usize = 0;
            for (k, &d) in projections[pi].iter().enumerate() {
                idx += (digits[d] * pstrides[pi][k]) as usize;
            }
            let s = slots[pi][idx];
            let id = if s < 0 {
                let v = next[pi];
                next[pi] = v
                    .checked_add(1)
                    .expect("Too many unique joint patterns to fit into i32");
                slots[pi][idx] = v;
                out_counts[pi].push(0);
                v
            } else {
                s
            };
            out_counts[pi][id as usize] += 1;
            out_codes[pi].push(id);
        }
    }

    Some(
        out_codes
            .into_iter()
            .zip(out_counts)
            .map(|(codes, counts)| (Array1::from(codes), counts))
            .collect(),
    )
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

/// Build a [`DiscreteDataset`] from dense codes plus their per-code counts,
/// skipping the usual recount pass. `dense_counts[i]` is the frequency of
/// code `i`; the alphabet is exactly `0..dense_counts.len()`.
pub(crate) fn dataset_from_dense_codes(
    codes: Array1<i32>,
    dense_counts: &[usize],
) -> DiscreteDataset {
    let n: usize = dense_counts.iter().sum();
    let k = dense_counts.len();
    let n_f = n as f64;
    let mut counts_map = FxHashMap::with_capacity_and_hasher(k, Default::default());
    let mut dist = FxHashMap::with_capacity_and_hasher(k, Default::default());
    for (i, &cnt) in dense_counts.iter().enumerate() {
        if cnt == 0 {
            continue;
        }
        counts_map.insert(i as i32, cnt);
        dist.insert(i as i32, cnt as f64 / n_f);
    }
    DiscreteDataset {
        data: codes,
        counts: counts_map,
        n,
        k,
        dist,
        has_data: true,
    }
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
    fn dense_matches_hashed_joint_reduction() {
        // Small alphabets take the dense direct-index path; scaling the codes by
        // 10^6 blows the capacity guard and takes the packed-u128 hash map. Ids
        // are assigned in first-occurrence order, so both must agree exactly.
        let cols = [
            Array1::from(vec![0, 1, 0, 1, 2, 0]),
            Array1::from(vec![5, 5, 3, 5, 5, 3]),
        ];
        let scaled: Vec<Array1<i32>> = cols.iter().map(|c| c.mapv(|v| v * 1_000_000)).collect();
        let dense =
            reduce_views_compact_counted(&cols.iter().map(|c| c.view()).collect::<Vec<_>>());
        let hashed =
            reduce_views_compact_counted(&scaled.iter().map(|c| c.view()).collect::<Vec<_>>());
        assert_eq!(dense.0, hashed.0);
        assert_eq!(dense.1, hashed.1);
        assert_eq!(dense.0, Array1::from(vec![0, 1, 2, 1, 3, 2]));
        assert_eq!(dense.1, vec![1, 2, 2, 1]);
    }

    #[test]
    fn borrowed_dataset_matches_owned_counts() {
        let data = vec![0, 1, 2, 1, 0, 3, 3, 2, 1, 0];
        let owned = DiscreteDataset::from_data(Array1::from(data.clone()));
        let borrowed = DiscreteDataset::from_borrowed(&data);
        assert_eq!(owned.n, borrowed.n);
        assert_eq!(owned.k, borrowed.k);
        assert_eq!(owned.counts, borrowed.counts);
        assert!(owned.has_data && !borrowed.has_data);
        assert_eq!(borrowed.map_probs().len(), 0);
    }

    #[test]
    fn joint_marginal_matches_per_projection() {
        // Marginalising one joint pass must give exactly the codes/counts that
        // counting each projection independently would (same first-occurrence
        // order), for every subset.
        let a = Array1::from(vec![0, 1, 0, 1, 2, 0, 2, 1]);
        let b = Array1::from(vec![1, 0, 1, 0, 1, 1, 0, 0]);
        let c = Array1::from(vec![0, 0, 1, 1, 2, 2, 3, 3]);
        let cols = [a.view(), b.view(), c.view()];
        let projections = vec![vec![0, 1], vec![1, 2], vec![0, 1, 2], vec![2], vec![0]];
        let got = joint_marginal_counts(&cols, &projections).expect("small joint");
        for (p, (codes, counts)) in projections.iter().zip(got) {
            let pc: Vec<ArrayView1<i32>> = p.iter().map(|&d| cols[d]).collect();
            let (exp_codes, exp_counts) = reduce_views_compact_counted(&pc);
            assert_eq!(codes, exp_codes, "codes for projection {p:?}");
            assert_eq!(counts, exp_counts, "counts for projection {p:?}");
        }
    }

    #[test]
    fn joint_marginal_falls_back_when_too_large() {
        // Full alphabet exceeds the cap -> None, caller falls back.
        let a = Array1::from(vec![0, 40_000, 40_001]);
        let b = Array1::from(vec![0, 40_000, 40_001]);
        let c = Array1::from(vec![0, 40_000, 40_001]);
        let cols = [a.view(), b.view(), c.view()];
        assert!(joint_marginal_counts(&cols, &[vec![0, 1, 2]]).is_none());
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

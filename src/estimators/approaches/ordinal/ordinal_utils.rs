// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use ndarray::Array1;
use std::collections::HashMap;

/// Argsort for f64 values.
///
/// Returns indices that would sort the slice in ascending order.
/// If stable is true, ties are resolved by the original index order (stable),
/// matching numpy.argsort(stable=True).
/// If stable is false, ties are resolved arbitrarily (unstable).
pub(crate) fn argsort(window: &[f64], idx: &mut [usize], stable: bool) {
    for (i, val) in idx.iter_mut().enumerate() {
        *val = i;
    }
    if stable {
        idx.sort_by(|&i, &j| {
            let a = window[i];
            let b = window[j];
            match a.partial_cmp(&b) {
                Some(ord) => {
                    if ord == core::cmp::Ordering::Equal {
                        i.cmp(&j)
                    } else {
                        ord
                    }
                }
                None => {
                    // One or both are NaN.
                    // To be consistent, NaNs are "greater" than everything.
                    if a.is_nan() && b.is_nan() {
                        i.cmp(&j)
                    } else if a.is_nan() {
                        core::cmp::Ordering::Greater
                    } else {
                        core::cmp::Ordering::Less
                    }
                }
            }
        });
    } else {
        idx.sort_unstable_by(|&i, &j| {
            let a = window[i];
            let b = window[j];
            match a.partial_cmp(&b) {
                Some(ord) => ord,
                None => {
                    if a.is_nan() && b.is_nan() {
                        core::cmp::Ordering::Equal
                    } else if a.is_nan() {
                        core::cmp::Ordering::Greater
                    } else {
                        core::cmp::Ordering::Less
                    }
                }
            }
        });
    }
}

/// Compute the Lehmer code (factoradic ranking) for a given permutation.
///
/// The input is a permutation of 0..m-1 represented as indices in the order
/// they would appear when sorting a window. This matches the output of argsort.
///
/// Panics if m > 20 (u64 overflow risk for factorial weights).
pub fn lehmer_code(perm: &[usize]) -> u64 {
    let n = perm.len();
    if n > 20 {
        panic!("For embedding dimensions larger than 20, the integer will be too large for u64.");
    }
    // Precompute factorials up to n
    let mut fact: Vec<u128> = vec![1u128; n];
    for i in 1..n {
        fact[i] = fact[i - 1] * (i as u128);
    }

    let mut acc: u128 = 0;
    for i in 0..n {
        let mut c = 0u128;
        for j in (i + 1)..n {
            if perm[i] > perm[j] {
                c += 1;
            }
        }
        let weight = fact[n - 1 - i];
        acc += c * weight;
    }
    // Fit into u64 (guaranteed for n <= 20)
    acc as u64
}

/// Internal version of lehmer_code that avoids recomputing factorials.
#[allow(dead_code)]
fn lehmer_code_with_fact(perm: &[usize], fact: &[u128]) -> u64 {
    let n = perm.len();
    let mut acc: u128 = 0;
    for i in 0..n {
        let mut c = 0u128;
        for j in (i + 1)..n {
            if perm[i] > perm[j] {
                c += 1;
            }
        }
        let weight = fact[n - 1 - i];
        acc += c * weight;
    }
    acc as u64
}

/// Remap u64 codes to compact i32 IDs for use with discrete estimators.
/// Each unique u64 code gets assigned a unique i32 ID based on first occurrence order.
pub(crate) fn remap_u64_to_i32(codes: &Array1<u64>) -> Array1<i32> {
    let mut map: HashMap<u64, i32> = HashMap::with_capacity(codes.len());
    let mut next_id: i32 = 0;
    let mut out = Vec::with_capacity(codes.len());
    for &c in codes.iter() {
        let id = *map.entry(c).or_insert_with(|| {
            let v = next_id;
            next_id = next_id
                .checked_add(1)
                .expect("Too many unique patterns to fit into i32");
            v
        });
        out.push(id);
    }
    Array1::from(out)
}

/// Convert a time series into compact i32 ordinal pattern codes.
///
/// This is a thin wrapper around `symbolize_series_u64` that remaps the raw Lehmer codes
/// to a compact i32 index space for integration with discrete estimators.
///
/// Note: The remapping is based on first-occurrence order, so the resulting IDs
/// will NOT match raw Lehmer codes and may differ from Python's symbolization
/// values (which use raw Lehmer codes or value-sorted remapping).
///
/// - order (m) ≥ 1
/// - step_size (τ) ≥ 1
/// - Supports orders up to 20 (Lehmer code fits in u64)
///
/// For parity testing against Python, use `symbolize_series_u64` instead.
pub fn symbolize_series_compact(
    series: &Array1<f64>,
    order: usize,
    step_size: usize,
    stable: bool,
) -> Array1<i32> {
    let codes_u64 = symbolize_series_u64(series, order, step_size, stable);
    remap_u64_to_i32(&codes_u64)
}

/// Return raw Lehmer codes (u64) for permutation patterns without remapping.
/// Useful for parity tests against Python utils.symbolize_series(to_int=True).
pub fn symbolize_series_u64(
    series: &Array1<f64>,
    order: usize,
    step_size: usize,
    stable: bool,
) -> Array1<u64> {
    if order < 1 {
        panic!("The embedding order must be a positive integer.");
    }
    if step_size < 1 {
        panic!("The step_size must be a positive integer.");
    }
    if order > 20 {
        panic!("For embedding dimensions larger than 20, the integer will be too large for u64.");
    }

    let n = series.len();
    if n == 0 {
        return Array1::<u64>::zeros(0);
    }

    let span = (order - 1) * step_size;
    if n <= span {
        return Array1::<u64>::zeros(0);
    }

    let n_windows = n - span;
    let mut out: Vec<u64> = Vec::with_capacity(n_windows);

    // Reuse buffers to avoid repeated allocations
    let mut w: Vec<f64> = vec![0.0; order];
    let mut idx: Vec<usize> = (0..order).collect();

    for t in 0..n_windows {
        for j in 0..order {
            w[j] = series[t + j * step_size];
        }

        argsort(&w, &mut idx, stable);
        let code_u64 = lehmer_code(&idx);
        out.push(code_u64);
    }
    Array1::from(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argsort_basic() {
        let window = [3.0, 1.0, 4.0, 2.0];
        let mut idx = [0usize; 4];
        argsort(&window, &mut idx, true);
        assert_eq!(idx, [1, 3, 0, 2]);

        argsort(&window, &mut idx, false);
        assert_eq!(idx, [1, 3, 0, 2]);
    }

    #[test]
    fn test_argsort_stable_with_ties() {
        let window = [1.0, 2.0, 1.0, 0.0];
        let mut idx = [0usize; 4];
        // Indices of 1.0 are 0 and 2.
        // Stable sort should keep them as 0 then 2.
        argsort(&window, &mut idx, true);
        assert_eq!(idx, [3, 0, 2, 1]);
    }

    #[test]
    fn test_argsort_unstable_with_ties() {
        let window = [1.0, 2.0, 1.0, 0.0];
        let mut idx = [0usize; 4];
        // Unstable sort doesn't guarantee order of ties.
        argsort(&window, &mut idx, false);

        // Verify that it is a valid sort
        for i in 0..3 {
            assert!(window[idx[i]] <= window[idx[i + 1]]);
        }

        // In this specific case, 0.0 is at index 3, 2.0 is at index 1.
        assert_eq!(idx[0], 3);
        assert_eq!(idx[3], 1);
        // idx[1] and idx[2] must be 0 and 2 in some order.
        assert!((idx[1] == 0 && idx[2] == 2) || (idx[1] == 2 && idx[2] == 0));
    }

    #[test]
    fn test_argsort_empty() {
        let window: [f64; 0] = [];
        let mut idx: [usize; 0] = [];
        argsort(&window, &mut idx, true);
        // Should not panic
    }

    #[test]
    fn test_argsort_single_element() {
        let window = [42.0];
        let mut idx = [0usize];
        argsort(&window, &mut idx, true);
        assert_eq!(idx, [0]);
    }

    #[test]
    fn test_argsort_nan() {
        let window = [1.0, f64::NAN, 0.0];
        let mut idx = [0usize; 3];
        // Our implementation treats NaN as "Greater" than everything.

        // Stable:
        argsort(&window, &mut idx, true);
        // 0.0 (idx 2) < 1.0 (idx 0) < NaN (idx 1)
        assert_eq!(idx, [2, 0, 1]);

        // Unstable:
        argsort(&window, &mut idx, false);
        assert_eq!(idx, [2, 0, 1]);
    }

    #[test]
    fn test_remap_u64_to_i32_parametrized() {
        use ndarray::array;

        let test_cases: [(&str, Array1<u64>, Array1<i32>); 6] = [
            (
                "basic",
                array![100, 200, 100, 300, 200, 400],
                array![0, 1, 0, 2, 1, 3],
            ),
            ("empty", Array1::<u64>::zeros(0), Array1::<i32>::zeros(0)),
            (
                "all_same",
                Array1::from_elem(5, 42u64),
                Array1::from_elem(5, 0i32),
            ),
            (
                "all_unique",
                array![10, 20, 30, 40, 50],
                array![0, 1, 2, 3, 4],
            ),
            (
                "first_occurrence_order",
                array![50, 10, 50, 30, 10, 30],
                array![0, 1, 0, 2, 1, 2],
            ),
            (
                "large_values",
                array![u64::MAX, u64::MIN, u64::MAX, 1234567890123456789u64],
                array![0, 1, 0, 2],
            ),
        ];

        for (name, input, expected) in test_cases.iter() {
            assert_eq!(
                remap_u64_to_i32(input),
                *expected,
                "test case {name:?} failed"
            );
        }
    }

    #[test]
    fn test_symbolize_series_u64_basic() {
        use ndarray::array;

        let series = array![1.0, 2.0, 3.0, 2.0, 1.0];

        // Order 1 - each point is independent
        let result = symbolize_series_u64(&series, 1, 1, true);
        assert_eq!(result.len(), 5);

        // Order 2 - pattern size 2
        let result = symbolize_series_u64(&series, 2, 1, true);
        assert_eq!(result.len(), 4); // 5 - (2-1)*1

        // Order 3 - pattern size 3
        let result = symbolize_series_u64(&series, 3, 1, true);
        assert_eq!(result.len(), 3); // 5 - (3-1)*1
    }

    #[test]
    fn test_symbolize_series_u64_empty() {
        use ndarray::array;

        let series = array![];
        let result = symbolize_series_u64(&series, 2, 1, true);
        assert_eq!(result.len(), 0);
    }

    #[test]
    #[should_panic(expected = "The embedding order must be a positive integer.")]
    fn test_symbolize_series_u64_invalid_order() {
        use ndarray::array;

        let series = array![1.0, 2.0, 3.0];
        symbolize_series_u64(&series, 0, 1, true);
    }

    #[test]
    #[should_panic(expected = "The step_size must be a positive integer.")]
    fn test_symbolize_series_u64_invalid_step() {
        use ndarray::array;

        let series = array![1.0, 2.0, 3.0];
        symbolize_series_u64(&series, 2, 0, true);
    }
}

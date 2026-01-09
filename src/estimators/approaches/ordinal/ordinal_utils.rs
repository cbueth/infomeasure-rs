use ndarray::Array1;
use std::collections::HashMap;

/// Argsort for f64 values.
///
/// Returns indices that would sort the slice in ascending order.
/// If stable is true, ties are resolved by the original index order (stable),
/// matching numpy.argsort(stable=True).
/// If stable is false, ties are resolved arbitrarily (unstable).
pub fn argsort(window: &[f64], idx: &mut [usize], stable: bool) {
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
            if perm[i] > perm[j] { c += 1; }
        }
        let weight = fact[n - 1 - i];
        acc += c * weight;
    }
    // Fit into u64 (guaranteed for n <= 20)
    acc as u64
}

/// Internal version of lehmer_code that avoids recomputing factorials.
fn lehmer_code_with_fact(perm: &[usize], fact: &[u128]) -> u64 {
    let n = perm.len();
    let mut acc: u128 = 0;
    for i in 0..n {
        let mut c = 0u128;
        for j in (i + 1)..n {
            if perm[i] > perm[j] { c += 1; }
        }
        let weight = fact[n - 1 - i];
        acc += c * weight;
    }
    acc as u64
}

/// Remap u64 codes to compact i32 IDs for use with discrete estimators.
/// Each unique u64 code gets assigned a unique i32 ID based on first occurrence order.
pub fn remap_u64_to_i32(codes: &Array1<u64>) -> Array1<i32> {
    use std::collections::HashMap;
    let mut map: HashMap<u64, i32> = HashMap::with_capacity(codes.len());
    let mut next_id: i32 = 0;
    let mut out = Vec::with_capacity(codes.len());
    for &c in codes.iter() {
        let id = *map.entry(c).or_insert_with(|| {
            let v = next_id;
            next_id = next_id.checked_add(1).expect("Too many unique patterns to fit into i32");
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
pub fn symbolize_series_compact(series: &Array1<f64>, order: usize, step_size: usize, stable: bool) -> Array1<i32> {
    let codes_u64 = symbolize_series_u64(series, order, step_size, stable);
    remap_u64_to_i32(&codes_u64)
}

/// Return raw Lehmer codes (u64) for permutation patterns without remapping.
/// Useful for parity tests against Python utils.symbolize_series(to_int=True).
pub fn symbolize_series_u64(series: &Array1<f64>, order: usize, step_size: usize, stable: bool) -> Array1<u64> {
    if order < 1 { panic!("The embedding order must be a positive integer."); }
    if step_size < 1 { panic!("The step_size must be a positive integer."); }
    if order > 20 { panic!("For embedding dimensions larger than 20, the integer will be too large for u64."); }

    let n = series.len();
    if n == 0 { return Array1::<u64>::zeros(0); }

    let span = (order - 1) * step_size;
    if n <= span { return Array1::<u64>::zeros(0); }

    let n_windows = n - span;
    let mut out: Vec<u64> = Vec::with_capacity(n_windows);

    // Reuse buffers to avoid repeated allocations
    let mut w: Vec<f64> = vec![0.0; order];
    let mut idx: Vec<usize> = (0..order).collect();

    // Precompute factorials up to order
    let mut fact: Vec<u128> = vec![1u128; order];
    for i in 1..order {
        fact[i] = fact[i - 1] * (i as u128);
    }

    for t in 0..n_windows {
        for j in 0..order {
            w[j] = series[t + j * step_size];
        }

        argsort(&w, &mut idx, stable);
        let code_u64 = lehmer_code_with_fact(&idx, &fact);
        out.push(code_u64);
    }
    Array1::from(out)
}

/// Reduce multiple code arrays (aligned by index) into a single compact joint code space.
///
/// Given k arrays of equal length containing compact i32 codes, this function produces a
/// single Array1<i32> where each position's tuple of codes is mapped to a unique compact i32 ID.
/// The mapping preserves first-occurrence order for determinism.
pub fn reduce_joint_space_compact(code_arrays: &[Array1<i32>]) -> Array1<i32> {
    if code_arrays.is_empty() { return Array1::zeros(0); }
    let len = code_arrays[0].len();
    for arr in code_arrays.iter() {
        assert_eq!(arr.len(), len, "All code arrays must have the same length for joint reduction");
    }
    let mut map: HashMap<Vec<i32>, i32> = HashMap::new();
    let mut next_id: i32 = 0;
    let mut out: Vec<i32> = Vec::with_capacity(len);
    let k = code_arrays.len();
    for i in 0..len {
        let mut key: Vec<i32> = Vec::with_capacity(k);
        for arr in code_arrays.iter() { key.push(arr[i]); }
        let id = *map.entry(key).or_insert_with(|| {
            let v = next_id;
            next_id = next_id.checked_add(1).expect("Too many unique joint patterns to fit into i32");
            v
        });
        out.push(id);
    }
    Array1::from(out)
}

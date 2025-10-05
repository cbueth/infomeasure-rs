use ndarray::Array1;

/// Stable argsort for f64 values within a window.
///
/// Returns indices that would sort the slice in ascending order. Ties are
/// resolved by the original index order (stable), matching numpy.argsort(stable=True).
pub fn stable_argsort(window: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..window.len()).collect();
    idx.sort_by(|&i, &j| {
        match window[i].partial_cmp(&window[j]) {
            Some(core::cmp::Ordering::Less) => core::cmp::Ordering::Less,
            Some(core::cmp::Ordering::Greater) => core::cmp::Ordering::Greater,
            Some(core::cmp::Ordering::Equal) | None => i.cmp(&j),
        }
    });
    idx
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

/// Remap u64 codes to compact i32 IDs for use with discrete estimators.
/// Each unique u64 code gets assigned a unique i32 ID based on first occurrence order.
fn remap_u64_to_i32(codes: &Array1<u64>) -> Array1<i32> {
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
/// - order (m) ≥ 1
/// - step_size (τ) ≥ 1  
/// - Supports orders up to 20 (Lehmer code fits in u64)
///
/// For parity testing against Python, use `symbolize_series_u64` instead.
pub fn symbolize_series_compact(series: &Array1<f64>, order: usize, step_size: usize, stable: bool) -> Array1<i32> {
    let codes_u64 = symbolize_series_u64(series, order, step_size, stable);
    remap_u64_to_i32(&codes_u64)
}

/// Legacy alias for `symbolize_series_compact` to maintain compatibility.
/// 
/// **Deprecated**: Use `symbolize_series_compact` for new code, or `symbolize_series_u64` 
/// for parity testing against Python.
pub fn symbolize_series(series: &Array1<f64>, order: usize, step_size: usize, stable: bool) -> Array1<i32> {
    symbolize_series_compact(series, order, step_size, stable)
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
    for t in 0..n_windows {
        let mut w: Vec<f64> = Vec::with_capacity(order);
        for j in 0..order { w.push(series[t + j * step_size]); }
        let perm = if stable { stable_argsort(&w) } else { stable_argsort(&w) };
        let code_u64 = lehmer_code(&perm);
        out.push(code_u64);
    }
    Array1::from(out)
}

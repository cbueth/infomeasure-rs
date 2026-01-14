// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

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

/// Convert a time series into integer ordinal pattern codes using permutation patterns.
///
/// - order (m) ≥ 1
/// - delay (τ) ≥ 1
/// - For minimal integration with existing discrete pipeline, we cap order ≤ 12 (so m! ≤ i32::MAX).
///   This can be generalized later by adopting u64 keys in discrete utilities.
pub fn symbolize_series(series: &Array1<f64>, order: usize, delay: usize, stable: bool) -> Array1<i32> {
    if order < 1 { panic!("The embedding order must be a positive integer."); }
    if delay < 1 { panic!("The delay must be a positive integer."); }
    if order > 12 { panic!("Temporary limitation: order must be ≤ 12 to fit pattern IDs into i32."); }

    let n = series.len();
    if n == 0 { return Array1::<i32>::zeros(0); }

    let span = (order - 1) * delay;
    if n <= span { return Array1::<i32>::zeros(0); }

    let n_windows = n - span;
    let mut out: Vec<i32> = Vec::with_capacity(n_windows);

    // Build each window, compute permutation (argsort), map to Lehmer code, store as i32
    for t in 0..n_windows {
        // Extract window values at positions t + j*delay
        let mut w: Vec<f64> = Vec::with_capacity(order);
        for j in 0..order { w.push(series[t + j * delay]); }
        // Permutation via (stable) argsort
        let perm = if stable { stable_argsort(&w) } else {
            // Not-stable variant: same comparator, but instability not strictly guaranteed in Rust
            // However, using stable sort here too is acceptable; parity issues arise only on ties.
            stable_argsort(&w)
        };
        let code_u64 = lehmer_code(&perm);
        if code_u64 > i32::MAX as u64 {
            panic!("Pattern code exceeds i32 range. Reduce order or generalize key type.");
        }
        out.push(code_u64 as i32);
    }

    Array1::from(out)
}

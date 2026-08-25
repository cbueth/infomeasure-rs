// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Utilities for slicing and embedding time series data for Transfer Entropy (TE) and
//! Conditional Transfer Entropy (CTE) estimators.
//!
//! This module provides functions to transform raw time series into the embedding formats
//! required by various estimators. It handles:
//! - Time-delay embeddings with configurable history lengths and step sizes.
//! - Slicing data into future, history, and (optionally) conditioning components.
//! - Random permutation of source history for significance testing and local TE/CTE calculations.
//! - Support for both scalar (`Array1`) and multivariate (`Array2`) time series.
//!
//! The core embedding logic follows the "oldest first" column ordering convention, where
//! column 0 contains the most distant past sample and the last column contains the most
//! recent past sample.

use ndarray::{Array1, Array2, ArrayView1};
use rand::seq::SliceRandom;
use rand::thread_rng;

/// Build the embedding arrays used by transfer-entropy estimators (const generic version).
///
/// This is the multivariate version of [`te_slices`], using const generics for
/// efficiency and working with [`Array2`] inputs.
///
/// # Returns
/// `(dest_future, dest_history, src_history)`:
/// - `dest_future`: shape $N \times D_{\mathrm{target}}$
/// - `dest_history`: shape $N \times (k \cdot D_{\mathrm{target}})$
/// - `src_history`: shape $N \times (l \cdot D_{\mathrm{source}})$
pub fn te_observations_const<
    T: Clone + Default,
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_JOINT: usize,
    const D_XP_YP: usize,
    const D_YP: usize,
    const D_YF_YP: usize,
>(
    source: &Array2<T>,
    destination: &Array2<T>,
    permute_src: bool,
) -> (Array2<T>, Array2<T>, Array2<T>) {
    let max_delay = SRC_HIST.max(DEST_HIST) * STEP_SIZE;
    let n = destination.nrows();

    if max_delay >= n {
        return (
            Array2::default((0, D_TARGET)),
            Array2::default((0, DEST_HIST * D_TARGET)),
            Array2::default((0, SRC_HIST * D_SOURCE)),
        );
    }

    let base_indices: Vec<usize> = (max_delay..n).step_by(STEP_SIZE).collect();
    let n_samples = base_indices.len();

    let mut dest_future = Array2::default((n_samples, D_TARGET));
    let mut dest_history = Array2::default((n_samples, DEST_HIST * D_TARGET));
    let mut src_history = Array2::default((n_samples, SRC_HIST * D_SOURCE));

    for (idx, &base_idx) in base_indices.iter().enumerate() {
        for d in 0..D_TARGET {
            dest_future[(idx, d)] = destination[(base_idx, d)].clone();
        }

        for j in 0..DEST_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_TARGET {
                dest_history[(idx, (DEST_HIST - 1 - j) * D_TARGET + d)] =
                    destination[(base_idx - offset, d)].clone();
            }
        }

        for j in 0..SRC_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_SOURCE {
                src_history[(idx, (SRC_HIST - 1 - j) * D_SOURCE + d)] =
                    source[(base_idx - offset, d)].clone();
            }
        }
    }

    if permute_src {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let mut permuted_src_history = Array2::default((n_samples, SRC_HIST * D_SOURCE));
        for (i, &new_idx) in indices.iter().enumerate() {
            for j in 0..SRC_HIST * D_SOURCE {
                permuted_src_history[(i, j)] = src_history[(new_idx, j)].clone();
            }
        }
        src_history = permuted_src_history;
    }

    (dest_future, dest_history, src_history)
}

/// Build the embedding arrays used by conditional transfer-entropy estimators (const generic version).
///
/// This is the multivariate version of [`cte_slices`], using const generics for
/// efficiency and working with [`Array2`] inputs.
///
/// # Returns
/// `(dest_future, dest_history, src_history, cond_history)`:
/// - `dest_future`: shape $N \times D_{\mathrm{target}}$
/// - `dest_history`: shape $N \times (k \cdot D_{\mathrm{target}})$
/// - `src_history`: shape $N \times (l \cdot D_{\mathrm{source}})$
/// - `cond_history`: shape $N \times (m \cdot D_{\mathrm{cond}})$
pub fn cte_observations_const<
    T: Clone + Default,
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const COND_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D_XP_YP_ZP: usize,
    const D_YP_ZP: usize,
    const D_YF_YP_ZP: usize,
>(
    source: &Array2<T>,
    destination: &Array2<T>,
    condition: &Array2<T>,
    permute_src: bool,
) -> (Array2<T>, Array2<T>, Array2<T>, Array2<T>) {
    let max_delay = SRC_HIST.max(DEST_HIST).max(COND_HIST) * STEP_SIZE;
    let n = destination.nrows();

    if max_delay >= n {
        return (
            Array2::default((0, D_TARGET)),
            Array2::default((0, DEST_HIST * D_TARGET)),
            Array2::default((0, SRC_HIST * D_SOURCE)),
            Array2::default((0, COND_HIST * D_COND)),
        );
    }

    let base_indices: Vec<usize> = (max_delay..n).step_by(STEP_SIZE).collect();
    let n_samples = base_indices.len();

    let mut dest_future = Array2::default((n_samples, D_TARGET));
    let mut dest_history = Array2::default((n_samples, DEST_HIST * D_TARGET));
    let mut src_history = Array2::default((n_samples, SRC_HIST * D_SOURCE));
    let mut cond_history = Array2::default((n_samples, COND_HIST * D_COND));

    for (idx, &base_idx) in base_indices.iter().enumerate() {
        for d in 0..D_TARGET {
            dest_future[(idx, d)] = destination[(base_idx, d)].clone();
        }

        for j in 0..DEST_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_TARGET {
                dest_history[(idx, (DEST_HIST - 1 - j) * D_TARGET + d)] =
                    destination[(base_idx - offset, d)].clone();
            }
        }

        for j in 0..SRC_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_SOURCE {
                src_history[(idx, (SRC_HIST - 1 - j) * D_SOURCE + d)] =
                    source[(base_idx - offset, d)].clone();
            }
        }

        for j in 0..COND_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_COND {
                cond_history[(idx, (COND_HIST - 1 - j) * D_COND + d)] =
                    condition[(base_idx - offset, d)].clone();
            }
        }
    }

    if permute_src {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let mut permuted_src_history = Array2::default((n_samples, SRC_HIST * D_SOURCE));
        for (i, &new_idx) in indices.iter().enumerate() {
            for j in 0..SRC_HIST * D_SOURCE {
                permuted_src_history[(i, j)] = src_history[(new_idx, j)].clone();
            }
        }
        src_history = permuted_src_history;
    }

    (dest_future, dest_history, src_history, cond_history)
}

/// Build the embedding arrays used by transfer-entropy estimators.
///
/// Given a source process $X$ and destination process $Y$, this returns the
/// triple $(Y_t,\\ \mathbf{y}_{t-1}^{(k,\tau)},\\ \mathbf{x}_{t-1}^{(l,\tau)})$
/// for every valid time index $t$, where $\tau$ is the embedding delay
/// (`step_size`), $k$ = `dest_hist_len`, and $l$ = `src_hist_len`.
///
/// Only time indices for which all required past samples exist are kept.
/// The retained base indices are `t = max_delay, max_delay + τ, max_delay + 2τ, …`
/// where `max_delay = max(k, l) · τ`, yielding
/// $N = \lceil (n - \max(k, l)\,\tau) / \tau \rceil$ rows (with $n$ the input length).
///
/// Returns `(dest_future, dest_history, src_history)`:
/// - `dest_future`: $Y_t$, shape $N \times 1$
/// - `dest_history`: the embedding vector of $Y$ ending one step before $t$,
///   with $k$ samples spaced by $\tau$:
///   $$\mathbf{y}_{t-1}^{(k,\tau)} = \bigl(Y_{t-k\tau},\ \ldots,\ Y_{t-2\tau},\ Y_{t-\tau}\bigr)$$
///   shape $N \times k$. Column ordering is **oldest first**: column 0 holds
///   $Y_{t-k\tau}$ and column $k-1$ holds the most recent past sample $Y_{t-\tau}$.
/// - `src_history`: analogously
///   $\mathbf{x}_{t-1}^{(l,\tau)} = (X_{t-l\tau}, \ldots, X_{t-2\tau}, X_{t-\tau})$,
///   shape $N \times l$, same column ordering.
///
/// Note that the embedding excludes the sample at time $t$ itself; $Y_t$ is
/// returned separately as `dest_future`. This matches the standard TE
/// convention where the source and destination histories are strictly past
/// observations used to predict the next destination value.
///
/// For `step_size = 1` and writing $n+1$ for the future index, this is the
/// usual embedding $\mathbf{y}_n^{(k)} = \{y_n, y_{n-1}, \ldots, y_{n-k+1}\}$
/// from the [Transfer Entropy Guide](crate::guide::transfer_entropy)
/// (modulo column order).
///
/// # Example
/// With `dest_hist_len = 3`, `step_size = 2`, one row of `dest_history`
/// (with `t` the corresponding base index) contains
/// $(Y_{t-6},\ Y_{t-4},\ Y_{t-2})$ in columns `[0, 1, 2]`, and the matching
/// `dest_future` entry is $Y_t$.
/// Zero-copy strided views equivalent to the columns of [`te_slices`]
/// output for `i32` code series, used by the discrete/ordinal TE/CTE
/// constructors so no intermediate arrays are materialised.
///
/// Columns are ordered oldest-first, matching the Array2 column semantics of
/// the slicing functions. Each history column is a stride-`step_size` slice
/// of the input; the future is a contiguous strided slice.
pub(crate) struct TeEmbeddingViews<'a> {
    pub dest_future: ndarray::ArrayView1<'a, i32>,
    pub dest_past_cols: Vec<ndarray::ArrayView1<'a, i32>>,
    pub src_past_cols: Vec<ndarray::ArrayView1<'a, i32>>,
}

/// Number of observation rows for the given series length and embedding
/// parameters (mirrors `te_observations`).
fn te_row_count(n: usize, max_delay: usize, step_size: usize) -> usize {
    if max_delay >= n {
        0
    } else {
        (n - max_delay).div_ceil(step_size)
    }
}

/// History column `col` (0 = oldest) as a strided view.
fn hist_col<'a>(
    arr: &'a Array1<i32>,
    col: usize,
    hist_len: usize,
    max_delay: usize,
    step_size: usize,
    n_samples: usize,
) -> ArrayView1<'a, i32> {
    let steps_back = (hist_len - col) * step_size;
    let first = max_delay - steps_back;
    strided_take(arr, first, step_size, n_samples)
}

/// View of `n_samples` elements starting at `first`, stride `step`.
fn strided_take<'a>(
    arr: &'a Array1<i32>,
    first: usize,
    step: usize,
    n_samples: usize,
) -> ArrayView1<'a, i32> {
    if n_samples == 0 {
        return arr.slice(ndarray::s![0..0]);
    }
    let last = first + (n_samples - 1) * step;
    debug_assert!(last < arr.len(), "embedding column exceeds series length");
    arr.slice(ndarray::s![first..=last.min(arr.len() - 1);step])
}

pub(crate) fn te_embedding_views<'a>(
    source: &'a Array1<i32>,
    destination: &'a Array1<i32>,
    src_hist_len: usize,
    dest_hist_len: usize,
    step_size: usize,
) -> TeEmbeddingViews<'a> {
    let max_delay = src_hist_len.max(dest_hist_len) * step_size;
    let n_samples = te_row_count(destination.len(), max_delay, step_size);

    TeEmbeddingViews {
        dest_future: {
            let first = max_delay.min(destination.len());
            strided_take(destination, first, step_size, n_samples)
        },
        dest_past_cols: (0..dest_hist_len)
            .map(|c| {
                hist_col(
                    destination,
                    c,
                    dest_hist_len,
                    max_delay,
                    step_size,
                    n_samples,
                )
            })
            .collect(),
        src_past_cols: (0..src_hist_len)
            .map(|c| hist_col(source, c, src_hist_len, max_delay, step_size, n_samples))
            .collect(),
    }
}

/// Zero-copy variant of [`cte_slices`] columns; see [`TeEmbeddingViews`].
pub(crate) struct CteEmbeddingViews<'a> {
    pub dest_future: ndarray::ArrayView1<'a, i32>,
    pub dest_past_cols: Vec<ndarray::ArrayView1<'a, i32>>,
    pub src_past_cols: Vec<ndarray::ArrayView1<'a, i32>>,
    pub cond_past_cols: Vec<ndarray::ArrayView1<'a, i32>>,
}

pub(crate) fn cte_embedding_views<'a>(
    source: &'a Array1<i32>,
    destination: &'a Array1<i32>,
    condition: &'a Array1<i32>,
    src_hist_len: usize,
    dest_hist_len: usize,
    cond_hist_len: usize,
    step_size: usize,
) -> CteEmbeddingViews<'a> {
    let max_delay = src_hist_len.max(dest_hist_len).max(cond_hist_len) * step_size;
    let n_samples = te_row_count(destination.len(), max_delay, step_size);

    CteEmbeddingViews {
        dest_future: {
            let first = max_delay.min(destination.len());
            strided_take(destination, first, step_size, n_samples)
        },
        dest_past_cols: (0..dest_hist_len)
            .map(|c| {
                hist_col(
                    destination,
                    c,
                    dest_hist_len,
                    max_delay,
                    step_size,
                    n_samples,
                )
            })
            .collect(),
        src_past_cols: (0..src_hist_len)
            .map(|c| hist_col(source, c, src_hist_len, max_delay, step_size, n_samples))
            .collect(),
        cond_past_cols: (0..cond_hist_len)
            .map(|c| hist_col(condition, c, cond_hist_len, max_delay, step_size, n_samples))
            .collect(),
    }
}

pub fn te_slices<T: Clone + Default>(
    source: &Array1<T>,
    destination: &Array1<T>,
    src_hist_len: usize,
    dest_hist_len: usize,
    step_size: usize,
) -> (Array2<T>, Array2<T>, Array2<T>) {
    te_observations(
        source,
        destination,
        src_hist_len,
        dest_hist_len,
        step_size,
        false,
    )
}

/// Build the embedding arrays used by transfer-entropy estimators and optionally permute the source.
///
/// Given a source process $X$ and destination process $Y$, this returns the
/// triple $(Y_t,\ \mathbf{y}_{t-1}^{(k,\tau)},\ \mathbf{x}_{t-1}^{(l,\tau)})$
/// for every valid time index $t$, where $\tau$ is the embedding delay
/// (`step_size`), $k$ = `dest_hist_len`, and $l$ = `src_hist_len`.
///
/// Only time indices for which all required past samples exist are kept.
/// The retained base indices are `t = max_delay, max_delay + τ, max_delay + 2τ, …`
/// where `max_delay = max(k, l) · τ`, yielding
/// $N = \lceil (n - \max(k, l)\,\tau) / \tau \rceil$ rows (with $n$ the input length).
///
/// Returns `(dest_future, dest_history, src_history)`:
/// - `dest_future`: $Y_t$, shape $N \times 1$
/// - `dest_history`: the embedding vector of $Y$ ending one step before $t$,
///   with $k$ samples spaced by $\tau$:
///   $$\mathbf{y}_{t-1}^{(k,\tau)} = \bigl(Y_{t-k\tau},\ \ldots,\ Y_{t-2\tau},\ Y_{t-\tau}\bigr)$$
///   shape $N \times k$. Column ordering is **oldest first**: column 0 holds
///   $Y_{t-k\tau}$ and column $k-1$ holds the most recent past sample $Y_{t-\tau}$.
/// - `src_history`: analogously
///   $\mathbf{x}_{t-1}^{(l,\tau)} = (X_{t-l\tau}, \ldots, X_{t-2\tau}, X_{t-\tau})$,
///   shape $N \times l$, same column ordering.
///   If `permute_src` is true, the rows of `src_history` are randomly shuffled.
///
/// Note that the embedding excludes the sample at time $t$ itself; $Y_t$ is
/// returned separately as `dest_future`. This matches the standard TE
/// convention where the source and destination histories are strictly past
/// observations used to predict the next destination value.
///
/// For `step_size = 1` and writing $n+1$ for the future index, this is the
/// usual embedding $\mathbf{y}_n^{(k)} = \{y_n, y_{n-1}, \ldots, y_{n-k+1}\}$
/// from the [Transfer Entropy Guide](crate::guide::transfer_entropy)
/// (modulo column order).
///
/// # Example
/// With `dest_hist_len = 3`, `step_size = 2`, one row of `dest_history`
/// (with `t` the corresponding base index) contains
/// $(Y_{t-6},\ Y_{t-4},\ Y_{t-2})$ in columns `[0, 1, 2]`, and the matching
/// `dest_future` entry is $Y_t$.
pub fn te_observations<T: Clone + Default>(
    source: &Array1<T>,
    destination: &Array1<T>,
    src_hist_len: usize,
    dest_hist_len: usize,
    step_size: usize,
    permute_src: bool,
) -> (Array2<T>, Array2<T>, Array2<T>) {
    let max_delay = src_hist_len.max(dest_hist_len) * step_size;
    let n = destination.len();

    if max_delay >= n {
        return (
            Array2::default((0, 1)),
            Array2::default((0, dest_hist_len)),
            Array2::default((0, src_hist_len)),
        );
    }

    let base_indices: Vec<usize> = (max_delay..n).step_by(step_size).collect();
    let n_samples = base_indices.len();

    let mut dest_future = Array2::default((n_samples, 1));
    let mut dest_history = Array2::default((n_samples, dest_hist_len));
    let mut src_history = Array2::default((n_samples, src_hist_len));

    for (idx, &base_idx) in base_indices.iter().enumerate() {
        dest_future[(idx, 0)] = destination[base_idx].clone();

        for j in 0..dest_hist_len {
            let offset = (j + 1) * step_size;
            dest_history[(idx, dest_hist_len - 1 - j)] = destination[base_idx - offset].clone();
        }

        for j in 0..src_hist_len {
            let offset = (j + 1) * step_size;
            src_history[(idx, src_hist_len - 1 - j)] = source[base_idx - offset].clone();
        }
    }

    if permute_src {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let mut permuted_src_history = Array2::default((n_samples, src_hist_len));
        for (i, &new_idx) in indices.iter().enumerate() {
            for j in 0..src_hist_len {
                permuted_src_history[(i, j)] = src_history[(new_idx, j)].clone();
            }
        }
        src_history = permuted_src_history;
    }

    (dest_future, dest_history, src_history)
}

/// Build the embedding arrays used by conditional transfer-entropy estimators.
///
/// Given a source process $X$, destination process $Y$, and conditioning
/// process $Z$, this returns the tuple
/// $(Y_t,\ \mathbf{y}_{t-1}^{(k,\tau)},\ \mathbf{x}_{t-1}^{(l,\tau)},\ \mathbf{z}_{t-1}^{(m,\tau)})$
/// for every valid time index $t$, where $\tau$ is the embedding delay
/// (`step_size`), $k$ = `dest_hist_len`, $l$ = `src_hist_len`, and
/// $m$ = `cond_hist_len`.
///
/// Only time indices for which all required past samples exist are kept.
/// The retained base indices are `t = max_delay, max_delay + τ, max_delay + 2τ, …`
/// where `max_delay = max(k, l, m) · τ`, yielding
/// $N = \lceil (n - \max(k, l, m)\,\tau) / \tau \rceil$ rows (with $n$ the input length).
///
/// Returns `(dest_future, dest_history, src_history, cond_history)`:
/// - `dest_future`: $Y_t$, shape $N \times 1$.
/// - `dest_history`: embedding of $Y$ ending one step before $t$:
///   $$\mathbf{y}_{t-1}^{(k,\tau)} = \bigl(Y_{t-k\tau},\ \ldots,\ Y_{t-2\tau},\ Y_{t-\tau}\bigr)$$
///   shape $N \times k$. Column ordering is **oldest first** (column 0 is
///   $Y_{t-k\tau}$, column $k-1$ is the most recent past sample $Y_{t-\tau}$).
/// - `src_history`: analogously
///   $\mathbf{x}_{t-1}^{(l,\tau)} = (X_{t-l\tau}, \ldots, X_{t-2\tau}, X_{t-\tau})$,
///   shape $N \times l$.
/// - `cond_history`: analogously
///   $\mathbf{z}_{t-1}^{(m,\tau)} = (Z_{t-m\tau}, \ldots, Z_{t-2\tau}, Z_{t-\tau})$,
///   shape $N \times m$.
///
/// All three input series are assumed to have the same length; out-of-bounds
/// indices would panic. The number of valid samples $N$ is governed by the
/// longest of the three embeddings.
///
/// For `step_size = 1` this matches the standard contiguous embeddings used
/// in the [Conditional Transfer Entropy Guide](crate::guide::cond_te),
/// modulo column order.
///
/// # Example
/// With `dest_hist_len = 3`, `cond_hist_len = 2`, `step_size = 2`, one row of
/// `dest_history` contains $(Y_{t-6},\ Y_{t-4},\ Y_{t-2})$ and the matching
/// row of `cond_history` contains $(Z_{t-4},\ Z_{t-2})$, while `dest_future`
/// holds $Y_t$.
pub fn cte_slices<T: Clone + Default>(
    source: &Array1<T>,
    destination: &Array1<T>,
    condition: &Array1<T>,
    src_hist_len: usize,
    dest_hist_len: usize,
    cond_hist_len: usize,
    step_size: usize,
) -> (Array2<T>, Array2<T>, Array2<T>, Array2<T>) {
    cte_observations(
        source,
        destination,
        condition,
        src_hist_len,
        dest_hist_len,
        cond_hist_len,
        step_size,
        false,
    )
}

/// Build the embedding arrays used by conditional transfer-entropy estimators and optionally permute the source.
///
/// Given a source process $X$, destination process $Y$, and conditioning
/// process $Z$, this returns the tuple
/// $(Y_t,\ \mathbf{y}_{t-1}^{(k,\tau)},\ \mathbf{x}_{t-1}^{(l,\tau)},\ \mathbf{z}_{t-1}^{(m,\tau)})$
/// for every valid time index $t$, where $\tau$ is the embedding delay
/// (`step_size`), $k$ = `dest_hist_len`, $l$ = `src_hist_len`, and
/// $m$ = `cond_hist_len`.
///
/// Only time indices for which all required past samples exist are kept.
/// The retained base indices are `t = max_delay, max_delay + τ, max_delay + 2τ, …`
/// where `max_delay = max(k, l, m) · τ`, yielding
/// $N = \lceil (n - \max(k, l, m)\,\tau) / \tau \rceil$ rows (with $n$ the input length).
///
/// Returns `(dest_future, dest_history, src_history, cond_history)`:
/// - `dest_future`: $Y_t$, shape $N \times 1$.
/// - `dest_history`: embedding of $Y$ ending one step before $t$:
///   $$\mathbf{y}_{t-1}^{(k,\tau)} = \bigl(Y_{t-k\tau},\ \ldots,\ Y_{t-2\tau},\ Y_{t-\tau}\bigr)$$
///   shape $N \times k$. Column ordering is **oldest first** (column 0 is
///   $Y_{t-k\tau}$, column $k-1$ is the most recent past sample $Y_{t-\tau}$).
/// - `src_history`: analogously
///   $\mathbf{x}_{t-1}^{(l,\tau)} = (X_{t-l\tau}, \ldots, X_{t-2\tau}, X_{t-\tau})$,
///   shape $N \times l$.
///   If `permute_src` is true, the rows of `src_history` are randomly shuffled.
/// - `cond_history`: analogously
///   $\mathbf{z}_{t-1}^{(m,\tau)} = (Z_{t-m\tau}, \ldots, Z_{t-2\tau}, Z_{t-\tau})$,
///   shape $N \times m$.
///
/// All three input series are assumed to have the same length; out-of-bounds
/// indices would panic. The number of valid samples $N$ is governed by the
/// longest of the three embeddings.
///
/// For `step_size = 1` this matches the standard contiguous embeddings used
/// in the [Conditional Transfer Entropy Guide](crate::guide::cond_te),
/// modulo column order.
///
/// # Example
/// With `dest_hist_len = 3`, `cond_hist_len = 2`, `step_size = 2`, one row of
/// `dest_history` contains $(Y_{t-6},\ Y_{t-4},\ Y_{t-2})$ and the matching
/// row of `cond_history` contains $(Z_{t-4},\ Z_{t-2})$, while `dest_future`
/// holds $Y_t$.
#[allow(clippy::too_many_arguments)]
pub fn cte_observations<T: Clone + Default>(
    source: &Array1<T>,
    destination: &Array1<T>,
    condition: &Array1<T>,
    src_hist_len: usize,
    dest_hist_len: usize,
    cond_hist_len: usize,
    step_size: usize,
    permute_src: bool,
) -> (Array2<T>, Array2<T>, Array2<T>, Array2<T>) {
    let max_delay = src_hist_len.max(dest_hist_len).max(cond_hist_len) * step_size;
    let n = destination.len();

    if max_delay >= n {
        return (
            Array2::default((0, 1)),
            Array2::default((0, dest_hist_len)),
            Array2::default((0, src_hist_len)),
            Array2::default((0, cond_hist_len)),
        );
    }

    let base_indices: Vec<usize> = (max_delay..n).step_by(step_size).collect();
    let n_samples = base_indices.len();

    let mut dest_future = Array2::default((n_samples, 1));
    let mut dest_history = Array2::default((n_samples, dest_hist_len));
    let mut src_history = Array2::default((n_samples, src_hist_len));
    let mut cond_history = Array2::default((n_samples, cond_hist_len));

    for (idx, &base_idx) in base_indices.iter().enumerate() {
        dest_future[(idx, 0)] = destination[base_idx].clone();

        for j in 0..dest_hist_len {
            let offset = (j + 1) * step_size;
            dest_history[(idx, dest_hist_len - 1 - j)] = destination[base_idx - offset].clone();
        }

        for j in 0..src_hist_len {
            let offset = (j + 1) * step_size;
            src_history[(idx, src_hist_len - 1 - j)] = source[base_idx - offset].clone();
        }

        for j in 0..cond_hist_len {
            let offset = (j + 1) * step_size;
            cond_history[(idx, cond_hist_len - 1 - j)] = condition[base_idx - offset].clone();
        }
    }

    if permute_src {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let mut permuted_src_history = Array2::default((n_samples, src_hist_len));
        for (i, &new_idx) in indices.iter().enumerate() {
            for j in 0..src_hist_len {
                permuted_src_history[(i, j)] = src_history[(new_idx, j)].clone();
            }
        }
        src_history = permuted_src_history;
    }

    (dest_future, dest_history, src_history, cond_history)
}

#[cfg(test)]
mod embedding_view_tests {
    use super::*;
    use crate::estimators::approaches::discrete::DiscreteConditionalMutualInformation;
    use crate::estimators::approaches::discrete::discrete_utils::{
        reduce_array2_compact, reduce_hist_columns_compact,
    };
    use crate::estimators::traits::GlobalValue;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;

    fn codes(n: usize, states: i32, seed: u64) -> Array1<i32> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array1::from((0..n).map(|_| rng.gen_range(0..states)).collect::<Vec<_>>())
    }

    /// Column views must reproduce the materialised Array2 columns exactly.
    #[rstest]
    #[case(1, 1, 1)]
    #[case(2, 1, 1)]
    #[case(1, 2, 2)]
    #[case(3, 2, 2)]
    #[case(2, 3, 5)]
    fn views_match_materialised_columns(#[case] l: usize, #[case] k: usize, #[case] tau: usize) {
        let source = codes(29, 5, 7);
        let destination = codes(29, 5, 11);
        let condition = codes(29, 5, 13);

        let (df, dh, sh) = te_slices(&source, &destination, l, k, tau);
        let v = te_embedding_views(&source, &destination, l, k, tau);

        assert_eq!(v.dest_future, df.column(0));
        for (c, col) in dh.columns().into_iter().enumerate() {
            assert_eq!(v.dest_past_cols[c], col, "dest history col {c}");
        }
        for (c, col) in sh.columns().into_iter().enumerate() {
            assert_eq!(v.src_past_cols[c], col, "src history col {c}");
        }

        let (df, _dh, _sh, ch) = cte_slices(&source, &destination, &condition, l, k, k, tau);
        let cv = cte_embedding_views(&source, &destination, &condition, l, k, k, tau);
        assert_eq!(cv.dest_future, df.column(0));
        for (c, col) in ch.columns().into_iter().enumerate() {
            assert_eq!(cv.cond_past_cols[c], col, "cond history col {c}");
        }
        let _ = dh;
    }

    fn legacy_te_global(s: &Array1<i32>, d: &Array1<i32>, l: usize, k: usize, tau: usize) -> f64 {
        let (df, dh, sh) = te_slices(s, d, l, k, tau);
        let src = reduce_array2_compact(&sh);
        let dst = reduce_array2_compact(&dh);
        let fut = df.column(0).to_owned();
        DiscreteConditionalMutualInformation::new(
            &[src, fut],
            &dst,
            crate::estimators::approaches::discrete::mle::DiscreteEntropy::new,
        )
        .global_value()
    }

    fn new_te_global(s: &Array1<i32>, d: &Array1<i32>, l: usize, k: usize, tau: usize) -> f64 {
        let v = te_embedding_views(s, d, l, k, tau);
        let src = reduce_hist_columns_compact(v.src_past_cols.iter().copied());
        let dst = reduce_hist_columns_compact(v.dest_past_cols.iter().copied());
        DiscreteConditionalMutualInformation::new(
            &[src, v.dest_future.to_owned()],
            &dst,
            crate::estimators::approaches::discrete::mle::DiscreteEntropy::new,
        )
        .global_value()
    }

    /// Estimator built on strided views must equal the materialised-array
    /// reference bit-for-bit (identical integer codes feed identical counting).
    #[rstest]
    #[case(1, 1, 1)]
    #[case(2, 1, 1)]
    #[case(1, 2, 2)]
    #[case(2, 3, 2)]
    #[case(3, 3, 3)]
    #[case(2, 2, 5)]
    fn te_matches_legacy_pipeline(#[case] l: usize, #[case] k: usize, #[case] tau: usize) {
        let source = codes(23, 4, 42);
        let dest = codes(23, 4, 99);
        let old = legacy_te_global(&source, &dest, l, k, tau);
        let new = new_te_global(&source, &dest, l, k, tau);
        assert_eq!(old.to_bits(), new.to_bits(), "l={l} k={k} tau={tau}");
    }

    /// Degenerate case where the window exceeds the series: both paths see
    /// empty embeddings.
    #[test]
    fn te_empty_window_matches_legacy() {
        let source = codes(4, 3, 1);
        let dest = codes(4, 3, 2);
        let old = legacy_te_global(&source, &dest, 3, 3, 3);
        let new = new_te_global(&source, &dest, 3, 3, 3);
        // Whatever the CMI machinery returns for empty embeddings (a finite
        // value, as it happens), both paths must return the identical bits.
        assert_eq!(old.to_bits(), new.to_bits(), "empty-window divergence");
    }
}

#[cfg(test)]
mod mle_fusion_tests {
    use crate::estimators::approaches::discrete::DiscreteConditionalTransferEntropy;
    use crate::estimators::approaches::discrete::DiscreteTransferEntropy;
    use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
    use crate::estimators::traits::GlobalValue;
    use ndarray::Array1;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;

    fn codes(n: usize, states: i32, seed: u64) -> Array1<i32> {
        let mut rng = StdRng::seed_from_u64(seed);
        Array1::from((0..n).map(|_| rng.gen_range(0..states)).collect::<Vec<_>>())
    }

    #[rstest]
    #[case(1, 1, 1)]
    #[case(2, 1, 2)]
    #[case(1, 3, 2)]
    #[case(3, 3, 3)]
    #[case(2, 2, 5)]
    fn fused_mle_te_matches_generic(#[case] l: usize, #[case] k: usize, #[case] tau: usize) {
        let source = codes(37, 6, 5);
        let dest = codes(37, 6, 17);

        let legacy = DiscreteTransferEntropy::new(&source, &dest, l, k, tau, DiscreteEntropy::new)
            .global_value();
        let fused = DiscreteTransferEntropy::new_mle(&source, &dest, l, k, tau).global_value();

        let scale = legacy.abs().max(1.0);
        assert!(
            (legacy - fused).abs() <= 1e-12 * scale,
            "l={l} k={k} tau={tau}: {legacy} vs {fused}"
        );
    }

    #[rstest]
    #[case(1, 1, 1)]
    #[case(2, 2, 2)]
    #[case(3, 1, 3)]
    fn fused_mle_cte_matches_generic(#[case] l: usize, #[case] k: usize, #[case] tau: usize) {
        let source = codes(31, 5, 21);
        let dest = codes(31, 5, 22);
        let cond = codes(31, 4, 23);

        let legacy = DiscreteConditionalTransferEntropy::new(
            &source,
            &dest,
            &cond,
            l,
            k,
            k,
            tau,
            DiscreteEntropy::new,
        )
        .global_value();
        let fused =
            DiscreteConditionalTransferEntropy::new_mle(&source, &dest, &cond, l, k, k, tau)
                .global_value();

        let scale = legacy.abs().max(1.0);
        assert!(
            (legacy - fused).abs() <= 1e-12 * scale,
            "l={l} k={k} tau={tau}: {legacy} vs {fused}"
        );
    }
}

// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared CPU data-parallelism helpers for the query-parallel estimator loops
//! (feature `parallel`).
//!
//! Each helper evaluates one independent output per index, so the result is
//! bit-identical to the sequential loop regardless of the thread count or
//! chunking. Callers keep their own inline sequential loop for the feature-off
//! and single-thread cases (see `PARALLEL_MIN_QUERIES`), so those paths stay
//! byte-for-byte identical to a non-`parallel` build.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Minimum number of items before a CPU loop is split across rayon workers.
/// Below it the call is µs-scale and pool overhead would dominate. The chunked
/// path is only taken when more than one thread is available, so a single-thread
/// run (e.g. the benchmark tracks, which pin `RAYON_NUM_THREADS=1`) executes the
/// exact same sequential loop as a build without the feature.
#[cfg_attr(not(feature = "parallel"), allow(dead_code))]
pub(crate) const PARALLEL_MIN_QUERIES: usize = 512;

/// Evaluates `query` for `0..n`, one chunk per rayon worker, reusing one scratch
/// buffer per chunk.
///
/// Every query writes exactly one independent output, so the result is
/// bit-identical to the sequential loop. Chunking is explicit (one chunk per
/// worker) rather than `map_init` so an allocating `init` runs ~`num_threads`
/// times instead of once per rayon split.
#[cfg(feature = "parallel")]
pub(crate) fn chunked_map<S, Init, F>(n: usize, init: Init, query: F) -> Vec<f64>
where
    S: Send,
    Init: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize) -> f64 + Sync + Send,
{
    let threads = rayon::current_num_threads().max(1);
    let chunk = n.div_ceil(threads);
    let mut out = vec![0.0f64; n];
    out.par_chunks_mut(chunk).enumerate().for_each(|(c, slot)| {
        let mut scratch = init();
        let base = c * chunk;
        for (k, value) in slot.iter_mut().enumerate() {
            *value = query(&mut scratch, base + k);
        }
    });
    out
}

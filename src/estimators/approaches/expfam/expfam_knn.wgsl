// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Dense brute-force k-th-nearest-neighbour distances for the exponential-family
// (expfam) kNN estimators. One invocation per query row scans every candidate
// in the data cloud and keeps an insertion-sorted top-`k_select` buffer, so no
// pairwise-distance matrix is ever materialised.
//
// Metric: 0 = squared Euclidean (output is the squared distance, the host takes
// the square root exactly like kiddo's `nearest_n` path), 1 = Chebyshev (the
// output is the distance). Both match the CPU radii bit-for-bit up to f32
// rounding. `self_query` skips the candidate at the query's own index.

const F32_MAX: f32 = 3.402823466e38;
// `best` is sized for the largest `k_select` (k + 1 with k <= 32).
const MAX_SELECT: u32 = 33u;

struct Config {
    n_data: u32,
    n_query: u32,
    dim: u32,
    k_select: u32,
    metric: u32,
    self_query: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> queries: array<f32>;
@group(0) @binding(2) var<uniform> config: Config;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= config.n_query) {
        return;
    }

    let dim = config.dim;
    let cap = config.k_select;
    let q_off = q * dim;

    var best: array<f32, MAX_SELECT>;
    for (var i = 0u; i < cap; i = i + 1u) {
        best[i] = F32_MAX;
    }

    for (var j = 0u; j < config.n_data; j = j + 1u) {
        if (config.self_query == 1u && j == q) {
            continue;
        }
        let d_off = j * dim;
        var d = 0.0;
        if (config.metric == 1u) {
            var m = 0.0;
            for (var t = 0u; t < dim; t = t + 1u) {
                m = max(m, abs(data[d_off + t] - queries[q_off + t]));
            }
            d = m;
        } else {
            var s = 0.0;
            for (var t = 0u; t < dim; t = t + 1u) {
                let diff = data[d_off + t] - queries[q_off + t];
                s = s + diff * diff;
            }
            d = s;
        }

        if (d < best[cap - 1u]) {
            var pos = cap - 1u;
            loop {
                if (pos == 0u || best[pos - 1u] <= d) {
                    break;
                }
                best[pos] = best[pos - 1u];
                pos = pos - 1u;
            }
            best[pos] = d;
        }
    }

    output[q] = best[cap - 1u];
}

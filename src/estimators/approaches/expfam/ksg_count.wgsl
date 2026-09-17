// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Per-query fixed-radius neighbour count for the KSG marginal and conditional
// spaces (tier T2). One invocation per query scans the whole candidate cloud
// (dense O(N^2)) and writes two raw counts including the query itself:
//
//   output[q]         count at `eps`
//   output[n + q]     count at `eps * (1 - margin)`
//
// The host compares them to detect boundary ambiguity. A candidate within
// `margin` of the radius may be classified differently in f32 than on the CPU,
// so those spaces are recomputed exactly on the CPU. Everything else uses the
// GPU count.
//
// metric:    0 = squared Euclidean (compares d2 against eps*eps),
//            1 = Chebyshev (compares max|diff| against eps),
//            mirroring `SortedSpace::count_within`.
// inclusive:  1 = d <= eps (KSG Type2), 0 = d < eps (KSG Type1).

struct Config {
    point_count: u32,
    dim: u32,
    inclusive: u32,
    chebyshev: u32,
    margin: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<storage, read> epsilons: array<f32>;
@group(0) @binding(2) var<uniform> config: Config;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn within_radius(value: f32, radius: f32, inclusive: bool) -> bool {
    return select(value < radius, value <= radius, inclusive);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    // The output holds two counts per query; only the first half computes.
    if (q >= config.point_count) {
        return;
    }

    let n = config.point_count;
    let dim = config.dim;
    let q_base = q * dim;
    let eps = epsilons[q];
    // Chebyshev compares against `eps`, Euclidean against the squared radius.
    let r_hi = select(eps * eps, eps, config.chebyshev == 1u);
    let scale = 1.0 - config.margin;
    let r_lo = select(r_hi * scale * scale, r_hi * scale, config.chebyshev == 1u);
    let inclusive = config.inclusive == 1u;

    var count_hi: f32 = 0.0;
    var count_lo: f32 = 0.0;
    for (var i: u32 = 0; i < n; i = i + 1) {
        let n_base = i * dim;
        var m = 0.0;
        if (config.chebyshev == 1u) {
            for (var t: u32 = 0; t < dim; t = t + 1u) {
                m = max(m, abs(points[q_base + t] - points[n_base + t]));
            }
        } else {
            var d2 = 0.0;
            for (var t: u32 = 0; t < dim; t = t + 1u) {
                let diff = points[q_base + t] - points[n_base + t];
                d2 = d2 + diff * diff;
            }
            m = d2;
        }
        if (within_radius(m, r_hi, inclusive)) {
            count_hi = count_hi + 1.0;
        }
        if (within_radius(m, r_lo, inclusive)) {
            count_lo = count_lo + 1.0;
        }
    }

    output[q] = count_hi;
    output[n + q] = count_lo;
}

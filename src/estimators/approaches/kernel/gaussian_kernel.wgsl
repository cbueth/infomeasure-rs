/*
 * SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */

// Gaussian kernel compute shader for entropy calculation.
//
// Works in WHITENED space: the host transforms the points y = L^-1(x - m) so
// the Mahalanobis distance becomes a plain Euclidean distance. The per-candidate
// inner loop is therefore an O(dim_count) dot product and the truncation radius
// is a data-independent constant (no precision matrix, no max_eigenvalue).
//
// Points are stored in a compact flat array of N * dim_count f32 values
// (row-major), indexed as points[i * dim_count + d], to cut upload and
// global-read traffic for low dimensions.

// Configuration parameters
struct GpuConfig {
    point_count: u32,
    dim_count: u32,
    normalization: f32,
    adaptive_radius: f32, // fixed squared truncation bound in whitened space
};

// Bind groups
@group(0) @binding(0) var<storage, read> points: array<f32>;
@group(0) @binding(1) var<uniform> config: GpuConfig;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

// Main compute shader entry point
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Check if this thread is within bounds
    if (idx >= config.point_count) {
        return;
    }

    let q_base = idx * config.dim_count;

    // Calculate density for this point
    var density: f32 = 0.0;
    var c_density: f32 = 0.0;

    // Loop through all other points
    for (var i: u32 = 0; i < config.point_count; i = i + 1) {
        let n_base = i * config.dim_count;

        // Squared Euclidean distance in whitened space (equals squared
        // Mahalanobis distance in the original space).
        var squared_dist: f32 = 0.0;
        for (var d: u32 = 0; d < config.dim_count; d = d + 1) {
            let diff = points[q_base + d] - points[n_base + d];
            squared_dist += diff * diff;
        }

        // Check if point is within the truncation ball
        if (squared_dist <= config.adaptive_radius) {
            // Gaussian kernel: exp(-squared_dist / 2)
            let term = exp(-0.5 * squared_dist);

            // Kahan summation for better precision
            let y = term - c_density;
            let t = density + y;
            c_density = (t - density) - y;
            density = t;
        }
    }

    // Normalize the density and return it directly
    // (the host applies -log when converting to entropy).
    output[idx] = density / config.normalization;
}

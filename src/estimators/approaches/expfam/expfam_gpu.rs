// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU host side for the exponential-family (expfam) kNN estimators.
//!
//! Tier T4 in the acceleration model: expfam radii come from *irregular* kiddo
//! traversal, which the GPU cannot accelerate directly. What it can do is the
//! dense $O(N^2)$ fallback — compute every pairwise distance and select the
//! k-th per row in a single dispatch — which beats the tree once `N` is large
//! (and, in higher dimensions, once the tree degenerates).
//!
//! A query row is one thread: it scans all candidates and keeps an
//! insertion-sorted top-`k` buffer, so no $N \times N$ matrix is materialised.
//! Only the k-th distance leaves the device; the entropy closing terms stay on
//! the CPU. Parity with the CPU path is exact up to f32 rounding of the
//! distance itself (see the module tests).

use crate::estimators::gpu::{BatchJob, GpuContext, ShaderKind};
use ndarray::ArrayView2;

/// Shared with the WGSL top-`k` buffer size.
pub(crate) const EXPFAM_GPU_MAX_K: usize = 32;

const EXPFAM_KNN_WGSL: &str = include_str!("expfam_knn.wgsl");

/// Metric selector for the shader. Squared Euclidean keeps the accumulation in
/// the same form kiddo uses on the CPU (the host takes the square root).
const METRIC_SQUARED_EUCLIDEAN: u32 = 0;
const METRIC_CHEBYSHEV: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExpfamKnnConfig {
    n_data: u32,
    n_query: u32,
    dim: u32,
    /// Number of smallest distances to retain; the last is returned. The
    /// shader skips the point itself for self-queries, so this is always `k`
    /// (the k-th neighbour excluding self).
    k_select: u32,
    metric: u32,
    self_query: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Packs a row-major `n x K` cloud into flat f32 bytes for the shader.
pub(crate) fn pack_cloud<const K: usize>(data: ArrayView2<'_, f64>) -> Vec<u8> {
    let mut flat: Vec<f32> = Vec::with_capacity(data.nrows() * K);
    if let Some(slice) = data.as_slice() {
        flat.extend(slice.iter().map(|&v| v as f32));
    } else {
        for row in data.rows() {
            for c in 0..K {
                flat.push(row[c] as f32);
            }
        }
    }
    bytemuck::cast_slice(&flat).to_vec()
}

/// Whether the dense tier may run for this query shape.
fn knn_eligible(n_data: usize, n_query: usize, k: usize, self_query: bool, dim: usize) -> bool {
    // Self-queries need one extra candidate because the point itself is skipped.
    let needed = if self_query { k + 1 } else { k };
    (1..=EXPFAM_GPU_MAX_K).contains(&k)
        && n_data >= needed
        && n_query > 0
        && dim >= crate::estimators::gpu::gpu_min_points_expfam_min_dim()
        && n_data.min(n_query) >= crate::estimators::gpu::gpu_min_points_expfam()
        && n_data <= u32::MAX as usize
        && n_query <= u32::MAX as usize
}

/// Dense k-th-neighbour radii on the GPU, or `None` when the tier does not
/// apply (below the gate, unsupported `k`/metric, no adapter, any GPU error).
///
/// Returns distances exactly like the CPU helpers: the Chebyshev distance for
/// the Chebyshev metric, and the Euclidean distance (square root applied on the
/// host) otherwise.
pub(crate) fn knn_radii_gpu<const K: usize>(
    data: ArrayView2<'_, f64>,
    k: usize,
    at: Option<ArrayView2<'_, f64>>,
    use_chebyshev: bool,
) -> Option<Vec<f64>> {
    if data.ncols() != K {
        return None;
    }
    let n_data = data.nrows();
    let queries = at.unwrap_or(data);
    if queries.ncols() != K {
        return None;
    }
    let n_query = queries.nrows();

    let self_query = at.is_none();
    if !knn_eligible(n_data, n_query, k, self_query, K) {
        return None;
    }

    let ctx = GpuContext::get()?;

    let config = ExpfamKnnConfig {
        n_data: n_data as u32,
        n_query: n_query as u32,
        dim: K as u32,
        k_select: k as u32,
        metric: if use_chebyshev {
            METRIC_CHEBYSHEV
        } else {
            METRIC_SQUARED_EUCLIDEAN
        },
        self_query: u32::from(self_query),
        _pad0: 0,
        _pad1: 0,
    };

    let job = BatchJob {
        kind: ShaderKind::Expfam,
        wgsl: EXPFAM_KNN_WGSL,
        points: pack_cloud::<K>(data),
        extra_storage: if self_query {
            None
        } else {
            Some(pack_cloud::<K>(queries))
        },
        config: bytemuck::bytes_of(&config).to_vec(),
        n_items: n_query as u32,
    };

    let mut results = ctx.run_compute_batch(&[job])?;
    if results.len() != 1 {
        return None;
    }
    let mut radii: Vec<f64> = results.pop()?.into_iter().map(f64::from).collect();
    if !use_chebyshev {
        for r in radii.iter_mut() {
            *r = r.sqrt();
        }
    }
    Some(radii)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn eligibility_requires_the_gate_and_a_valid_k() {
        let gate = crate::estimators::gpu::gpu_min_points_expfam();
        let dim = crate::estimators::gpu::gpu_min_points_expfam_min_dim();

        // A valid shape at or above the gates is eligible...
        assert!(knn_eligible(gate, gate, 3, false, dim));
        // ...while below the size gate, low `k`, oversized `k`, and low
        // dimensionality all decline the tier.
        assert!(!knn_eligible(gate - 1, gate - 1, 3, false, dim));
        assert!(!knn_eligible(gate, gate, 0, false, dim));
        assert!(!knn_eligible(gate, gate, EXPFAM_GPU_MAX_K + 1, false, dim));
        assert!(!knn_eligible(2, gate, 3, false, dim));
        if dim > 0 {
            assert!(!knn_eligible(gate, gate, 3, false, dim - 1));
        }
    }

    #[test]
    fn pack_cloud_is_row_major_f32() {
        let data: ndarray::Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let bytes = pack_cloud::<2>(data.view());
        let floats: &[f32] = bytemuck::cast_slice(&bytes);
        assert_eq!(floats, &[1.0, 2.0, 3.0, 4.0]);
    }
}

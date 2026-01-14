// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use rand_distr::Normal;
use std::num::NonZeroUsize;

/// Add Gaussian noise to a 2D array.
pub fn add_noise(mut data: Array2<f64>, noise_level: f64) -> Array2<f64> {
    if noise_level <= 0.0 {
        return data;
    }
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, noise_level).unwrap();
    for x in data.iter_mut() {
        *x += normal.sample(&mut rng);
    }
    data
}

/// Compute the volume of the unit m-ball in R^m.
/// c_m = pi^{m/2} / Gamma(m/2 + 1)
/// We use a simple implementation via statrs gamma function.
pub fn unit_ball_volume(m: usize) -> f64 {
    use statrs::function::gamma::gamma;
    let m_f = m as f64;
    let numerator = std::f64::consts::PI.powf(m_f / 2.0);
    let denom = gamma(m_f / 2.0 + 1.0);
    numerator / denom
}

/// Convert an ArrayView2<f64> with exactly K columns into Vec<[f64; K]> points.
fn to_points<const K: usize>(data: ArrayView2<'_, f64>) -> Vec<[f64; K]> {
    assert!(data.ncols() == K, "data.ncols() must equal K");
    let n = data.nrows();
    let mut points: Vec<[f64; K]> = Vec::with_capacity(n);
    if let Some(slice) = data.as_slice() {
        for chunk in slice.chunks_exact(K) {
            let mut p = [0.0; K];
            p.copy_from_slice(&chunk[..K]);
            points.push(p);
        }
    } else {
        for r in 0..n {
            let mut p = [0.0; K];
            for c in 0..K {
                p[c] = data[(r, c)];
            }
            points.push(p);
        }
    }
    points
}

/// Compute kNN radii (Euclidean distances to the k-th nearest neighbor).
///
/// If `at` is None, it computes distances within the same dataset `data` (excluding self).
/// If `at` is Some(target), it computes distances from points in `target` to their k-th neighbors in `data`.
///
/// - data is shape (N, K)
/// - target is shape (M, K)
/// - k >= 1
pub fn knn_radii_at<const K: usize>(
    data: ArrayView2<'_, f64>,
    k: usize,
    at: Option<ArrayView2<'_, f64>>,
) -> Vec<f64> {
    assert!(k >= 1, "k must be >= 1");
    assert!(data.ncols() == K, "data.ncols() must equal K");
    let n = data.nrows();
    if n == 0 {
        return Vec::new();
    }

    let points = to_points::<K>(data);
    let tree: ImmutableKdTree<f64, K> = ImmutableKdTree::new_from_slice(&points);

    if let Some(target_data) = at {
        assert!(target_data.ncols() == K, "target_data.ncols() must equal K");
        let m = target_data.nrows();
        let target_points = to_points::<K>(target_data);
        let mut radii = Vec::with_capacity(m);
        for p in target_points.iter() {
            // No need to exclude self if we are querying another dataset
            let mut neigh = tree.nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(k).unwrap());
            let kth = neigh.remove(k - 1);
            let (dist2, _idx): (f64, u64) = kth.into();
            radii.push(dist2.sqrt());
        }
        radii
    } else {
        assert!(
            k <= n - 1,
            "k must be <= N-1 when querying within the same dataset"
        );
        // Query k+1 neighbors (including self), take index k (0-based) and sqrt distance
        let mut radii = Vec::with_capacity(n);
        for p in points.iter() {
            let mut neigh =
                tree.nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(k + 1).unwrap());
            let kth = neigh.remove(k);
            let (dist2, _idx): (f64, u64) = kth.into();
            radii.push(dist2.sqrt());
        }
        radii
    }
}

/// Compute kNN radii (Euclidean distances to the k-th nearest neighbor), excluding self.
pub fn knn_radii<const K: usize>(data: ArrayView2<'_, f64>, k: usize) -> Vec<f64> {
    knn_radii_at::<K>(data, k, None)
}

/// Compute common components used by exponential-family kNN estimators.
/// Mirrors Python calculate_common_entropy_components.
pub fn calculate_common_entropy_components_at<const K: usize>(
    data: ArrayView2<'_, f64>,
    k: usize,
    at: Option<ArrayView2<'_, f64>>,
) -> (f64, Vec<f64>, usize, usize) {
    let v_m = unit_ball_volume(K);
    let rho_k = knn_radii_at::<K>(data, k, at);
    let n = rho_k.len(); // N if at is None, M if at is Some(target)
    (v_m, rho_k, n, K)
}

/// Compute common components used by exponential-family kNN estimators for self-evaluation.
pub fn calculate_common_entropy_components<const K: usize>(
    data: ArrayView2<'_, f64>,
    k: usize,
) -> (f64, Vec<f64>, usize, usize) {
    calculate_common_entropy_components_at::<K>(data, k, None)
}

/// Helper to ensure 2D view from 1D or 2D input (placeholder for future API generalization).
pub fn as_2d(
    data: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
) -> ArrayView2<'_, f64> {
    data.view()
}

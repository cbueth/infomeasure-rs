// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::estimators::approaches::common_nd::KdTreeExpfam;
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use kiddo::{Chebyshev, SquaredEuclidean};
use ndarray::{Array2, ArrayView2};
use rand::prelude::*;
use rand_distr::Normal;
use std::num::NonZeroUsize;

/// KSG Mutual Information Estimator Type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KsgType {
    /// Type I (Algorithm 1) uses strict inequality (dist < eps) for neighbor counting in marginal spaces.
    Type1,
    /// Type II (Algorithm 2) uses non-strict inequality (dist <= eps) and a modified formula.
    Type2,
}

/// Add Gaussian noise to a 2D array.
///
/// The noise exists only to break degenerate ties in neighbour distances, so
/// cryptographic-quality entropy is wasted cost: a per-call seeded [`SmallRng`]
/// avoids ThreadRng's periodic OsRng reseeds, which profiler scans showed
/// dominating KSG-family estimates at $N \geq 10^4$ (up to two thirds of wall
/// time on pure entropy workloads).
pub fn add_noise(mut data: Array2<f64>, noise_level: f64) -> Array2<f64> {
    if noise_level <= 0.0 {
        return data;
    }
    let mut rng = SmallRng::from_rng(thread_rng()).expect("system RNG unavailable");
    let normal = Normal::new(0.0, noise_level).unwrap();
    for x in data.iter_mut() {
        *x += normal.sample(&mut rng);
    }
    data
}

/// Compute the volume of the unit $m$-ball in $R^m$ with radius $r$.
/// This matches Python's `unit_ball_volume(d, r=r, p=p)`.
/// For KL entropy, Python uses $r=1/2$, which gives:
/// - For $p=∞$: $c_d = 1$ (since $(2r)^d = 1$ when $r=1/2$)
/// - For $p=2$: $c_d = π^{d/2} / (Γ(1+d/2) 2^d)$
pub fn unit_ball_volume(m: usize, p: f64) -> f64 {
    unit_ball_volume_with_radius(m, p, 1.0)
}

/// Compute the unit ball volume with a specific radius $r$.
/// This matches Python's `unit_ball_volume(d, r=r, p=p)`.
pub fn unit_ball_volume_with_radius(m: usize, p: f64, r: f64) -> f64 {
    if p.is_infinite() {
        return unit_ball_volume_chebyshev_with_radius(m, r);
    }
    if p == 2.0 {
        use statrs::function::gamma::gamma;
        let m_f = m as f64;
        let numerator = std::f64::consts::PI.powf(m_f / 2.0) * r.powf(m_f);
        let denom = gamma(m_f / 2.0 + 1.0);
        numerator / denom
    } else {
        use statrs::function::gamma::gamma;
        let m_f = m as f64;
        let numerator = (2.0 * r * gamma(1.0 + 1.0 / p)).powf(m_f);
        let denom = gamma(1.0 + m_f / p);
        numerator / denom
    }
}

/// Compute the volume of the unit $m$-ball in $R^m$ under $L_{∞}$ norm with radius $r$.
/// $V_m = (2r)^m$
pub fn unit_ball_volume_chebyshev(m: usize) -> f64 {
    unit_ball_volume_chebyshev_with_radius(m, 1.0)
}

/// Compute the volume of the unit $m$-ball in $R^m$ under $L_{∞}$ norm with radius $r$.
/// $V_m = (2r)^m$
/// For KL entropy ($r=1/2$), this gives $c_d = 1$
pub fn unit_ball_volume_chebyshev_with_radius(m: usize, r: f64) -> f64 {
    (2.0 * r).powf(m as f64)
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

/// Compute kNN radii using specified metric.
///
/// If `at` is None, it computes distances within the same dataset `data` (excluding self).
/// If `at` is Some(target), it computes distances from points in `target` to their k-th neighbors in `data`.
///
/// `allow_gpu` opts into the dense GPU tier; with `false` (e.g. an estimator
/// with `force_cpu` set), or without the `gpu` feature / a usable adapter /
/// above the size gate, this is the plain kiddo path.
#[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
pub(crate) fn knn_radii_at_with_metric<const K: usize>(
    data: ArrayView2<'_, f64>,
    k: usize,
    at: Option<ArrayView2<'_, f64>>,
    use_chebyshev: bool,
    allow_gpu: bool,
) -> Vec<f64> {
    assert!(k >= 1, "k must be >= 1");
    assert!(data.ncols() == K, "data.ncols() must equal K");
    let n = data.nrows();
    if n == 0 {
        return Vec::new();
    }
    if at.is_none() {
        assert!(
            k < n,
            "k must be <= N-1 when querying within the same dataset"
        );
    }

    // Dense O(N²) tier: above the expfam gate the pairwise scan beats kiddo's
    // irregular traversal. Any ineligibility or GPU error falls through.
    #[cfg(feature = "gpu")]
    if allow_gpu
        && let Some(radii) = super::expfam_gpu::knn_radii_gpu::<K>(data, k, at, use_chebyshev)
    {
        return radii;
    }

    let points = to_points::<K>(data);
    let tree: KdTreeExpfam<K> = KdTreeExpfam::<K>::new_from_slice(&points).unwrap();
    let at_points = match at {
        Some(target_data) => {
            assert!(target_data.ncols() == K, "target_data.ncols() must equal K");
            Some(to_points::<K>(target_data))
        }
        None => None,
    };
    radii_from_tree(&tree, &points, k, at_points.as_deref(), use_chebyshev)
}

/// Like [`knn_radii_at_with_metric`], but takes a [`NdDataset`] whose kd-tree is
/// already built and reuses it.
///
/// This is the hot path for the expfam estimators: materialising points and
/// rebuilding the tree on every call is pure overhead because each estimator
/// already owns its dataset and tree. `at = None` queries the dataset against
/// itself (excluding self); `at = Some(q)` queries `q`'s points against
/// `data`'s tree.
#[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
pub(crate) fn knn_radii_at_dataset<const K: usize>(
    data: &NdDataset<K>,
    k: usize,
    at: Option<&NdDataset<K>>,
    use_chebyshev: bool,
    allow_gpu: bool,
) -> Vec<f64> {
    assert!(k >= 1, "k must be >= 1");
    if data.n == 0 {
        return Vec::new();
    }
    if at.is_none() {
        assert!(
            k < data.n,
            "k must be <= N-1 when querying within the same dataset"
        );
    }

    // Dense O(N²) tier: above the expfam gate the pairwise scan beats kiddo's
    // irregular traversal. Any ineligibility or GPU error falls through.
    #[cfg(feature = "gpu")]
    if allow_gpu
        && let Some(radii) = super::expfam_gpu::knn_radii_gpu::<K>(
            data.view(),
            k,
            at.map(|q| q.view()),
            use_chebyshev,
        )
    {
        return radii;
    }

    radii_from_tree(
        &data.tree,
        &data.points,
        k,
        at.map(|q| q.points.as_slice()),
        use_chebyshev,
    )
}

/// Euclidean/Chebyshev k-th-neighbour radii answered by an existing tree.
///
/// `points` are the tree's own points (self-queries); `at_points`, when given,
/// is the query cloud for a cross query. Keeping this separate lets both the
/// view-based and the [`NdDataset`]-based entry points share one implementation
/// without rebuilding a tree.
fn radii_from_tree<const K: usize>(
    tree: &KdTreeExpfam<K>,
    points: &[[f64; K]],
    k: usize,
    at_points: Option<&[[f64; K]]>,
    use_chebyshev: bool,
) -> Vec<f64> {
    if let Some(target_points) = at_points {
        let m = target_points.len();
        let mut radii = Vec::with_capacity(m);
        if use_chebyshev {
            let mut scratch = tree.create_scratch::<Chebyshev<f64>>();
            for p in target_points.iter() {
                let neigh = tree
                    .query(p)
                    .nearest_n::<Chebyshev<f64>>(NonZeroUsize::new(k).unwrap())
                    .with_scratch(&mut scratch)
                    .execute();
                radii.push(neigh[k - 1].distance);
            }
        } else {
            let mut scratch = tree.create_scratch::<SquaredEuclidean<f64>>();
            for p in target_points.iter() {
                let neigh = tree
                    .query(p)
                    .nearest_n::<SquaredEuclidean<f64>>(NonZeroUsize::new(k).unwrap())
                    .with_scratch(&mut scratch)
                    .execute();
                radii.push(neigh[k - 1].distance.sqrt());
            }
        }
        radii
    } else {
        let mut radii = Vec::with_capacity(points.len());
        let max_qty = NonZeroUsize::new(k + 1).unwrap();
        if use_chebyshev {
            let mut scratch = tree.create_scratch::<Chebyshev<f64>>();
            for p in points.iter() {
                let neigh = tree
                    .query(p)
                    .nearest_n::<Chebyshev<f64>>(max_qty)
                    .with_scratch(&mut scratch)
                    .execute();
                radii.push(neigh[k].distance);
            }
        } else {
            let mut scratch = tree.create_scratch::<SquaredEuclidean<f64>>();
            for p in points.iter() {
                let neigh = tree
                    .query(p)
                    .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                    .with_scratch(&mut scratch)
                    .execute();
                radii.push(neigh[k].distance.sqrt());
            }
        }
        radii
    }
}

/// Compute kNN radii (Euclidean distances to the k-th nearest neighbor).
pub(crate) fn knn_radii_at<const K: usize>(
    data: ArrayView2<'_, f64>,
    k: usize,
    at: Option<ArrayView2<'_, f64>>,
    allow_gpu: bool,
) -> Vec<f64> {
    knn_radii_at_with_metric::<K>(data, k, at, false, allow_gpu)
}

/// Compute kNN radii using Chebyshev metric.
pub(crate) fn knn_radii_at_chebyshev<const K: usize>(
    data: ArrayView2<'_, f64>,
    k: usize,
    at: Option<ArrayView2<'_, f64>>,
    allow_gpu: bool,
) -> Vec<f64> {
    knn_radii_at_with_metric::<K>(data, k, at, true, allow_gpu)
}

/// Compute kNN radii (Euclidean distances to the k-th nearest neighbor), excluding self.
pub fn knn_radii<const K: usize>(data: ArrayView2<'_, f64>, k: usize) -> Vec<f64> {
    knn_radii_at::<K>(data, k, None, true)
}

/// Compute kNN radii (Chebyshev distances to the k-th nearest neighbor), excluding self.
pub fn knn_radii_chebyshev<const K: usize>(data: ArrayView2<'_, f64>, k: usize) -> Vec<f64> {
    knn_radii_at_chebyshev::<K>(data, k, None, true)
}

/// Compute common components used by exponential-family kNN estimators.
/// This is the standard version (r=1) used by Rényi and Tsallis entropy.
/// Mirrors Python calculate_common_entropy_components. Kept as the view-based
/// reference (tests/parity); the estimators use
/// [`calculate_common_entropy_components_at_dataset`].
#[allow(dead_code)]
pub(crate) fn calculate_common_entropy_components_at<const K: usize>(
    data: ArrayView2<'_, f64>,
    k: usize,
    at: Option<ArrayView2<'_, f64>>,
    allow_gpu: bool,
) -> (f64, Vec<f64>, usize, usize) {
    let v_m = unit_ball_volume_with_radius(K, 2.0, 1.0);
    let rho_k = knn_radii_at::<K>(data, k, at, allow_gpu);
    let n = rho_k.len(); // N if at is None, M if at is Some(target)
    (v_m, rho_k, n, K)
}

/// Compute common components used by exponential-family kNN estimators using Chebyshev metric.
/// This is the standard version (r=1) used by Rényi and Tsallis entropy.
#[allow(dead_code)]
pub(crate) fn calculate_common_entropy_components_at_chebyshev<const K: usize>(
    data: ArrayView2<'_, f64>,
    k: usize,
    at: Option<ArrayView2<'_, f64>>,
    allow_gpu: bool,
) -> (f64, Vec<f64>, usize, usize) {
    let v_m = unit_ball_volume_chebyshev_with_radius(K, 1.0);
    let rho_k = knn_radii_at_chebyshev::<K>(data, k, at, allow_gpu);
    let n = rho_k.len();
    (v_m, rho_k, n, K)
}

/// Dataset-reusing variant of [`calculate_common_entropy_components_at`] (r=1,
/// Euclidean). Used by the estimators, which already own their kd-tree.
pub(crate) fn calculate_common_entropy_components_at_dataset<const K: usize>(
    data: &NdDataset<K>,
    k: usize,
    at: Option<&NdDataset<K>>,
    allow_gpu: bool,
) -> (f64, Vec<f64>, usize, usize) {
    let v_m = unit_ball_volume_with_radius(K, 2.0, 1.0);
    let rho_k = knn_radii_at_dataset::<K>(data, k, at, false, allow_gpu);
    let n = rho_k.len(); // N if at is None, M if at is Some(target)
    (v_m, rho_k, n, K)
}

/// Dataset-reusing variant of [`calculate_common_entropy_components_at_kl`]
/// (r=1/2, Euclidean).
pub(crate) fn calculate_common_entropy_components_at_kl_dataset<const K: usize>(
    data: &NdDataset<K>,
    k: usize,
    at: Option<&NdDataset<K>>,
    allow_gpu: bool,
) -> (f64, Vec<f64>, usize, usize) {
    let v_m = unit_ball_volume_with_radius(K, 2.0, 0.5);
    let rho_k = knn_radii_at_dataset::<K>(data, k, at, false, allow_gpu);
    let n = rho_k.len();
    (v_m, rho_k, n, K)
}

/// Dataset-reusing variant of
/// [`calculate_common_entropy_components_at_chebyshev_kl`] (r=1/2, Chebyshev).
pub(crate) fn calculate_common_entropy_components_at_chebyshev_kl_dataset<const K: usize>(
    data: &NdDataset<K>,
    k: usize,
    at: Option<&NdDataset<K>>,
    allow_gpu: bool,
) -> (f64, Vec<f64>, usize, usize) {
    let v_m = unit_ball_volume_chebyshev_with_radius(K, 0.5);
    let rho_k = knn_radii_at_dataset::<K>(data, k, at, true, allow_gpu);
    let n = rho_k.len();
    (v_m, rho_k, n, K)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array2, array};
    use rstest::rstest;

    /// Compute common components used by exponential-family kNN estimators for self-evaluation.
    pub(crate) fn calculate_common_entropy_components<const K: usize>(
        data: ArrayView2<'_, f64>,
        k: usize,
    ) -> (f64, Vec<f64>, usize, usize) {
        calculate_common_entropy_components_at::<K>(data, k, None, true)
    }

    #[test]
    fn test_add_noise_basic() {
        let data: Array2<f64> = array![[1.0, 2.0], [3.0, 4.0]];
        let noisy = add_noise(data.clone(), 0.1);

        // Should be different (unless very unlikely same noise)
        assert!(noisy != data);

        // With zero noise should be identical
        let no_noise = add_noise(data.clone(), 0.0);
        assert_eq!(no_noise, data);
    }

    #[rstest]
    #[case(1, 2.0, 2.0)]
    #[case(2, 2.0, std::f64::consts::PI)]
    #[case(3, 2.0, 4.0 * std::f64::consts::PI / 3.0)]
    #[case(2, f64::INFINITY, 4.0)]
    #[case(2, 1.0, 2.0)]
    #[case(3, f64::INFINITY, 8.0)] // (2*1)^3 = 8
    #[case(3, 1.0, 4.0 / 3.0)] // (2^3 * gamma(2)^3) / gamma(4) = 8 * 1 / 6 = 1.333...
    fn unit_ball_volume_known_values(#[case] d: usize, #[case] p: f64, #[case] expected: f64) {
        assert_abs_diff_eq!(unit_ball_volume(d, p), expected, epsilon = 1e-12);
    }

    #[test]
    fn knn_radii_1d_simple_cases() {
        // [[1.0], [2.0]] with k=1 -> [1.0, 1.0]
        let d2: Array2<f64> = array![[1.0], [2.0]];
        let r = knn_radii::<1>(d2.view(), 1);
        assert_abs_diff_eq!(r[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], 1.0, epsilon = 1e-12);

        // [[1.0],[2.0],[3.0]] with k=1 -> [1.0, 1.0, 1.0]
        let d3: Array2<f64> = array![[1.0], [2.0], [3.0]];
        let r = knn_radii::<1>(d3.view(), 1);
        assert_eq!(r.len(), 3);
        for v in r {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-12);
        }

        // k=2 -> [2.0, 1.0, 2.0]
        let d3: Array2<f64> = array![[1.0], [2.0], [3.0]];
        let r = knn_radii::<1>(d3.view(), 2);
        assert_abs_diff_eq!(r[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[2], 2.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic]
    fn knn_radii_panics_when_k_too_large() {
        let d: Array2<f64> = array![[1.0], [2.0], [3.0]];
        // Here N=3, but k must be <= N-1 when querying within same dataset
        let _ = knn_radii::<1>(d.view(), 3);
    }

    #[test]
    fn calculate_common_entropy_components_matches_python_cases() {
        // Case: ([[1.0],[2.0]], k=1) -> V_m=2.0, rho_k=[1,1], N=2, m=1
        let d: Array2<f64> = array![[1.0], [2.0]];
        let (v_m, rho_k, n, m) = calculate_common_entropy_components::<1>(d.view(), 1);
        assert_abs_diff_eq!(v_m, 2.0, epsilon = 1e-12);
        assert_eq!(n, 2);
        assert_eq!(m, 1);
        assert_eq!(rho_k.len(), 2);
        assert_abs_diff_eq!(rho_k[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rho_k[1], 1.0, epsilon = 1e-12);

        // Case: ([[1.0],[2.0],[3.0]], k=1) -> V_m=2.0, rho_k=[1,1,1], N=3, m=1
        let d: Array2<f64> = array![[1.0], [2.0], [3.0]];
        let (v_m, rho_k, n, m) = calculate_common_entropy_components::<1>(d.view(), 1);
        assert_abs_diff_eq!(v_m, 2.0, epsilon = 1e-12);
        assert_eq!(n, 3);
        assert_eq!(m, 1);
        assert_eq!(rho_k.len(), 3);
        for v in rho_k {
            assert_abs_diff_eq!(v, 1.0, epsilon = 1e-12);
        }

        // Case: ([[1.0],[2.0],[3.0]], k=2) -> rho_k=[2,1,2]
        let d: Array2<f64> = array![[1.0], [2.0], [3.0]];
        let (_v_m, rho_k, _n, _m) = calculate_common_entropy_components::<1>(d.view(), 2);
        assert_abs_diff_eq!(rho_k[0], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rho_k[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(rho_k[2], 2.0, epsilon = 1e-12);

        // Case: ([[1.0,2.0],[2.0,3.0]], k=1) -> V_m=pi, rho_k=[sqrt(2), sqrt(2)], N=2, m=2
        let d: Array2<f64> = array![[1.0, 2.0], [2.0, 3.0]];
        let (v_m, rho_k, n, m) = calculate_common_entropy_components::<2>(d.view(), 1);
        assert_abs_diff_eq!(v_m, std::f64::consts::PI, epsilon = 1e-12);
        assert_eq!(n, 2);
        assert_eq!(m, 2);
        assert_eq!(rho_k.len(), 2);
        let s2 = 2.0_f64.sqrt();
        assert_abs_diff_eq!(rho_k[0], s2, epsilon = 1e-12);
        assert_abs_diff_eq!(rho_k[1], s2, epsilon = 1e-12);

        // Case: ([[1.0,2.0,3.0],[2.0,3.0,4.0]], k=1) -> V_m≈4.188790, rho_k=[sqrt(3), sqrt(3)], N=2, m=3
        let d: Array2<f64> = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let (v_m, rho_k, n, m) = calculate_common_entropy_components::<3>(d.view(), 1);
        assert_abs_diff_eq!(v_m, 4.0 * std::f64::consts::PI / 3.0, epsilon = 1e-6);
        assert_eq!(n, 2);
        assert_eq!(m, 3);
        assert_eq!(rho_k.len(), 2);
        let s3 = 3.0_f64.sqrt();
        assert_abs_diff_eq!(rho_k[0], s3, epsilon = 1e-12);
        assert_abs_diff_eq!(rho_k[1], s3, epsilon = 1e-12);
    }

    #[test]
    fn calculate_common_entropy_components_panics_when_k_too_large() {
        // ([[0.0]], 1) -> k too large because N-1 = 0
        let d: Array2<f64> = array![[0.0]];
        assert!(
            std::panic::catch_unwind(|| {
                let _ = calculate_common_entropy_components::<1>(d.view(), 1);
            })
            .is_err()
        );

        // ([[1.0],[2.0],[3.0]], 3) and 4
        let d: Array2<f64> = array![[1.0], [2.0], [3.0]];
        assert!(
            std::panic::catch_unwind(|| {
                let _ = calculate_common_entropy_components::<1>(d.view(), 3);
            })
            .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| {
                let _ = calculate_common_entropy_components::<1>(d.view(), 4);
            })
            .is_err()
        );

        // ([[1.0,2.0,3.0],[2.0,3.0,4.0]], 3)
        let d: Array2<f64> = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        assert!(
            std::panic::catch_unwind(|| {
                let _ = calculate_common_entropy_components::<3>(d.view(), 3);
            })
            .is_err()
        );
    }
}

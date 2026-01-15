// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use kiddo::traits::DistanceMetric;
use kiddo::{ImmutableKdTree, Manhattan, SquaredEuclidean};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::num::NonZeroUsize;

/// Chebyshev distance metric (L-infinity norm) for kiddo.
pub struct Chebyshev;

impl<const K: usize> DistanceMetric<f64, K> for Chebyshev {
    fn dist(a: &[f64; K], b: &[f64; K]) -> f64 {
        let mut max = 0.0;
        for i in 0..K {
            let diff = (a[i] - b[i]).abs();
            if diff > max {
                max = diff;
            }
        }
        max
    }

    fn dist1(a: f64, b: f64) -> f64 {
        (a - b).abs()
    }
}

/// Shared N-D dataset container with KD-tree for fast neighbor queries.
pub struct NdDataset<const K: usize> {
    pub points: Vec<[f64; K]>,
    pub n: usize,
    pub tree: ImmutableKdTree<f64, K>,
}

impl<const K: usize> NdDataset<K> {
    pub fn from_points(points: Vec<[f64; K]>) -> Self {
        let n = points.len();
        let tree = ImmutableKdTree::new_from_slice(&points);
        Self { points, n, tree }
    }

    pub fn from_array2(data: Array2<f64>) -> Self {
        assert!(data.ncols() == K, "data.ncols() must equal K");
        let points = Self::to_points(data.view());
        Self::from_points(points)
    }

    pub fn from_array1(data: Array1<f64>) -> NdDataset<1> {
        let n = data.len();
        let a2 = data.into_shape_with_order((n, 1)).expect("reshape 1d->2d");
        NdDataset::<1>::from_array2(a2)
    }

    /// Return a view of the dataset as a 2D array (samples x dimensions)
    pub fn view(&self) -> ArrayView2<'_, f64> {
        unsafe { ArrayView2::from_shape_ptr((self.n, K), self.points.as_ptr() as *const f64) }
    }

    fn to_points(data: ArrayView2<'_, f64>) -> Vec<[f64; K]> {
        let n = data.nrows();
        assert!(data.ncols() == K);
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

    /// Euclidean metric (p=2): distance to k-th neighbor per point (self-excluded)
    pub fn kth_neighbor_radii_euclidean(&self, k: usize) -> Vec<f64> {
        assert!(k >= 1);
        if self.n == 0 {
            return Vec::new();
        }
        assert!(k < self.n, "k must be <= N-1 for self-queries");

        let mut radii = Vec::with_capacity(self.n);
        for p in self.points.iter() {
            let mut neigh = self
                .tree
                .nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(k + 1).unwrap());
            let kth = neigh.remove(k);
            let (dist2, _idx): (f64, u64) = kth.into();
            radii.push(dist2.sqrt());
        }
        radii
    }

    /// Manhattan metric (p=1): distance to k-th neighbor per point (self-excluded)
    pub fn kth_neighbor_radii_manhattan(&self, k: usize) -> Vec<f64> {
        assert!(k >= 1);
        if self.n == 0 {
            return Vec::new();
        }
        assert!(k < self.n, "k must be <= N-1 for self-queries");

        let mut radii = Vec::with_capacity(self.n);
        for p in self.points.iter() {
            let mut neigh = self
                .tree
                .nearest_n::<Manhattan>(p, NonZeroUsize::new(k + 1).unwrap());
            let kth = neigh.remove(k);
            let (dist, _idx): (f64, u64) = kth.into();
            // Manhattan metric returns actual L1 distance (not squared)
            radii.push(dist);
        }
        radii
    }

    /// General Minkowski p (>0): O(N^2) fallback for arbitrary p (including fractional) to ensure parity.
    pub fn kth_neighbor_radii_minkowski(&self, p: f64, k: usize) -> Vec<f64> {
        assert!(p.is_finite() && p > 0.0, "Minkowski p must be > 0");
        assert!(k >= 1);
        if self.n == 0 {
            return Vec::new();
        }
        assert!(k < self.n, "k must be <= N-1 for self-queries");

        // Fast paths for p≈1 and p≈2 using KD-tree
        if (p - 1.0).abs() < 1e-12 {
            return self.kth_neighbor_radii_manhattan(k);
        }
        if (p - 2.0).abs() < 1e-12 {
            return self.kth_neighbor_radii_euclidean(k);
        }

        // Brute-force compute Lp distances row-wise
        let mut out = Vec::with_capacity(self.n);
        for i in 0..self.n {
            // Collect distances to all other points
            let mut dists: Vec<f64> = Vec::with_capacity(self.n - 1);
            let xi = &self.points[i];
            for j in 0..self.n {
                if i == j {
                    continue;
                }
                let mut acc = 0.0f64;
                for (dim, &xi_val) in xi.iter().enumerate() {
                    acc += (xi_val - self.points[j][dim]).abs().powf(p);
                }
                dists.push(acc.powf(1.0 / p));
            }
            // Select k-th smallest
            dists.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap());
            out.push(dists[k - 1]);
        }
        out
    }

    /// Split a 2D array into a Vec of owned [f64; K] points for batch processing.
    pub fn points_as_vec(data: Array2<f64>) -> Vec<[f64; K]> {
        data.axis_iter(Axis(0))
            .map(|row| {
                let mut pt = [0.0; K];
                for j in 0..K {
                    pt[j] = row[j];
                }
                pt
            })
            .collect()
    }
}

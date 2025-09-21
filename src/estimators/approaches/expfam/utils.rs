use ndarray::{Array2, ArrayView2};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use std::num::NonZeroUsize;

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
            for c in 0..K { p[c] = data[(r, c)]; }
            points.push(p);
        }
    }
    points
}

/// Compute kNN radii (Euclidean distances to the k-th nearest neighbor), excluding self.
///
/// - data is shape (N, K) with K known at compile-time
/// - k >= 1 and must be <= N-1
/// - Returns a Vec of length N with the distance to the k-th nearest neighbor for each row
pub fn knn_radii<const K: usize>(data: ArrayView2<'_, f64>, k: usize) -> Vec<f64> {
    assert!(k >= 1, "k must be >= 1");
    assert!(data.ncols() == K, "data.ncols() must equal K");
    let n = data.nrows();
    if n == 0 { return Vec::new(); }
    assert!(k <= n - 1, "k must be <= N-1 when querying within the same dataset");

    let points = to_points::<K>(data);
    let tree: ImmutableKdTree<f64, K> = ImmutableKdTree::new_from_slice(&points);

    // Query k+1 neighbors (including self), take index k (0-based) and sqrt distance
    let mut radii = Vec::with_capacity(n);
    for p in points.iter() {
        let mut neigh = tree.nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(k + 1).unwrap());
        let kth = neigh.remove(k);
        let (dist2, _idx): (f64, u64) = kth.into();
        radii.push(dist2.sqrt());
    }
    radii
}

/// Compute common components used by exponential-family kNN estimators.
/// Mirrors Python calculate_common_entropy_components for the self-evaluation case.
pub fn calculate_common_entropy_components<const K: usize>(data: ArrayView2<'_, f64>, k: usize) -> (f64, Vec<f64>, usize, usize) {
    assert!(data.ncols() == K, "data.ncols() must equal K");
    let n = data.nrows();
    let v_m = unit_ball_volume(K);
    let rho_k = knn_radii::<K>(data, k);
    (v_m, rho_k, n, K)
}

/// Helper to ensure 2D view from 1D or 2D input (placeholder for future API generalization).
pub fn as_2d(data: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>) -> ArrayView2<'_, f64> {
    data.view()
}

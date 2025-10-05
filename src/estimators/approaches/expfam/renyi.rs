use ndarray::{Array1, Array2};
use kiddo::SquaredEuclidean;
use std::num::NonZeroUsize;

use crate::estimators::traits::LocalValues;
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use super::utils::unit_ball_volume;

/// Rényi entropy estimator (kNN-based, exponential-family formulation)
pub struct RenyiEntropy<const K: usize> {
    pub nd: NdDataset<K>,
    pub k: usize,
    pub alpha: f64,
    pub base: f64,
}

impl<const K: usize> RenyiEntropy<K> {
    /// Construct from 2D data (rows = samples, cols = dimensions)
    pub fn new(data: Array2<f64>, k: usize, alpha: f64) -> Self {
        assert!(data.ncols() == K, "data.ncols() must equal K");
        let nd = NdDataset::<K>::from_array2(data);
        assert!(nd.n == 0 || k <= nd.n - 1, "k must be <= N-1 for self-queries");
        Self { nd, k, alpha, base: std::f64::consts::E }
    }

    /// Construct from a vector of K-dimensional points (already materialized)
    pub fn from_points(points: Vec<[f64; K]>, k: usize, alpha: f64) -> Self {
        assert!(k >= 1);
        let nd = NdDataset::<K>::from_points(points);
        assert!(nd.n == 0 || k <= nd.n - 1, "k must be <= N-1 for self-queries");
        Self { nd, k, alpha, base: std::f64::consts::E }
    }

    /// Construct from 1D data (convenience)
    pub fn new_1d(data: Array1<f64>, k: usize, alpha: f64) -> RenyiEntropy<1> {
        let nd = NdDataset::<1>::from_array1(data);
        assert!(nd.n == 0 || k <= nd.n - 1, "k must be <= N-1 for self-queries");
        RenyiEntropy { nd, k, alpha, base: std::f64::consts::E }
    }

    /// Set logarithm base (default e)
    pub fn with_base(mut self, base: f64) -> Self { self.base = base; self }
}

impl<const K: usize> LocalValues for RenyiEntropy<K> {
    fn local_values(&self) -> Array1<f64> {
        // Local contributions for Rényi are not exposed in the Python estimator; return empty.
        Array1::zeros(0)
    }

    fn global_value(&self) -> f64 {
        use statrs::function::gamma::{gamma, digamma};

        if self.nd.n == 0 { return 0.0; }
        let v_m = unit_ball_volume(K);

        // Compute kNN radii via KD-tree (exclude self by requesting k+1 and skipping first)
        let mut rho_k: Vec<f64> = Vec::with_capacity(self.nd.n);
        for p in self.nd.points.iter() {
            let mut neigh = self.nd.tree.nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(self.k + 1).unwrap());
            let kth = neigh.remove(self.k); // position k after including self
            let (dist2, _idx): (f64, u64) = kth.into();
            rho_k.push(dist2.sqrt());
        }
        // Effective sample count when self is excluded in neighbor queries
        let n_eff = (self.nd.n as f64) - 1.0;

        // Log with chosen base
        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };

        let q = self.alpha;
        if (q - 1.0).abs() < 1e-12 {
            // Shannon limit (q -> 1)
            let c = n_eff * (-digamma(self.k as f64)).exp() * v_m;
            let mut acc = 0.0f64;
            let mut cnt = 0usize;
            for &r in &rho_k {
                if r <= 0.0 { continue; }
                let zeta = c * r.powi(K as i32);
                if zeta > 0.0 { acc += log_b(zeta); cnt += 1; }
            }
            if cnt == 0 { return 0.0; }
            return acc / (rho_k.len() as f64);
        }

        // General Rényi case (q != 1)
        if (q - (self.k as f64 + 1.0)).abs() < 1e-12 {
            return 0.0;
        }
        let c_k = (gamma(self.k as f64) / gamma(self.k as f64 + 1.0 - q)).powf(1.0 / (1.0 - q));
        let prefactor = (n_eff * c_k * v_m).powf(1.0 - q);
        let mut sum_term = 0.0f64;
        for &r in &rho_k { if r > 0.0 { sum_term += r.powi(K as i32).powf(1.0 - q); } }
        let i_q = prefactor * sum_term / (rho_k.len() as f64);
        if i_q <= 0.0 { return 0.0; }
        log_b(i_q) / (1.0 - q)
    }
}

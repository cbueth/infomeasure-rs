use ndarray::{Array1, Array2};
use kiddo::SquaredEuclidean;
use std::num::NonZeroUsize;

use crate::estimators::traits::LocalValues;
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use super::utils::unit_ball_volume;

/// Tsallis entropy estimator (kNN-based, exponential-family formulation)
pub struct TsallisEntropy<const K: usize> {
    pub nd: NdDataset<K>,
    pub k: usize,
    pub q: f64,
    pub base: f64,
}

impl<const K: usize> TsallisEntropy<K> {
    /// Construct from 2D data
    pub fn new(data: Array2<f64>, k: usize, q: f64) -> Self {
        assert!(data.ncols() == K, "data.ncols() must equal K");
        let nd = NdDataset::<K>::from_array2(data);
        assert!(nd.n == 0 || k <= nd.n - 1, "k must be <= N-1 for self-queries");
        Self { nd, k, q, base: std::f64::consts::E }
    }

    /// Construct from 1D data (convenience)
    pub fn new_1d(data: Array1<f64>, k: usize, q: f64) -> TsallisEntropy<1> {
        let nd = NdDataset::<1>::from_array1(data);
        assert!(nd.n == 0 || k <= nd.n - 1, "k must be <= N-1 for self-queries");
        TsallisEntropy { nd, k, q, base: std::f64::consts::E }
    }

    /// Construct from points
    pub fn from_points(points: Vec<[f64; K]>, k: usize, q: f64) -> Self {
        assert!(k >= 1);
        let nd = NdDataset::<K>::from_points(points);
        assert!(nd.n == 0 || k <= nd.n - 1, "k must be <= N-1 for self-queries");
        Self { nd, k, q, base: std::f64::consts::E }
    }

    /// Set logarithm base (default e)
    pub fn with_base(mut self, base: f64) -> Self { self.base = base; self }
}

impl<const K: usize> LocalValues for TsallisEntropy<K> {
    fn local_values(&self) -> Array1<f64> { Array1::zeros(0) }

    fn global_value(&self) -> f64 {
        use statrs::function::gamma::digamma;
        if self.nd.n == 0 { return 0.0; }

        let v_m = unit_ball_volume(K);

        // Compute kNN radii via KD-tree (exclude self by requesting k+1 and skipping self)
        let mut rho_k: Vec<f64> = Vec::with_capacity(self.nd.n);
        for p in self.nd.points.iter() {
            let mut neigh = self.nd.tree.nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(self.k + 1).unwrap());
            let kth = neigh.remove(self.k);
            let (dist2, _idx): (f64, u64) = kth.into();
            rho_k.push(dist2.sqrt());
        }
        // Effective sample count when self is excluded in neighbor queries
        let n_eff = (self.nd.n as f64) - 1.0;

        let q = self.q;
        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };

        // Shannon limit for q -> 1
        if (q - 1.0).abs() < 1e-12 {
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

        // For q == k+1, follow Renyi handling and return 0.0 to avoid pathological C_k
        if (q - (self.k as f64 + 1.0)).abs() < 1e-12 { return 0.0; }

        // General Tsallis case via exponential-family I_q
        use statrs::function::gamma::gamma;
        let c_k = (gamma(self.k as f64) / gamma(self.k as f64 + 1.0 - q)).powf(1.0 / (1.0 - q));
        let prefactor = (n_eff * c_k * v_m).powf(1.0 - q);
        let mut sum_term = 0.0f64;
        for &r in &rho_k { if r > 0.0 { sum_term += r.powi(K as i32).powf(1.0 - q); } }
        let i_q = prefactor * sum_term / (rho_k.len() as f64);
        if i_q <= 0.0 { return 0.0; }
        (i_q - 1.0) / (1.0 - q)
    }
}

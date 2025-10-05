use ndarray::{Array1, Array2};
use kiddo::SquaredEuclidean;
use std::num::NonZeroUsize;

use crate::estimators::traits::LocalValues;
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use super::utils::unit_ball_volume;

/// Kozachenko–Leonenko (KL) differential entropy estimator (kNN-based, Euclidean metric)
///
/// H_hat = psi(N) - psi(k) + ln(V_m) + (m/N) * sum_i ln(rho_k,i)
/// where V_m is the m-dimensional unit-ball volume (Euclidean) and rho_k,i is the
/// distance to the k-th nearest neighbor of point i (self excluded).
pub struct KozachenkoLeonenkoEntropy<const K: usize> {
    pub nd: NdDataset<K>,
    pub k: usize,
    pub base: f64,
}

impl<const K: usize> KozachenkoLeonenkoEntropy<K> {
    /// Construct from 2D data (rows = samples, cols = dimensions)
    pub fn new(data: Array2<f64>, k: usize) -> Self {
        assert!(data.ncols() == K, "data.ncols() must equal K");
        let nd = NdDataset::<K>::from_array2(data);
        assert!(nd.n == 0 || k <= nd.n - 1, "k must be <= N-1 for self-queries");
        Self { nd, k, base: std::f64::consts::E }
    }

    /// Construct from 1D data (convenience)
    pub fn new_1d(data: Array1<f64>, k: usize) -> KozachenkoLeonenkoEntropy<1> {
        let nd = NdDataset::<1>::from_array1(data);
        assert!(nd.n == 0 || k <= nd.n - 1, "k must be <= N-1 for self-queries");
        KozachenkoLeonenkoEntropy { nd, k, base: std::f64::consts::E }
    }

    /// Construct from a vector of K-dimensional points (already materialized)
    pub fn from_points(points: Vec<[f64; K]>, k: usize) -> Self {
        assert!(k >= 1);
        let nd = NdDataset::<K>::from_points(points);
        assert!(nd.n == 0 || k <= nd.n - 1, "k must be <= N-1 for self-queries");
        Self { nd, k, base: std::f64::consts::E }
    }

    /// Set logarithm base (default e)
    pub fn with_base(mut self, base: f64) -> Self { self.base = base; self }
}

impl<const K: usize> LocalValues for KozachenkoLeonenkoEntropy<K> {
    fn local_values(&self) -> Array1<f64> {
        // Local values often exposed as -ln f_hat(x_i); here we return per-sample
        // contributions (m * ln rho_k,i) up to additive constants, but to keep parity
        // simple and because Python exposes local values via the estimator, we compute
        // full local contributions such that their mean equals the global value.
        // H = A + (m/N) * sum ln rho_k,i  => per-sample h_i = A + m * ln rho_k,i
        use statrs::function::gamma::digamma;
        if self.nd.n == 0 { return Array1::zeros(0); }
        let v_m = unit_ball_volume(K);
        let n_f = self.nd.n as f64;
        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };
        let a_const = (digamma(n_f) - digamma(self.k as f64)) / ln_base
            + log_b(v_m);

        let mut out = Array1::<f64>::zeros(self.nd.n);
        for (i, p) in self.nd.points.iter().enumerate() {
            let mut neigh = self.nd.tree.nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(self.k + 1).unwrap());
            let kth = neigh.remove(self.k);
            let (dist2, _idx): (f64, u64) = kth.into();
            let r = dist2.sqrt();
            if r > 0.0 {
                out[i] = a_const + (K as f64) * log_b(r);
            } else {
                // If zero radius due to duplicates, drop the ln term
                out[i] = a_const;
            }
        }
        out
    }

    fn global_value(&self) -> f64 {
        use statrs::function::gamma::digamma;
        if self.nd.n == 0 { return 0.0; }
        let v_m = unit_ball_volume(K);

        // Gather kNN radii
        let mut sum_ln_r = 0.0f64;
        let mut cnt = 0usize;
        for p in self.nd.points.iter() {
            let mut neigh = self.nd.tree.nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(self.k + 1).unwrap());
            let kth = neigh.remove(self.k);
            let (dist2, _idx): (f64, u64) = kth.into();
            let r = dist2.sqrt();
            if r > 0.0 { sum_ln_r += r.ln(); cnt += 1; }
        }
        if cnt == 0 { return 0.0; }

        let n_f = self.nd.n as f64;
        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };

        let term_digamma = (digamma(n_f) - digamma(self.k as f64)) / ln_base;
        let term_volume = log_b(v_m);
        let term_radii = (K as f64) * (sum_ln_r / (cnt as f64)) / ln_base; // average ln(r)
        term_digamma + term_volume + term_radii
    }
}

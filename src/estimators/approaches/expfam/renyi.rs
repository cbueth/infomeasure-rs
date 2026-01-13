use kiddo::SquaredEuclidean;
use ndarray::{Array1, Array2};
use std::num::NonZeroUsize;

use super::utils::{calculate_common_entropy_components_at, unit_ball_volume};
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use crate::estimators::traits::{
    CrossEntropy, GlobalValue, JointEntropy, LocalValues, OptionalLocalValues,
};

/// Rényi entropy estimator (kNN-based, exponential-family formulation)
pub struct RenyiEntropy<const K: usize> {
    pub nd: NdDataset<K>,
    pub k: usize,
    pub alpha: f64,
    pub base: f64,
    pub noise_level: f64,
}

impl<const K: usize> JointEntropy for RenyiEntropy<K> {
    type Source = Array1<f64>;
    type Params = (usize, f64, f64); // k, alpha, noise_level

    fn joint_entropy(series: &[Self::Source], params: Self::Params) -> f64 {
        assert_eq!(
            series.len(),
            K,
            "Number of series must match dimensionality K"
        );
        if series.is_empty() {
            return 0.0;
        }

        let n_samples = series[0].len();
        let mut data = Array2::zeros((n_samples, K));
        for (j, s) in series.iter().enumerate() {
            for i in 0..n_samples {
                data[[i, j]] = s[i];
            }
        }

        let estimator = RenyiEntropy::<K>::new(data, params.0, params.1, params.2);
        estimator.global_value()
    }
}

impl<const K: usize> CrossEntropy for RenyiEntropy<K> {
    fn cross_entropy(&self, other: &RenyiEntropy<K>) -> f64 {
        use statrs::function::gamma::{digamma, gamma};
        // H_alpha(P||Q) evaluated by taking points from self (P) and k-neighbors in other (Q)
        let (v_m, rho_k, m_samples, dimension) = calculate_common_entropy_components_at::<K>(
            other.nd.view(),
            self.k,
            Some(self.nd.view()),
        );

        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };
        let n_eff = m_samples as f64; // actually M samples in Q

        let q = self.alpha;
        if (q - 1.0).abs() < 1e-12 {
            // Shannon cross-entropy limit
            let c = n_eff * (-digamma(self.k as f64)).exp() * v_m;
            let mut acc = 0.0f64;
            let mut cnt = 0usize;
            for &r in &rho_k {
                if r <= 0.0 {
                    continue;
                }
                let zeta = c * r.powi(dimension as i32);
                if zeta > 0.0 {
                    acc += log_b(zeta);
                    cnt += 1;
                }
            }
            if cnt == 0 {
                return 0.0;
            }
            return acc / (rho_k.len() as f64);
        }

        // General Rényi cross-entropy
        if (q - (self.k as f64 + 1.0)).abs() < 1e-12 {
            return 0.0;
        }
        let c_k = (gamma(self.k as f64) / gamma(self.k as f64 + 1.0 - q)).powf(1.0 / (1.0 - q));
        let prefactor = (n_eff * c_k * v_m).powf(1.0 - q);
        let mut sum_term = 0.0f64;
        for &r in &rho_k {
            if r > 0.0 {
                sum_term += r.powi(dimension as i32).powf(1.0 - q);
            }
        }
        let i_q = prefactor * sum_term / (rho_k.len() as f64);
        if i_q <= 0.0 {
            return 0.0;
        }
        log_b(i_q) / (1.0 - q)
    }
}

impl<const K: usize> RenyiEntropy<K> {
    /// Construct from 2D data (rows = samples, cols = dimensions)
    pub fn new(data: Array2<f64>, k: usize, alpha: f64, noise_level: f64) -> Self {
        assert!(data.ncols() == K, "data.ncols() must equal K");
        let data = super::utils::add_noise(data, noise_level);
        let nd = NdDataset::<K>::from_array2(data);
        assert!(
            nd.n == 0 || k <= nd.n - 1,
            "k must be <= N-1 for self-queries"
        );
        Self {
            nd,
            k,
            alpha,
            base: std::f64::consts::E,
            noise_level,
        }
    }

    /// Build a vector of RenyiEntropy estimators, one per row of a 2D array.
    pub fn from_rows(data: Array2<f64>, k: usize, alpha: f64, noise_level: f64) -> Vec<Self> {
        let n_rows = data.nrows();
        let mut out = Vec::with_capacity(n_rows);
        for row in data.axis_iter(ndarray::Axis(0)) {
            let row_a2 = row
                .to_owned()
                .into_shape_with_order((1, K))
                .expect("reshape row");
            out.push(Self::new(row_a2, k, alpha, noise_level));
        }
        out
    }

    /// Construct from a vector of K-dimensional points (already materialized)
    pub fn from_points(points: Vec<[f64; K]>, k: usize, alpha: f64, noise_level: f64) -> Self {
        assert!(k >= 1);
        let n = points.len();
        let mut data = Array2::zeros((n, K));
        for i in 0..n {
            for j in 0..K {
                data[[i, j]] = points[i][j];
            }
        }
        Self::new(data, k, alpha, noise_level)
    }

    /// Construct from 1D data (convenience)
    pub fn new_1d(data: Array1<f64>, k: usize, alpha: f64, noise_level: f64) -> RenyiEntropy<1> {
        let n = data.len();
        let a2 = data.into_shape_with_order((n, 1)).expect("reshape 1d->2d");
        RenyiEntropy::new(a2, k, alpha, noise_level)
    }

    /// Set logarithm base (default e)
    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }
}

impl<const K: usize> GlobalValue for RenyiEntropy<K> {
    fn global_value(&self) -> f64 {
        use statrs::function::gamma::{digamma, gamma};

        if self.nd.n == 0 {
            return 0.0;
        }
        let v_m = unit_ball_volume(K);

        // Compute kNN radii via KD-tree (exclude self by requesting k+1 and skipping first)
        let mut rho_k: Vec<f64> = Vec::with_capacity(self.nd.n);
        for p in self.nd.points.iter() {
            let mut neigh = self
                .nd
                .tree
                .nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(self.k + 1).unwrap());
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
                if r <= 0.0 {
                    continue;
                }
                let zeta = c * r.powi(K as i32);
                if zeta > 0.0 {
                    acc += log_b(zeta);
                    cnt += 1;
                }
            }
            if cnt == 0 {
                return 0.0;
            }
            return acc / (rho_k.len() as f64);
        }

        // General Rényi case (q != 1)
        if (q - (self.k as f64 + 1.0)).abs() < 1e-12 {
            return 0.0;
        }
        let c_k = (gamma(self.k as f64) / gamma(self.k as f64 + 1.0 - q)).powf(1.0 / (1.0 - q));
        let prefactor = (n_eff * c_k * v_m).powf(1.0 - q);
        let mut sum_term = 0.0f64;
        for &r in &rho_k {
            if r > 0.0 {
                sum_term += r.powi(K as i32).powf(1.0 - q);
            }
        }
        let i_q = prefactor * sum_term / (rho_k.len() as f64);
        if i_q <= 0.0 {
            return 0.0;
        }
        log_b(i_q) / (1.0 - q)
    }
}

impl<const K: usize> LocalValues for RenyiEntropy<K> {
    fn local_values(&self) -> Array1<f64> {
        Array1::zeros(0)
    }
}

impl<const K: usize> OptionalLocalValues for RenyiEntropy<K> {
    fn supports_local(&self) -> bool {
        false
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not implemented for kNN-based Renyi entropy.")
    }
}

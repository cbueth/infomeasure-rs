use kiddo::SquaredEuclidean;
use ndarray::{Array1, Array2};
use std::num::NonZeroUsize;

use super::utils::unit_ball_volume;
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use crate::estimators::traits::{CrossEntropy, GlobalValue, JointEntropy, LocalValues};

/// Kozachenko–Leonenko (KL) differential entropy estimator (kNN-based, Euclidean metric)
///
/// H_hat = psi(N) - psi(k) + ln(V_m) + (m/N) * sum_i ln(rho_k,i)
/// where V_m is the m-dimensional unit-ball volume (Euclidean) and rho_k,i is the
/// distance to the k-th nearest neighbor of point i (self excluded).
pub struct KozachenkoLeonenkoEntropy<const K: usize> {
    pub nd: NdDataset<K>,
    pub k: usize,
    pub base: f64,
    pub noise_level: f64,
}

impl<const K: usize> JointEntropy for KozachenkoLeonenkoEntropy<K> {
    type Source = Array1<f64>;
    type Params = (usize, f64); // k, noise_level

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

        let estimator = KozachenkoLeonenkoEntropy::<K>::new(data, params.0, params.1);
        estimator.global_value()
    }
}

impl<const K: usize> CrossEntropy for KozachenkoLeonenkoEntropy<K> {
    fn cross_entropy(&self, other: &KozachenkoLeonenkoEntropy<K>) -> f64 {
        use super::utils::calculate_common_entropy_components_at;
        use statrs::function::gamma::digamma;

        // H(P||Q) evaluated by taking points from self (P) and k-neighbors in other (Q)
        let (v_m, rho_k, _n_p, dimension) = calculate_common_entropy_components_at::<K>(
            other.nd.view(),
            self.k,
            Some(self.nd.view()),
        );

        let n_q = other.nd.n as f64;
        let ln_base = self.base.ln();

        let mut sum_ln_rho = 0.0;
        let mut count = 0usize;
        for &r in &rho_k {
            if r > 0.0 {
                sum_ln_rho += r.ln();
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        // Parity with Python implementation: uses n_q (M) in digamma and denominator
        let hx = -digamma(self.k as f64)
            + digamma(n_q)
            + v_m.ln()
            + (dimension as f64) * sum_ln_rho / n_q;

        hx / ln_base
    }
}

impl<const K: usize> KozachenkoLeonenkoEntropy<K> {
    /// Construct from 2D data (rows = samples, cols = dimensions)
    pub fn new(data: Array2<f64>, k: usize, noise_level: f64) -> Self {
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
            base: std::f64::consts::E,
            noise_level,
        }
    }

    /// Construct from 1D data (convenience)
    pub fn new_1d(data: Array1<f64>, k: usize, noise_level: f64) -> KozachenkoLeonenkoEntropy<1> {
        let n = data.len();
        let a2 = data.into_shape_with_order((n, 1)).expect("reshape 1d->2d");
        KozachenkoLeonenkoEntropy::new(a2, k, noise_level)
    }

    /// Construct from a vector of K-dimensional points (already materialized)
    pub fn from_points(points: Vec<[f64; K]>, k: usize, noise_level: f64) -> Self {
        assert!(k >= 1);
        let n = points.len();
        let mut data = Array2::zeros((n, K));
        for i in 0..n {
            for j in 0..K {
                data[[i, j]] = points[i][j];
            }
        }
        Self::new(data, k, noise_level)
    }

    /// Set logarithm base (default e)
    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }
}

impl<const K: usize> GlobalValue for KozachenkoLeonenkoEntropy<K> {
    fn global_value(&self) -> f64 {
        use statrs::function::gamma::digamma;
        if self.nd.n == 0 {
            return 0.0;
        }
        let v_m = unit_ball_volume(K);

        // Gather kNN radii
        let mut sum_ln_r = 0.0f64;
        let mut cnt = 0usize;
        for p in self.nd.points.iter() {
            let mut neigh = self
                .nd
                .tree
                .nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(self.k + 1).unwrap());
            let kth = neigh.remove(self.k);
            let (dist2, _idx): (f64, u64) = kth.into();
            let r = dist2.sqrt();
            if r > 0.0 {
                sum_ln_r += r.ln();
                cnt += 1;
            }
        }
        if cnt == 0 {
            return 0.0;
        }

        let n_f = self.nd.n as f64;
        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };

        let term_digamma = (digamma(n_f) - digamma(self.k as f64)) / ln_base;
        let term_volume = log_b(v_m);
        let term_radii = (K as f64) * (sum_ln_r / (cnt as f64)) / ln_base; // average ln(r)
        term_digamma + term_volume + term_radii
    }
}

impl<const K: usize> LocalValues for KozachenkoLeonenkoEntropy<K> {
    fn local_values(&self) -> Array1<f64> {
        // Local values often exposed as -ln f_hat(x_i); here we return per-sample
        // contributions (m * ln rho_k,i) up to additive constants, but to keep parity
        // simple and because Python exposes local values via the estimator, we compute
        // full local contributions such that their mean equals the global value.
        // H = A + (m/N) * sum ln rho_k,i  => per-sample h_i = A + m * ln rho_k,i
        use statrs::function::gamma::digamma;
        if self.nd.n == 0 {
            return Array1::zeros(0);
        }
        let v_m = unit_ball_volume(K);
        let n_f = self.nd.n as f64;
        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };
        let a_const = (digamma(n_f) - digamma(self.k as f64)) / ln_base + log_b(v_m);

        let mut out = Array1::<f64>::zeros(self.nd.n);
        for (i, p) in self.nd.points.iter().enumerate() {
            let mut neigh = self
                .nd
                .tree
                .nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(self.k + 1).unwrap());
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
}

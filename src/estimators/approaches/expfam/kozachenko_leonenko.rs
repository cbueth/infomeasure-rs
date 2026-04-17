// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Kozachenko-Leonenko (KL) Estimators
//!
//! This module implements the Kozachenko-Leonenko differential entropy estimator
//! and its extensions to Mutual Information and Transfer Entropy using
//! k-nearest neighbor (kNN) distances.
//!
//! ## Theory
//!
//! The Kozachenko-Leonenko estimator provides an asymptotically unbiased estimate
//! of differential entropy for continuous variables [Kozachenko & Leonenko, 1987](../../../../guide/references/index.html#kozachenko1987):
//!
//! $$H_{KL}(X) = -\psi(k) + \psi(N) + \log(V_m) + \frac{m}{N} \sum_{i=1}^{N} \log(\rho_{i,k})$$
//!
//! where:
//! - $\psi$ is the digamma function.
//! - $k$ is the number of nearest neighbors.
//! - $N$ is the number of data points.
//! - $m$ is the dimensionality of the data.
//! - $V_m$ is the volume of the $m$-dimensional unit ball.
//! - $\rho_{i,k}$ is the distance to the $k$-th nearest neighbor of point $i$.
//!
//! The method works by exploiting the relationship between nearest neighbor
//! distances and local density, making it effective for high-dimensional data
//! where traditional histogram-based methods fail.
//!
//! ## Measures Implemented
//!
//! - **Differential Entropy**: $H_{KL}(X)$
//! - **Mutual Information**: $I_{KL}(X; Y) = H_{KL}(X) + H_{KL}(Y) - H_{KL}(X, Y)$
//! - **Conditional MI**: $I_{KL}(X; Y | Z) = H_{KL}(X, Z) + H_{KL}(Y, Z) - H_{KL}(X, Y, Z) - H_{KL}(Z)$
//! - **Transfer Entropy**: $T_{KL}(X \to Y)$ estimated via the CMI entropy-summation formula.
//!
//! ## See Also
//! - [Entropy Guide](crate::guide::entropy) — Conceptual background
//! - [KSG Estimators](super::ksg) — kNN-based MI optimized to cancel bias
//!
//! ## References
//!
//! - [Kozachenko & Leonenko, 1987](../../../../guide/references/index.html#kozachenko1987)
//! - [Kraskov et al., 2004](../../../../guide/references/index.html#ksg2004)

use crate::estimators::doc_macros::doc_snippets;
use kiddo::Chebyshev;
use ndarray::{Array1, Array2};
use statrs::function::gamma::digamma;

use super::utils::{KsgType, unit_ball_volume_chebyshev_with_radius, unit_ball_volume_with_radius};
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use crate::estimators::traits::{
    ConditionalMutualInformationEstimator, ConditionalTransferEntropyEstimator, CrossEntropy,
    GlobalValue, JointEntropy, LocalValues, MutualInformationEstimator, OptionalLocalValues,
    TransferEntropyEstimator,
};
use crate::estimators::utils::te_slicing::{cte_observations_const, te_observations_const};
use ndarray::{Axis, concatenate};

/// Kozachenko–Leonenko (KL) differential entropy estimator (kNN-based)
///
/// ## Theory
///
/// The Kozachenko-Leonenko estimator provides an asymptotically unbiased estimate
/// of differential entropy for continuous variables [Kozachenko & Leonenko, 1987](../../../../guide/references/index.html#kozachenko1987):
///
/// $$H_{KL}(X) = -\psi(k) + \psi(N) + \log(V_m) + \frac{m}{N} \sum_{i=1}^{N} \log(\rho_{i,k})$$
///
/// where:
/// - $\psi$ is the digamma function.
/// - $k$ is the number of nearest neighbors.
/// - $N$ is the number of data points.
/// - $m$ is the dimensionality of the data.
/// - $V_m$ is the volume of the $m$-dimensional unit ball.
/// - $\rho_{i,k}$ is the distance to the $k$-th nearest neighbor of point $i$.
///
/// **Important Note**: This estimator requires the logarithm base to be specified during
/// construction via the `base` field or `with_base()` method. Unlike other entropy
/// estimators in this library, results cannot be converted to a different base
/// afterwards using simple logarithmic conversion due to the internal mathematical
/// formulation of the KL estimator.
pub struct KozachenkoLeonenkoEntropy<const K: usize> {
    pub nd: NdDataset<K>,
    pub k: usize,
    pub ksg_type: KsgType,
    pub base: f64,
    pub noise_level: f64,
    pub use_chebyshev: bool,
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
        use statrs::function::gamma::digamma;

        // H(P||Q) evaluated by taking points from self (P) and k-neighbors in other (Q)
        let (v_m, rho_k, _n_p, dimension) = if self.use_chebyshev {
            super::utils::calculate_common_entropy_components_at_chebyshev_kl::<K>(
                other.nd.view(),
                self.k,
                Some(self.nd.view()),
            )
        } else {
            super::utils::calculate_common_entropy_components_at_kl::<K>(
                other.nd.view(),
                self.k,
                Some(self.nd.view()),
            )
        };

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

        let mut psi_k = digamma(self.k as f64);
        if self.ksg_type == KsgType::Type2 {
            psi_k -= 1.0 / (self.k as f64);
        }

        let hx = -psi_k
            + digamma(n_q)
            + v_m.ln()
            + (dimension as f64) * (sum_ln_rho / (count as f64) + 2.0f64.ln());

        hx / ln_base
    }
}

impl<const K: usize> KozachenkoLeonenkoEntropy<K> {
    /// Construct from 2D data (rows = samples, cols = dimensions)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn new(data: Array2<f64>, k: usize, noise_level: f64) -> Self {
        assert!(data.ncols() == K, "data.ncols() must equal K");
        let data = super::utils::add_noise(data, noise_level);
        let nd = NdDataset::<K>::from_array2(data);
        assert!(nd.n == 0 || k < nd.n, "k must be <= N-1 for self-queries");
        Self {
            nd,
            k,
            ksg_type: KsgType::Type1,
            base: std::f64::consts::E,
            noise_level,
            use_chebyshev: true,
        }
    }

    pub fn with_type(mut self, ksg_type: KsgType) -> Self {
        self.ksg_type = ksg_type;
        self
    }

    pub fn with_chebyshev(mut self, use_chebyshev: bool) -> Self {
        self.use_chebyshev = use_chebyshev;
        self
    }

    /// Construct from 1D data (convenience)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn new_1d(data: Array1<f64>, k: usize, noise_level: f64) -> KozachenkoLeonenkoEntropy<1> {
        let n = data.len();
        let a2 = data.into_shape_with_order((n, 1)).expect("reshape 1d->2d");
        KozachenkoLeonenkoEntropy::new(a2, k, noise_level)
    }

    /// Construct from a vector of K-dimensional points (already materialized)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
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
        if self.nd.n <= 1 {
            return 0.0;
        }
        let n_samples = self.nd.n;
        let n_f = n_samples as f64;
        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };

        let c_d = if self.use_chebyshev {
            unit_ball_volume_chebyshev_with_radius(K, 0.5)
        } else {
            unit_ball_volume_with_radius(K, 2.0, 0.5)
        };

        let mut sum_ln_eps = 0.0f64;
        let mut cnt = 0usize;
        for i in 0..n_samples {
            let p = &self.nd.points[i];

            let max_qty = std::num::NonZeroUsize::new(self.k + 1).unwrap();
            let neighbors = if self.use_chebyshev {
                self.nd.tree.nearest_n::<Chebyshev>(p, max_qty)
            } else {
                self.nd
                    .tree
                    .nearest_n::<kiddo::SquaredEuclidean>(p, max_qty)
            };

            let dist = neighbors[self.k].distance;
            let r = if self.use_chebyshev {
                dist
            } else {
                dist.sqrt()
            };

            if r > 0.0 {
                sum_ln_eps += (2.0 * r).ln();
                cnt += 1;
            }
        }
        if cnt == 0 {
            return 0.0;
        }

        let mut psi_k = digamma(self.k as f64);
        if self.ksg_type == KsgType::Type2 {
            psi_k -= 1.0 / (self.k as f64);
        }

        let term_digamma = (statrs::function::gamma::digamma(n_f) - psi_k) / ln_base;
        let term_volume = log_b(c_d);
        let term_radii = (K as f64) * (sum_ln_eps / (cnt as f64)) / ln_base;
        term_digamma + term_volume + term_radii
    }
}

impl<const K: usize> LocalValues for KozachenkoLeonenkoEntropy<K> {
    fn local_values(&self) -> Array1<f64> {
        if self.nd.n <= 1 {
            return Array1::zeros(self.nd.n);
        }
        let n_samples = self.nd.n;
        let n_f = n_samples as f64;
        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };

        let c_d = if self.use_chebyshev {
            unit_ball_volume_chebyshev_with_radius(K, 0.5)
        } else {
            unit_ball_volume_with_radius(K, 2.0, 0.5)
        };

        let mut psi_k = digamma(self.k as f64);
        if self.ksg_type == KsgType::Type2 {
            psi_k -= 1.0 / (self.k as f64);
        }

        let a_const = (statrs::function::gamma::digamma(n_f) - psi_k) / ln_base + log_b(c_d);

        let mut out = Array1::<f64>::zeros(n_samples);
        for i in 0..n_samples {
            let p = &self.nd.points[i];

            let max_qty = std::num::NonZeroUsize::new(self.k + 1).unwrap();
            let neighbors = if self.use_chebyshev {
                self.nd.tree.nearest_n::<Chebyshev>(p, max_qty)
            } else {
                self.nd
                    .tree
                    .nearest_n::<kiddo::SquaredEuclidean>(p, max_qty)
            };

            let dist = neighbors[self.k].distance;
            let r = if self.use_chebyshev {
                dist
            } else {
                dist.sqrt()
            };

            if r > 0.0 {
                out[i] = a_const + (K as f64) * log_b(2.0 * r);
            } else {
                out[i] = a_const;
            }
        }
        out
    }
}

impl<const K: usize> OptionalLocalValues for KozachenkoLeonenkoEntropy<K> {
    fn supports_local(&self) -> bool {
        true
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Ok(self.local_values())
    }
}

macro_rules! impl_kl_mi {
    ($name:ident, $num_rvs:expr, ($($d_param:ident),*), ($($d_idx:expr),*)) => {
        #[doc = concat!("Kozachenko-Leonenko mutual information estimator for ", stringify!($num_rvs), " random variables")]
        ///
        /// ## Theory
        ///
        #[doc = doc_snippets!(mi_formula "KL-based", r"_{KL}", "")]
        pub struct $name<const D_JOINT: usize, $(const $d_param: usize),*> {
            pub k: usize,
            pub ksg_type: KsgType,
            pub data: Vec<Array2<f64>>,
            pub base: f64,
            pub noise_level: f64,
            pub use_chebyshev: bool,
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> $name<D_JOINT, $($d_param),*> {
            pub fn new(series: &[Array2<f64>], k: usize, noise_level: f64) -> Self {
                assert_eq!(series.len(), $num_rvs, "Number of series must match estimator type");
                let noisy_data = series.iter().map(|s| super::utils::add_noise(s.clone(), noise_level)).collect();
                Self {
                    k,
                    ksg_type: KsgType::Type1,
                    data: noisy_data,
                    base: std::f64::consts::E,
                    noise_level,
                    use_chebyshev: true,
                }
            }

            pub fn with_type(mut self, ksg_type: KsgType) -> Self {
                self.ksg_type = ksg_type;
                self
            }

            pub fn with_base(mut self, base: f64) -> Self {
                self.base = base;
                self
            }

            pub fn with_chebyshev(mut self, use_chebyshev: bool) -> Self {
                self.use_chebyshev = use_chebyshev;
                self
            }
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> GlobalValue for $name<D_JOINT, $($d_param),*> {
            fn global_value(&self) -> f64 {
                let mut marginal_sum = 0.0;
                $(
                    let m_est = KozachenkoLeonenkoEntropy::<$d_param>::new(self.data[$d_idx].clone(), self.k, 0.0)
                        .with_type(self.ksg_type)
                        .with_chebyshev(self.use_chebyshev)
                        .with_base(self.base);
                    marginal_sum += m_est.global_value();
                )*

                let joint_data = concatenate(
                    Axis(1),
                    &self.data.iter().map(|d| d.view()).collect::<Vec<_>>()
                ).unwrap();
                let joint_est = KozachenkoLeonenkoEntropy::<D_JOINT>::new(joint_data, self.k, 0.0)
                    .with_type(self.ksg_type)
                    .with_chebyshev(self.use_chebyshev)
                    .with_base(self.base);

                marginal_sum - joint_est.global_value()
            }
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> OptionalLocalValues for $name<D_JOINT, $($d_param),*> {
            fn supports_local(&self) -> bool { false }
            fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
                Err("Local values are not implemented for KL-based mutual information.")
            }
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> MutualInformationEstimator for $name<D_JOINT, $($d_param),*> {}
    }
}

impl_kl_mi!(KozachenkoLeonenkoMutualInformation2, 2, (D1, D2), (0, 1));
impl_kl_mi!(
    KozachenkoLeonenkoMutualInformation3,
    3,
    (D1, D2, D3),
    (0, 1, 2)
);
impl_kl_mi!(
    KozachenkoLeonenkoMutualInformation4,
    4,
    (D1, D2, D3, D4),
    (0, 1, 2, 3)
);
impl_kl_mi!(
    KozachenkoLeonenkoMutualInformation5,
    5,
    (D1, D2, D3, D4, D5),
    (0, 1, 2, 3, 4)
);
impl_kl_mi!(
    KozachenkoLeonenkoMutualInformation6,
    6,
    (D1, D2, D3, D4, D5, D6),
    (0, 1, 2, 3, 4, 5)
);

/// Kozachenko-Leonenko conditional mutual information estimator
///
/// ## Theory
///
#[doc = doc_snippets!(cmi_formula "KL-based", r"_{KL}", "")]
pub struct KozachenkoLeonenkoConditionalMutualInformation<
    const D1: usize,
    const D2: usize,
    const DZ: usize,
    const D_JOINT: usize,
    const D1Z: usize,
    const D2Z: usize,
> {
    pub k: usize,
    pub ksg_type: KsgType,
    pub data: [Array2<f64>; 2],
    pub cond: Array2<f64>,
    pub base: f64,
    pub noise_level: f64,
    pub use_chebyshev: bool,
}

impl<
    const D1: usize,
    const D2: usize,
    const DZ: usize,
    const D_JOINT: usize,
    const D1Z: usize,
    const D2Z: usize,
> KozachenkoLeonenkoConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z>
{
    pub fn new(series: &[Array2<f64>], cond: &Array2<f64>, k: usize, noise_level: f64) -> Self {
        assert_eq!(series.len(), 2);
        let noisy_data = [
            super::utils::add_noise(series[0].clone(), noise_level),
            super::utils::add_noise(series[1].clone(), noise_level),
        ];
        let noisy_cond = super::utils::add_noise(cond.clone(), noise_level);
        Self {
            k,
            ksg_type: KsgType::Type1,
            data: noisy_data,
            cond: noisy_cond,
            base: std::f64::consts::E,
            noise_level,
            use_chebyshev: true,
        }
    }

    pub fn with_type(mut self, ksg_type: KsgType) -> Self {
        self.ksg_type = ksg_type;
        self
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }

    pub fn with_chebyshev(mut self, use_chebyshev: bool) -> Self {
        self.use_chebyshev = use_chebyshev;
        self
    }
}

impl<
    const D1: usize,
    const D2: usize,
    const DZ: usize,
    const D_JOINT: usize,
    const D1Z: usize,
    const D2Z: usize,
> GlobalValue for KozachenkoLeonenkoConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z>
{
    fn global_value(&self) -> f64 {
        // I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)
        let xz = concatenate(Axis(1), &[self.data[0].view(), self.cond.view()]).unwrap();
        let yz = concatenate(Axis(1), &[self.data[1].view(), self.cond.view()]).unwrap();
        let xyz = concatenate(
            Axis(1),
            &[self.data[0].view(), self.data[1].view(), self.cond.view()],
        )
        .unwrap();

        let h_xz = KozachenkoLeonenkoEntropy::<D1Z>::new(xz, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();
        let h_yz = KozachenkoLeonenkoEntropy::<D2Z>::new(yz, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();
        let h_xyz = KozachenkoLeonenkoEntropy::<D_JOINT>::new(xyz, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();
        let h_z = KozachenkoLeonenkoEntropy::<DZ>::new(self.cond.clone(), self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();

        h_xz + h_yz - h_xyz - h_z
    }
}

impl<
    const D1: usize,
    const D2: usize,
    const DZ: usize,
    const D_JOINT: usize,
    const D1Z: usize,
    const D2Z: usize,
> OptionalLocalValues
    for KozachenkoLeonenkoConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z>
{
    fn supports_local(&self) -> bool {
        false
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not implemented for KL-based conditional mutual information.")
    }
}

impl<
    const D1: usize,
    const D2: usize,
    const DZ: usize,
    const D_JOINT: usize,
    const D1Z: usize,
    const D2Z: usize,
> ConditionalMutualInformationEstimator
    for KozachenkoLeonenkoConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z>
{
}

/// Kozachenko-Leonenko transfer entropy estimator
///
/// ## Theory
///
#[doc = doc_snippets!(te_formula "KL-based", r"_{KL}", "")]
pub struct KozachenkoLeonenkoTransferEntropy<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_JOINT: usize,
    const D_XP_YP: usize,
    const D_YP: usize,
    const D_YF_YP: usize,
> {
    pub k: usize,
    pub ksg_type: KsgType,
    pub source: Array2<f64>,
    pub dest: Array2<f64>,
    pub base: f64,
    pub noise_level: f64,
    pub use_chebyshev: bool,
}

impl<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_JOINT: usize,
    const D_XP_YP: usize,
    const D_YP: usize,
    const D_YF_YP: usize,
>
    KozachenkoLeonenkoTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_JOINT,
        D_XP_YP,
        D_YP,
        D_YF_YP,
    >
{
    pub fn new(source: &Array2<f64>, dest: &Array2<f64>, k: usize, noise_level: f64) -> Self {
        Self {
            k,
            ksg_type: KsgType::Type1,
            source: source.clone(),
            dest: dest.clone(),
            base: std::f64::consts::E,
            noise_level,
            use_chebyshev: true,
        }
    }

    pub fn with_type(mut self, ksg_type: KsgType) -> Self {
        self.ksg_type = ksg_type;
        self
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }

    pub fn with_chebyshev(mut self, use_chebyshev: bool) -> Self {
        self.use_chebyshev = use_chebyshev;
        self
    }
}

impl<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_JOINT: usize,
    const D_XP_YP: usize,
    const D_YP: usize,
    const D_YF_YP: usize,
> GlobalValue
    for KozachenkoLeonenkoTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_JOINT,
        D_XP_YP,
        D_YP,
        D_YF_YP,
    >
{
    fn global_value(&self) -> f64 {
        let noisy_src = super::utils::add_noise(self.source.clone(), self.noise_level);
        let noisy_dest = super::utils::add_noise(self.dest.clone(), self.noise_level);

        let (yf, yp, xp) = te_observations_const::<
            f64,
            SRC_HIST,
            DEST_HIST,
            STEP_SIZE,
            D_SOURCE,
            D_TARGET,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >(&noisy_src, &noisy_dest, false);

        // TE(X -> Y) = I(X_past; Y_future | Y_past) = H(X_past, Y_past) + H(Y_future, Y_past) - H(X_past, Y_future, Y_past) - H(Y_past)
        let xpyp = concatenate(Axis(1), &[xp.view(), yp.view()]).unwrap();
        let yfyp = concatenate(Axis(1), &[yf.view(), yp.view()]).unwrap();
        let xpyfyp = concatenate(Axis(1), &[xp.view(), yf.view(), yp.view()]).unwrap();

        let h_xpyp = KozachenkoLeonenkoEntropy::<D_XP_YP>::new(xpyp, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();
        let h_yfyp = KozachenkoLeonenkoEntropy::<D_YF_YP>::new(yfyp, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();
        let h_xpyfyp = KozachenkoLeonenkoEntropy::<D_JOINT>::new(xpyfyp, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();
        let h_yp = KozachenkoLeonenkoEntropy::<D_YP>::new(yp, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();

        h_xpyp + h_yfyp - h_xpyfyp - h_yp
    }
}

impl<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_JOINT: usize,
    const D_XP_YP: usize,
    const D_YP: usize,
    const D_YF_YP: usize,
> OptionalLocalValues
    for KozachenkoLeonenkoTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_JOINT,
        D_XP_YP,
        D_YP,
        D_YF_YP,
    >
{
    fn supports_local(&self) -> bool {
        false
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not implemented for KL-based transfer entropy.")
    }
}

impl<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_JOINT: usize,
    const D_XP_YP: usize,
    const D_YP: usize,
    const D_YF_YP: usize,
> TransferEntropyEstimator
    for KozachenkoLeonenkoTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_JOINT,
        D_XP_YP,
        D_YP,
        D_YF_YP,
    >
{
}

/// Kozachenko-Leonenko conditional transfer entropy estimator
///
/// ## Theory
///
#[doc = doc_snippets!(cte_formula "KL-based", r"_{KL}", "")]
pub struct KozachenkoLeonenkoConditionalTransferEntropy<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const COND_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D_XP_YP_ZP: usize,
    const D_YP_ZP: usize,
    const D_YF_YP_ZP: usize,
> {
    pub k: usize,
    pub ksg_type: KsgType,
    pub source: Array2<f64>,
    pub dest: Array2<f64>,
    pub cond: Array2<f64>,
    pub base: f64,
    pub noise_level: f64,
    pub use_chebyshev: bool,
}

impl<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const COND_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D_XP_YP_ZP: usize,
    const D_YP_ZP: usize,
    const D_YF_YP_ZP: usize,
>
    KozachenkoLeonenkoConditionalTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        COND_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_COND,
        D_JOINT,
        D_XP_YP_ZP,
        D_YP_ZP,
        D_YF_YP_ZP,
    >
{
    pub fn new(
        source: &Array2<f64>,
        dest: &Array2<f64>,
        cond: &Array2<f64>,
        k: usize,
        noise_level: f64,
    ) -> Self {
        Self {
            k,
            ksg_type: KsgType::Type1,
            source: source.clone(),
            dest: dest.clone(),
            cond: cond.clone(),
            base: std::f64::consts::E,
            noise_level,
            use_chebyshev: true,
        }
    }

    pub fn with_type(mut self, ksg_type: KsgType) -> Self {
        self.ksg_type = ksg_type;
        self
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }

    pub fn with_chebyshev(mut self, use_chebyshev: bool) -> Self {
        self.use_chebyshev = use_chebyshev;
        self
    }
}

impl<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const COND_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D_XP_YP_ZP: usize,
    const D_YP_ZP: usize,
    const D_YF_YP_ZP: usize,
> GlobalValue
    for KozachenkoLeonenkoConditionalTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        COND_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_COND,
        D_JOINT,
        D_XP_YP_ZP,
        D_YP_ZP,
        D_YF_YP_ZP,
    >
{
    fn global_value(&self) -> f64 {
        let noisy_src = super::utils::add_noise(self.source.clone(), self.noise_level);
        let noisy_dest = super::utils::add_noise(self.dest.clone(), self.noise_level);
        let noisy_cond = super::utils::add_noise(self.cond.clone(), self.noise_level);

        let (yf, yp, xp, zp) = cte_observations_const::<
            f64,
            SRC_HIST,
            DEST_HIST,
            COND_HIST,
            STEP_SIZE,
            D_SOURCE,
            D_TARGET,
            D_COND,
            D_JOINT,
            D_XP_YP_ZP,
            D_YP_ZP,
            D_YF_YP_ZP,
        >(&noisy_src, &noisy_dest, &noisy_cond, false);

        // CTE(X -> Y | Z) = I(X_past; Y_future | Y_past, Z_past) = H(X_past, Y_past, Z_past) + H(Y_future, Y_past, Z_past) - H(X_past, Y_future, Y_past, Z_past) - H(Y_past, Z_past)
        let xpypzp = concatenate(Axis(1), &[xp.view(), yp.view(), zp.view()]).unwrap();
        let yfypzp = concatenate(Axis(1), &[yf.view(), yp.view(), zp.view()]).unwrap();
        let xpyfypzp = concatenate(Axis(1), &[xp.view(), yf.view(), yp.view(), zp.view()]).unwrap();
        let ypzp = concatenate(Axis(1), &[yp.view(), zp.view()]).unwrap();

        let h_xpypzp = KozachenkoLeonenkoEntropy::<D_XP_YP_ZP>::new(xpypzp, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();
        let h_yfypzp = KozachenkoLeonenkoEntropy::<D_YF_YP_ZP>::new(yfypzp, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();
        let h_xpyfypzp = KozachenkoLeonenkoEntropy::<D_JOINT>::new(xpyfypzp, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();
        let h_ypzp = KozachenkoLeonenkoEntropy::<D_YP_ZP>::new(ypzp, self.k, 0.0)
            .with_type(self.ksg_type)
            .with_chebyshev(self.use_chebyshev)
            .with_base(self.base)
            .global_value();

        h_xpypzp + h_yfypzp - h_xpyfypzp - h_ypzp
    }
}

impl<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const COND_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D_XP_YP_ZP: usize,
    const D_YP_ZP: usize,
    const D_YF_YP_ZP: usize,
> OptionalLocalValues
    for KozachenkoLeonenkoConditionalTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        COND_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_COND,
        D_JOINT,
        D_XP_YP_ZP,
        D_YP_ZP,
        D_YF_YP_ZP,
    >
{
    fn supports_local(&self) -> bool {
        false
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not implemented for KL-based conditional transfer entropy.")
    }
}

impl<
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const COND_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D_XP_YP_ZP: usize,
    const D_YP_ZP: usize,
    const D_YF_YP_ZP: usize,
> ConditionalTransferEntropyEstimator
    for KozachenkoLeonenkoConditionalTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        COND_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_COND,
        D_JOINT,
        D_XP_YP_ZP,
        D_YP_ZP,
        D_YF_YP_ZP,
    >
{
}

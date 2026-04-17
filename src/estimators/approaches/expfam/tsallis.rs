// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Tsallis Entropy Estimators (kNN-based)
//!
//! This module implements estimators for Tsallis entropy and its derived measures
//! using k-nearest neighbor distances.
//!
//! ## Theory
//!
//! Tsallis entropy (q-order entropy) is a non-additive generalization of Shannon entropy:
//!
//! $$S_q(X) = \frac{1}{q-1} \left( 1 - \sum_{i=1}^{n} p_i^q \right)$$
//!
//! For continuous variables, it is estimated using kNN distances as:
//!
//! $$S_q(X) = \frac{1 - I_q}{q - 1}$$
//!
//! where $I_q$ is the same exponential-family integral used in Rényi entropy estimation:
//!
//! $$I_q = \frac{1}{N} \sum_{i=1}^N ( (N-1) C_{k,q} V_m \rho_{i,k}^m )^{1-q}$$
//!
//! In the limit $q \to 1$, Tsallis entropy reduces to Shannon entropy.
//!
//! ## Measures Implemented
//!
//! - **Entropy**: $S_q(X)$
//! - **Mutual Information**: $I_q(X; Y) = S_q(X) + S_q(Y) - S_q(X, Y)$
//! - **Conditional MI**: $I_q(X; Y | Z) = S_q(X, Z) + S_q(Y, Z) - S_q(X, Y, Z) - S_q(Z)$
//! - **Transfer Entropy**: $T_q(X \to Y)$ estimated via the CMI entropy-summation formula.
//!
//! ## See Also
//! - [Entropy Guide](crate::guide::entropy) — Conceptual background
//! - [Rényi Estimators](super::renyi) — Additive generalized entropy
//! - [KSG Estimators](super::ksg) — kNN-based MI optimized for Shannon entropy
//!
//! ## References
//!
//! - [Tsallis, 1988](../../../../guide/references/index.html#tsallis1988)
//! - [Leonenko et al., 2008](../../../../guide/references/index.html#leonenko2008)

use crate::estimators::doc_macros::doc_snippets;
use kiddo::SquaredEuclidean;
use ndarray::{Array1, Array2, Axis, concatenate};
use std::num::NonZeroUsize;

use super::utils::{add_noise, calculate_common_entropy_components_at, unit_ball_volume};
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use crate::estimators::traits::{
    ConditionalMutualInformationEstimator, ConditionalTransferEntropyEstimator, CrossEntropy,
    GlobalValue, JointEntropy, LocalValues, MutualInformationEstimator, OptionalLocalValues,
    TransferEntropyEstimator,
};
use crate::estimators::utils::te_slicing::{cte_observations_const, te_observations_const};

/// Tsallis entropy estimator (kNN-based, exponential-family formulation)
///
/// ## Theory
///
/// For continuous variables, the Tsallis $q$-entropy is estimated using kNN distances as:
///
/// $$S_q(X) = \frac{1 - I_q}{q - 1}$$
///
/// where $I_q$ is the exponential-family integral:
///
/// $$I_q = \frac{1}{N} \sum_{i=1}^N ( (N-1) C_{k,q} V_m \rho_{i,k}^m )^{1-q}$$
///
/// **Important Note**: This estimator requires the logarithm base to be specified during
/// construction via the `base` field or `with_base()` method. Unlike other entropy
/// estimators in this library, results cannot be converted to a different base
/// afterwards using simple logarithmic conversion due to the internal mathematical
/// formulation of the Tsallis entropy in the exponential family.
pub struct TsallisEntropy<const K: usize> {
    pub nd: NdDataset<K>,
    pub k: usize,
    pub q: f64,
    pub base: f64,
    pub noise_level: f64,
}

impl<const K: usize> JointEntropy for TsallisEntropy<K> {
    type Source = Array1<f64>;
    type Params = (usize, f64, f64); // k, q, noise_level

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

        let estimator = TsallisEntropy::<K>::new(data, params.0, params.1, params.2);
        estimator.global_value()
    }
}

impl<const K: usize> CrossEntropy for TsallisEntropy<K> {
    fn cross_entropy(&self, other: &TsallisEntropy<K>) -> f64 {
        use statrs::function::gamma::digamma;
        // H_q(P||Q) evaluated by taking points from self (P) and k-neighbors in other (Q)
        let (v_m, rho_k, m_samples, dimension) = calculate_common_entropy_components_at::<K>(
            other.nd.view(),
            self.k,
            Some(self.nd.view()),
        );

        let ln_base = self.base.ln();
        let log_b = |x: f64| -> f64 { x.ln() / ln_base };
        let n_eff = m_samples as f64; // actually M samples in Q

        let q = self.q;
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

        // General Tsallis cross-entropy
        if (q - (self.k as f64 + 1.0)).abs() < 1e-12 {
            return 0.0;
        }
        use statrs::function::gamma::gamma;
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
        (i_q - 1.0) / (1.0 - q)
    }
}

impl<const K: usize> TsallisEntropy<K> {
    /// Construct from 2D data
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn new(data: Array2<f64>, k: usize, q: f64, noise_level: f64) -> Self {
        assert!(data.ncols() == K, "data.ncols() must equal K");
        let data = super::utils::add_noise(data, noise_level);
        let nd = NdDataset::<K>::from_array2(data);
        assert!(nd.n == 0 || k < nd.n, "k must be <= N-1 for self-queries");
        Self {
            nd,
            k,
            q,
            base: std::f64::consts::E,
            noise_level,
        }
    }

    /// Build a vector of TsallisEntropy estimators, one per row of a 2D array.
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn from_rows(data: Array2<f64>, k: usize, q: f64, noise_level: f64) -> Vec<Self> {
        let n_rows = data.nrows();
        let mut out = Vec::with_capacity(n_rows);
        for row in data.axis_iter(ndarray::Axis(0)) {
            let row_a2 = row
                .to_owned()
                .into_shape_with_order((1, K))
                .expect("reshape row");
            out.push(Self::new(row_a2, k, q, noise_level));
        }
        out
    }

    /// Construct from 1D data (convenience)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn new_1d(data: Array1<f64>, k: usize, q: f64, noise_level: f64) -> TsallisEntropy<1> {
        let n = data.len();
        let a2 = data.into_shape_with_order((n, 1)).expect("reshape 1d->2d");
        TsallisEntropy::new(a2, k, q, noise_level)
    }

    /// Construct from points
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn from_points(points: Vec<[f64; K]>, k: usize, q: f64, noise_level: f64) -> Self {
        assert!(k >= 1);
        let n = points.len();
        let mut data = Array2::zeros((n, K));
        for i in 0..n {
            for j in 0..K {
                data[[i, j]] = points[i][j];
            }
        }
        Self::new(data, k, q, noise_level)
    }

    /// Set logarithm base (default e)
    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
        self
    }
}

impl<const K: usize> GlobalValue for TsallisEntropy<K> {
    fn global_value(&self) -> f64 {
        use statrs::function::gamma::digamma;
        if self.nd.n == 0 {
            return 0.0;
        }

        let v_m = unit_ball_volume(K, 2.0);

        // Compute kNN radii via KD-tree (exclude self by requesting k+1 and skipping self)
        let mut rho_k: Vec<f64> = Vec::with_capacity(self.nd.n);
        for p in self.nd.points.iter() {
            let mut neigh = self
                .nd
                .tree
                .nearest_n::<SquaredEuclidean>(p, NonZeroUsize::new(self.k + 1).unwrap());
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

        // For q == k+1, follow Renyi handling and return 0.0 to avoid pathological C_k
        if (q - (self.k as f64 + 1.0)).abs() < 1e-12 {
            return 0.0;
        }

        // General Tsallis case via exponential-family I_q
        use statrs::function::gamma::gamma;
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
        (i_q - 1.0) / (1.0 - q)
    }
}

impl<const K: usize> LocalValues for TsallisEntropy<K> {
    fn local_values(&self) -> Array1<f64> {
        Array1::zeros(0)
    }
}

impl<const K: usize> OptionalLocalValues for TsallisEntropy<K> {
    fn supports_local(&self) -> bool {
        false
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not implemented for kNN-based Tsallis entropy.")
    }
}

macro_rules! impl_tsallis_mi {
    ($name:ident, $num_rvs:expr, ($($d_param:ident),*), ($($d_idx:expr),*)) => {
        #[doc = concat!("Tsallis mutual information estimator for ", stringify!($num_rvs), " random variables")]
        ///
        /// ## Theory
        ///
        #[doc = doc_snippets!(mi_formula "Tsallis", "_q", "", "S")]
        pub struct $name<const D_JOINT: usize, $(const $d_param: usize),*> {
            pub k: usize,
            pub q: f64,
            pub data: Vec<Array2<f64>>,
            pub base: f64,
            pub noise_level: f64,
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> $name<D_JOINT, $($d_param),*> {
            pub fn new(series: &[Array2<f64>], k: usize, q: f64, noise_level: f64) -> Self {
                assert_eq!(series.len(), $num_rvs, "Number of series must match estimator type");
                let noisy_data = series.iter().map(|s| add_noise(s.clone(), noise_level)).collect();
                Self {
                    k,
                    q,
                    data: noisy_data,
                    base: std::f64::consts::E,
                    noise_level,
                }
            }

            pub fn with_base(mut self, base: f64) -> Self {
                self.base = base;
                self
            }
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> GlobalValue for $name<D_JOINT, $($d_param),*> {
            fn global_value(&self) -> f64 {
                let mut marginal_sum = 0.0;
                $(
                    let m_est = TsallisEntropy::<$d_param>::new(self.data[$d_idx].clone(), self.k, self.q, 0.0)
                        .with_base(self.base);
                    marginal_sum += m_est.global_value();
                )*

                let joint_data = concatenate(
                    Axis(1),
                    &self.data.iter().map(|d| d.view()).collect::<Vec<_>>()
                ).unwrap();
                let joint_est = TsallisEntropy::<D_JOINT>::new(joint_data, self.k, self.q, 0.0)
                    .with_base(self.base);

                marginal_sum - joint_est.global_value()
            }
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> OptionalLocalValues for $name<D_JOINT, $($d_param),*> {
            fn supports_local(&self) -> bool { false }
            fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
                Err("Local values are not implemented for Tsallis mutual information.")
            }
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> MutualInformationEstimator for $name<D_JOINT, $($d_param),*> {}
    }
}

impl_tsallis_mi!(TsallisMutualInformation2, 2, (D1, D2), (0, 1));
impl_tsallis_mi!(TsallisMutualInformation3, 3, (D1, D2, D3), (0, 1, 2));
impl_tsallis_mi!(TsallisMutualInformation4, 4, (D1, D2, D3, D4), (0, 1, 2, 3));
impl_tsallis_mi!(
    TsallisMutualInformation5,
    5,
    (D1, D2, D3, D4, D5),
    (0, 1, 2, 3, 4)
);
impl_tsallis_mi!(
    TsallisMutualInformation6,
    6,
    (D1, D2, D3, D4, D5, D6),
    (0, 1, 2, 3, 4, 5)
);

/// Tsallis conditional mutual information estimator
///
/// ## Theory
///
#[doc = doc_snippets!(cmi_formula "Tsallis", "_q", "", "S")]
pub struct TsallisConditionalMutualInformation<
    const D1: usize,
    const D2: usize,
    const DZ: usize,
    const D_JOINT: usize,
    const D1Z: usize,
    const D2Z: usize,
> {
    pub k: usize,
    pub q: f64,
    pub data: [Array2<f64>; 2],
    pub cond: Array2<f64>,
    pub base: f64,
    pub noise_level: f64,
}

impl<
    const D1: usize,
    const D2: usize,
    const DZ: usize,
    const D_JOINT: usize,
    const D1Z: usize,
    const D2Z: usize,
> TsallisConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z>
{
    pub fn new(
        series: &[Array2<f64>],
        cond: &Array2<f64>,
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> Self {
        assert_eq!(series.len(), 2);
        let noisy_data = [
            add_noise(series[0].clone(), noise_level),
            add_noise(series[1].clone(), noise_level),
        ];
        let noisy_cond = add_noise(cond.clone(), noise_level);
        Self {
            k,
            q,
            data: noisy_data,
            cond: noisy_cond,
            base: std::f64::consts::E,
            noise_level,
        }
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
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
> GlobalValue for TsallisConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z>
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

        let h_xz = TsallisEntropy::<D1Z>::new(xz, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();
        let h_yz = TsallisEntropy::<D2Z>::new(yz, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();
        let h_xyz = TsallisEntropy::<D_JOINT>::new(xyz, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();
        let h_z = TsallisEntropy::<DZ>::new(self.cond.clone(), self.k, self.q, 0.0)
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
> OptionalLocalValues for TsallisConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z>
{
    fn supports_local(&self) -> bool {
        false
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Err("Local values are not implemented for Tsallis conditional mutual information.")
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
    for TsallisConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z>
{
}

/// Tsallis transfer entropy estimator
///
/// ## Theory
///
#[doc = doc_snippets!(te_formula "Tsallis", "_q", "", "S")]
pub struct TsallisTransferEntropy<
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
    pub q: f64,
    pub source: Array2<f64>,
    pub dest: Array2<f64>,
    pub base: f64,
    pub noise_level: f64,
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
    TsallisTransferEntropy<
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
    pub fn new(
        source: &Array2<f64>,
        dest: &Array2<f64>,
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> Self {
        Self {
            k,
            q,
            source: source.clone(),
            dest: dest.clone(),
            base: std::f64::consts::E,
            noise_level,
        }
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
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
    for TsallisTransferEntropy<
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
        // TE(X -> Y) = H(X_past, Y_past) + H(Y_future, Y_past) - H(X_past, Y_future, Y_past) - H(Y_past)
        let noisy_src = add_noise(self.source.clone(), self.noise_level);
        let noisy_dest = add_noise(self.dest.clone(), self.noise_level);

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

        let xpyp = concatenate(Axis(1), &[xp.view(), yp.view()]).unwrap();
        let yfyp = concatenate(Axis(1), &[yf.view(), yp.view()]).unwrap();
        let xpyfyp = concatenate(Axis(1), &[xp.view(), yf.view(), yp.view()]).unwrap();

        let h_xpyp = TsallisEntropy::<D_XP_YP>::new(xpyp, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();
        let h_yfyp = TsallisEntropy::<D_YF_YP>::new(yfyp, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();
        let h_xpyfyp = TsallisEntropy::<D_JOINT>::new(xpyfyp, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();
        let h_yp = TsallisEntropy::<D_YP>::new(yp, self.k, self.q, 0.0)
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
    for TsallisTransferEntropy<
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
        Err("Local values are not implemented for Tsallis transfer entropy.")
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
    for TsallisTransferEntropy<
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

/// Tsallis conditional transfer entropy estimator
///
/// ## Theory
///
#[doc = doc_snippets!(cte_formula "Tsallis", "_q", "", "S")]
pub struct TsallisConditionalTransferEntropy<
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
    pub q: f64,
    pub source: Array2<f64>,
    pub dest: Array2<f64>,
    pub cond: Array2<f64>,
    pub base: f64,
    pub noise_level: f64,
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
    TsallisConditionalTransferEntropy<
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
        q: f64,
        noise_level: f64,
    ) -> Self {
        Self {
            k,
            q,
            source: source.clone(),
            dest: dest.clone(),
            cond: cond.clone(),
            base: std::f64::consts::E,
            noise_level,
        }
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.base = base;
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
    for TsallisConditionalTransferEntropy<
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
        // CTE(X -> Y | Z) = H(X_past, Y_past, Z_past) + H(Y_future, Y_past, Z_past) - H(X_past, Y_future, Y_past, Z_past) - H(Y_past, Z_past)
        let noisy_src = add_noise(self.source.clone(), self.noise_level);
        let noisy_dest = add_noise(self.dest.clone(), self.noise_level);
        let noisy_cond = add_noise(self.cond.clone(), self.noise_level);

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

        let xpyfzp = concatenate(Axis(1), &[xp.view(), yp.view(), zp.view()]).unwrap();
        let yfypzp = concatenate(Axis(1), &[yf.view(), yp.view(), zp.view()]).unwrap();
        let xpyfypzp = concatenate(Axis(1), &[xp.view(), yf.view(), yp.view(), zp.view()]).unwrap();
        let ypzp = concatenate(Axis(1), &[yp.view(), zp.view()]).unwrap();

        let h_xpyfzp = TsallisEntropy::<D_XP_YP_ZP>::new(xpyfzp, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();
        let h_yfypzp = TsallisEntropy::<D_YF_YP_ZP>::new(yfypzp, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();
        let h_xpyfypzp = TsallisEntropy::<D_JOINT>::new(xpyfypzp, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();
        let h_ypzp = TsallisEntropy::<D_YP_ZP>::new(ypzp, self.k, self.q, 0.0)
            .with_base(self.base)
            .global_value();

        h_xpyfzp + h_yfypzp - h_xpyfypzp - h_ypzp
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
    for TsallisConditionalTransferEntropy<
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
        Err("Local values are not implemented for Tsallis conditional transfer entropy.")
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
    for TsallisConditionalTransferEntropy<
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

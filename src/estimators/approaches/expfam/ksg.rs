// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Kraskov-Stögbauer-Grassberger (KSG) Estimators
//!
//! The Kraskov-Stögbauer-Grassberger (KSG) method is a non-parametric estimation technique
//! for mutual information and related measures based on k-nearest neighbor (kNN) distances.
//!
//! ## Overview
//!
//! KSG avoids explicit density estimation by leveraging properties of kNN distances,
//! similar to the Kozachenko-Leonenko entropy estimator. It is designed to cancel out
//! errors in marginal and joint entropy estimates that would otherwise arise from
//! different dimensionalities.
//!
//! This module implements:
//! - **Mutual Information (MI)**: $I(X; Y)$
//! - **Conditional Mutual Information (CMI)**: $I(X; Y | Z)$
//! - **Transfer Entropy (TE)**: $T_{X \to Y}$
//! - **Conditional Transfer Entropy (CTE)**: $T_{X \to Y | Z}$
//!
//! ## Algorithms: Type I and Type II
//!
//! The KSG method supports two variants that differ in how they count neighbors in
//! marginal spaces and the specific formula used:
//!
//! ### Type I (Algorithm 1)
//! Uses strict inequality for neighbor counting in marginal spaces (distance $< \epsilon$).
//!
//! For MI, the formula is:
//! $$I(X; Y) = \psi(k) + \psi(N) - \frac{1}{N} \sum_{i=1}^{N} [\psi(n_x(i) + 1) + \psi(n_y(i) + 1)]$$
//!
//! where $n_x(i)$ is the number of points in the $X$-marginal space with distance strictly
//! less than the distance to the $k$-th neighbor in the joint space.
//!
//! ### Type II (Algorithm 2)
//! Uses non-strict inequality (distance $\le \epsilon$) and a modified formula:
//! $$I(X; Y) = \psi(k) - 1/k + \psi(N) - \frac{1}{N} \sum_{i=1}^{N} [\psi(n_x(i)) + \psi(n_y(i))]$$
//!
//! where $n_x(i)$ now includes points at distance $\le \epsilon$.
//!
//! ## Conditional Measures
//!
//! ### Conditional Mutual Information (CMI)
//! For CMI, the KSG estimator uses:
//! $$I(X; Y | Z) = \psi(k) + \langle \psi(n_z(i) + 1) - \psi(n_{xz}(i) + 1) - \psi(n_{yz}(i) + 1) \rangle$$
//!
//! ### Transfer Entropy (TE)
//! TE is estimated as a conditional mutual information $I(Y_{future}; X_{past} | Y_{past})$:
//! $$T_{X \to Y} = \psi(k) + \langle \psi(n_{Y_{past}} + 1) - \psi(n_{Y_{future}, Y_{past}} + 1) - \psi(n_{Y_{past}, X_{past}} + 1) \rangle$$
//!
//! ## See Also
//! - [Mutual Information Guide](crate::guide::mutual_information) — Conceptual background
//! - [Transfer Entropy Guide](crate::guide::transfer_entropy) — Directed information flow
//! - [Kozachenko-Leonenko](super::kozachenko_leonenko) — kNN-based entropy
//!
//! ## References
//!
//! - [Kraskov et al., 2004](../../../../guide/references/index.html#ksg2004)
//! - [Frenzel & Pompe, 2007](../../../../guide/references/index.html#frenzel2007)

use kiddo::{Chebyshev, SquaredEuclidean};
use ndarray::{Array1, Array2, Axis, concatenate};
use statrs::function::gamma::digamma;

pub use super::utils::KsgType;
use super::utils::add_noise;
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use crate::estimators::traits::{
    ConditionalTransferEntropyEstimator, GlobalValue, LocalValues, MutualInformationEstimator,
    OptionalLocalValues, TransferEntropyEstimator,
};
use crate::estimators::utils::te_slicing::{cte_observations_const, te_observations_const};

/// A helper trait to allow using the same metric across different dimensions in KD-trees.
pub trait KsgMetric<F64, const K: usize>: kiddo::traits::DistanceMetric<F64, K> {}
impl<const K: usize> KsgMetric<f64, K> for Chebyshev {}
impl<const K: usize> KsgMetric<f64, K> for SquaredEuclidean {}

macro_rules! impl_ksg_mi {
    ($name:ident, $num_rvs:expr, ($($d_param:ident),*), ($($d_idx:expr),*)) => {
        #[doc = concat!("KSG (kNN-based) mutual information estimator for ", stringify!($num_rvs), " random variables")]
        ///
        /// ## Theory
        ///
        /// For two variables, the KSG Type I formula is:
        /// $$I(X; Y) = \psi(k) + \psi(N) - \frac{1}{N} \sum_{i=1}^{N} [\psi(n_x(i) + 1) + \psi(n_y(i) + 1)]$$
        ///
        /// For $m$ variables, this generalizes to:
        /// $$I(X_1; \ldots; X_m) = \psi(k) + (m-1)\psi(N) - \left\langle \sum_{j=1}^m \psi(n_j + 1) \right\rangle$$
        ///
        /// See the [Mutual Information Guide](crate::guide::mutual_information) for conceptual background.
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
                let noisy_data = series.iter().map(|s| add_noise(s.clone(), noise_level)).collect();
                Self {
                    k,
                    ksg_type: KsgType::Type1,
                    data: noisy_data,
                    base: std::f64::consts::E,
                    noise_level,
                    use_chebyshev: true, // Chebyshev is standard for KSG
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

            fn compute_local_mi_with_metric<M>(&self) -> Array1<f64>
            where
                M: KsgMetric<f64, D_JOINT> + $(KsgMetric<f64, $d_param> +)* 'static
            {
                let n_samples = self.data[0].nrows();
                let joint_data = concatenate(
                    Axis(1),
                    &self.data.iter().map(|d| d.view()).collect::<Vec<_>>(),
                ).unwrap();

                // 1. Find k-th neighbor distance in joint space
                let joint_points = NdDataset::<D_JOINT>::points_as_vec(joint_data);
                let joint_tree = kiddo::ImmutableKdTree::new_from_slice(&joint_points);

                let mut epsilons = Vec::with_capacity(n_samples);
                let max_qty = std::num::NonZeroUsize::new(self.k + 1).unwrap();
                for i in 0..n_samples {
                    let p = &joint_points[i];
                    let neighbors = joint_tree.nearest_n::<M>(p, max_qty);
                    let dist = neighbors[self.k].distance;
                    let eps = if std::any::TypeId::of::<M>() == std::any::TypeId::of::<SquaredEuclidean>() {
                        dist.sqrt()
                    } else {
                        dist
                    };
                    epsilons.push(eps);
                }

                // 2. Count neighbours in marginal spaces within epsilon
                let mut marginal_counts = Vec::new();
                $(
                    let m_data = self.data[$d_idx].view();
                    let m_points = NdDataset::<$d_param>::points_as_vec(m_data.to_owned());
                    let m_tree = kiddo::ImmutableKdTree::new_from_slice(&m_points);

                    let mut counts = Vec::with_capacity(n_samples);
                    for i in 0..n_samples {
                        let p = &m_points[i];
                        let eps = epsilons[i];

                        let count = if self.ksg_type == KsgType::Type1 {
                            // Type 1 uses strict inequality: dist < eps
                            // Python uses: query_ball_point(r=nextafter(eps, -inf)) - (eps > 0 ? 1 : 0)
                            if eps > 0.0 {
                                // Use strict inequality via within_exclusive
                                let strict_count = if self.use_chebyshev {
                                    m_tree.within_exclusive::<Chebyshev>(p, eps, false).len()
                                } else {
                                    m_tree.within_exclusive::<SquaredEuclidean>(p, eps.powi(2), false).len()
                                };
                                // Subtract 1 to exclude the point itself (same as Python)
                                strict_count - 1
                            } else {
                                0
                            }
                        } else {
                            if self.use_chebyshev {
                                m_tree.within::<Chebyshev>(p, eps).len()
                            } else {
                                m_tree.within::<SquaredEuclidean>(p, eps.powi(2)).len()
                            }
                        };

                        counts.push(count as f64);
                    }
                    marginal_counts.push(counts);
                )*

                let mut local_mi = Array1::zeros(n_samples);
                let ln_base = self.base.ln();

                for i in 0..n_samples {
                    if self.ksg_type == KsgType::Type1 {
                        let mut sum_psi_ni_plus_1 = 0.0;
                        for m_idx in 0..$num_rvs {
                            let ni = marginal_counts[m_idx][i];
                            sum_psi_ni_plus_1 += digamma(ni + 1.0);
                        }
                        // Type I: I = psi(k) - <sum psi(ni+1)> + (m-1)psi(N)
                        local_mi[i] = (digamma(self.k as f64) - sum_psi_ni_plus_1 + ($num_rvs as f64 - 1.0) * digamma(n_samples as f64)) / ln_base;
                    } else {
                        // Type II: I = psi(k) - 1/k - <sum psi(ni)> + (m-1)psi(N)
                        let mut sum_psi_ni = 0.0;
                        for m_idx in 0..$num_rvs {
                            let ni = marginal_counts[m_idx][i];
                            sum_psi_ni += digamma(ni);
                        }
                        local_mi[i] = (digamma(self.k as f64) - 1.0 / (self.k as f64) - sum_psi_ni + ($num_rvs as f64 - 1.0) * digamma(n_samples as f64)) / ln_base;
                    }
                }
                local_mi
            }
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> GlobalValue for $name<D_JOINT, $($d_param),*> {
            fn global_value(&self) -> f64 {
                self.local_values().mean().unwrap_or(0.0)
            }
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> OptionalLocalValues for $name<D_JOINT, $($d_param),*> {
            fn supports_local(&self) -> bool { true }
            fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
                Ok(self.local_values())
            }
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> MutualInformationEstimator for $name<D_JOINT, $($d_param),*> {}

        impl<const D_JOINT: usize, $(const $d_param: usize),*> LocalValues for $name<D_JOINT, $($d_param),*> {
            fn local_values(&self) -> Array1<f64> {
                if self.use_chebyshev {
                    self.compute_local_mi_with_metric::<Chebyshev>()
                } else {
                    self.compute_local_mi_with_metric::<SquaredEuclidean>()
                }
            }
        }
    };
}

impl_ksg_mi!(KsgMutualInformation2, 2, (D1, D2), (0, 1));
impl_ksg_mi!(KsgMutualInformation3, 3, (D1, D2, D3), (0, 1, 2));
impl_ksg_mi!(KsgMutualInformation4, 4, (D1, D2, D3, D4), (0, 1, 2, 3));
impl_ksg_mi!(
    KsgMutualInformation5,
    5,
    (D1, D2, D3, D4, D5),
    (0, 1, 2, 3, 4)
);
impl_ksg_mi!(
    KsgMutualInformation6,
    6,
    (D1, D2, D3, D4, D5, D6),
    (0, 1, 2, 3, 4, 5)
);

/// KSG (kNN-based) conditional mutual information estimator.
///
/// ## Theory
///
/// The KSG estimator for CMI uses the following formula:
/// $$I(X; Y \mid Z) = \psi(k) + \langle \psi(n_z + 1) - \psi(n_{xz} + 1) - \psi(n_{yz} + 1) \rangle$$
///
/// where $n_z, n_{xz}, n_{yz}$ are neighbor counts in the respective subspaces defined
/// by the distance to the $k$-th neighbor in the joint $(X, Y, Z)$ space.
///
/// See the [Conditional MI Guide](crate::guide::cond_mi) for conceptual background.
pub struct KsgConditionalMutualInformation<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> {
    pub k: usize,
    pub ksg_type: KsgType,
    pub data: Vec<Array2<f64>>,
    pub cond: Array2<f64>,
    pub base: f64,
    pub noise_level: f64,
    pub use_chebyshev: bool,
}

impl<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> KsgConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
{
    pub fn new(series: &[Array2<f64>], cond: &Array2<f64>, k: usize, noise_level: f64) -> Self {
        assert_eq!(series.len(), 2, "CMI expects 2 random variables");
        let noisy_data = series
            .iter()
            .map(|s| add_noise(s.clone(), noise_level))
            .collect();
        let noisy_cond = add_noise(cond.clone(), noise_level);
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

    fn compute_local_cmi_with_metric<M>(&self) -> Array1<f64>
    where
        M: KsgMetric<f64, D_JOINT>
            + KsgMetric<f64, D1_COND>
            + KsgMetric<f64, D2_COND>
            + KsgMetric<f64, D_COND>
            + 'static,
    {
        let n_samples = self.data[0].nrows();
        // Joint: (X, Y, Z)
        let joint_all = concatenate(
            Axis(1),
            &[self.data[0].view(), self.data[1].view(), self.cond.view()],
        )
        .unwrap();

        let joint_points = NdDataset::<D_JOINT>::points_as_vec(joint_all);
        let joint_tree = kiddo::ImmutableKdTree::new_from_slice(&joint_points);

        let mut epsilons = Vec::with_capacity(n_samples);
        let max_qty = std::num::NonZeroUsize::new(self.k + 1).unwrap();
        for p in joint_points.iter().take(n_samples) {
            let neighbors = joint_tree.nearest_n::<M>(p, max_qty);
            let dist = neighbors[self.k].distance;
            let eps = if std::any::TypeId::of::<M>() == std::any::TypeId::of::<SquaredEuclidean>() {
                dist.sqrt()
            } else {
                dist
            };
            epsilons.push(eps);
        }

        // Marginal/Conditional spaces: (X, Z), (Y, Z), (Z)
        let xz = concatenate(Axis(1), &[self.data[0].view(), self.cond.view()]).unwrap();
        let yz = concatenate(Axis(1), &[self.data[1].view(), self.cond.view()]).unwrap();
        let z = self.cond.view();

        let xz_points = NdDataset::<D1_COND>::points_as_vec(xz);
        let yz_points = NdDataset::<D2_COND>::points_as_vec(yz);
        let z_points = NdDataset::<D_COND>::points_as_vec(z.to_owned());

        let xz_tree = kiddo::ImmutableKdTree::new_from_slice(&xz_points);
        let yz_tree = kiddo::ImmutableKdTree::new_from_slice(&yz_points);
        let z_tree = kiddo::ImmutableKdTree::new_from_slice(&z_points);

        let mut local_cmi = Array1::zeros(n_samples);
        let ln_base = self.base.ln();

        for i in 0..n_samples {
            let eps = epsilons[i];

            let (count_xz, count_yz, count_z) = if self.ksg_type == KsgType::Type1 {
                if eps > 0.0 {
                    let p_xz = &xz_points[i];
                    let p_yz = &yz_points[i];
                    let p_z = &z_points[i];

                    // Algorithm 1 uses strict inequality (dist < eps)
                    // Python: query_ball_point(r=nextafter(eps, -inf)) - (eps > 0 ? 1 : 0)
                    let c_xz = if self.use_chebyshev {
                        xz_tree
                            .within_exclusive::<Chebyshev>(p_xz, eps, false)
                            .len()
                            - 1
                    } else {
                        xz_tree
                            .within_exclusive::<SquaredEuclidean>(p_xz, eps.powi(2), false)
                            .len()
                            - 1
                    };

                    let c_yz = if self.use_chebyshev {
                        yz_tree
                            .within_exclusive::<Chebyshev>(p_yz, eps, false)
                            .len()
                            - 1
                    } else {
                        yz_tree
                            .within_exclusive::<SquaredEuclidean>(p_yz, eps.powi(2), false)
                            .len()
                            - 1
                    };

                    let c_z = if self.use_chebyshev {
                        z_tree.within_exclusive::<Chebyshev>(p_z, eps, false).len() - 1
                    } else {
                        z_tree
                            .within_exclusive::<SquaredEuclidean>(p_z, eps.powi(2), false)
                            .len()
                            - 1
                    };

                    (c_xz as i32, c_yz as i32, c_z as i32)
                } else {
                    (0, 0, 0)
                }
            } else {
                let p_xz = &xz_points[i];
                let p_yz = &yz_points[i];
                let p_z = &z_points[i];

                // Algorithm 2 uses inclusive inequality (distance <= eps).
                // Python: query_ball_point(..., r=eps, p=inf, ...)
                let c_xz = if self.use_chebyshev {
                    xz_tree.within::<Chebyshev>(p_xz, eps).len()
                } else {
                    xz_tree.within::<SquaredEuclidean>(p_xz, eps.powi(2)).len()
                };

                let c_yz = if self.use_chebyshev {
                    yz_tree.within::<Chebyshev>(p_yz, eps).len()
                } else {
                    yz_tree.within::<SquaredEuclidean>(p_yz, eps.powi(2)).len()
                };

                let c_z = if self.use_chebyshev {
                    z_tree.within::<Chebyshev>(p_z, eps).len()
                } else {
                    z_tree.within::<SquaredEuclidean>(p_z, eps.powi(2)).len()
                };

                (c_xz as i32, c_yz as i32, c_z as i32)
            };

            let (cxz, cyz, cz) = (count_xz, count_yz, count_z);

            if self.ksg_type == KsgType::Type1 {
                // local_cmi = digamma(k) + [digamma(cz + 1) - sum(digamma(c + 1) for c in counts)]
                local_cmi[i] = (digamma(self.k as f64) + digamma(cz as f64 + 1.0)
                    - digamma(cxz as f64 + 1.0)
                    - digamma(cyz as f64 + 1.0))
                    / ln_base;
            } else {
                // local_cmi = digamma(k) - 1.0/k + [digamma(cz) - sum(digamma(c) for c in counts)]
                local_cmi[i] = (digamma(self.k as f64) - 1.0 / (self.k as f64)
                    + digamma(cz as f64)
                    - digamma(cxz as f64)
                    - digamma(cyz as f64))
                    / ln_base;
            }
        }
        local_cmi
    }
}

impl<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> GlobalValue for KsgConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
{
    fn global_value(&self) -> f64 {
        self.local_values().mean().unwrap_or(0.0)
    }
}

impl<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> OptionalLocalValues
    for KsgConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
{
    fn supports_local(&self) -> bool {
        true
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        Ok(self.local_values())
    }
}

impl<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> crate::estimators::traits::ConditionalMutualInformationEstimator
    for KsgConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
{
}

impl<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> LocalValues for KsgConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
{
    fn local_values(&self) -> Array1<f64> {
        if self.use_chebyshev {
            self.compute_local_cmi_with_metric::<Chebyshev>()
        } else {
            self.compute_local_cmi_with_metric::<SquaredEuclidean>()
        }
    }
}

/// KSG-based transfer entropy estimator.
///
/// ## Theory
///
/// Transfer entropy is estimated as a conditional mutual information $I(Y_{future}; X_{past} | Y_{past})$:
/// $$T_{X \to Y} = \psi(k) + \langle \psi(n_{Y_{past}} + 1) - \psi(n_{Y_{future}, Y_{past}} + 1) - \psi(n_{Y_{past}, X_{past}} + 1) \rangle$$
///
/// See the [Transfer Entropy Guide](crate::guide::transfer_entropy) for conceptual background.
pub struct KsgTransferEntropy<
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
    pub internal_cmi:
        KsgConditionalMutualInformation<D_SOURCE, D_TARGET, D_YP, D_JOINT, D_XP_YP, D_YF_YP>,
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
    KsgTransferEntropy<
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
        >(source, dest, false);

        let cmi = KsgConditionalMutualInformation::new(&[xp, yf], &yp, k, noise_level);
        Self { internal_cmi: cmi }
    }

    pub fn with_type(mut self, ksg_type: KsgType) -> Self {
        self.internal_cmi = self.internal_cmi.with_type(ksg_type);
        self
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.internal_cmi = self.internal_cmi.with_base(base);
        self
    }

    pub fn with_chebyshev(mut self, use_chebyshev: bool) -> Self {
        self.internal_cmi = self.internal_cmi.with_chebyshev(use_chebyshev);
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
    for KsgTransferEntropy<
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
        self.internal_cmi.global_value()
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
    for KsgTransferEntropy<
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
        true
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        self.internal_cmi.local_values_opt()
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
> LocalValues
    for KsgTransferEntropy<
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
    fn local_values(&self) -> Array1<f64> {
        self.internal_cmi.local_values()
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
    for KsgTransferEntropy<
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

/// KSG-based conditional transfer entropy estimator.
///
/// ## Theory
///
/// Conditional transfer entropy is estimated as:
/// $$TE(X \to Y \mid Z) = \psi(k) + \langle \psi(n_{Y_{past}, Z_{past}} + 1) - \psi(n_{Y_{future}, Y_{past}, Z_{past}} + 1) - \psi(n_{X_{past}, Y_{past}, Z_{past}} + 1) \rangle$$
///
/// See the [Conditional TE Guide](crate::guide::cond_te) for conceptual background.
pub struct KsgConditionalTransferEntropy<
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
    pub internal_cmi: KsgConditionalMutualInformation<
        D_SOURCE,
        D_TARGET,
        D_YP_ZP,
        D_JOINT,
        D_XP_YP_ZP,
        D_YF_YP_ZP,
    >,
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
    KsgConditionalTransferEntropy<
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
        >(source, dest, cond, false);

        let yp_zp = concatenate(Axis(1), &[yp.view(), zp.view()]).unwrap();

        let cmi = KsgConditionalMutualInformation::new(&[xp, yf], &yp_zp, k, noise_level);
        Self { internal_cmi: cmi }
    }

    pub fn with_type(mut self, ksg_type: KsgType) -> Self {
        self.internal_cmi = self.internal_cmi.with_type(ksg_type);
        self
    }

    pub fn with_base(mut self, base: f64) -> Self {
        self.internal_cmi = self.internal_cmi.with_base(base);
        self
    }

    pub fn with_chebyshev(mut self, use_chebyshev: bool) -> Self {
        self.internal_cmi = self.internal_cmi.with_chebyshev(use_chebyshev);
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
    for KsgConditionalTransferEntropy<
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
        self.internal_cmi.global_value()
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
    for KsgConditionalTransferEntropy<
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
        true
    }
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str> {
        self.internal_cmi.local_values_opt()
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
> LocalValues
    for KsgConditionalTransferEntropy<
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
    fn local_values(&self) -> Array1<f64> {
        self.internal_cmi.local_values()
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
    for KsgConditionalTransferEntropy<
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

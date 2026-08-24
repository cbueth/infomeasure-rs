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
//! $$I(X; Y | Z) = \psi(k) + \langle \psi(n_{z}(i) + 1) - \psi(n_{xz}(i) + 1) - \psi(n_{yz}(i) + 1) \rangle$$
//!
//! ### Transfer Entropy (TE)
//! TE is estimated as a conditional mutual information $I(Y_{\mathrm{future}}; X_{\mathrm{past}} | Y_{\mathrm{past}})$:
//! $$T_{X \to Y} = \psi(k) + \langle \psi(n_{Y_{\mathrm{past}}} + 1) - \psi(n_{Y_{\mathrm{future}}, Y_{\mathrm{past}}} + 1) - \psi(n_{Y_{\mathrm{past}}, X_{\mathrm{past}}} + 1) \rangle$$
//!
//! ## See Also
//! - [Mutual Information Guide](crate::guide::mutual_information) — Conceptual background
//! - [Transfer Entropy Guide](crate::guide::transfer_entropy) — Directed information flow
//! - [Kozachenko-Leonenko](super::kozachenko_leonenko) — kNN-based entropy
//!
//! ## References
//!
//! - [Kraskov et al., 2004](crate::guide::references#ksg2004)
//! - [Frenzel & Pompe, 2007](crate::guide::references#frenzel2007)

use kiddo::{Chebyshev, SquaredEuclidean};
use ndarray::{Array1, Array2, Axis, concatenate};
use statrs::function::gamma::digamma;

pub use super::utils::KsgType;
use super::utils::add_noise;
use crate::estimators::approaches::common_nd::KdTreeExpfam;
use crate::estimators::approaches::common_nd::dataset::NdDataset;
use crate::estimators::traits::{
    ConditionalTransferEntropyEstimator, GlobalValue, LocalValues, MutualInformationEstimator,
    OptionalLocalValues, TransferEntropyEstimator,
};
use crate::estimators::utils::te_slicing::{cte_observations_const, te_observations_const};

/// Counts points within `eps` of `query` in `tree` (Chebyshev or squared-Euclidean,
/// matching `use_chebyshev`), using strict-exclusive boundaries when `exclusive`.
///
/// Count-only: uses kiddo's public `.visit()` so no result `Vec` is materialised —
/// the KSG marginal/conditional neighbour counting only needs the count.
/// Reference ball counter kept as the test oracle for [`SortedSpace`]:
/// production paths all go through the sorted-window scan.
#[cfg(test)]
use kiddo::{Donnelly, QueryScratch};

#[cfg(test)]
fn count_neighbors_within<const K: usize>(
    tree: &KdTreeExpfam<K>,
    query: &[f64; K],
    eps: f64,
    use_chebyshev: bool,
    exclusive: bool,
    scratch: &mut QueryScratch<Donnelly<3>, f64, K>,
) -> usize {
    let mut count = 0usize;
    if use_chebyshev {
        if exclusive {
            tree.query(query)
                .within::<Chebyshev<f64>>(eps)
                .exclusive_boundaries()
                .unsorted()
                .with_scratch(scratch)
                .visit(|_| count += 1);
        } else {
            tree.query(query)
                .within::<Chebyshev<f64>>(eps)
                .unsorted()
                .with_scratch(scratch)
                .visit(|_| count += 1);
        }
    } else if exclusive {
        tree.query(query)
            .within::<SquaredEuclidean<f64>>(eps.powi(2))
            .exclusive_boundaries()
            .unsorted()
            .with_scratch(scratch)
            .visit(|_| count += 1);
    } else {
        tree.query(query)
            .within::<SquaredEuclidean<f64>>(eps.powi(2))
            .unsorted()
            .with_scratch(scratch)
            .visit(|_| count += 1);
    }
    count
}

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
        /// $$I(X_1; \ldots; X_m) = \psi(k) + (m-1)\psi(N) - \left\langle \sum_{j=1}^m \psi(n_{j} + 1) \right\rangle$$
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

            fn compute_local_mi(&self) -> Array1<f64>
            {
                let n_samples = self.data[0].nrows();
                let joint_data = concatenate(
                    Axis(1),
                    &self.data.iter().map(|d| d.view()).collect::<Vec<_>>(),
                ).unwrap();

                // 1. Find k-th neighbor distance in joint space
                let joint_points = NdDataset::<D_JOINT>::points_as_vec(joint_data);
                let joint_tree = KdTreeExpfam::<D_JOINT>::new_from_slice(&joint_points).unwrap();

                let mut epsilons = Vec::with_capacity(n_samples);
                let max_qty = std::num::NonZeroUsize::new(self.k + 1).unwrap();
                if self.use_chebyshev {
                    let mut scratch = joint_tree.create_scratch::<Chebyshev<f64>>();
                    for i in 0..n_samples {
                        let p = &joint_points[i];
                        let neighbors = joint_tree
                            .query(p)
                            .nearest_n::<Chebyshev<f64>>(max_qty)
                            .with_scratch(&mut scratch)
                            .execute();
                        let dist = neighbors[self.k].distance;
                        epsilons.push(dist);
                    }
                } else {
                    let mut scratch = joint_tree.create_scratch::<SquaredEuclidean<f64>>();
                    for i in 0..n_samples {
                        let p = &joint_points[i];
                        let neighbors = joint_tree
                            .query(p)
                            .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                            .with_scratch(&mut scratch)
                            .execute();
                        let dist = neighbors[self.k].distance;
                        epsilons.push(dist.sqrt());
                    }
                }

                // 2. Count neighbours in marginal spaces within epsilon
                let mut marginal_counts = Vec::new();
                $(
                    let m_data = self.data[$d_idx].view();
                    let m_points = NdDataset::<$d_param>::points_as_vec(m_data.to_owned());
                    let m_sorted = SortedSpace::new(m_points.clone());

                    let mut counts = Vec::with_capacity(n_samples);
                    for i in 0..n_samples {
                        let p = &m_points[i];
                        let eps = epsilons[i];

                        let count = if self.ksg_type == KsgType::Type1 {
                            // Type 1 uses strict inequality: dist < eps
                            // Python uses: query_ball_point(r=nextafter(eps, -inf)) - (eps > 0 ? 1 : 0)
                            if eps > 0.0 {
                                let raw = m_sorted.count_within(p, eps, self.use_chebyshev, true);
                                // Subtract 1 to exclude the point itself (same as Python)
                                raw - 1
                            } else {
                                0
                            }
                        } else {
                            // Type 2 uses inclusive inequality: dist <= eps
                            m_sorted.count_within(p, eps, self.use_chebyshev, false)
                        };

                        counts.push(count as f64);
                    }
                    marginal_counts.push(counts);
                )*

                let mut local_mi = Array1::zeros(n_samples);
                let ln_base = self.base.ln();
                let digamma_k = digamma(self.k as f64);
                let inv_ln_base = 1.0 / ln_base;
                let inv_k = 1.0 / (self.k as f64);
                let term_n = ($num_rvs as f64 - 1.0) * digamma(n_samples as f64);

                for i in 0..n_samples {
                    if self.ksg_type == KsgType::Type1 {
                        let mut sum_psi_ni_plus_1 = 0.0;
                        for m_idx in 0..$num_rvs {
                            let ni = marginal_counts[m_idx][i];
                            sum_psi_ni_plus_1 += digamma(ni + 1.0);
                        }
                        // Type I: I = psi(k) - <sum psi(ni+1)> + (m-1)psi(N)
                        local_mi[i] = (digamma_k - sum_psi_ni_plus_1 + term_n) * inv_ln_base;
                    } else {
                        // Type II: I = psi(k) - 1/k - <sum psi(ni)> + (m-1)psi(N)
                        let mut sum_psi_ni = 0.0;
                        for m_idx in 0..$num_rvs {
                            let ni = marginal_counts[m_idx][i];
                            sum_psi_ni += digamma(ni);
                        }
                        local_mi[i] = (digamma_k - inv_k - sum_psi_ni + term_n) * inv_ln_base;
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
                self.compute_local_mi()
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

/// Smallest representable `f64` strictly greater than `x`.
fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == -0.0 {
        return 0.0;
    }
    let bits = x.to_bits();
    if x >= 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// A point cloud sorted once by its first coordinate, turning ball counting
/// into a window scan instead of a per-query kd-tree traversal.
///
/// Any Chebyshev or squared-Euclidean ball of radius `eps` around a query is
/// contained in the axis-0 slab $\left[q_0 - \varepsilon, q_0 +
/// \varepsilon\right]$ (widened by one ulp per side), so scanning that slab
/// and filtering with the same floating-point expressions kiddo evaluates
/// yields identical integer counts at every dimensionality.
struct SortedSpace<const D: usize> {
    /// Points ascending by `[0]` (`total_cmp`; input contains no NaNs).
    points: Vec<[f64; D]>,
}

impl<const D: usize> SortedSpace<D> {
    fn new(mut points: Vec<[f64; D]>) -> Self {
        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        Self { points }
    }

    /// Raw neighbour count within `eps` of `query`, self included, matching
    /// [`count_neighbors_within`] bit-for-bit for both metrics and both
    /// boundary modes.
    ///
    /// The scan range is delimited by partitioning on the same rounded
    /// per-axis subtraction kiddo evaluates, widened by two ulps of `eps`.
    /// Widening the *radius* instead of materialising interval bounds
    /// $\left[q_0 \pm \varepsilon\right]$ sidesteps catastrophic cancellation
    /// when $q_0 \approx \mp\varepsilon$, where a naively rounded sum would
    /// truncate the slab near zero.
    fn count_within(
        &self,
        query: &[f64; D],
        eps: f64,
        use_chebyshev: bool,
        exclusive: bool,
    ) -> usize {
        let slack = next_up(next_up(eps));
        let q0 = query[0];
        // Prefix predicates are monotone because rounding is monotone.
        let start = self.points.partition_point(|p| q0 - p[0] > slack);
        let end = start + self.points[start..].partition_point(|p| p[0] - q0 <= slack);

        self.points[start..end]
            .iter()
            .filter(|p| {
                if use_chebyshev {
                    let d = p
                        .iter()
                        .zip(query.iter())
                        .map(|(pi, qi)| (pi - qi).abs())
                        .fold(0.0_f64, f64::max);
                    if exclusive { d < eps } else { d <= eps }
                } else {
                    let mut d2 = 0.0;
                    for (pi, qi) in p.iter().zip(query.iter()) {
                        d2 += (pi - qi) * (pi - qi);
                    }
                    let e2 = eps * eps;
                    if exclusive { d2 < e2 } else { d2 <= e2 }
                }
            })
            .count()
    }
}

/// KSG (kNN-based) conditional mutual information estimator.
///
/// ## Theory
///
/// The KSG estimator for CMI uses the following formula:
/// $$I(X; Y \mid Z) = \psi(k) + \langle \psi(n_{z} + 1) - \psi(n_{xz} + 1) - \psi(n_{yz} + 1) \rangle$$
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

    fn compute_local_cmi(&self) -> Array1<f64> {
        let n_samples = self.data[0].nrows();
        // Joint: (X, Y, Z)
        let joint_all = concatenate(
            Axis(1),
            &[self.data[0].view(), self.data[1].view(), self.cond.view()],
        )
        .unwrap();

        let joint_points = NdDataset::<D_JOINT>::points_as_vec(joint_all);
        let joint_tree = KdTreeExpfam::<D_JOINT>::new_from_slice(&joint_points).unwrap();

        let mut epsilons = Vec::with_capacity(n_samples);
        let max_qty = std::num::NonZeroUsize::new(self.k + 1).unwrap();
        if self.use_chebyshev {
            let mut scratch = joint_tree.create_scratch::<Chebyshev<f64>>();
            for p in joint_points.iter().take(n_samples) {
                let neighbors = joint_tree
                    .query(p)
                    .nearest_n::<Chebyshev<f64>>(max_qty)
                    .with_scratch(&mut scratch)
                    .execute();
                epsilons.push(neighbors[self.k].distance);
            }
        } else {
            let mut scratch = joint_tree.create_scratch::<SquaredEuclidean<f64>>();
            for p in joint_points.iter().take(n_samples) {
                let neighbors = joint_tree
                    .query(p)
                    .nearest_n::<SquaredEuclidean<f64>>(max_qty)
                    .with_scratch(&mut scratch)
                    .execute();
                epsilons.push(neighbors[self.k].distance.sqrt());
            }
        }

        // Marginal/Conditional spaces: (X, Z), (Y, Z), (Z)
        let xz = concatenate(Axis(1), &[self.data[0].view(), self.cond.view()]).unwrap();
        let yz = concatenate(Axis(1), &[self.data[1].view(), self.cond.view()]).unwrap();
        let z = self.cond.view();

        let xz_points = NdDataset::<D1_COND>::points_as_vec(xz);
        let yz_points = NdDataset::<D2_COND>::points_as_vec(yz);
        let z_points = NdDataset::<D_COND>::points_as_vec(z.to_owned());

        let xz_sorted = SortedSpace::new(xz_points.clone());
        let yz_sorted = SortedSpace::new(yz_points.clone());
        let z_sorted = SortedSpace::new(z_points.clone());

        let mut local_cmi = Array1::zeros(n_samples);
        let ln_base = self.base.ln();
        let digamma_k = digamma(self.k as f64);
        let inv_ln_base = 1.0 / ln_base;
        let inv_k = 1.0 / (self.k as f64);

        for i in 0..n_samples {
            let eps = epsilons[i];

            let (count_xz, count_yz, count_z) = if self.ksg_type == KsgType::Type1 {
                // Algorithm 1 uses strict inequality (dist < eps)
                // Python: query_ball_point(r=nextafter(eps, -inf)) - (eps > 0 ? 1 : 0)
                if eps > 0.0 {
                    let raw_xz =
                        xz_sorted.count_within(&xz_points[i], eps, self.use_chebyshev, true);
                    let raw_yz =
                        yz_sorted.count_within(&yz_points[i], eps, self.use_chebyshev, true);
                    let raw_z = z_sorted.count_within(&z_points[i], eps, self.use_chebyshev, true);
                    (raw_xz as i32 - 1, raw_yz as i32 - 1, raw_z as i32 - 1)
                } else {
                    (0, 0, 0)
                }
            } else {
                // Algorithm 2 uses inclusive inequality (distance <= eps).
                // Python: query_ball_point(..., r=eps, p=inf, ...)
                let raw_xz = xz_sorted.count_within(&xz_points[i], eps, self.use_chebyshev, false);
                let raw_yz = yz_sorted.count_within(&yz_points[i], eps, self.use_chebyshev, false);
                let raw_z = z_sorted.count_within(&z_points[i], eps, self.use_chebyshev, false);
                (raw_xz as i32, raw_yz as i32, raw_z as i32)
            };

            let (cxz, cyz, cz) = (count_xz, count_yz, count_z);

            if self.ksg_type == KsgType::Type1 {
                // local_cmi = digamma(k) + [digamma(cz + 1) - sum(digamma(c + 1) for c in counts)]
                local_cmi[i] = (digamma_k + digamma(cz as f64 + 1.0)
                    - digamma(cxz as f64 + 1.0)
                    - digamma(cyz as f64 + 1.0))
                    * inv_ln_base;
            } else {
                // local_cmi = digamma(k) - 1.0/k + [digamma(cz) - sum(digamma(c) for c in counts)]
                local_cmi[i] = (digamma_k - inv_k + digamma(cz as f64)
                    - digamma(cxz as f64)
                    - digamma(cyz as f64))
                    * inv_ln_base;
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
        self.compute_local_cmi()
    }
}

/// KSG-based transfer entropy estimator.
///
/// ## Theory
///
/// Transfer entropy is estimated as a conditional mutual information $I(Y_{\mathrm{future}}; X_{\mathrm{past}} | Y_{\mathrm{past}})$:
/// $$T_{X \to Y} = \psi(k) + \langle \psi(n_{Y_{\mathrm{past}}} + 1) - \psi(n_{Y_{\mathrm{future}}, Y_{\mathrm{past}}} + 1) - \psi(n_{Y_{\mathrm{past}}, X_{\mathrm{past}}} + 1) \rangle$$
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
/// $$\mathrm{TE}(X \to Y \mid Z) = \psi(k) + \langle \psi(n_{Y_{\mathrm{past}}, Z_{\mathrm{past}}} + 1) - \psi(n_{Y_{\mathrm{future}}, Y_{\mathrm{past}}, Z_{\mathrm{past}}} + 1) - \psi(n_{X_{\mathrm{past}}, Y_{\mathrm{past}}, Z_{\mathrm{past}}} + 1) \rangle$$
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

#[cfg(test)]
mod ksg_count_tests {
    #![allow(unused_imports)]
    use super::*;
    use ndarray::array;
    use rstest::rstest;

    /// Reference implementation with the same floating-point expressions the
    /// kiddo queries evaluate: per-dimension absolute differences folded into
    /// Chebyshev max or squared-Euclidean sum, compared against `eps`.
    fn brute_force_count<const D: usize>(
        points: &[[f64; D]],
        query: &[f64; D],
        eps: f64,
        use_chebyshev: bool,
        exclusive: bool,
    ) -> usize {
        points
            .iter()
            .filter(|p| {
                if use_chebyshev {
                    let d = p
                        .iter()
                        .zip(query)
                        .map(|(pi, qi)| (pi - qi).abs())
                        .fold(0.0_f64, f64::max);
                    if exclusive { d < eps } else { d <= eps }
                } else {
                    let mut d2 = 0.0;
                    for (pi, qi) in p.iter().zip(query) {
                        d2 += (pi - qi) * (pi - qi);
                    }
                    let e2 = eps * eps;
                    if exclusive { d2 < e2 } else { d2 <= e2 }
                }
            })
            .count()
    }

    fn seeded_points(n: usize, dim: usize, seed: u64, duplicates: bool) -> Array2<f64> {
        use rand::Rng;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut data = Array2::<f64>::zeros((n, dim));
        for v in data.iter_mut() {
            *v = if duplicates {
                // Coarse grid forces many exact coordinate collisions.
                let r: f64 = rng.gen_range(0.0..10.0);
                (r.floor()) / 4.0
            } else {
                rng.gen_range(-5.0..5.0)
            };
        }
        data
    }

    fn check_sorted_space<const D: usize>(n: usize, seed: u64, duplicates: bool) {
        let data = seeded_points(n, D, seed, duplicates);
        let points = NdDataset::<D>::points_as_vec(data);
        let tree = KdTreeExpfam::<D>::new_from_slice(&points).unwrap();
        let sorted = SortedSpace::new(points.clone());
        let mut scratch = Default::default();

        for (i, q) in points.iter().enumerate() {
            let eps = 0.37 + 0.11 * i as f64;
            for use_chebyshev in [true, false] {
                for exclusive in [true, false] {
                    let want = brute_force_count(&points, q, eps, use_chebyshev, exclusive);
                    let got_sorted = sorted.count_within(q, eps, use_chebyshev, exclusive);
                    let got_kiddo = count_neighbors_within(
                        &tree,
                        q,
                        eps,
                        use_chebyshev,
                        exclusive,
                        &mut scratch,
                    );
                    assert_eq!(
                        got_sorted, want,
                        "brute mismatch i={i} cheb={use_chebyshev} excl={exclusive}"
                    );
                    assert_eq!(
                        got_sorted, got_kiddo,
                        "kiddo mismatch i={i} cheb={use_chebyshev} excl={exclusive}"
                    );
                }
            }
        }
    }

    #[rstest]
    fn sorted_space_d1_matches_brute_force_and_kiddo(#[values(true, false)] duplicates: bool) {
        check_sorted_space::<1>(64, 7, duplicates);
    }

    #[rstest]
    fn sorted_space_d3_matches_brute_force_and_kiddo(#[values(true, false)] duplicates: bool) {
        check_sorted_space::<3>(48, 11, duplicates);
    }

    /// Points sitting *exactly* at distance `eps` must move together across the
    /// strict/inclusive boundary — the tie scenario the Python parity suite
    /// stresses, reproduced here at the counting layer.
    #[test]
    fn boundary_ties_flip_only_at_inclusive() {
        let raw = array![[0.0], [0.25], [-0.5], [0.75]];
        let points = NdDataset::<1>::points_as_vec(raw);
        let sorted = SortedSpace::new(points.clone());

        // Query at origin, eps = 0.25: only the self point and +0.25 qualify.
        assert_eq!(sorted.count_within(&[0.0], 0.25, true, true), 1);
        assert_eq!(sorted.count_within(&[0.0], 0.25, true, false), 2);
    }

    /// `eps == 0` with inclusive boundaries still counts coincident points
    /// (distance 0 ≤ 0); exclusive counts nothing, matching Type-1's early
    /// `eps > 0` guard upstream.
    #[test]
    fn zero_epsilon_counts_coincident_points_when_inclusive() {
        let raw = array![[1.5], [1.5], [1.5], [2.0]];
        let points = NdDataset::<1>::points_as_vec(raw);
        let sorted = SortedSpace::new(points.clone());

        assert_eq!(sorted.count_within(&[1.5], 0.0, true, true), 0);
        assert_eq!(sorted.count_within(&[1.5], 0.0, true, false), 3);
    }

    /// When $q_0 = -\varepsilon$ the sum $q_0 + \varepsilon$ cancels to zero,
    /// so a slab built from rounded interval bounds would truncate near zero
    /// and drop neighbours kiddo counts. Radius-slack partitioning must keep
    /// them.
    #[test]
    fn window_survives_catastrophic_cancellation() {
        let raw = array![[-0.1], [1e-300], [5e-17], [0.5]];
        let points = NdDataset::<1>::points_as_vec(raw);
        let sorted = SortedSpace::new(points.clone());

        // Query sits at exactly -eps: the far edge of its ball is ~0.
        // Self and 1e-300 are within rounded distance eps; 5e-17 already
        // lands ~4 ulps above eps after rounding, so it is out for kiddo too.
        assert_eq!(sorted.count_within(&[-0.1], 0.1, true, false), 2);
        assert_eq!(sorted.count_within(&[-0.1], 0.1, true, true), 1);
    }
}

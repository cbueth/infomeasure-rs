// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Kernel Density Estimation for Entropy Calculation
//!
//! This module implements entropy estimation using kernel density estimation (KDE) techniques.
//! KDE is a non-parametric way to estimate the probability density function of a random variable,
//! which is then used to calculate differential entropy.
//!
//! ## Theoretical Background
//!
//! The differential entropy of a continuous random variable X is defined as:
//!
//! H(X) = -∫ f(x) log(f(x)) dx
//!
//! where f(x) is the probability density function (PDF) of X.
//!
//! In practice, we don't know the true PDF, so we estimate it using kernel density estimation:
//!
//! f̂(x) = (1/Nh^d) ∑ K((x - x_i)/h)
//!
//! where:
//! - N is the number of data points
//! - h is the bandwidth parameter
//! - d is the dimensionality of the data
//! - K is the kernel function
//! - x_i are the observed data points
//!
//! The entropy is then estimated as:
//!
//! Ĥ(X) = -(1/N) ∑ log(f̂(x_i))
//!
//! ## Supported Kernel Types
//!
//! This implementation supports two types of kernels:
//!
//! 1. **Box Kernel**: A uniform kernel where all points within a certain distance contribute equally.
//!    Simple and computationally efficient, but can produce discontinuities.
//!
//! 2. **Gaussian Kernel**: A smooth kernel based on the normal distribution. Provides smoother
//!    density estimates but is more computationally intensive. The Gaussian kernel implementation
//!    now uses the full covariance matrix of the data, matching the behavior of scipy.stats.gaussian_kde
//!    exactly. This accounts for correlations between dimensions.
//!
//! ## Bandwidth Selection
//!
//! The bandwidth parameter (h) controls the smoothness of the density estimate:
//! - Small bandwidth: More detail but potentially noisy (high variance)
//! - Large bandwidth: Smoother but potentially over-smoothed (high bias)
//!
//! Choosing an appropriate bandwidth is crucial for accurate entropy estimation.
//!
//! ## Implementation Details
//!
//! This implementation uses a KD-tree for efficient nearest-neighbor queries, making it
//! Suitable for large datasets. The Gaussian kernel implementation includes full covariance
//! handling, proper dimension-dependent normalization, and bandwidth scaling to match
//! the behavior of scipy.stats.gaussian_kde.
//!
//! ## GPU Acceleration
//!
//! When compiled with the `gpu` feature flag, this implementation can use GPU
//! acceleration for both Gaussian and Box kernel calculations, providing significant
//! performance improvements for large datasets:
//!
//! - **Gaussian Kernel**: GPU acceleration is used for datasets with 500 or more points,
//!   providing speedups of up to 340x for large datasets. The adaptive radius for neighbor
//!   search is larger when using GPU acceleration, especially for small bandwidths.
//!
//! - **Box Kernel**: GPU acceleration is used for datasets with 2000 or more points,
//!   providing speedups of up to 37x for large datasets. For smaller datasets, the CPU
//!   implementation is faster due to the overhead of GPU setup.

use crate::estimators::approaches::common_nd::KdTreeKernel;
use crate::estimators::doc_macros::doc_snippets;
use crate::estimators::traits::{
    ConditionalMutualInformationEstimator, ConditionalTransferEntropyEstimator,
    MutualInformationEstimator, TransferEntropyEstimator,
};
use crate::estimators::traits::{
    CrossEntropy, GlobalValue, JointEntropy, LocalValues, OptionalLocalValues,
};
use crate::estimators::utils::te_slicing::{cte_observations_const, te_observations_const};
use kiddo::{Chebyshev, SquaredEuclidean};
use ndarray::{Array1, Array2, Axis, concatenate};
use ndarray_linalg::{Cholesky, Inverse, UPLO};
use ndarray_stats::CorrelationExt;

/// Kernel-based transfer entropy estimator.
///
/// $TE(X \to Y) = I(Y\_{\mathrm{future}}; X\_{\mathrm{past}} | Y\_{\mathrm{past}})$
///
/// # Const Generics
/// - `SRC_HIST`: Number of past source observations to include.
/// - `DEST_HIST`: Number of past destination observations to include.
/// - `STEP_SIZE`: Delay between observations.
/// - `D_SOURCE`: Dimensionality of source variable.
/// - `D_TARGET`: Dimensionality of destination variable.
/// - `D_JOINT`: $D_{\mathrm{target}} + (\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{source}}) + (\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{target}})$
/// - `D_XP_YP`: $(\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{source}}) + (\mathrm{DEST}\_\mathrm{HIST} \times D_{\mathrm{target}})$
/// - `D_YP`: $\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{target}}$
/// - `D_YF_YP`: $D_{\mathrm{target}} + (\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{target}})$
///
/// Kernel-based transfer entropy estimator.
///
/// ## Theory
///
#[doc = doc_snippets!(te_formula "Kernel-based", "", "")]
///
/// # Note on Dimensions
/// These dimensions must satisfy the mathematical relations defined by the slicing logic.
/// It is recommended to use the `new_kernel_te!` macro to instantiate this struct,
/// as it handles the dimension calculations automatically.
pub struct KernelTransferEntropy<
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
    pub kernel_type: String,
    pub bandwidth: f64,
    pub source: Array2<f64>,
    pub dest: Array2<f64>,
    pub force_cpu: bool,
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
    KernelTransferEntropy<
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
    /// Creates a new `KernelTransferEntropy` estimator.
    ///
    /// # Arguments
    /// * `source`: Source time series.
    /// * `dest`: Destination time series.
    /// * `_src_hist_len`: Source history length (unused, rely on const generic).
    /// * `_dest_hist_len`: Destination history length (unused, rely on const generic).
    /// * `_step_size`: Step size (unused, rely on const generic).
    /// * `kernel_type`: "box" or "gaussian".
    /// * `bandwidth`: Kernel bandwidth.
    pub fn new(
        source: &Array2<f64>,
        dest: &Array2<f64>,
        _src_hist_len: usize,
        _dest_hist_len: usize,
        _step_size: usize,
        kernel_type: String,
        bandwidth: f64,
    ) -> Self {
        Self {
            kernel_type,
            bandwidth,
            source: source.clone(),
            dest: dest.clone(),
            force_cpu: false,
        }
    }

    /// Sets whether to force CPU implementation even if GPU support is available.
    pub fn set_force_cpu(&mut self, force_cpu: bool) {
        self.force_cpu = force_cpu;
    }

    /// Helper method to compute density for a multi-dimensional dataset.
    fn compute_density<const D: usize>(&self, data: Array2<f64>) -> Array1<f64> {
        let mut est = KernelEntropy::<D>::new_with_kernel_type(
            data,
            self.kernel_type.clone(),
            self.bandwidth,
        );
        est.set_force_cpu(self.force_cpu);
        est.kde_probability_density()
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
    for KernelTransferEntropy<
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
        self.local_values().mean().unwrap_or(0.0)
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
    for KernelTransferEntropy<
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
        Ok(self.local_values())
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
    for KernelTransferEntropy<
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
    for KernelTransferEntropy<
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
    /// Computes local transfer entropy values.
    fn local_values(&self) -> Array1<f64> {
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
        >(&self.source, &self.dest, false);

        let joint_all = concatenate(Axis(1), &[yf.view(), xp.view(), yp.view()]).unwrap();
        let xp_yp = concatenate(Axis(1), &[xp.view(), yp.view()]).unwrap();
        let yf_yp = concatenate(Axis(1), &[yf.view(), yp.view()]).unwrap();

        let p_joint_all = self.compute_density::<D_JOINT>(joint_all);
        let p_yp = self.compute_density::<D_YP>(yp);
        let p_xp_yp = self.compute_density::<D_XP_YP>(xp_yp);
        let p_yf_yp = self.compute_density::<D_YF_YP>(yf_yp);

        let n = p_joint_all.len();
        let mut local_te = Array1::zeros(n);
        for i in 0..n {
            let num = p_joint_all[i] * p_yp[i];
            let den = p_xp_yp[i] * p_yf_yp[i];
            if num > 0.0 && den > 0.0 {
                local_te[i] = (num / den).ln();
            }
        }
        local_te
    }
}

/// Kernel-based conditional transfer entropy estimator.
///
/// $CTE(X \to Y | Z) = I(Y\_{\\mathrm{future}}; X\_{\\mathrm{past}} | Y\_{\\mathrm{past}}, Z\_{\\mathrm{past}})$
///
/// # Const Generics
/// - `SRC_HIST`, `DEST_HIST`, `COND_HIST`: History lengths.
/// - `STEP_SIZE`: Delay between observations.
/// - `D_SOURCE`, `D_TARGET`, `D_COND`: Input dimensionality.
/// - `D_JOINT`: $D_{\mathrm{target}} + (\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{source}}) + (\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{target}}) + (\mathrm{COND}\_\mathrm{HIST} \times D_{\\mathrm{cond}})$
/// - `D_XP_YP_ZP`: $(\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{source}}) + (\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{target}}) + (\mathrm{COND}\_\mathrm{HIST} \times D_{\\mathrm{cond}})$
/// - `D_YP_ZP`: $(\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{target}}) + (\mathrm{COND}\_\mathrm{HIST} \times D_{\\mathrm{cond}})$
/// - `D_YF_YP_ZP`: $D_{\mathrm{target}} + (\mathrm{SRC}\_\mathrm{HIST} \times D_{\mathrm{target}}) + (\mathrm{COND}\_\mathrm{HIST} \times D_{\\mathrm{cond}})$
///
/// Kernel-based conditional transfer entropy estimator.
///
/// ## Theory
///
#[doc = doc_snippets!(cte_formula "Kernel-based", "", "")]
///
/// # Note on Dimensions
/// These dimensions must satisfy the mathematical relations defined by the slicing logic.
/// It is recommended to use the `new_kernel_cte!` macro to instantiate this struct,
/// as it handles the dimension calculations automatically.
pub struct KernelConditionalTransferEntropy<
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
    pub kernel_type: String,
    pub bandwidth: f64,
    pub source: Array2<f64>,
    pub dest: Array2<f64>,
    pub cond: Array2<f64>,
    pub force_cpu: bool,
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
    KernelConditionalTransferEntropy<
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
    /// Creates a new `KernelConditionalTransferEntropy` estimator.
    ///
    /// # Arguments
    /// * `source`: Source time series.
    /// * `dest`: Destination time series.
    /// * `cond`: Conditioning time series.
    /// * `_src_hist_len`, `_dest_hist_len`, `_cond_hist_len`, `_step_size`: (Unused, rely on const generics).
    /// * `kernel_type`: "box" or "gaussian".
    /// * `bandwidth`: Kernel bandwidth.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: &Array2<f64>,
        dest: &Array2<f64>,
        cond: &Array2<f64>,
        _src_hist_len: usize,
        _dest_hist_len: usize,
        _cond_hist_len: usize,
        _step_size: usize,
        kernel_type: String,
        bandwidth: f64,
    ) -> Self {
        Self {
            kernel_type,
            bandwidth,
            source: source.clone(),
            dest: dest.clone(),
            cond: cond.clone(),
            force_cpu: false,
        }
    }

    /// Sets whether to force CPU implementation even if GPU support is available.
    pub fn set_force_cpu(&mut self, force_cpu: bool) {
        self.force_cpu = force_cpu;
    }

    /// Helper method to compute density for a multi-dimensional dataset.
    fn compute_density<const D: usize>(&self, data: Array2<f64>) -> Array1<f64> {
        let mut est = KernelEntropy::<D>::new_with_kernel_type(
            data,
            self.kernel_type.clone(),
            self.bandwidth,
        );
        est.set_force_cpu(self.force_cpu);
        est.kde_probability_density()
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
    for KernelConditionalTransferEntropy<
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
        self.local_values().mean().unwrap_or(0.0)
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
    for KernelConditionalTransferEntropy<
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
        Ok(self.local_values())
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
    for KernelConditionalTransferEntropy<
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
    for KernelConditionalTransferEntropy<
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
    /// Computes local conditional transfer entropy values.
    fn local_values(&self) -> Array1<f64> {
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
        >(&self.source, &self.dest, &self.cond, false);

        let joint_all =
            concatenate(Axis(1), &[yf.view(), xp.view(), yp.view(), zp.view()]).unwrap();
        let xp_yp_zp = concatenate(Axis(1), &[xp.view(), yp.view(), zp.view()]).unwrap();
        let yp_zp = concatenate(Axis(1), &[yp.view(), zp.view()]).unwrap();
        let yf_yp_zp = concatenate(Axis(1), &[yf.view(), yp.view(), zp.view()]).unwrap();

        let p_joint_all = self.compute_density::<D_JOINT>(joint_all);
        let p_xp_yp_zp = self.compute_density::<D_XP_YP_ZP>(xp_yp_zp);
        let p_yp_zp = self.compute_density::<D_YP_ZP>(yp_zp);
        let p_yf_yp_zp = self.compute_density::<D_YF_YP_ZP>(yf_yp_zp);

        let n = p_joint_all.len();
        let mut local_cte = Array1::zeros(n);
        for i in 0..n {
            let num = p_joint_all[i] * p_yp_zp[i];
            let den = p_xp_yp_zp[i] * p_yf_yp_zp[i];
            if num > 0.0 && den > 0.0 {
                local_cte[i] = (num / den).ln();
            }
        }
        local_cte
    }
}

macro_rules! impl_kernel_mi {
    ($name:ident, $num_rvs:expr, ($($d_param:ident),*), ($($d_idx:expr),*)) => {
        #[doc = concat!("Kernel-based mutual information estimator for ", stringify!($num_rvs), " random variables")]
        ///
        /// ## Theory
        ///
        #[doc = doc_snippets!(mi_formula "Kernel-based", "", "")]
        pub struct $name<const D_JOINT: usize, $(const $d_param: usize),*> {
            pub kernel_type: String,
            pub bandwidth: f64,
            pub data: Vec<Array2<f64>>,
            pub force_cpu: bool,
        }

        impl<const D_JOINT: usize, $(const $d_param: usize),*> $name<D_JOINT, $($d_param),*> {
            pub fn new(series: &[Array2<f64>], kernel_type: String, bandwidth: f64) -> Self {
                Self {
                    kernel_type,
                    bandwidth,
                    data: series.to_vec(),
                    force_cpu: false,
                }
            }

            /// Sets whether to force CPU implementation even if GPU support is available
            pub fn set_force_cpu(&mut self, force_cpu: bool) {
                self.force_cpu = force_cpu;
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
                let joint_data = concatenate(
                    Axis(1),
                    &self.data.iter().map(|d| d.view()).collect::<Vec<_>>(),
                ).unwrap();

                let mut joint_est = KernelEntropy::<D_JOINT>::new_with_kernel_type(
                    joint_data,
                    self.kernel_type.clone(),
                    self.bandwidth,
                );
                joint_est.set_force_cpu(self.force_cpu);
                let joint_density = joint_est.kde_probability_density();

                let mut marginal_densities = Vec::new();
                $(
                    let mut m_est = KernelEntropy::<$d_param>::new_with_kernel_type(
                        self.data[$d_idx].clone(),
                        self.kernel_type.clone(),
                        self.bandwidth
                    );
                    m_est.set_force_cpu(self.force_cpu);
                    marginal_densities.push(m_est.kde_probability_density());
                )*

                let n = joint_density.len();
                let mut local_mi = Array1::zeros(n);
                for i in 0..n {
                    let mut sum_log_p_marginals = 0.0;
                    for densities in &marginal_densities {
                        if densities[i] > 0.0 {
                            sum_log_p_marginals += densities[i].ln();
                        }
                    }
                    if joint_density[i] > 0.0 {
                        local_mi[i] = joint_density[i].ln() - sum_log_p_marginals;
                    }
                }
                local_mi
            }
        }
    };
}

impl_kernel_mi!(KernelMutualInformation2, 2, (D1, D2), (0, 1));
impl_kernel_mi!(KernelMutualInformation3, 3, (D1, D2, D3), (0, 1, 2));
impl_kernel_mi!(KernelMutualInformation4, 4, (D1, D2, D3, D4), (0, 1, 2, 3));
impl_kernel_mi!(
    KernelMutualInformation5,
    5,
    (D1, D2, D3, D4, D5),
    (0, 1, 2, 3, 4)
);
impl_kernel_mi!(
    KernelMutualInformation6,
    6,
    (D1, D2, D3, D4, D5, D6),
    (0, 1, 2, 3, 4, 5)
);

/// Kernel-based conditional mutual information estimator for continuous data
///
/// ## Theory
///
#[doc = doc_snippets!(cmi_formula "Kernel-based", "", "")]
///
/// # Const Generics
/// - `D1`, `D2`, `D_COND`: Dimensions of input variables.
/// - `D_JOINT`: $D_1 + D_2 + D_{\mathrm{cond}}$
/// - `D1_COND`: $D_1 + D_{\mathrm{cond}}$
/// - `D2_COND`: $D_2 + D_{\mathrm{cond}}$
pub struct KernelConditionalMutualInformation<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> {
    pub kernel_type: String,
    pub bandwidth: f64,
    pub series: Vec<Array2<f64>>,
    pub cond: Array2<f64>,
}

impl<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> KernelConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
{
    /// Create a new KernelConditionalMutualInformation estimator.
    pub fn new(
        series: &[Array2<f64>],
        cond: &Array2<f64>,
        _kernel_type: String,
        _bandwidth: f64,
    ) -> Self {
        Self {
            kernel_type: _kernel_type,
            bandwidth: _bandwidth,
            series: series.to_vec(),
            cond: cond.clone(),
        }
    }
}

impl<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> GlobalValue for KernelConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
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
    for KernelConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
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
> ConditionalMutualInformationEstimator
    for KernelConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
{
}

impl<
    const D1: usize,
    const D2: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D1_COND: usize,
    const D2_COND: usize,
> LocalValues for KernelConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND>
{
    fn local_values(&self) -> Array1<f64> {
        let mut all_data_vec = self.series.iter().map(|d| d.view()).collect::<Vec<_>>();
        all_data_vec.push(self.cond.view());
        let all_data = concatenate(Axis(1), &all_data_vec).unwrap();

        let p_joint_all = KernelEntropy::<D_JOINT>::new_with_kernel_type(
            all_data,
            self.kernel_type.clone(),
            self.bandwidth,
        )
        .kde_probability_density();

        let p_cond = KernelEntropy::<D_COND>::new_with_kernel_type(
            self.cond.clone(),
            self.kernel_type.clone(),
            self.bandwidth,
        )
        .kde_probability_density();

        let xi_z1 = concatenate(Axis(1), &[self.series[0].view(), self.cond.view()]).unwrap();
        let p_marg1 = KernelEntropy::<D1_COND>::new_with_kernel_type(
            xi_z1,
            self.kernel_type.clone(),
            self.bandwidth,
        )
        .kde_probability_density();

        let xi_z2 = concatenate(Axis(1), &[self.series[1].view(), self.cond.view()]).unwrap();
        let p_marg2 = KernelEntropy::<D2_COND>::new_with_kernel_type(
            xi_z2,
            self.kernel_type.clone(),
            self.bandwidth,
        )
        .kde_probability_density();

        let n = p_joint_all.len();
        let mut local_cmi = Array1::zeros(n);
        for i in 0..n {
            let num = p_joint_all[i] * p_cond[i];
            let den = p_marg1[i] * p_marg2[i];
            if p_joint_all[i] > 0.0 && p_cond[i] > 0.0 && den > 0.0 {
                local_cmi[i] = (num / den).ln();
            }
        }
        local_cmi
    }
}

/// Input data representation for kernel entropy estimation
///
/// This enum allows the kernel entropy estimator to accept both 1D and 2D data arrays,
/// providing flexibility in how data is passed to the estimator.
pub enum KernelData {
    /// One-dimensional data: `Array1<f64>` where each element is a data point
    OneDimensional(Array1<f64>),

    /// Two-dimensional data: `Array2<f64>` where rows are data points and columns are dimensions
    /// First dimension (rows) = samples, second dimension (columns) = features/dimensions
    TwoDimensional(Array2<f64>),
}

impl From<Array1<f64>> for KernelData {
    fn from(array: Array1<f64>) -> Self {
        KernelData::OneDimensional(array)
    }
}

impl From<Array2<f64>> for KernelData {
    fn from(array: Array2<f64>) -> Self {
        KernelData::TwoDimensional(array)
    }
}

/// Kernel-based entropy estimator for continuous data
///
/// This struct implements entropy estimation using kernel density estimation (KDE).
/// It supports both box (uniform) and Gaussian kernels, and can handle data of any dimensionality
/// (specified by the generic parameter K).
///
/// # Features
///
/// - Supports both box and Gaussian kernels
/// - Handles multi-dimensional data efficiently
/// - Uses KD-tree for fast nearest-neighbor queries
/// - Implements proper bandwidth scaling for Gaussian kernels
/// - Provides both global and local entropy values
/// - Supports GPU acceleration when compiled with the `gpu` feature flag
///
/// # Bandwidth Scaling
///
/// The two kernel types handle bandwidth differently:
///
/// - **Box Kernel**: Uses the raw bandwidth value without scaling. The bandwidth directly
///   determines the size of the hypercube within which points are counted.
///
/// - **Gaussian Kernel**: Scales the full covariance matrix of the data by the squared bandwidth,
///   matching the behavior of scipy.stats.gaussian_kde. This makes the estimator
///   adaptive to the scale and correlation of the data across all dimensions.
///
/// # GPU Acceleration
///
/// When compiled with the `gpu` feature flag, this implementation can use GPU
/// acceleration for both kernel types:
///
/// - **Gaussian Kernel**: GPU acceleration is automatically used for datasets with 500 or more points.
///   The adaptive radius for neighbor search is larger when using GPU acceleration, especially for
///   small bandwidths (< 0.5):
///   - For large datasets (> 5000 points) with small bandwidths: 4σ radius
///   - For smaller datasets with small bandwidths: 5σ radius
///   - For large datasets with normal bandwidths: 3σ radius
///   - For smaller datasets with normal bandwidths: 4σ radius
///
/// - **Box Kernel**: GPU acceleration is automatically used for datasets with 2000 or more points.
///   For smaller datasets, the CPU implementation is used as it's faster due to the overhead of GPU setup.
///
/// # Examples
///
/// ```
/// use infomeasure::estimators::entropy::Entropy;
/// use infomeasure::estimators::entropy::LocalValues;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::Array1;
///
/// // Create some 1D data
/// let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
///
/// // Calculate entropy using box kernel
/// let box_entropy = Entropy::new_kernel(data.clone(), 0.5);
/// let box_global_value = box_entropy.global_value();
/// let box_local_values = box_entropy.local_values();
///
/// // Calculate entropy using Gaussian kernel
/// let gaussian_entropy = Entropy::new_kernel_with_type(data, "gaussian".to_string(), 0.5);
/// let gaussian_global_value = gaussian_entropy.global_value();
/// let gaussian_local_values = gaussian_entropy.local_values();
/// ```
pub struct KernelEntropy<const K: usize> {
    /// Data points stored in a format suitable for KD-tree operations
    pub points: Vec<[f64; K]>,
    /// Number of data points
    pub n_samples: usize,
    /// Type of kernel to use ("box" or "gaussian")
    pub kernel_type: String,
    /// Bandwidth parameter controlling the smoothness of the density estimate
    pub bandwidth: f64,
    /// KD-tree for efficient nearest-neighbor queries
    pub tree: KdTreeKernel<K>,
    /// Standard deviations of the data in each dimension (used for Gaussian kernel scaling)
    pub std_devs: [f64; K],
    /// Lower triangular matrix L from Cholesky decomposition of scaled covariance matrix (Σ * h^2),
    /// stored flat row-major (length K*K) for cache-friendly Mahalanobis forward substitution
    pub cholesky_factor: Option<Vec<f64>>,
    /// Precision matrix (inverse of scaled covariance matrix) for GPU or direct Mahalanobis distance
    pub precision_matrix: Option<Array2<f64>>,
    /// Largest eigenvalue of the scaled covariance matrix, used for KD-tree search radius
    pub max_eigenvalue: f64,
    /// Sample mean per dimension (used by the Gaussian whitening transform); `None` for box kernel
    pub mean: Option<[f64; K]>,
    /// Data points transformed by `y = L⁻¹(x - mean)`; `Some` only for the Gaussian kernel
    /// when the full Cholesky factor is available. In whitened space the Mahalanobis distance
    /// is a plain Euclidean distance and the KD-tree search radius is data-independent.
    pub whitened_points: Option<Vec<[f64; K]>>,
    /// KD-tree built on `whitened_points`; `Some` only for the Gaussian kernel with Cholesky
    pub whitened_tree: Option<KdTreeKernel<K>>,
    /// Force CPU implementation even if GPU support is available
    pub force_cpu: bool,
}
impl<const K: usize> CrossEntropy for KernelEntropy<K> {
    fn cross_entropy(&self, other: &KernelEntropy<K>) -> f64 {
        // H(P||Q) = -1/N_p * sum(ln q(x_i))
        // where q(x_i) is the kernel density estimate of the second distribution (other)
        // evaluated at the points of the first distribution (self).

        let mut sum_ln_q = 0.0;
        let n_q = other.n_samples as f64;
        let bw = other.bandwidth;

        // Pre-calculate normalization for other if it's Gaussian
        let (normalization_q, adaptive_radius_q) = if other.kernel_type == "gaussian" {
            let det_scaled_cov_q = if let Some(ref l) = other.cholesky_factor {
                let mut diag_prod = 1.0;
                for i in 0..K {
                    diag_prod *= l[i * (K + 1)];
                }
                diag_prod * diag_prod
            } else {
                other.std_devs.iter().map(|&s| (bw * s).powi(2)).product()
            };
            let norm =
                n_q * (2.0 * std::f64::consts::PI).powf(K as f64 / 2.0) * det_scaled_cov_q.sqrt();
            let base_radius = if other.n_samples > 5000 { 36.0 } else { 64.0 };
            // In whitened space the covariance is identity, so the search radius is
            // the constant truncation bound; otherwise scale by λ_max as before.
            let radius = if other.whitened_tree.is_some() {
                base_radius
            } else {
                base_radius * other.max_eigenvalue
            };
            (norm, radius)
        } else {
            (0.0, 0.0)
        };

        let mut scratch = Default::default();
        let box_vol_inv = 1.0 / (n_q * bw.powi(K as i32));
        let mut capacity = 64;
        for (i, query_point) in self.points.iter().enumerate() {
            let density = if other.kernel_type == "gaussian" {
                // Gaussian kernel density at query_point using other's covariance
                let mut local_density = 0.0;

                if let (Some(wtree), Some(wpoints), Some(mean), Some(l)) = (
                    &other.whitened_tree,
                    &other.whitened_points,
                    &other.mean,
                    &other.cholesky_factor,
                ) {
                    // Whitened fast path: whiten the query point with other's transform,
                    // then accumulate exp(-½‖y_q - y_p‖²) directly in the visitor.
                    let mut yq = [0.0; K];
                    for dim in 0..K {
                        let base = dim * K;
                        let mut sum = 0.0;
                        for j in 0..dim {
                            sum += l[base + j] * yq[j];
                        }
                        yq[dim] = (query_point[dim] - mean[dim] - sum) / l[base + dim];
                    }
                    wtree
                        .query(&yq)
                        .within::<SquaredEuclidean<f64>>(adaptive_radius_q)
                        .unsorted()
                        .with_scratch(&mut scratch)
                        .visit(|item| {
                            let p = &wpoints[item.item as usize];
                            let mut dist_sq = 0.0;
                            for dim in 0..K {
                                let d = yq[dim] - p[dim];
                                dist_sq += d * d;
                            }
                            local_density += (-0.5 * dist_sq).exp();
                        });
                } else {
                    let neighbors = other
                        .tree
                        .query(query_point)
                        .within::<SquaredEuclidean<f64>>(adaptive_radius_q)
                        .unsorted()
                        .with_result_capacity(capacity)
                        .with_scratch(&mut scratch)
                        .execute();
                    let neighbors_len = neighbors.len();
                    for neighbor in neighbors {
                        let neighbor_point = &other.points[neighbor.item as usize];
                        let dist_sq =
                            other.calculate_mahalanobis_distance(query_point, neighbor_point);
                        local_density += (-0.5 * dist_sq).exp();
                    }
                    if i == 0 {
                        capacity = (neighbors_len as f64 * 1.2) as usize;
                        if capacity == 0 {
                            capacity = 1;
                        }
                    }
                }

                local_density / normalization_q
            } else {
                // Box kernel density at query_point
                let r = bw / 2.0;
                let r_eps = r + 1e-15;

                let candidates = other
                    .tree
                    .query(query_point)
                    .within::<Chebyshev<f64>>(r_eps)
                    .unsorted()
                    .with_result_capacity(capacity)
                    .with_scratch(&mut scratch)
                    .execute();

                let count = candidates.len();
                if i == 0 {
                    capacity = (count as f64 * 1.2) as usize;
                    if capacity == 0 {
                        capacity = 1;
                    }
                }
                (count as f64) * box_vol_inv
            };

            if density > 0.0 {
                sum_ln_q += density.ln();
            } else {
                // Parity with Python: points with zero density contribute 0.0 to the sum of logs.
                // This is effectively ignoring them in the average calculation.
                sum_ln_q += 0.0;
            }
        }

        -sum_ln_q / (self.n_samples as f64)
    }
}

impl<const K: usize> JointEntropy for KernelEntropy<K> {
    type Source = Array1<f64>;
    type Params = (String, f64); // kernel_type, bandwidth

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
        for s in series {
            assert_eq!(s.len(), n_samples, "All series must have the same length");
        }

        let mut data = Array2::zeros((n_samples, K));
        for (j, s) in series.iter().enumerate() {
            for i in 0..n_samples {
                data[[i, j]] = s[i];
            }
        }

        let estimator = KernelEntropy::<K>::new_with_kernel_type(data, params.0, params.1);
        GlobalValue::global_value(&estimator)
    }
}
impl<const K: usize> KernelEntropy<K> {
    /// Creates a new KernelEntropy estimator with the default "box" kernel
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A new KernelEntropy instance configured with a box kernel
    pub fn new(data: impl Into<KernelData>, bandwidth: f64) -> Self {
        Self::new_with_kernel_type(data, "box".to_string(), bandwidth)
    }

    /// Creates a new KernelEntropy estimator with a specified kernel type
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A new KernelEntropy instance configured with the specified kernel type
    ///
    /// # Notes
    ///
    /// The bandwidth parameter is interpreted differently depending on the kernel type:
    /// - For box kernels, it's used directly as the radius of the hypercube
    /// - For Gaussian kernels, it's scaled by the standard deviation in each dimension
    pub fn new_with_kernel_type(
        data: impl Into<KernelData>,
        kernel_type: String,
        bandwidth: f64,
    ) -> Self {
        let data = data.into();
        // for kernel type, lowercase it
        let kernel_type = kernel_type.to_lowercase();
        // for bandwidth, ensure it's a positive number
        assert!(bandwidth > 0.0);
        // for data array, assure second dimension == K
        assert!(match &data {
            KernelData::OneDimensional(_) => K == 1,
            KernelData::TwoDimensional(d) => d.ncols() == K,
        });

        // Convert the data into points suitable for the KD-tree
        let points: Vec<[f64; K]> = match &data {
            KernelData::OneDimensional(arr) => {
                // For 1D data, we can directly use as_slice()
                arr.as_slice()
                    .expect("Array must be contiguous")
                    .chunks(1)
                    .map(|chunk| {
                        let mut point = [0.0; K];
                        point[0] = chunk[0];
                        point
                    })
                    .collect()
            }
            KernelData::TwoDimensional(arr) => {
                if let Some(slice) = arr.as_slice() {
                    // If the array is contiguous, we can process it as a flat slice
                    slice
                        .chunks(K)
                        .map(|chunk| {
                            let mut point = [0.0; K];
                            point.copy_from_slice(&chunk[..K]);
                            point
                        })
                        .collect()
                } else {
                    // Fallback for non-contiguous arrays
                    arr.rows()
                        .into_iter()
                        .map(|row| {
                            let mut point = [0.0; K];
                            for (i, &val) in row.iter().enumerate() {
                                point[i] = val;
                            }
                            point
                        })
                        .collect()
                }
            }
        };

        let n_samples = points.len();
        let tree = KdTreeKernel::<K>::new_from_slice(&points).unwrap();

        // Calculate standard deviations and covariance for kernels
        let mut std_devs = [0.0; K];
        let mut cholesky_factor = None;
        let mut precision_matrix = None;

        // Calculate standard deviations (always done as it's cheap and used by Box kernel too)
        let mut means = [0.0; K];
        for dim in 0..K {
            let mut mean = 0.0;
            let mut m2 = 0.0;
            let mut count = 0.0;
            for point in &points {
                count += 1.0;
                let delta = point[dim] - mean;
                mean += delta / count;
                let delta2 = point[dim] - mean;
                m2 += delta * delta2;
            }
            means[dim] = mean;
            let variance = if count < 2.0 { 0.0 } else { m2 / (count - 1.0) };
            std_devs[dim] = variance.sqrt();
        }

        // Max eigenvalue for search radius (ensures KD-tree captures all neighbors in the ellipsoid)
        let max_eigenvalue = if kernel_type == "gaussian" {
            // Convert points to Array2 for full covariance calculation
            // Shape: (K, n_samples) where K is number of variables
            let mut data_for_cov = Array2::<f64>::zeros((K, n_samples));
            for (i, point) in points.iter().enumerate() {
                for dim in 0..K {
                    data_for_cov[[dim, i]] = point[dim];
                }
            }

            // Calculate covariance matrix
            let cov = data_for_cov
                .view()
                .cov(1.0)
                .expect("Failed to calculate covariance matrix");

            // Scale covariance matrix by bandwidth squared
            let scaled_cov = cov * (bandwidth * bandwidth);

            // Cholesky decomposition (Lower triangular L)
            // L * L^T = Σ_scaled
            let l = scaled_cov.cholesky(UPLO::Lower)
                .expect("Covariance matrix must be positive definite. Check for redundant dimensions or duplicate data.");
            // Store L flat row-major for cache-friendly Mahalanobis forward substitution
            let flat: Vec<f64> = l.iter().copied().collect();
            cholesky_factor = Some(flat);

            // Precision matrix (inverse of scaled covariance) for GPU
            let inv = scaled_cov
                .inv()
                .expect("Failed to invert covariance matrix");
            precision_matrix = Some(inv);

            use ndarray_linalg::EigValsh;
            let eigenvalues = scaled_cov
                .eigvalsh(UPLO::Lower)
                .expect("Failed to calculate eigenvalues");
            eigenvalues
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max)
        } else {
            // For Box kernel, use the max standard deviation to maintain consistency with previous scaling behavior
            let max_std = std_devs.iter().cloned().fold(0.0f64, f64::max);
            (max_std * bandwidth).powi(2)
        };

        // Whitening transform for the Gaussian kernel: y = L⁻¹(x - m) where L Lᵀ = Σ_scaled.
        // In whitened space d_M²(p,q) = ‖y_p - y_q‖² exactly, so the density inner loop
        // becomes an O(K) dot product and the KD-tree radius is a data-independent constant.
        let (whitened_points, whitened_tree) = if kernel_type == "gaussian" {
            if let Some(ref l) = cholesky_factor {
                let wpoints: Vec<[f64; K]> = points
                    .iter()
                    .map(|p| {
                        // Solve L y = (p - m) via forward substitution.
                        let mut y = [0.0; K];
                        for i in 0..K {
                            let base = i * K;
                            let mut sum = 0.0;
                            for j in 0..i {
                                sum += l[base + j] * y[j];
                            }
                            y[i] = (p[i] - means[i] - sum) / l[base + i];
                        }
                        y
                    })
                    .collect();
                let wtree = KdTreeKernel::<K>::new_from_slice(&wpoints).unwrap();
                (Some(wpoints), Some(wtree))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let is_gaussian = kernel_type == "gaussian";

        Self {
            points,
            n_samples,
            kernel_type,
            bandwidth,
            tree,
            std_devs,
            cholesky_factor,
            precision_matrix,
            max_eigenvalue,
            mean: is_gaussian.then_some(means),
            whitened_points,
            whitened_tree,
            force_cpu: false,
        }
    }

    /// Convenience constructor for 1D data
    ///
    /// # Arguments
    ///
    /// * `data` - One-dimensional data array
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter
    pub fn new_1d(data: Array1<f64>, kernel_type: String, bandwidth: f64) -> Self {
        Self::new_with_kernel_type(KernelData::OneDimensional(data), kernel_type, bandwidth)
    }

    /// Convenience constructor for 2D data
    ///
    /// # Arguments
    ///
    /// * `data` - Two-dimensional data array (rows = samples, columns = dimensions)
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter
    pub fn new_2d(data: Array2<f64>, kernel_type: String, bandwidth: f64) -> Self {
        Self::new_with_kernel_type(KernelData::TwoDimensional(data), kernel_type, bandwidth)
    }

    /// Sets whether to force CPU implementation even if GPU support is available
    pub fn set_force_cpu(&mut self, force_cpu: bool) {
        self.force_cpu = force_cpu;
    }

    /// Calculates the squared Mahalanobis distance between two points
    ///
    /// If full covariance is available (for Gaussian kernel), it computes:
    /// d_M^2 = (p1 - p2)^T Σ_scaled^-1 (p1 - p2) = ||L^-1 (p1 - p2)||^2
    /// where L is the lower triangular Cholesky factor.
    ///
    /// If only diagonal covariance is available, it falls back to scaled Euclidean distance.
    pub fn calculate_mahalanobis_distance(&self, p1: &[f64; K], p2: &[f64; K]) -> f64 {
        if let Some(ref l) = self.cholesky_factor {
            // Solve L z = (p1 - p2) using forward substitution.
            // L is flat row-major (length K*K); with const generic K the loops unroll.
            let mut diff = [0.0; K];
            for dim in 0..K {
                diff[dim] = p1[dim] - p2[dim];
            }

            let mut z = [0.0; K];
            for i in 0..K {
                let mut sum = 0.0;
                let base = i * K;
                for j in 0..i {
                    sum += l[base + j] * z[j];
                }
                z[i] = (diff[i] - sum) / l[base + i];
            }

            // dist_sq = ||z||^2
            z.iter().map(|&val| val * val).sum()
        } else {
            // Fallback to diagonal scaled Euclidean distance
            let mut sum = 0.0;
            for dim in 0..K {
                let scale = self.bandwidth * self.std_devs[dim];
                if scale > 0.0 {
                    let diff = (p1[dim] - p2[dim]) / scale;
                    sum += diff * diff;
                } else {
                    let diff = (p1[dim] - p2[dim]) / self.bandwidth;
                    sum += diff * diff;
                }
            }
            sum
        }
    }

    /// Computes local probability density values for each data point
    ///
    /// Note: The GPU size thresholds (1600 for Gaussian, 5000 for box kernel)
    /// are architecture-dependent and may vary across GPU hardware generations
    /// and driver versions.
    pub fn kde_probability_density(&self) -> Array1<f64> {
        #[cfg(feature = "gpu")]
        {
            if !self.force_cpu {
                if self.kernel_type == "box" && self.n_samples >= 5000 {
                    return self.box_kernel_density_gpu();
                } else if self.kernel_type == "gaussian" && self.n_samples >= 1600 {
                    return self.gaussian_kernel_density_gpu();
                }
            }
        }

        if self.kernel_type == "box" {
            self.box_kernel_density_cpu()
        } else {
            self.gaussian_kernel_density_cpu()
        }
    }

    /// Internal method to compute density using box kernel on CPU
    pub fn box_kernel_density_cpu(&self) -> Array1<f64> {
        let volume = self.bandwidth.powi(K as i32);
        let n_volume = self.n_samples as f64 * volume;
        let mut densities = Array1::<f64>::zeros(self.n_samples);

        let r = self.bandwidth / 2.0;
        let r_eps = r + 1e-15;
        let mut scratch = Default::default();
        let mut capacity = 64;

        for (i, query_point) in self.points.iter().enumerate() {
            let candidates = self
                .tree
                .query(query_point)
                .within::<Chebyshev<f64>>(r_eps)
                .unsorted()
                .with_result_capacity(capacity)
                .with_scratch(&mut scratch)
                .execute();

            let count = candidates.len();
            densities[i] = count as f64 / n_volume;
            if i == 0 {
                capacity = (count as f64 * 1.2) as usize;
                if capacity == 0 {
                    capacity = 1;
                }
            }
        }
        densities
    }

    /// Internal method to compute density using Gaussian kernel on CPU
    ///
    /// Uses the whitened-space fast path when the Cholesky factor is available
    /// (O(K) dot-product inner loop, data-independent radius); otherwise falls
    /// back to the Mahalanobis forward-substitution path.
    pub fn gaussian_kernel_density_cpu(&self) -> Array1<f64> {
        if let Some(ref wtree) = self.whitened_tree {
            self.gaussian_kernel_density_cpu_whitened(wtree)
        } else {
            self.gaussian_kernel_density_cpu_mahalanobis()
        }
    }

    /// Whitened-space Gaussian density: `d_M² = ‖y_p - y_q‖²`, so candidates are
    /// queried in a Euclidean ball of fixed squared radius and the inner loop is a
    /// plain dot product. Identical density to `gaussian_kernel_density_cpu_mahalanobis`
    /// up to fp rounding (the transform is an isometry for the Mahalanobis metric).
    fn gaussian_kernel_density_cpu_whitened(&self, wtree: &KdTreeKernel<K>) -> Array1<f64> {
        let n = self.n_samples as f64;
        let bw = self.bandwidth;

        let det_scaled_cov = if let Some(ref l) = self.cholesky_factor {
            let mut diag_prod = 1.0;
            for i in 0..K {
                diag_prod *= l[i * (K + 1)];
            }
            diag_prod * diag_prod
        } else {
            self.std_devs.iter().map(|&s| (bw * s).powi(2)).product()
        };

        let normalization =
            n * (2.0 * std::f64::consts::PI).powf(K as f64 / 2.0) * det_scaled_cov.sqrt();
        let mut densities = Array1::<f64>::zeros(self.n_samples);

        // In whitened space the covariance is the identity, so the search radius is
        // the squared Mahalanobis truncation bound (64 for N<=5000, 36 otherwise),
        // matching the original `64·λ_max` / `36·λ_max` thresholds.
        let adaptive_radius = if self.n_samples > 5000 { 36.0 } else { 64.0 };

        let wpoints = self
            .whitened_points
            .as_deref()
            .expect("whitened tree implies whitened points");
        let mut scratch = Default::default();
        for (i, query_point) in wpoints.iter().enumerate() {
            let mut sum_k = 0.0;
            wtree
                .query(query_point)
                .within::<SquaredEuclidean<f64>>(adaptive_radius)
                .unsorted()
                .with_scratch(&mut scratch)
                .visit(|item| {
                    let p = &wpoints[item.item as usize];
                    let mut dist_sq = 0.0;
                    for dim in 0..K {
                        let d = query_point[dim] - p[dim];
                        dist_sq += d * d;
                    }
                    #[cfg(feature = "fast_exp")]
                    {
                        sum_k += self.fast_exp(-0.5 * dist_sq);
                    }
                    #[cfg(not(feature = "fast_exp"))]
                    {
                        sum_k += (-0.5 * dist_sq).exp();
                    }
                });
            densities[i] = sum_k / normalization;
        }
        densities
    }

    /// Mahalanobis-space Gaussian density (fallback when no Cholesky factor exists,
    /// e.g. manually constructed estimators). Original forward-substitution path.
    fn gaussian_kernel_density_cpu_mahalanobis(&self) -> Array1<f64> {
        let n = self.n_samples as f64;
        let bw = self.bandwidth;

        let det_scaled_cov = if let Some(ref l) = self.cholesky_factor {
            let mut diag_prod = 1.0;
            for i in 0..K {
                diag_prod *= l[i * (K + 1)];
            }
            diag_prod * diag_prod
        } else {
            self.std_devs.iter().map(|&s| (bw * s).powi(2)).product()
        };

        let normalization =
            n * (2.0 * std::f64::consts::PI).powf(K as f64 / 2.0) * det_scaled_cov.sqrt();
        let mut densities = Array1::<f64>::zeros(self.n_samples);

        let adaptive_radius = if self.n_samples > 5000 {
            36.0 * self.max_eigenvalue
        } else {
            64.0 * self.max_eigenvalue
        };

        let mut scratch = Default::default();
        let mut capacity = 64;
        for (i, query_point) in self.points.iter().enumerate() {
            let candidates = self
                .tree
                .query(query_point)
                .within::<SquaredEuclidean<f64>>(adaptive_radius)
                .unsorted()
                .with_result_capacity(capacity)
                .with_scratch(&mut scratch)
                .execute();

            let candidates_len = candidates.len();
            let mut sum_k = 0.0;
            for candidate in candidates {
                let p = &self.points[candidate.item as usize];
                let dist_sq = self.calculate_mahalanobis_distance(query_point, p);
                #[cfg(feature = "fast_exp")]
                {
                    sum_k += self.fast_exp(-0.5 * dist_sq);
                }
                #[cfg(not(feature = "fast_exp"))]
                {
                    sum_k += (-0.5 * dist_sq).exp();
                }
            }
            densities[i] = sum_k / normalization;
            if i == 0 {
                capacity = (candidates_len as f64 * 1.2) as usize;
                if capacity == 0 {
                    capacity = 1;
                }
            }
        }
        densities
    }

    /// Computes local entropy values using a box (uniform) kernel
    pub fn box_kernel_local_values(&self) -> Array1<f64> {
        let densities = self.kde_probability_density();
        densities.mapv(|d| if d > 0.0 { -d.ln() } else { 0.0 })
    }

    /// Computes local entropy values using a Gaussian kernel
    pub fn gaussian_kernel_local_values(&self) -> Array1<f64> {
        let densities = self.kde_probability_density();
        densities.mapv(|d| if d > 0.0 { -d.ln() } else { 0.0 })
    }

    /// Computes local entropy values using a box (uniform) kernel (CPU only)
    pub fn box_kernel_local_values_cpu(&self) -> Array1<f64> {
        let densities = self.box_kernel_density_cpu();
        densities.mapv(|d| if d > 0.0 { -d.ln() } else { 0.0 })
    }

    /// Computes local entropy values using a Gaussian kernel (CPU only)
    pub fn gaussian_kernel_local_values_cpu(&self) -> Array1<f64> {
        let densities = self.gaussian_kernel_density_cpu();
        densities.mapv(|d| if d > 0.0 { -d.ln() } else { 0.0 })
    }

    /// Computes local entropy values using a box (uniform) kernel (LEGACY - DO NOT USE FOR NEW CODE)
    pub fn box_kernel_local_values_legacy(&self) -> Array1<f64> {
        // Calculate volume = bandwidth^d (where d = K)
        // This is the volume of the hypercube with side length = bandwidth
        let volume = self.bandwidth.powi(K as i32);

        // Normalization factor: N * volume
        // This is the denominator in the KDE formula: f̂(x) = (1/Nh^d) ∑ K((x - x_i)/h)
        // where K is the box kernel (uniform within the bandwidth)
        let n_volume = self.n_samples as f64 * volume;

        // Initialize array to store local entropy values
        let mut local_values = Array1::<f64>::zeros(self.n_samples);

        // Process points in batches of 4 (for f64x4) or 8 (for f64x8)
        let batch_size = 4;
        let num_batches = self.n_samples / batch_size;
        let mut scratch = Default::default();
        let mut capacity = 64;

        // Process complete batches
        for batch in 0..num_batches {
            let start_idx = batch * batch_size;

            let r = self.bandwidth / 2.0;
            let r_eps = r + 1e-15;

            for i in 0..batch_size {
                let idx = start_idx + i;
                let query_point = &self.points[idx];

                let count = self
                    .tree
                    .query(query_point)
                    .within::<Chebyshev<f64>>(r_eps)
                    .unsorted()
                    .with_result_capacity(capacity)
                    .with_scratch(&mut scratch)
                    .execute()
                    .len();
                local_values[idx] = count as f64;
                if idx == 0 {
                    capacity = (count as f64 * 1.2) as usize;
                    if capacity == 0 {
                        capacity = 1;
                    }
                }
            }
        }

        // Process remaining points
        let r = self.bandwidth / 2.0;
        let r_eps = r + 1e-15;

        for i in (num_batches * batch_size)..self.n_samples {
            let query_point = &self.points[i];

            let count = self
                .tree
                .query(query_point)
                .within::<Chebyshev<f64>>(r_eps)
                .unsorted()
                .with_result_capacity(capacity)
                .with_scratch(&mut scratch)
                .execute()
                .len();
            local_values[i] = count as f64;
        }

        local_values.mapv_inplace(|x| if x > 0.0 { -(x / n_volume).ln() } else { 0.0 });

        local_values
    }

    /// Computes local entropy values using a Gaussian kernel
    ///
    /// The Gaussian kernel uses a normal distribution centered at each query point
    /// to weight the contribution of neighboring points to the density estimate.
    /// This provides a smoother density estimate compared to the box kernel.
    ///
    /// # Implementation Details
    ///
    /// 1. For each data point, find all neighbors within a reasonable distance
    /// 2. Calculate the Gaussian kernel contribution from each neighbor
    /// 3. Normalize by the product of (bandwidth * std_dev) in each dimension and the number of samples
    /// 4. Apply logarithm and dimension-dependent normalization to get entropy values
    ///
    /// # Adaptive Radius for Neighbor Search
    ///
    /// The Gaussian kernel uses an adaptive radius to limit the search for neighbors:
    ///
    /// - For large datasets (>5000 points): 3σ radius (9 * max_scaled_bandwidth²)
    /// - For smaller datasets: 4σ radius (16 * max_scaled_bandwidth²)
    ///
    /// When compiled with the `gpu` feature flag, the GPU implementation uses
    /// a larger adaptive radius, especially for small bandwidths (< 0.5):
    ///
    /// - For large datasets (>5000 points) with small bandwidths: 4σ radius
    /// - For smaller datasets with small bandwidths: 5σ radius
    /// - For large datasets with normal bandwidths: 3σ radius
    /// - For smaller datasets with normal bandwidths: 4σ radius
    ///
    /// Points beyond this distance have a negligible contribution to the density estimate.
    ///
    /// # Bandwidth Scaling and Covariance
    ///
    /// Unlike the box kernel, the Gaussian kernel uses the full covariance matrix
    /// of the data, scaled by the squared bandwidth. This makes the estimator adaptive to
    /// both the scale and the correlation of the data in each dimension, matching the
    /// behavior of scipy.stats.gaussian_kde exactly.
    ///
    /// The scaling is applied in three places:
    /// 1. When calculating the search radius for finding neighbors (using the largest eigenvalue)
    /// 2. When calculating the Mahalanobis distance for the Gaussian kernel
    /// 3. When calculating the normalization factor (using the determinant of the covariance matrix)
    ///
    /// # Normalization
    ///
    /// The Gaussian kernel entropy includes a proper normalization factor based on the
    /// determinant of the scaled covariance matrix: (2π)^(d/2) * sqrt(det(Σ_scaled)).
    /// This ensures that the entropy estimate is consistent with the theoretical
    /// definition of differential entropy for a multivariate Gaussian distribution.
    /// Fast approximation of the exponential function
    #[cfg(feature = "fast_exp")]
    fn fast_exp(&self, x: f64) -> f64 {
        // Handle extreme values to prevent overflow/underflow
        if x < -700.0 {
            return 0.0;
        }
        if x > 700.0 {
            return f64::INFINITY;
        }

        // For very small negative values, use a Taylor series approximation
        if x > -0.5 {
            // Use a 5th-order Taylor series approximation for small negative values
            // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
            return 1.0
                + x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0)))));
        }

        // For medium negative values, use a rational approximation
        if x > -2.5 {
            // This is a minimax approximation that provides a good balance between accuracy and performance
            // exp(x) ≈ 1 / (1 - x + x²/2 - x³/6) for x < 0
            return 1.0 / (1.0 - x + x * x / 2.0 - x * x * x / 6.0);
        }

        // For large negative values, use a higher-order rational approximation
        // This provides better accuracy for inputs far from 0
        // exp(x) ≈ 1 / (1 - x + x²/2 - x³/6 + x⁴/24 - x⁵/120 + x⁶/720)
        1.0 / (1.0 - x + x * x / 2.0 - x * x * x / 6.0 + x * x * x * x / 24.0
            - x * x * x * x * x / 120.0
            + x * x * x * x * x * x / 720.0)
    }
}

/// Implementation of the LocalValues trait for KernelEntropy
///
/// This allows KernelEntropy to be used with the entropy estimation framework,
/// which expects implementors to provide local entropy values that can be
/// aggregated to compute the global entropy.
impl<const K: usize> GlobalValue for KernelEntropy<K> {
    fn global_value(&self) -> f64 {
        self.global_from_local()
    }
}

impl<const K: usize> LocalValues for KernelEntropy<K> {
    /// Computes the local entropy values for each data point
    ///
    /// This method dispatches to the appropriate kernel implementation based on
    /// the kernel_type specified during construction. It returns an array of
    /// local entropy values, one for each data point.
    ///
    /// # Returns
    ///
    /// An array of local entropy values. The mean of these values gives the
    /// global entropy estimate.
    ///
    /// # Notes
    ///
    /// - For the box kernel, local values represent the entropy contribution from
    ///   counting points within a hypercube centered at each data point.
    ///
    /// - For the Gaussian kernel, local values represent the entropy contribution
    ///   from a Gaussian-weighted sum of distances to neighboring points, with
    ///   bandwidth scaled by the standard deviation in each dimension.
    ///
    /// # GPU Acceleration
    ///
    /// When the `gpu` feature flag is enabled, both the Gaussian and box kernel
    /// calculations can use GPU acceleration, which provides significant performance
    /// improvements for large datasets and high-dimensional data:
    ///
    /// - **Gaussian Kernel**: GPU acceleration is used for datasets with 500 or more points.
    ///   For smaller datasets, the implementation automatically falls back to the CPU version.
    ///   The adaptive radius for neighbor search is larger when using GPU acceleration.
    ///
    /// - **Box Kernel**: GPU acceleration is used for datasets with 2000 or more points.
    ///   For smaller datasets, the implementation automatically falls back to the CPU version
    ///   as the overhead of GPU setup outweighs the benefits for small data sizes.
    fn local_values(&self) -> Array1<f64> {
        // The implementation now uses the centralized kde_probability_density method
        // which handles CPU/GPU dispatching and kernel type selection.
        let densities = self.kde_probability_density();
        densities.mapv(|d| if d > 0.0 { -d.ln() } else { 0.0 })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    /// Build a KernelEntropy<2> with a hand-specified (flat row-major) Cholesky factor
    /// or None to exercise the diagonal fallback. The tree/points are not used by
    /// `calculate_mahalanobis_distance`.
    fn kernel_2d(
        cholesky: Option<Vec<f64>>,
        std_devs: [f64; 2],
        bandwidth: f64,
    ) -> KernelEntropy<2> {
        let points = vec![[0.0, 0.0], [1.0, 1.0]];
        let tree = KdTreeKernel::<2>::new_from_slice(&points).unwrap();
        KernelEntropy {
            points,
            n_samples: 2,
            kernel_type: "gaussian".to_string(),
            bandwidth,
            tree,
            std_devs,
            cholesky_factor: cholesky,
            precision_matrix: None,
            max_eigenvalue: 1.0,
            mean: None,
            whitened_points: None,
            whitened_tree: None,
            force_cpu: false,
        }
    }

    /// Diagonal Cholesky L = [[a, 0], [0, b]]: d_M^2 = (dx/a)^2 + (dy/b)^2.
    /// Flat row-major: [a, 0.0, 0.0, b].
    #[rstest]
    #[case([0.0, 0.0], [3.0, 4.0], vec![2.0, 0.0, 0.0, 4.0], 3.25)]
    #[case([1.0, 1.0], [2.0, 3.0], vec![2.0, 0.0, 0.0, 4.0], 0.5)]
    #[case([0.0, 0.0], [1.0, 0.0], vec![2.0, 0.0, 0.0, 4.0], 0.25)]
    #[case([0.0, 0.0], [3.0, 4.0], vec![5.0, 0.0, 0.0, 1.0], 0.36 + 16.0)]
    fn mahalanobis_diagonal_cholesky(
        #[case] p1: [f64; 2],
        #[case] p2: [f64; 2],
        #[case] l: Vec<f64>,
        #[case] expected: f64,
    ) {
        let kernel = kernel_2d(Some(l), [1.0, 1.0], 1.0);
        assert_relative_eq!(
            kernel.calculate_mahalanobis_distance(&p1, &p2),
            expected,
            epsilon = 1e-12
        );
    }

    /// Full lower-triangular Cholesky L = [[a, 0], [c, b]] exercises the flat
    /// forward-substitution indexing across both rows. Flat row-major: [a, 0.0, c, b].
    #[rstest]
    #[case([0.0, 0.0], [4.0, 8.0], vec![2.0, 0.0, 1.0, 4.0], 4.0 + 2.25)]
    #[case([0.0, 0.0], [0.0, 8.0], vec![2.0, 0.0, 1.0, 4.0], 4.0)]
    #[case([0.0, 0.0], [2.0, 6.0], vec![2.0, 0.0, 1.0, 4.0], 1.0 + 1.5625)]
    #[case([0.0, 0.0], [3.0, 6.0], vec![3.0, 0.0, 2.0, 2.0], 1.0 + 4.0)]
    fn mahalanobis_full_cholesky(
        #[case] p1: [f64; 2],
        #[case] p2: [f64; 2],
        #[case] l: Vec<f64>,
        #[case] expected: f64,
    ) {
        let kernel = kernel_2d(Some(l), [1.0, 1.0], 1.0);
        assert_relative_eq!(
            kernel.calculate_mahalanobis_distance(&p1, &p2),
            expected,
            epsilon = 1e-12
        );
    }

    /// Fallback (cholesky_factor = None): scaled Euclidean with scale = bandwidth * std_dev.
    #[rstest]
    #[case([0.0, 0.0], [3.0, 4.0], [1.0, 1.0], 1.0, 25.0)]
    #[case([0.0, 0.0], [1.0, 1.0], [1.0, 1.0], 1.0, 2.0)]
    #[case([2.0, 2.0], [2.0, 6.0], [1.0, 1.0], 1.0, 16.0)]
    #[case([0.0, 0.0], [2.0, 4.0], [1.0, 1.0], 2.0, 5.0)]
    #[case([0.0, 0.0], [6.0, 0.0], [1.0, 2.0], 1.0, 36.0)]
    fn mahalanobis_diagonal_fallback(
        #[case] p1: [f64; 2],
        #[case] p2: [f64; 2],
        #[case] std_devs: [f64; 2],
        #[case] bandwidth: f64,
        #[case] expected: f64,
    ) {
        let kernel = kernel_2d(None, std_devs, bandwidth);
        assert_relative_eq!(
            kernel.calculate_mahalanobis_distance(&p1, &p2),
            expected,
            epsilon = 1e-12
        );
    }

    /// The Mahalanobis distance is symmetric: d(p1, p2) == d(p2, p1).
    #[test]
    fn mahalanobis_symmetric() {
        let kernel = kernel_2d(Some(vec![2.0, 1.0, 0.0, 4.0]), [1.0, 1.0], 1.0);
        let pairs = [
            ([0.0, 0.0], [4.0, 8.0]),
            ([1.5, -2.0], [-3.0, 7.25]),
            ([0.1, 0.2], [-9.0, 1.0]),
        ];
        for (p1, p2) in pairs {
            let d12 = kernel.calculate_mahalanobis_distance(&p1, &p2);
            let d21 = kernel.calculate_mahalanobis_distance(&p2, &p1);
            assert_relative_eq!(d12, d21, epsilon = 1e-12);
        }
    }

    /// The whitened-space density path must reproduce the Mahalanobis
    /// forward-substitution path to fp rounding (the transform is an isometry for
    /// the Mahalanobis metric), for well-conditioned data.
    #[rstest]
    fn whitened_density_matches_mahalanobis(#[values(100, 500)] n: usize) {
        use rand::SeedableRng;
        use rand::distributions::Distribution;
        use rand::rngs::StdRng;
        use rand_distr::Normal;

        type Case = (u64, [f64; 2], [f64; 2], f64, f64);
        let cases: [Case; 3] = [
            (42u64, [0.0, 0.0], [1.0, 1.0], 0.0, 0.9),
            (7u64, [3.0, -2.0], [1.0, 1.0], 0.6, 0.5),
            (99u64, [-5.0, 8.0], [0.25, 4.0], -0.8, 1.0),
        ];

        for (seed, mean, var, corr, bw) in cases {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut data = Array2::<f64>::zeros((n, 2));
            let zx = Normal::new(0.0, var[0].sqrt()).unwrap();
            let zy = Normal::new(0.0, var[1].sqrt()).unwrap();
            for i in 0..n {
                let x = zx.sample(&mut rng);
                let y = corr * x + (1.0 - corr * corr).sqrt() * zy.sample(&mut rng);
                data[[i, 0]] = x + mean[0];
                data[[i, 1]] = y + mean[1];
            }
            let kernel = KernelEntropy::<2>::new_2d(data, "gaussian".to_string(), bw);
            let densities_w = kernel.gaussian_kernel_density_cpu();
            let densities_m = kernel.gaussian_kernel_density_cpu_mahalanobis();
            for i in 0..n {
                assert_relative_eq!(
                    densities_w[i],
                    densities_m[i],
                    max_relative = 1e-9,
                    epsilon = 1e-12
                );
            }
        }
    }
}

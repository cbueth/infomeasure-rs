// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::estimators::approaches::discrete::ansb::AnsbEntropy;
use crate::estimators::approaches::discrete::bayes::{AlphaParam, BayesEntropy};
use crate::estimators::approaches::discrete::bonachela::BonachelaEntropy;
use crate::estimators::approaches::discrete::chao_shen::ChaoShenEntropy;
use crate::estimators::approaches::discrete::chao_wang_jost::ChaoWangJostEntropy;
use crate::estimators::approaches::discrete::grassberger::GrassbergerEntropy;
use crate::estimators::approaches::discrete::miller_madow::MillerMadowEntropy;
use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
use crate::estimators::approaches::discrete::nsb::NsbEntropy;
use crate::estimators::approaches::discrete::shrink::ShrinkEntropy;
use crate::estimators::approaches::discrete::zhang::ZhangEntropy;
use crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
use crate::estimators::approaches::expfam::renyi::RenyiEntropy;
use crate::estimators::approaches::expfam::tsallis::TsallisEntropy;
use crate::estimators::approaches::kernel;
use crate::estimators::approaches::ordinal::ordinal_estimator::OrdinalEntropy;
pub use crate::estimators::traits::{CrossEntropy, GlobalValue, JointEntropy, LocalValues};
use ndarray::{Array1, Array2};

/// Facade for creating various entropy estimators.
///
/// This struct provides a unified interface for all entropy estimation techniques supported
/// by the library. It includes methods for discrete, kernel-based, ordinal, and
/// exponential family (k-NN) estimators.
///
/// Each estimator can be used to compute the global entropy value or local entropy values
/// (if supported) using the [`GlobalValue`] and [`LocalValues`] traits.
///
/// # Mathematical Notation
/// - $H(X)$: Shannon entropy of random variable $X$.
/// - $H(X, Y)$: Joint entropy of random variables $X$ and $Y$.
/// - $H(P||Q)$: Cross-entropy between distributions $P$ and $Q$.
///
/// # Examples
///
/// ```rust
/// use infomeasure::estimators::entropy::Entropy;
/// use infomeasure::estimators::traits::{GlobalValue, JointEntropy};
/// use ndarray::{array, Array1, Array2};
///
/// // 1. Discrete Shannon Entropy (MLE)
/// let data = array![1, 2, 1, 3, 2, 1];
/// let entropy = Entropy::new_discrete(data).global_value();
/// println!("MLE Entropy: {}", entropy);
///
/// // 2. Kernel Density Entropy (continuous data)
/// let continuous_data = array![[1.0, 1.5], [2.0, 3.0], [4.0, 5.0]];
/// // Specify 2D points via const generic
/// let kernel_entropy = Entropy::nd_kernel::<2>(continuous_data, 1.0);
/// println!("Kernel entropy: {}", kernel_entropy.global_value());
///
/// // 3. Ordinal entropy for time series
/// let time_series = array![1.0, 2.0, 1.5, 3.0, 2.5];
/// let ordinal_entropy = Entropy::new_ordinal(time_series, 3);
/// println!("Ordinal entropy: {}", ordinal_entropy.global_value());
///
/// // 4. Joint Entropy of two discrete variables
/// let x = array![1, 2, 1, 3, 2, 1];
/// let y = array![0, 1, 0, 1, 0, 1];
/// // Using any discrete estimator that implements JointEntropy
/// let joint = Entropy::joint_discrete(&[x, y], ());
/// println!("Joint entropy: {}", joint);
/// ```
pub struct Entropy;

// Non-generic implementation (1D default case)
impl Entropy {
    /// Creates a new discrete entropy estimator for 1D integer data
    ///
    /// # Arguments
    ///
    /// * `data` - One-dimensional array of integer data
    ///
    /// # Returns
    ///
    /// A discrete entropy estimator configured for the provided data
    /// Create a Maximum-Likelihood (Shannon) discrete entropy estimator for integer data.
    ///
    /// The MLE entropy is computed as:
    /// $H(X) = - \sum_{i} \hat{p}_i \ln \hat{p}_i$
    /// where $\hat{p}_i = n_i / N$ is the empirical probability of the $i$-th bin.
    ///
    /// This estimator is asymptotically unbiased but exhibits significant negative bias
    /// for finite samples, especially when $N/K$ is small.
    ///
    /// Supports local values via [`LocalValues`].
    pub fn new_discrete(data: Array1<i32>) -> DiscreteEntropy {
        DiscreteEntropy::new(data)
    }

    /// Create a Miller–Madow bias-corrected discrete entropy estimator.
    ///
    /// The Miller-Madow correction adds $(K-1)/(2N)$ to the MLE estimate:
    /// $\hat{H}_{MM} = \hat{H}_{MLE} + \frac{K-1}{2N}$
    /// where $K$ is the number of bins with non-zero counts and $N$ is the sample size.
    ///
    /// This is the simplest bias correction and works well when $N \gg K$.
    /// Supports local values (uniformly offset by the correction term).
    pub fn new_miller_madow(data: Array1<i32>) -> MillerMadowEntropy {
        MillerMadowEntropy::new(data)
    }

    /// Create a James–Stein shrinkage discrete entropy estimator.
    ///
    /// The shrinkage estimator shrinks the empirical distribution towards a uniform
    /// target distribution using a data-driven shrinkage parameter $\lambda \in [0, 1]$:
    /// $\hat{p}_i^{\text{shrink}} = \lambda \hat{p}_i^{\text{uniform}} + (1-\lambda) \hat{p}_i^{\text{MLE}}$
    ///
    /// This estimator provides a good trade-off between bias and variance and is
    /// particularly robust in high-dimensional or undersampled regimes.
    ///
    /// Supports local values via [`LocalValues`].
    pub fn new_shrink(data: Array1<i32>) -> ShrinkEntropy {
        ShrinkEntropy::new(data)
    }

    /// Create a Grassberger (Gr88) discrete entropy estimator.
    ///
    /// Uses digamma-based bias correction per-count; supports local values.
    pub fn new_grassberger(data: Array1<i32>) -> GrassbergerEntropy {
        GrassbergerEntropy::new(data)
    }

    /// Create a Zhang discrete entropy estimator (Lozano 2017 fast formulation).
    ///
    /// Efficient series-based correction; supports local values.
    pub fn new_zhang(data: Array1<i32>) -> ZhangEntropy {
        ZhangEntropy::new(data)
    }

    /// Create a Bayesian discrete entropy estimator with Dirichlet prior.
    ///
    /// Choose alpha via AlphaParam; optional k_override specifies support size K.
    /// Global-only (no local values).
    pub fn new_bayes(
        data: Array1<i32>,
        alpha: AlphaParam,
        k_override: Option<usize>,
    ) -> BayesEntropy {
        BayesEntropy::new(data, alpha, k_override)
    }

    /// Create a Bonachela (de-noised) discrete entropy estimator.
    ///
    /// Harmonic-sum based correction; global-only.
    pub fn new_bonachela(data: Array1<i32>) -> BonachelaEntropy {
        BonachelaEntropy::new(data)
    }

    /// Create a Chao–Shen coverage-adjusted discrete entropy estimator.
    ///
    /// The Chao-Shen estimator accounts for unobserved species through coverage estimation:
    /// $\hat{H}_{CS} = - \sum_{i=1}^{K} \frac{\hat{p}_i^{CS} \ln \hat{p}_i^{CS}}{1 - (1 - \hat{p}_i^{CS})^N}$
    /// where $\hat{p}_i^{CS} = C \hat{p}_i^{MLE}$ and $C$ is the estimated sample coverage.
    ///
    /// Recommended for undersampled regimes with many singletons. Global-only.
    /// Matches the implementation in the Python `infomeasure` package.
    pub fn new_chao_shen(data: Array1<i32>) -> ChaoShenEntropy {
        ChaoShenEntropy::new(data)
    }

    /// Create a Chao–Wang–Jost discrete entropy estimator.
    ///
    /// Coverage-based correction using f1, f2 singletons/doubletons; global-only.
    pub fn new_chao_wang_jost(data: Array1<i32>) -> ChaoWangJostEntropy {
        ChaoWangJostEntropy::new(data)
    }

    /// Create an ANSB (asymptotic NSB) discrete entropy estimator.
    ///
    /// Requires optional K override; uses default undersampled threshold of 0.1; global-only.
    pub fn new_ansb(data: Array1<i32>, k_override: Option<usize>) -> AnsbEntropy {
        AnsbEntropy::new(data, k_override, 0.1)
    }

    /// Create an ANSB (asymptotic NSB) discrete entropy estimator.
    ///
    /// Requires optional K override and custom undersampled threshold parameter; global-only.
    pub fn new_ansb_with_threshold(
        data: Array1<i32>,
        k_override: Option<usize>,
        undersampled_threshold: f64,
    ) -> AnsbEntropy {
        AnsbEntropy::new(data, k_override, undersampled_threshold)
    }

    /// Create an NSB (Nemenman–Shafee–Bialek) discrete entropy estimator.
    ///
    /// Prior-averaged estimator via 1/K mixture and numerical integration; global-only.
    pub fn new_nsb(data: Array1<i32>, k_override: Option<usize>) -> NsbEntropy {
        NsbEntropy::new(data, k_override)
    }

    // Batch (rows) constructors for 2D integer data
    // These mirror the per-row constructors found in approaches::<estimator>::from_rows
    // and provide a convenient facade-level API.

    /// Create a vector of MLE discrete entropy estimators, one per row.
    pub fn new_discrete_rows(data: Array2<i32>) -> Vec<DiscreteEntropy> {
        DiscreteEntropy::from_rows(data)
    }

    /// Create a vector of Miller–Madow estimators, one per row.
    pub fn new_miller_madow_rows(data: Array2<i32>) -> Vec<MillerMadowEntropy> {
        MillerMadowEntropy::from_rows(data)
    }

    /// Create a vector of Shrinkage estimators, one per row.
    pub fn new_shrink_rows(data: Array2<i32>) -> Vec<ShrinkEntropy> {
        ShrinkEntropy::from_rows(data)
    }

    /// Create a vector of Grassberger estimators, one per row.
    pub fn new_grassberger_rows(data: Array2<i32>) -> Vec<GrassbergerEntropy> {
        GrassbergerEntropy::from_rows(data)
    }

    /// Create a vector of Zhang estimators, one per row.
    pub fn new_zhang_rows(data: Array2<i32>) -> Vec<ZhangEntropy> {
        ZhangEntropy::from_rows(data)
    }

    /// Create a vector of Bonachela estimators (global-only), one per row.
    pub fn new_bonachela_rows(data: Array2<i32>) -> Vec<BonachelaEntropy> {
        BonachelaEntropy::from_rows(data)
    }

    /// Create a vector of Chao–Shen estimators (global-only), one per row.
    pub fn new_chao_shen_rows(data: Array2<i32>) -> Vec<ChaoShenEntropy> {
        ChaoShenEntropy::from_rows(data)
    }

    /// Create a vector of Chao–Wang–Jost estimators (global-only), one per row.
    pub fn new_chao_wang_jost_rows(data: Array2<i32>) -> Vec<ChaoWangJostEntropy> {
        ChaoWangJostEntropy::from_rows(data)
    }

    /// Create a vector of ANSB estimators (global-only), one per row.
    ///
    /// Uses default undersampled threshold of 0.1.
    pub fn new_ansb_rows(data: Array2<i32>, k_override: Option<usize>) -> Vec<AnsbEntropy> {
        AnsbEntropy::from_rows(data, k_override, 0.1)
    }

    /// Create a vector of ANSB estimators (global-only), one per row.
    ///
    /// Requires custom undersampled threshold parameter.
    pub fn new_ansb_rows_with_threshold(
        data: Array2<i32>,
        k_override: Option<usize>,
        undersampled_threshold: f64,
    ) -> Vec<AnsbEntropy> {
        AnsbEntropy::from_rows(data, k_override, undersampled_threshold)
    }

    /// Create a vector of Bayesian estimators (global-only), one per row.
    pub fn new_bayes_rows(
        data: Array2<i32>,
        alpha: AlphaParam,
        k_override: Option<usize>,
    ) -> Vec<BayesEntropy> {
        BayesEntropy::from_rows(data, alpha, k_override)
    }

    /// Create a vector of NSB estimators (global-only), one per row.
    pub fn new_nsb_rows(data: Array2<i32>, k_override: Option<usize>) -> Vec<NsbEntropy> {
        NsbEntropy::from_rows(data, k_override)
    }

    /// Creates a new kernel entropy estimator for 1D data using the default box kernel
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A kernel entropy estimator configured with a box kernel
    ///
    /// # GPU Acceleration
    ///
    /// When compiled with the `gpu` feature flag, this method will use GPU
    /// acceleration for datasets with 2000 or more points, providing significant
    /// performance improvements for large datasets.
    pub fn new_kernel(
        data: impl Into<kernel::KernelData>,
        bandwidth: f64,
    ) -> kernel::KernelEntropy<1> {
        kernel::KernelEntropy::new(data, bandwidth)
    }

    /// Creates a new kernel entropy estimator for 1D data with a specified kernel type
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A kernel entropy estimator configured with the specified kernel type
    ///
    /// # GPU Acceleration
    ///
    /// When compiled with the `gpu` feature flag, this method will use GPU
    /// acceleration based on the kernel type and dataset size:
    ///
    /// - For Gaussian kernel: GPU is used for datasets with 500 or more points
    /// - For Box kernel: GPU is used for datasets with 2000 or more points
    ///
    /// The Gaussian kernel with GPU acceleration uses an enhanced adaptive radius
    /// calculation, especially for small bandwidths (< 0.5).
    pub fn new_kernel_with_type(
        data: impl Into<kernel::KernelData>,
        kernel_type: String,
        bandwidth: f64,
    ) -> kernel::KernelEntropy<1> {
        kernel::KernelEntropy::new_with_kernel_type(data, kernel_type, bandwidth)
    }

    /// Creates a new kernel entropy estimator for N-dimensional data using the default box kernel
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A kernel entropy estimator configured with a box kernel for N-dimensional data
    ///
    /// # GPU Acceleration
    ///
    /// When compiled with the `gpu` feature flag, this method will use GPU
    /// acceleration for datasets with 2000 or more points, providing significant
    /// performance improvements for large datasets and high-dimensional data.
    pub fn nd_kernel<const K: usize>(
        data: impl Into<kernel::KernelData>,
        bandwidth: f64,
    ) -> kernel::KernelEntropy<K> {
        kernel::KernelEntropy::new(data, bandwidth)
    }

    /// Creates a new kernel entropy estimator for N-dimensional data with a specified kernel type
    ///
    /// # Arguments
    ///
    /// * `data` - Input data for entropy estimation (1D or 2D)
    /// * `kernel_type` - Type of kernel to use ("box" or "gaussian")
    /// * `bandwidth` - Bandwidth parameter controlling the smoothness of the density estimate
    ///
    /// # Returns
    ///
    /// A kernel entropy estimator configured with the specified kernel type for N-dimensional data
    ///
    /// # GPU Acceleration
    ///
    /// When compiled with the `gpu` feature flag, this method will use GPU
    /// acceleration based on the kernel type and dataset size:
    ///
    /// - For Gaussian kernel: GPU is used for datasets with 500 or more points, providing
    ///   speedups of up to 340x for large datasets
    /// - For Box kernel: GPU is used for datasets with 2000 or more points, providing
    ///   speedups of up to 37x for large datasets
    ///
    /// The GPU implementation is particularly beneficial for high-dimensional data,
    /// where it can complete calculations that would timeout on the CPU.
    pub fn nd_kernel_with_type<const K: usize>(
        data: impl Into<kernel::KernelData>,
        kernel_type: String,
        bandwidth: f64,
    ) -> kernel::KernelEntropy<K> {
        kernel::KernelEntropy::new_with_kernel_type(data, kernel_type, bandwidth)
    }
}

impl Entropy {
    /// Create an Ordinal (permutation) entropy estimator from a 1D series.
    ///
    /// Parameters:
    /// - data: series as f64
    /// - order: embedding dimension m (temporarily limited to ≤ 12)
    pub fn new_ordinal(data: Array1<f64>, order: usize) -> OrdinalEntropy {
        OrdinalEntropy::new(data, order)
    }

    /// Create an Ordinal entropy estimator with a configurable step size (delay).
    pub fn new_ordinal_with_step(
        data: Array1<f64>,
        order: usize,
        step_size: usize,
    ) -> OrdinalEntropy {
        OrdinalEntropy::new_with_step(data, order, step_size)
    }

    /// Create an Ordinal entropy estimator with a configurable step size and stability.
    pub fn new_ordinal_with_step_and_stable(
        data: Array1<f64>,
        order: usize,
        step_size: usize,
        stable: bool,
    ) -> OrdinalEntropy {
        OrdinalEntropy::new_with_step_and_stable(data, order, step_size, stable)
    }

    /// Compute joint ordinal entropy for multiple 1D series.
    pub fn ordinal_joint_entropy(
        series_list: &[Array1<f64>],
        order: usize,
        step_size: usize,
    ) -> f64 {
        <OrdinalEntropy as JointEntropy>::joint_entropy(series_list, (order, step_size, true))
    }

    /// Compute joint ordinal entropy with configurable stability.
    pub fn ordinal_joint_entropy_with_stable(
        series_list: &[Array1<f64>],
        order: usize,
        step_size: usize,
        stable: bool,
    ) -> f64 {
        <OrdinalEntropy as JointEntropy>::joint_entropy(series_list, (order, step_size, stable))
    }

    /// Compute ordinal cross-entropy H(p||q) between two series' ordinal pattern distributions.
    pub fn ordinal_cross_entropy(
        x: &Array1<f64>,
        y: &Array1<f64>,
        order: usize,
        step_size: usize,
    ) -> f64 {
        let ex = OrdinalEntropy::new_with_step_and_stable(x.clone(), order, step_size, true);
        let ey = OrdinalEntropy::new_with_step_and_stable(y.clone(), order, step_size, true);
        ex.cross_entropy(&ey)
    }

    /// Compute ordinal cross-entropy with configurable stability.
    pub fn ordinal_cross_entropy_with_stable(
        x: &Array1<f64>,
        y: &Array1<f64>,
        order: usize,
        step_size: usize,
        stable: bool,
    ) -> f64 {
        let ex = OrdinalEntropy::new_with_step_and_stable(x.clone(), order, step_size, stable);
        let ey = OrdinalEntropy::new_with_step_and_stable(y.clone(), order, step_size, stable);
        ex.cross_entropy(&ey)
    }

    /// Create a Rényi entropy estimator (1D convenience constructor)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn new_renyi_1d(
        data: Array1<f64>,
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiEntropy<1> {
        RenyiEntropy::<1>::new_1d(data, k, alpha, noise_level)
    }

    /// Create a Rényi entropy estimator for N-dimensional data (const-generic K)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn renyi_nd<const K: usize>(
        data: Array2<f64>,
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiEntropy<K> {
        RenyiEntropy::<K>::new(data, k, alpha, noise_level)
    }

    /// Create a Tsallis entropy estimator (1D convenience constructor)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn new_tsallis_1d(
        data: Array1<f64>,
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisEntropy<1> {
        TsallisEntropy::<1>::new_1d(data, k, q, noise_level)
    }

    /// Create a Tsallis entropy estimator for N-dimensional data (const-generic K)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn tsallis_nd<const K: usize>(
        data: Array2<f64>,
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisEntropy<K> {
        TsallisEntropy::<K>::new(data, k, q, noise_level)
    }

    /// Create a Kozachenko–Leonenko entropy estimator (1D convenience constructor)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn new_kl_1d(
        data: Array1<f64>,
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoEntropy<1> {
        KozachenkoLeonenkoEntropy::<1>::new_1d(data, k, noise_level)
    }

    /// Create a Kozachenko–Leonenko entropy estimator for N-dimensional data (const-generic K)
    ///
    /// Uses natural logarithm (base e) by default. Use `with_base()` to change the logarithm base.
    pub fn kl_nd<const K: usize>(
        data: Array2<f64>,
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoEntropy<K> {
        KozachenkoLeonenkoEntropy::<K>::new(data, k, noise_level)
    }
}

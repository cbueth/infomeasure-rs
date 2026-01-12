use ndarray::{Array1, Array2};
use crate::estimators::approaches::kernel;
use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
use crate::estimators::approaches::discrete::miller_madow::MillerMadowEntropy;
use crate::estimators::approaches::discrete::shrink::ShrinkEntropy;
use crate::estimators::approaches::discrete::grassberger::GrassbergerEntropy;
use crate::estimators::approaches::discrete::zhang::ZhangEntropy;
use crate::estimators::approaches::discrete::bayes::{BayesEntropy, AlphaParam};
use crate::estimators::approaches::discrete::bonachela::BonachelaEntropy;
use crate::estimators::approaches::discrete::chao_shen::ChaoShenEntropy;
use crate::estimators::approaches::discrete::chao_wang_jost::ChaoWangJostEntropy;
use crate::estimators::approaches::discrete::ansb::AnsbEntropy;
use crate::estimators::approaches::discrete::nsb::NsbEntropy;
use crate::estimators::approaches::ordinal::ordinal::OrdinalEntropy;
use crate::estimators::approaches::expfam::renyi::RenyiEntropy;
use crate::estimators::approaches::expfam::tsallis::TsallisEntropy;
use crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
pub use crate::estimators::traits::LocalValues;

/// Entropy estimation methods for various data types
///
/// This struct provides static methods for creating entropy estimators
/// for different types of data and estimation approaches.
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
    /// Uses the empirical distribution p_i = n_i / N. Supports local values via LocalValues.
    pub fn new_discrete(data: Array1<i32>) -> DiscreteEntropy {
        DiscreteEntropy::new(data)
    }

    /// Create a Miller–Madow bias-corrected discrete entropy estimator.
    ///
    /// Adds $(K-1)/(2N)$ to the MLE estimate; supports local values (uniformly offset).
    pub fn new_miller_madow(data: Array1<i32>) -> MillerMadowEntropy {
        MillerMadowEntropy::new(data)
    }

    /// Create a James–Stein shrinkage discrete entropy estimator.
    ///
    /// Shrinks the empirical distribution towards the uniform target using a data-driven
    /// lambda in $\[0,1\]$; supports local values.
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
    /// Good for unseen-mass correction in undersampled regimes; global-only.
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
    /// Requires optional K override and undersampled threshold parameter; global-only.
    pub fn new_ansb(
        data: Array1<i32>,
        k_override: Option<usize>,
        undersampled_threshold: f64,
    ) -> AnsbEntropy {
        AnsbEntropy::new(data, k_override, undersampled_threshold)
    }

    /// Create an NSB (Nemenman–Shafee–Bialek) discrete entropy estimator.
    ///
    /// Prior-averaged estimator via 1/K mixture and numerical integration; global-only.
    pub fn new_nsb(
        data: Array1<i32>,
        k_override: Option<usize>,
    ) -> NsbEntropy {
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
    pub fn new_ansb_rows(
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
    pub fn new_nsb_rows(
        data: Array2<i32>,
        k_override: Option<usize>,
    ) -> Vec<NsbEntropy> {
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
    /// When compiled with the `gpu_support` feature flag, this method will use GPU
    /// acceleration for datasets with 2000 or more points, providing significant
    /// performance improvements for large datasets.
    pub fn new_kernel(data: impl Into<kernel::KernelData>, bandwidth: f64) -> kernel::KernelEntropy<1> {
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
    /// When compiled with the `gpu_support` feature flag, this method will use GPU
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
        bandwidth: f64
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
    /// When compiled with the `gpu_support` feature flag, this method will use GPU
    /// acceleration for datasets with 2000 or more points, providing significant
    /// performance improvements for large datasets and high-dimensional data.
    pub fn nd_kernel<const K: usize>(
        data: impl Into<kernel::KernelData>,
        bandwidth: f64
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
    /// When compiled with the `gpu_support` feature flag, this method will use GPU
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
        bandwidth: f64
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
    pub fn new_ordinal_with_step(data: Array1<f64>, order: usize, step_size: usize) -> OrdinalEntropy {
        OrdinalEntropy::new_with_step(data, order, step_size)
    }

    /// Create an Ordinal entropy estimator with a configurable step size and stability.
    pub fn new_ordinal_with_step_and_stable(data: Array1<f64>, order: usize, step_size: usize, stable: bool) -> OrdinalEntropy {
        OrdinalEntropy::new_with_step_and_stable(data, order, step_size, stable)
    }

    /// Compute joint ordinal entropy for multiple 1D series.
    pub fn ordinal_joint_entropy(series_list: &[Array1<f64>], order: usize, step_size: usize) -> f64 {
        OrdinalEntropy::joint_entropy(series_list, order, step_size, true)
    }

    /// Compute joint ordinal entropy with configurable stability.
    pub fn ordinal_joint_entropy_with_stable(series_list: &[Array1<f64>], order: usize, step_size: usize, stable: bool) -> f64 {
        OrdinalEntropy::joint_entropy(series_list, order, step_size, stable)
    }

    /// Compute ordinal cross-entropy H(p||q) between two series' ordinal pattern distributions.
    pub fn ordinal_cross_entropy(x: &Array1<f64>, y: &Array1<f64>, order: usize, step_size: usize) -> f64 {
        OrdinalEntropy::cross_entropy(x, y, order, step_size, true)
    }

    /// Compute ordinal cross-entropy with configurable stability.
    pub fn ordinal_cross_entropy_with_stable(x: &Array1<f64>, y: &Array1<f64>, order: usize, step_size: usize, stable: bool) -> f64 {
        OrdinalEntropy::cross_entropy(x, y, order, step_size, stable)
    }

    /// Create a Rényi entropy estimator (1D convenience constructor)
    pub fn new_renyi_1d(data: Array1<f64>, k: usize, alpha: f64, noise_level: f64) -> RenyiEntropy<1> {
        RenyiEntropy::<1>::new_1d(data, k, alpha, noise_level)
    }

    /// Create a Rényi entropy estimator for N-dimensional data (const-generic K)
    pub fn renyi_nd<const K: usize>(data: Array2<f64>, k: usize, alpha: f64, noise_level: f64) -> RenyiEntropy<K> {
        RenyiEntropy::<K>::new(data, k, alpha, noise_level)
    }

    /// Create a Tsallis entropy estimator (1D convenience constructor)
    pub fn new_tsallis_1d(data: Array1<f64>, k: usize, q: f64, noise_level: f64) -> TsallisEntropy<1> {
        TsallisEntropy::<1>::new_1d(data, k, q, noise_level)
    }

    /// Create a Tsallis entropy estimator for N-dimensional data (const-generic K)
    pub fn tsallis_nd<const K: usize>(data: Array2<f64>, k: usize, q: f64, noise_level: f64) -> TsallisEntropy<K> {
        TsallisEntropy::<K>::new(data, k, q, noise_level)
    }

    /// Create a Kozachenko–Leonenko entropy estimator (1D convenience constructor)
    pub fn new_kl_1d(data: Array1<f64>, k: usize, noise_level: f64) -> KozachenkoLeonenkoEntropy<1> {
        KozachenkoLeonenkoEntropy::<1>::new_1d(data, k, noise_level)
    }

    /// Create a Kozachenko–Leonenko entropy estimator for N-dimensional data (const-generic K)
    pub fn kl_nd<const K: usize>(data: Array2<f64>, k: usize, noise_level: f64) -> KozachenkoLeonenkoEntropy<K> {
        KozachenkoLeonenkoEntropy::<K>::new(data, k, noise_level)
    }
}

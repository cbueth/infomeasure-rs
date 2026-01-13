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
use crate::estimators::approaches::discrete::{
    DiscreteConditionalMutualInformation, DiscreteMutualInformation,
};

/// Macro for creating a new `KernelMutualInformation` estimator.
///
/// This macro automatically calculates the required joint dimensions
/// based on the input dimensionalities of the provided random variables.
/// It selects the correct struct (`KernelMutualInformation2` through `KernelMutualInformation6`)
/// based on the number of provided dimensions.
///
/// # Arguments
/// * `$series`: `&[Array2<f64>]` - Slice of input random variables.
/// * `$kernel`: `String` - Kernel type ("box" or "gaussian").
/// * `$bw`: `f64` - Bandwidth parameter.
/// * `$d1`, `$d2`, ...: `usize` - Dimensionality of each random variable.
#[macro_export]
macro_rules! new_kernel_mi {
    ($series:expr, $kernel:expr, $bw:expr, $d1:expr, $d2:expr) => {{
        const D_JOINT: usize = $d1 + $d2;
        $crate::estimators::approaches::kernel::KernelMutualInformation2::<D_JOINT, $d1, $d2>::new($series, $kernel, $bw)
    }};
    ($series:expr, $kernel:expr, $bw:expr, $d1:expr, $d2:expr, $d3:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3;
        $crate::estimators::approaches::kernel::KernelMutualInformation3::<D_JOINT, $d1, $d2, $d3>::new($series, $kernel, $bw)
    }};
    ($series:expr, $kernel:expr, $bw:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4;
        $crate::estimators::approaches::kernel::KernelMutualInformation4::<D_JOINT, $d1, $d2, $d3, $d4>::new($series, $kernel, $bw)
    }};
    ($series:expr, $kernel:expr, $bw:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5;
        $crate::estimators::approaches::kernel::KernelMutualInformation5::<D_JOINT, $d1, $d2, $d3, $d4, $d5>::new($series, $kernel, $bw)
    }};
    ($series:expr, $kernel:expr, $bw:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr, $d6:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5 + $d6;
        $crate::estimators::approaches::kernel::KernelMutualInformation6::<D_JOINT, $d1, $d2, $d3, $d4, $d5, $d6>::new($series, $kernel, $bw)
    }};
}

/// Macro for creating a new `KernelConditionalMutualInformation` estimator.
///
/// This macro automatically calculates the required joint and marginal dimensions
/// based on the input dimensionalities of the provided random variables and the condition.
///
/// # Arguments
/// * `$series`: `&[Array2<f64>]` - Slice of input random variables (X, Y).
/// * `$cond`: `&Array2<f64>` - Conditioning random variable (Z).
/// * `$kernel`: `String` - Kernel type ("box" or "gaussian").
/// * `$bw`: `f64` - Bandwidth parameter.
/// * `$d1`: `usize` - Dimensionality of the first random variable (X).
/// * `$d2`: `usize` - Dimensionality of the second random variable (Y).
/// * `$d_cond`: `usize` - Dimensionality of the conditioning variable (Z).
#[macro_export]
macro_rules! new_kernel_cmi {
    ($series:expr, $cond:expr, $kernel:expr, $bw:expr, $d1:expr, $d2:expr, $d_cond:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d_cond;
        const D1_COND: usize = $d1 + $d_cond;
        const D2_COND: usize = $d2 + $d_cond;
        $crate::estimators::approaches::kernel::KernelConditionalMutualInformation::<
            $d1,
            $d2,
            $d_cond,
            D_JOINT,
            D1_COND,
            D2_COND,
        >::new($series, $cond, $kernel, $bw)
    }};
}

pub struct MutualInformation;

impl MutualInformation {
    /// Create a Maximum-Likelihood (Shannon) discrete mutual information estimator.
    pub fn new_discrete_mle(series: &[Array1<i32>]) -> DiscreteMutualInformation<DiscreteEntropy> {
        DiscreteMutualInformation::new(series, DiscreteEntropy::new)
    }

    /// Create a Kernel-based mutual information estimator.
    pub fn new_kernel(series: &[Array1<f64>], bandwidth: f64) -> KernelMutualInformation2<2, 1, 1> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        KernelMutualInformation2::new(&series_2d, "box".to_string(), bandwidth)
    }

    /// Create a Kernel-based mutual information estimator with specific kernel type.
    pub fn new_kernel_with_type(
        series: &[Array1<f64>],
        kernel_type: String,
        bandwidth: f64,
    ) -> KernelMutualInformation2<2, 1, 1> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        KernelMutualInformation2::new(&series_2d, kernel_type, bandwidth)
    }

    /// Create a Multi-dimensional Kernel-based mutual information estimator.
    pub fn nd_kernel<const D_JOINT: usize, const D1: usize, const D2: usize>(
        series: &[Array2<f64>],
        bandwidth: f64,
    ) -> KernelMutualInformation2<D_JOINT, D1, D2> {
        KernelMutualInformation2::new(series, "box".to_string(), bandwidth)
    }

    /// Create a Multi-dimensional Kernel-based mutual information estimator with specific kernel type.
    pub fn nd_kernel_with_type<const D_JOINT: usize, const D1: usize, const D2: usize>(
        series: &[Array2<f64>],
        kernel_type: String,
        bandwidth: f64,
    ) -> KernelMutualInformation2<D_JOINT, D1, D2> {
        KernelMutualInformation2::new(series, kernel_type, bandwidth)
    }

    /// Create a Miller-Madow bias-corrected discrete mutual information estimator.
    pub fn new_discrete_miller_madow(
        series: &[Array1<i32>],
    ) -> DiscreteMutualInformation<MillerMadowEntropy> {
        DiscreteMutualInformation::new(series, MillerMadowEntropy::new)
    }

    /// Create a James-Stein shrinkage discrete mutual information estimator.
    pub fn new_discrete_shrink(series: &[Array1<i32>]) -> DiscreteMutualInformation<ShrinkEntropy> {
        DiscreteMutualInformation::new(series, ShrinkEntropy::new)
    }
    
    /// Create a Maximum-Likelihood (Shannon) discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_mle(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<DiscreteEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, DiscreteEntropy::new)
    }

    /// Create a Miller-Madow bias-corrected discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_miller_madow(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<MillerMadowEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, MillerMadowEntropy::new)
    }

    /// Create a James-Stein shrinkage discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_shrink(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<ShrinkEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, ShrinkEntropy::new)
    }

    /// Create a Kernel-based conditional mutual information estimator.
    pub fn new_cmi_kernel(
        series: &[Array1<f64>],
        cond: &Array1<f64>,
        bandwidth: f64,
    ) -> KernelConditionalMutualInformation<1, 1, 1, 3, 2, 2> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        let cond_2d = cond.clone().insert_axis(Axis(1));
        KernelConditionalMutualInformation::new(&series_2d, &cond_2d, "box".to_string(), bandwidth)
    }

    /// Create a Kernel-based conditional mutual information estimator with specific kernel type.
    pub fn new_cmi_kernel_with_type(
        series: &[Array1<f64>],
        cond: &Array1<f64>,
        kernel_type: String,
        bandwidth: f64,
    ) -> KernelConditionalMutualInformation<1, 1, 1, 3, 2, 2> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        let cond_2d = cond.clone().insert_axis(Axis(1));
        KernelConditionalMutualInformation::new(&series_2d, &cond_2d, kernel_type, bandwidth)
    }

    /// Create a Multi-dimensional Kernel-based conditional mutual information estimator.
    pub fn nd_cmi_kernel<
        const D1: usize,
        const D2: usize,
        const D_COND: usize,
        const D_JOINT: usize,
        const D1_COND: usize,
        const D2_COND: usize,
    >(
        series: &[Array2<f64>],
        cond: &Array2<f64>,
        bandwidth: f64,
    ) -> KernelConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND> {
        KernelConditionalMutualInformation::new(series, cond, "box".to_string(), bandwidth)
    }

    /// Create a Multi-dimensional Kernel-based conditional mutual information estimator with specific kernel type.
    pub fn nd_cmi_kernel_with_type<
        const D1: usize,
        const D2: usize,
        const D_COND: usize,
        const D_JOINT: usize,
        const D1_COND: usize,
        const D2_COND: usize,
    >(
        series: &[Array2<f64>],
        cond: &Array2<f64>,
        kernel_type: String,
        bandwidth: f64,
    ) -> KernelConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND> {
        KernelConditionalMutualInformation::new(series, cond, kernel_type, bandwidth)
    }

    /// Create a Chao-Shen discrete mutual information estimator.
    pub fn new_discrete_chao_shen(
        series: &[Array1<i32>],
    ) -> DiscreteMutualInformation<ChaoShenEntropy> {
        DiscreteMutualInformation::new(series, ChaoShenEntropy::new)
    }

    /// Create a Chao-Shen discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_chao_shen(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<ChaoShenEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, ChaoShenEntropy::new)
    }

    /// Create a Chao-Wang-Jost discrete mutual information estimator.
    pub fn new_discrete_chao_wang_jost(
        series: &[Array1<i32>],
    ) -> DiscreteMutualInformation<ChaoWangJostEntropy> {
        DiscreteMutualInformation::new(series, ChaoWangJostEntropy::new)
    }

    /// Create a Chao-Wang-Jost discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_chao_wang_jost(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<ChaoWangJostEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, ChaoWangJostEntropy::new)
    }

    /// Create a NSB discrete mutual information estimator.
    pub fn new_discrete_nsb(series: &[Array1<i32>]) -> DiscreteMutualInformation<NsbEntropy> {
        DiscreteMutualInformation::new(series, |data| NsbEntropy::new(data, None))
    }

    /// Create a NSB discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_nsb(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<NsbEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, |data| NsbEntropy::new(data, None))
    }

    /// Create a ANSB discrete mutual information estimator.
    pub fn new_discrete_ansb(series: &[Array1<i32>]) -> DiscreteMutualInformation<AnsbEntropy> {
        DiscreteMutualInformation::new(series, |data| AnsbEntropy::new(data, None, 0.0))
    }

    /// Create a ANSB discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_ansb(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<AnsbEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, |data| {
            AnsbEntropy::new(data, None, 0.0)
        })
    }

    /// Create a Bonachela discrete mutual information estimator.
    pub fn new_discrete_bonachela(
        series: &[Array1<i32>],
    ) -> DiscreteMutualInformation<BonachelaEntropy> {
        DiscreteMutualInformation::new(series, BonachelaEntropy::new)
    }

    /// Create a Bonachela discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_bonachela(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<BonachelaEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, BonachelaEntropy::new)
    }

    /// Create a Grassberger discrete mutual information estimator.
    pub fn new_discrete_grassberger(
        series: &[Array1<i32>],
    ) -> DiscreteMutualInformation<GrassbergerEntropy> {
        DiscreteMutualInformation::new(series, GrassbergerEntropy::new)
    }

    /// Create a Grassberger discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_grassberger(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<GrassbergerEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, GrassbergerEntropy::new)
    }

    /// Create a Zhang discrete mutual information estimator.
    pub fn new_discrete_zhang(series: &[Array1<i32>]) -> DiscreteMutualInformation<ZhangEntropy> {
        DiscreteMutualInformation::new(series, ZhangEntropy::new)
    }

    /// Create a Zhang discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_zhang(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<ZhangEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, ZhangEntropy::new)
    }

    /// Create a Bayes discrete mutual information estimator.
    pub fn new_discrete_bayes(series: &[Array1<i32>]) -> DiscreteMutualInformation<BayesEntropy> {
        DiscreteMutualInformation::new(series, |data| {
            BayesEntropy::new(data, AlphaParam::Jeffrey, None)
        })
    }

    /// Create a Bayes discrete conditional mutual information estimator.
    pub fn new_cmi_discrete_bayes(
        series: &[Array1<i32>],
        cond: &Array1<i32>,
    ) -> DiscreteConditionalMutualInformation<BayesEntropy> {
        DiscreteConditionalMutualInformation::new(series, cond, |data| {
            BayesEntropy::new(data, AlphaParam::Jeffrey, None)
        })
    }
}
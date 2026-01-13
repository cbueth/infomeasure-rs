use crate::estimators::approaches::discrete::{
    DiscreteConditionalTransferEntropy, DiscreteTransferEntropy,
};
use ndarray::{Array1, Array2, Axis};

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

use crate::estimators::approaches::kernel::KernelConditionalTransferEntropy;
use crate::estimators::approaches::kernel::KernelTransferEntropy;

/// Macro for creating a new `KernelTransferEntropy` estimator.
///
/// This macro automatically calculates the required joint and marginal dimensions
/// based on the history lengths and input dimensionalities.
///
/// # Arguments
/// * `$source`: `&Array2<f64>` - Source time series (samples x D_src)
/// * `$dest`: `&Array2<f64>` - Destination time series (samples x D_target)
/// * `$src_hist`: `usize` - Number of past source observations to include.
/// * `$dest_hist`: `usize` - Number of past destination observations to include.
/// * `$step`: `usize` - Delay between observations.
/// * `$d_src`: `usize` - Dimensionality of source variable.
/// * `$d_target`: `usize` - Dimensionality of destination variable.
/// * `$kernel`: `String` - Kernel type ("box" or "gaussian").
/// * `$bw`: `f64` - Bandwidth parameter.
#[macro_export]
macro_rules! new_kernel_te {
    ($source:expr, $dest:expr, $src_hist:expr, $dest_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $kernel:expr, $bw:expr) => {{
        const D_JOINT: usize = $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_XP_YP: usize = ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_YP: usize = $dest_hist * $d_target;
        const D_YF_YP: usize = $d_target + ($dest_hist * $d_target);

        $crate::estimators::approaches::kernel::KernelTransferEntropy::<
            $src_hist,
            $dest_hist,
            $step,
            $d_src,
            $d_target,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >::new($source, $dest, $src_hist, $dest_hist, $step, $kernel, $bw)
    }};
}

/// Macro for creating a new `KernelConditionalTransferEntropy` estimator.
///
/// This macro automatically calculates the required joint and marginal dimensions
/// based on the history lengths and input dimensionalities.
///
/// # Arguments
/// * `$source`: `&Array2<f64>` - Source time series (samples x D_src)
/// * `$dest`: `&Array2<f64>` - Destination time series (samples x D_target)
/// * `$cond`: `&Array2<f64>` - Conditioning time series (samples x D_cond)
/// * `$src_hist`: `usize` - Number of past source observations to include.
/// * `$dest_hist`: `usize` - Number of past destination observations to include.
/// * `$cond_hist`: `usize` - Number of past conditioning observations to include.
/// * `$step`: `usize` - Delay between observations.
/// * `$d_src`: `usize` - Dimensionality of source variable.
/// * `$d_target`: `usize` - Dimensionality of destination variable.
/// * `$d_cond`: `usize` - Dimensionality of conditioning variable.
/// * `$kernel`: `String` - Kernel type ("box" or "gaussian").
/// * `$bw`: `f64` - Bandwidth parameter.
#[macro_export]
macro_rules! new_kernel_cte {
    ($source:expr, $dest:expr, $cond:expr, $src_hist:expr, $dest_hist:expr, $cond_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $d_cond:expr, $kernel:expr, $bw:expr) => {{
        const D_JOINT: usize =
            $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_XP_YP_ZP: usize =
            ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YP_ZP: usize = ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YF_YP_ZP: usize = $d_target + ($dest_hist * $d_target) + ($cond_hist * $d_cond);

        $crate::estimators::approaches::kernel::KernelConditionalTransferEntropy::<
            $src_hist,
            $dest_hist,
            $cond_hist,
            $step,
            $d_src,
            $d_target,
            $d_cond,
            D_JOINT,
            D_XP_YP_ZP,
            D_YP_ZP,
            D_YF_YP_ZP,
        >::new(
            $source, $dest, $cond, $src_hist, $dest_hist, $cond_hist, $step, $kernel, $bw,
        )
    }};
}

pub struct TransferEntropy;

impl TransferEntropy {
    // pub fn new_discrete(data_source: Vec<i32>, data_dest: Vec<i32>) -> discrete::DiscreteTransferEntropy {
    //     discrete::DiscreteTransferEntropy::new(data_source, data_dest)
    // }
    // 
    // pub fn new_kernel(data_source: Vec<f64>, data_dest: Vec<f64>) -> kernel::KernelTransferEntropy {
    //     kernel::KernelTransferEntropy::new(data_source, data_dest)
    // }
    // 
    // pub fn new_ordinal(data_source: Vec<i32>, data_dest: Vec<i32>) -> ordinal::OrdinalTransferEntropy {
    //     ordinal::OrdinalTransferEntropy::new(data_source, data_dest)
    // }
}
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
    /// Create a Maximum-Likelihood (Shannon) discrete transfer entropy estimator.
    pub fn new_discrete_mle(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<DiscreteEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            DiscreteEntropy::new,
        )
    }

    /// Create a Kernel-based transfer entropy estimator.
    pub fn new_kernel(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
        bandwidth: f64,
    ) -> KernelTransferEntropy<1, 1, 1, 1, 1, 3, 2, 1, 2> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let dest_2d = destination.clone().insert_axis(Axis(1));
        KernelTransferEntropy::new(
            &source_2d,
            &dest_2d,
            src_hist_len,
            dest_hist_len,
            step_size,
            "box".to_string(),
            bandwidth,
        )
    }

    /// Create a Kernel-based transfer entropy estimator with specific kernel type.
    pub fn new_kernel_with_type(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
        kernel_type: String,
        bandwidth: f64,
    ) -> KernelTransferEntropy<1, 1, 1, 1, 1, 3, 2, 1, 2> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let dest_2d = destination.clone().insert_axis(Axis(1));
        KernelTransferEntropy::new(
            &source_2d,
            &dest_2d,
            src_hist_len,
            dest_hist_len,
            step_size,
            kernel_type,
            bandwidth,
        )
    }

    /// Create a Multi-dimensional Kernel-based transfer entropy estimator.
    pub fn nd_kernel<
        const SRC_HIST: usize,
        const DEST_HIST: usize,
        const STEP_SIZE: usize,
        const D_SOURCE: usize,
        const D_TARGET: usize,
        const D_JOINT: usize,
        const D_XP_YP: usize,
        const D_YP: usize,
        const D_YF_YP: usize,
    >(
        source: &Array2<f64>,
        destination: &Array2<f64>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
        bandwidth: f64,
    ) -> KernelTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_JOINT,
        D_XP_YP,
        D_YP,
        D_YF_YP,
    > {
        KernelTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            "box".to_string(),
            bandwidth,
        )
    }

    /// Create a Multi-dimensional Kernel-based transfer entropy estimator with specific kernel type.
    pub fn nd_kernel_with_type<
        const SRC_HIST: usize,
        const DEST_HIST: usize,
        const STEP_SIZE: usize,
        const D_SOURCE: usize,
        const D_TARGET: usize,
        const D_JOINT: usize,
        const D_XP_YP: usize,
        const D_YP: usize,
        const D_YF_YP: usize,
    >(
        source: &Array2<f64>,
        destination: &Array2<f64>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
        kernel_type: String,
        bandwidth: f64,
    ) -> KernelTransferEntropy<
        SRC_HIST,
        DEST_HIST,
        STEP_SIZE,
        D_SOURCE,
        D_TARGET,
        D_JOINT,
        D_XP_YP,
        D_YP,
        D_YF_YP,
    > {
        KernelTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            kernel_type,
            bandwidth,
        )
    }

    /// Create a James-Stein shrinkage discrete transfer entropy estimator.
    pub fn new_discrete_shrink(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<ShrinkEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            ShrinkEntropy::new,
        )
    }

    /// Create a Miller-Madow bias-corrected discrete transfer entropy estimator.
    pub fn new_discrete_miller_madow(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<MillerMadowEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            MillerMadowEntropy::new,
        )
    }

    /// Create a Chao-Shen discrete transfer entropy estimator.
    pub fn new_discrete_chao_shen(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<ChaoShenEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            ChaoShenEntropy::new,
        )
    }

    /// Create a NSB discrete transfer entropy estimator.
    pub fn new_discrete_nsb(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<NsbEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            |data| NsbEntropy::new(data, None),
        )
    }

    /// Create a ANSB discrete transfer entropy estimator.
    pub fn new_discrete_ansb(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<AnsbEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            |data| AnsbEntropy::new(data, None, 0.0),
        )
    }

    /// Create a Bonachela discrete transfer entropy estimator.
    pub fn new_discrete_bonachela(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<BonachelaEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            BonachelaEntropy::new,
        )
    }

    /// Create a Grassberger discrete transfer entropy estimator.
    pub fn new_discrete_grassberger(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<GrassbergerEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            GrassbergerEntropy::new,
        )
    }

    /// Create a Zhang discrete transfer entropy estimator.
    pub fn new_discrete_zhang(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<ZhangEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            ZhangEntropy::new,
        )
    }

    /// Create a Chao-Wang-Jost discrete transfer entropy estimator.
    pub fn new_discrete_chao_wang_jost(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<ChaoWangJostEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            ChaoWangJostEntropy::new,
        )
    }

    /// Create a Bayes discrete transfer entropy estimator.
    pub fn new_discrete_bayes(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
    ) -> DiscreteTransferEntropy<BayesEntropy> {
        DiscreteTransferEntropy::new(
            source,
            destination,
            src_hist_len,
            dest_hist_len,
            step_size,
            |data| BayesEntropy::new(data, AlphaParam::Jeffrey, None),
        )
    }

    /// Create a Maximum-Likelihood (Shannon) discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_mle(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<DiscreteEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            DiscreteEntropy::new,
        )
    }

    /// Create a James-Stein shrinkage discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_shrink(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<ShrinkEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            ShrinkEntropy::new,
        )
    }

    /// Create a Kernel-based conditional transfer entropy estimator.
    pub fn new_cte_kernel(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        condition: &Array1<f64>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
        bandwidth: f64,
    ) -> KernelConditionalTransferEntropy<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let dest_2d = destination.clone().insert_axis(Axis(1));
        let cond_2d = condition.clone().insert_axis(Axis(1));
        KernelConditionalTransferEntropy::new(
            &source_2d,
            &dest_2d,
            &cond_2d,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            "box".to_string(),
            bandwidth,
        )
    }

    /// Create a Kernel-based conditional transfer entropy estimator with specific kernel type.
    pub fn new_cte_kernel_with_type(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        condition: &Array1<f64>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
        kernel_type: String,
        bandwidth: f64,
    ) -> KernelConditionalTransferEntropy<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let dest_2d = destination.clone().insert_axis(Axis(1));
        let cond_2d = condition.clone().insert_axis(Axis(1));
        KernelConditionalTransferEntropy::new(
            &source_2d,
            &dest_2d,
            &cond_2d,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            kernel_type,
            bandwidth,
        )
    }

    /// Create a Multi-dimensional Kernel-based conditional transfer entropy estimator.
    pub fn nd_cte_kernel<
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
    >(
        source: &Array2<f64>,
        destination: &Array2<f64>,
        condition: &Array2<f64>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
        bandwidth: f64,
    ) -> KernelConditionalTransferEntropy<
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
    > {
        KernelConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            "box".to_string(),
            bandwidth,
        )
    }

    /// Create a Multi-dimensional Kernel-based conditional transfer entropy estimator with specific kernel type.
    pub fn nd_cte_kernel_with_type<
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
    >(
        source: &Array2<f64>,
        destination: &Array2<f64>,
        condition: &Array2<f64>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
        kernel_type: String,
        bandwidth: f64,
    ) -> KernelConditionalTransferEntropy<
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
    > {
        KernelConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            kernel_type,
            bandwidth,
        )
    }

    /// Create a Miller-Madow bias-corrected discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_miller_madow(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<MillerMadowEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            MillerMadowEntropy::new,
        )
    }

    /// Create a Chao-Shen discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_chao_shen(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<ChaoShenEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            ChaoShenEntropy::new,
        )
    }

    /// Create a NSB discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_nsb(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<NsbEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            |data| NsbEntropy::new(data, None),
        )
    }

    /// Create a ANSB discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_ansb(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<AnsbEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            |data| AnsbEntropy::new(data, None, 0.0),
        )
    }

    /// Create a Bonachela discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_bonachela(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<BonachelaEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            BonachelaEntropy::new,
        )
    }

    /// Create a Grassberger discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_grassberger(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<GrassbergerEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            GrassbergerEntropy::new,
        )
    }

    /// Create a Zhang discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_zhang(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<ZhangEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            ZhangEntropy::new,
        )
    }

    /// Create a Chao-Wang-Jost discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_chao_wang_jost(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<ChaoWangJostEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            ChaoWangJostEntropy::new,
        )
    }

    /// Create a Bayes discrete conditional transfer entropy estimator.
    pub fn new_cte_discrete_bayes(
        source: &Array1<i32>,
        destination: &Array1<i32>,
        condition: &Array1<i32>,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
    ) -> DiscreteConditionalTransferEntropy<BayesEntropy> {
        DiscreteConditionalTransferEntropy::new(
            source,
            destination,
            condition,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            |data| BayesEntropy::new(data, AlphaParam::Jeffrey, None),
        )
    }
}

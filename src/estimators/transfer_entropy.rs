// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Transfer Entropy Estimators
//!
//! This module provides estimators for computing transfer entropy $T_{X \to Y}$ and
//! conditional transfer entropy $T_{X \to Y|Z}$.
//!
//! For usage examples and guidance, see the [Estimator Usage Guide](../guide/estimator_usage/index.html).
//! For macro convenience functions, see the [Macros Guide](../guide/macros/index.html).
//!
//! ## Const Generics and Dimensions
//!
//! Due to Rust's current limitations with constant expressions in generic arguments (without
//! `generic_const_exprs`), all dimensionalities—including derived ones—must be passed as explicit
//! const generics. This ensures that the dimensionality of every internal estimator is known
//! at compile-time, allowing for significant optimizations and type safety.
//!
//! ### Transfer Entropy (TE)
//!
//! $TE(X \to Y) = I(Y_{future}; X_{past} | Y_{past})$
//!
//! The `KernelTransferEntropy` and `KsgTransferEntropy` structs use:
//! - `SRC_HIST`, `DEST_HIST`: History lengths.
//! - `STEP_SIZE`: Delay between observations.
//! - `D_SOURCE`, `D_TARGET`: Dimensionality of individual samples in X and Y.
//! - `D_JOINT`: $D_{target} + (SRC\_HIST \times D_{source}) + (DEST\_HIST \times D_{target})$
//! - `D_XP_YP`: $(SRC\_HIST \times D_{source}) + (DEST\_HIST \times D_{target})$
//! - `D_YP`: $DEST\_HIST \times D_{target}$
//! - `D_YF_YP`: $D_{target} + (DEST\_HIST \times D_{target})$
//!
//! **Dimension Relations for TE**:
//! - `D_JOINT` = `D_YF_YP` + `SRC_HIST` * `D_SOURCE`
//! - `D_XP_YP` = `D_YP` + `SRC_HIST` * `D_SOURCE`
//! - `D_YF_YP` = `D_TARGET` + `D_YP`
//! - `D_YP` = `DEST_HIST` * `D_TARGET`
//!
//! ### Conditional Transfer Entropy (CTE)
//!
//! $CTE(X \to Y | Z) = I(Y_{future}; X_{past} | Y_{past}, Z_{past})$
//!
//! The `KernelConditionalTransferEntropy` and `KsgConditionalTransferEntropy` structs use:
//! - `SRC_HIST`, `DEST_HIST`, `COND_HIST`: History lengths.
//! - `STEP_SIZE`: Delay between observations.
//! - `D_SOURCE`, `D_TARGET`, `D_COND`: Input dimensionality.
//! - `D_JOINT`: $D_{target} + (SRC\_HIST \times D_{source}) + (DEST\_HIST \times D_{target}) + (COND\_HIST \times D_{cond})$
//! - `D_XP_YP_ZP`: $(SRC\_HIST \times D_{source}) + (DEST\_HIST \times D_{target}) + (COND\_HIST \times D_{cond})$
//! - `D_YP_ZP`: $(DEST\_HIST \times D_{target}) + (COND\_HIST \times D_{cond})$
//! - `D_YF_YP_ZP`: $D_{target} + (DEST\_HIST \times D_{target}) + (COND\_HIST \times D_{cond})$
//!
//! **Dimension Relations for CTE**:
//! - `D_JOINT` = `D_YF_YP_ZP` + `SRC_HIST` * `D_SOURCE`
//! - `D_XP_YP_ZP` = `D_YP_ZP` + `SRC_HIST` * `D_SOURCE`
//! - `D_YF_YP_ZP` = `D_TARGET` + `D_YP_ZP`
//! - `D_YP_ZP` = (`DEST_HIST` * `D_TARGET`) + (`COND_HIST` * `D_COND`)
//!
//! ### Helper Macros
//!
//! To simplify instantiation and automatically calculate these dimensions, use the following macros:
//! - `new_kernel_te!` - Creates a `KernelTransferEntropy` estimator
//! - `new_kernel_cte!` - Creates a `KernelConditionalTransferEntropy` estimator
//! - `new_ksg_te!` - Creates a `KsgTransferEntropy` estimator
//! - `new_ksg_cte!` - Creates a `KsgConditionalTransferEntropy` estimator
//! - `new_renyi_te!` - Creates a `RenyiTransferEntropy` estimator
//! - `new_renyi_cte!` - Creates a `RenyiConditionalTransferEntropy` estimator
//! - `new_tsallis_te!` - Creates a `TsallisTransferEntropy` estimator
//! - `new_tsallis_cte!` - Creates a `TsallisConditionalTransferEntropy` estimator
//! - `new_ordinal_te!` - Creates an `OrdinalTransferEntropy` estimator
//! - `new_ordinal_cte!` - Creates an `OrdinalConditionalTransferEntropy` estimator
//!
//! These macros handle the dimension calculations automatically based on the history lengths
//! and input dimensionalities you provide.

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

use crate::estimators::approaches::expfam::kozachenko_leonenko::{
    KozachenkoLeonenkoConditionalTransferEntropy, KozachenkoLeonenkoTransferEntropy,
};
use crate::estimators::approaches::expfam::ksg::{
    KsgConditionalTransferEntropy, KsgTransferEntropy,
};
use crate::estimators::approaches::expfam::renyi::{
    RenyiConditionalTransferEntropy, RenyiTransferEntropy,
};
use crate::estimators::approaches::expfam::tsallis::{
    TsallisConditionalTransferEntropy, TsallisTransferEntropy,
};
use crate::estimators::approaches::ordinal::ordinal_estimator::{
    OrdinalConditionalTransferEntropy, OrdinalTransferEntropy,
};

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

/// Macro for creating a new `KsgTransferEntropy` estimator.
#[macro_export]
macro_rules! new_ksg_te {
    ($source:expr, $dest:expr, $src_hist:expr, $dest_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $k:expr, $noise:expr) => {{
        const D_JOINT: usize = $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_XP_YP: usize = ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_YP: usize = $dest_hist * $d_target;
        const D_YF_YP: usize = $d_target + ($dest_hist * $d_target);

        $crate::estimators::approaches::expfam::ksg_te::KsgTransferEntropy::<
            $src_hist,
            $dest_hist,
            $step,
            $d_src,
            $d_target,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >::new($source, $dest, $k, $noise)
    }};
}

/// Macro for creating a new `KsgConditionalTransferEntropy` estimator.
#[macro_export]
macro_rules! new_ksg_cte {
    ($source:expr, $dest:expr, $cond:expr, $src_hist:expr, $dest_hist:expr, $cond_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $d_cond:expr, $k:expr, $noise:expr) => {{
        const D_JOINT: usize =
            $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_XP_YP_ZP: usize =
            ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YP_ZP: usize = ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YF_YP_ZP: usize = $d_target + ($dest_hist * $d_target) + ($cond_hist * $d_cond);

        $crate::estimators::approaches::expfam::ksg_te::KsgConditionalTransferEntropy::<
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
        >::new($source, $dest, $cond, $k, $noise)
    }};
}

/// Macro for creating a new `RenyiTransferEntropy` estimator.
#[macro_export]
macro_rules! new_renyi_te {
    ($source:expr, $dest:expr, $src_hist:expr, $dest_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $k:expr, $alpha:expr, $noise:expr) => {{
        const D_JOINT: usize = $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_XP_YP: usize = ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_YP: usize = $dest_hist * $d_target;
        const D_YF_YP: usize = $d_target + ($dest_hist * $d_target);

        $crate::estimators::approaches::expfam::renyi::RenyiTransferEntropy::<
            $src_hist,
            $dest_hist,
            $step,
            $d_src,
            $d_target,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >::new($source, $dest, $k, $alpha, $noise)
    }};
}

/// Macro for creating a new `RenyiConditionalTransferEntropy` estimator.
#[macro_export]
macro_rules! new_renyi_cte {
    ($source:expr, $dest:expr, $cond:expr, $src_hist:expr, $dest_hist:expr, $cond_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $d_cond:expr, $k:expr, $alpha:expr, $noise:expr) => {{
        const D_JOINT: usize =
            $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_XP_YP_ZP: usize =
            ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YP_ZP: usize = ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YF_YP_ZP: usize = $d_target + ($dest_hist * $d_target) + ($cond_hist * $d_cond);

        $crate::estimators::approaches::expfam::renyi::RenyiConditionalTransferEntropy::<
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
        >::new($source, $dest, $cond, $k, $alpha, $noise)
    }};
}

/// Macro for creating a new `TsallisTransferEntropy` estimator.
#[macro_export]
macro_rules! new_tsallis_te {
    ($source:expr, $dest:expr, $src_hist:expr, $dest_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $k:expr, $q:expr, $noise:expr) => {{
        const D_JOINT: usize = $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_XP_YP: usize = ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_YP: usize = $dest_hist * $d_target;
        const D_YF_YP: usize = $d_target + ($dest_hist * $d_target);

        $crate::estimators::approaches::expfam::tsallis::TsallisTransferEntropy::<
            $src_hist,
            $dest_hist,
            $step,
            $d_src,
            $d_target,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >::new($source, $dest, $k, $q, $noise)
    }};
}

/// Macro for creating a new `TsallisConditionalTransferEntropy` estimator.
#[macro_export]
macro_rules! new_tsallis_cte {
    ($source:expr, $dest:expr, $cond:expr, $src_hist:expr, $dest_hist:expr, $cond_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $d_cond:expr, $k:expr, $q:expr, $noise:expr) => {{
        const D_JOINT: usize =
            $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_XP_YP_ZP: usize =
            ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YP_ZP: usize = ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YF_YP_ZP: usize = $d_target + ($dest_hist * $d_target) + ($cond_hist * $d_cond);

        $crate::estimators::approaches::expfam::tsallis::TsallisConditionalTransferEntropy::<
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
        >::new($source, $dest, $cond, $k, $q, $noise)
    }};
}

/// Macro for creating a new `KozachenkoLeonenkoTransferEntropy` estimator.
#[macro_export]
macro_rules! new_kl_te {
    ($source:expr, $dest:expr, $src_hist:expr, $dest_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $k:expr, $noise:expr) => {{
        const D_JOINT: usize = $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_XP_YP: usize = ($src_hist * $d_src) + ($dest_hist * $d_target);
        const D_YP: usize = $dest_hist * $d_target;
        const D_YF_YP: usize = $d_target + ($dest_hist * $d_target);

        $crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoTransferEntropy::<
            $src_hist,
            $dest_hist,
            $step,
            $d_src,
            $d_target,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >::new($source, $dest, $k, $noise)
    }};
}

/// Macro for creating a new `KozachenkoLeonenkoConditionalTransferEntropy` estimator.
#[macro_export]
macro_rules! new_kl_cte {
    ($source:expr, $dest:expr, $cond:expr, $src_hist:expr, $dest_hist:expr, $cond_hist:expr, $step:expr, $d_src:expr, $d_target:expr, $d_cond:expr, $k:expr, $noise:expr) => {{
        const D_JOINT: usize =
            $d_target + ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_XP_YP_ZP: usize =
            ($src_hist * $d_src) + ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YP_ZP: usize = ($dest_hist * $d_target) + ($cond_hist * $d_cond);
        const D_YF_YP_ZP: usize = $d_target + ($dest_hist * $d_target) + ($cond_hist * $d_cond);

        $crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoConditionalTransferEntropy::<
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
        >::new($source, $dest, $cond, $k, $noise)
    }};
}

pub struct TransferEntropy;

/// Facade for creating transfer entropy (TE) and conditional transfer entropy (CTE) estimators.
///
/// This struct provides a unified interface for all TE/CTE estimation techniques supported
/// by the library. It includes methods for discrete, kernel-based, ordinal, and
/// exponential family (k-NN) estimators.
///
/// Each estimator can be used to compute the global TE value or local TE values
/// (if supported) using the [`GlobalValue`](crate::estimators::traits::GlobalValue) and [`LocalValues`](crate::estimators::traits::LocalValues) traits.
///
/// # Relationship to Other Measures
///
/// Transfer entropy is a directional (asymmetric) measure of information flow between time series:
///
/// - **Mutual Information**: $I(X;Y)$ - non-directional dependence
/// - **Time-lagged MI**: $I(X_{t-u}; Y_t)$ - directional but without conditioning
/// - **Conditional MI**: $I(X;Y|Z)$ - MI with conditioning (but not time-lagged)
/// - **Transfer Entropy**: $T_{X \to Y} = I(X^{(k)}; Y_{t+1} | Y^{(l)})$ - MI with time lags + conditioning on target's past
/// - **Conditional TE**: $T_{X \to Y|Z}$ - TE with additional conditioning
///
/// For a detailed conceptual guide, see the [Transfer Entropy Guide](crate::guide::transfer_entropy).
///
/// # Examples
///
/// This section provides examples for all TE/CTE estimators available through the `TransferEntropy` facade.
///
/// ## Discrete TE Estimators
///
/// ### Maximum Likelihood (MLE)
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0, 1, 0, 1, 0, 1, 0, 1];
/// let dest = array![0, 0, 1, 0, 1, 0, 1, 0];
/// let te = TransferEntropy::new_discrete_mle(&source, &dest, 1, 1, 1).global_value();
/// assert!(te.is_finite());
/// ```
///
/// ### Miller–Madow
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0, 1, 0, 1, 0, 1, 0, 1];
/// let dest = array![0, 0, 1, 0, 1, 0, 1, 0];
/// let te = TransferEntropy::new_discrete_miller_madow(&source, &dest, 1, 1, 1).global_value();
/// assert!(te.is_finite());
/// ```
///
/// ### Shrinkage
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0, 1, 0, 1, 0, 1, 0, 1];
/// let dest = array![0, 0, 1, 0, 1, 0, 1, 0];
/// let te = TransferEntropy::new_discrete_shrink(&source, &dest, 1, 1, 1).global_value();
/// assert!(te.is_finite());
/// ```
///
/// ### Grassberger
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0, 1, 0, 1, 0, 1, 0, 1];
/// let dest = array![0, 0, 1, 0, 1, 0, 1, 0];
/// let te = TransferEntropy::new_discrete_grassberger(&source, &dest, 1, 1, 1).global_value();
/// assert!(te.is_finite());
/// ```
///
/// ### NSB
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0, 1, 0, 1, 0, 1, 0, 1];
/// let dest = array![0, 0, 1, 0, 1, 0, 1, 0];
/// let te = TransferEntropy::new_discrete_nsb(&source, &dest, 1, 1, 1).global_value();
/// assert!(te.is_finite());
/// ```
///
/// ## Continuous TE Estimators
///
/// **When to use**: Continuous TE for real-valued time series:
/// - **Kernel**: When you need explicit control over bandwidth
/// - **KSG**: Generally preferred (more robust to parameter choices)
/// - **Rényi/Tsallis**: For generalized entropy approaches
///
/// ### Kernel TE
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let dest = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let te = TransferEntropy::new_kernel(&source, &dest, 1, 1, 1, 1.0).global_value();
/// assert!(te.is_finite());
/// ```
///
/// ### KSG TE
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let dest = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let te = TransferEntropy::new_ksg(&source, &dest, 1, 1, 1, 3, 1e-10).global_value();
/// assert!(te.is_finite());
/// ```
///
/// ### Rényi TE
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let dest = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let te = TransferEntropy::new_renyi(&source, &dest, 3, 2.0, 1e-10).global_value();
/// assert!(te.is_finite());
/// ```
///
/// ## Ordinal TE Estimators
///
/// **When to use**: Ordinal TE is ideal for time series because:
/// - Robust to amplitude variations and monotonic transformations
/// - Captures temporal dynamics through permutation patterns
/// - Computationally efficient for long time series
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
/// let dest = array![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
/// let te = TransferEntropy::new_ordinal(&source, &dest, 3, 1, 1, 1, true).global_value();
/// assert!(te.is_finite());
/// ```
///
/// ## Conditional TE (CTE) Estimators
///
/// **When to use**: Use CTE to:
/// - Measure directed information flow while controlling for confounders
/// - Identify true causal drivers vs spurious correlations from common causes
/// - Analyze brain connectivity or gene regulatory networks with known confounders
///
/// ### Discrete CTE
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0, 1, 0, 1, 0, 1, 0, 1];
/// let dest = array![0, 0, 1, 0, 1, 0, 1, 0];
/// let cond = array![0, 0, 0, 1, 0, 1, 1, 1];
/// let cte = TransferEntropy::new_cte_discrete_mle(&source, &dest, &cond, 1, 1, 1, 1).global_value();
/// assert!(cte.is_finite());
/// ```
///
/// ### Kernel CTE
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let dest = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let cond = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
/// let cte = TransferEntropy::new_cte_kernel(&source, &dest, &cond, 1, 1, 1, 1, 1.0).global_value();
/// assert!(cte.is_finite());
/// ```
///
/// ### KSG CTE
///
/// ```rust
/// use infomeasure::estimators::transfer_entropy::TransferEntropy;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let source = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let dest = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let cond = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
/// let cte = TransferEntropy::new_cte_ksg(&source, &dest, &cond, 1, 1, 1, 1, 3, 1e-10).global_value();
/// assert!(cte.is_finite());
/// ```
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

    /// Create a Miller-Madow discrete conditional transfer entropy estimator.
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

    /// Create an NSB discrete conditional transfer entropy estimator.
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

    /// Create an ANSB discrete conditional transfer entropy estimator.
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
            |data| AnsbEntropy::new(data, None, 0.1),
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

    /// Create an Ordinal-based transfer entropy estimator.
    pub fn new_ordinal(
        source: &Array1<f64>,
        dest: &Array1<f64>,
        order: usize,
        src_hist_len: usize,
        dest_hist_len: usize,
        step_size: usize,
        stable: bool,
    ) -> OrdinalTransferEntropy {
        OrdinalTransferEntropy::new(
            source,
            dest,
            order,
            src_hist_len,
            dest_hist_len,
            step_size,
            stable,
        )
    }

    /// Create an Ordinal-based conditional transfer entropy estimator.
    #[allow(clippy::too_many_arguments)]
    pub fn new_cte_ordinal(
        source: &Array1<f64>,
        dest: &Array1<f64>,
        cond: &Array1<f64>,
        order: usize,
        src_hist_len: usize,
        dest_hist_len: usize,
        cond_hist_len: usize,
        step_size: usize,
        stable: bool,
    ) -> OrdinalConditionalTransferEntropy {
        OrdinalConditionalTransferEntropy::new(
            source,
            dest,
            cond,
            order,
            src_hist_len,
            dest_hist_len,
            cond_hist_len,
            step_size,
            stable,
        )
    }

    /// Create a Kernel-based conditional transfer entropy estimator.
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
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

    /// Create a KSG-based transfer entropy estimator.
    pub fn new_ksg(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        _src_hist_len: usize,
        _dest_hist_len: usize,
        _step_size: usize,
        k: usize,
        noise_level: f64,
    ) -> KsgTransferEntropy<1, 1, 1, 1, 1, 3, 2, 1, 2> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let destination_2d = destination.clone().insert_axis(Axis(1));
        KsgTransferEntropy::<1, 1, 1, 1, 1, 3, 2, 1, 2>::new(
            &source_2d,
            &destination_2d,
            k,
            noise_level,
        )
    }

    /// Create a multi-dimensional KSG-based transfer entropy estimator.
    pub fn nd_ksg<
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
        k: usize,
        noise_level: f64,
    ) -> KsgTransferEntropy<
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
        KsgTransferEntropy::<
            SRC_HIST,
            DEST_HIST,
            STEP_SIZE,
            D_SOURCE,
            D_TARGET,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >::new(source, destination, k, noise_level)
    }

    /// Create a KSG-based conditional transfer entropy estimator.
    #[allow(clippy::too_many_arguments)]
    pub fn new_cte_ksg(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        condition: &Array1<f64>,
        _src_hist_len: usize,
        _dest_hist_len: usize,
        _cond_hist_len: usize,
        _step_size: usize,
        k: usize,
        noise_level: f64,
    ) -> KsgConditionalTransferEntropy<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let destination_2d = destination.clone().insert_axis(Axis(1));
        let condition_2d = condition.clone().insert_axis(Axis(1));
        KsgConditionalTransferEntropy::<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3>::new(
            &source_2d,
            &destination_2d,
            &condition_2d,
            k,
            noise_level,
        )
    }

    /// Create a multi-dimensional KSG-based conditional transfer entropy estimator.
    pub fn nd_cte_ksg<
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
        k: usize,
        noise_level: f64,
    ) -> KsgConditionalTransferEntropy<
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
        KsgConditionalTransferEntropy::<
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
        >::new(source, destination, condition, k, noise_level)
    }

    /// Create a Rényi-based transfer entropy estimator.
    pub fn new_renyi(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiTransferEntropy<1, 1, 1, 1, 1, 3, 2, 1, 2> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let destination_2d = destination.clone().insert_axis(Axis(1));
        RenyiTransferEntropy::<1, 1, 1, 1, 1, 3, 2, 1, 2>::new(
            &source_2d,
            &destination_2d,
            k,
            alpha,
            noise_level,
        )
    }

    /// Create a multi-dimensional Rényi-based transfer entropy estimator.
    pub fn nd_renyi<
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
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiTransferEntropy<
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
        RenyiTransferEntropy::<
            SRC_HIST,
            DEST_HIST,
            STEP_SIZE,
            D_SOURCE,
            D_TARGET,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >::new(source, destination, k, alpha, noise_level)
    }

    /// Create a Tsallis-based transfer entropy estimator.
    pub fn new_tsallis(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisTransferEntropy<1, 1, 1, 1, 1, 3, 2, 1, 2> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let destination_2d = destination.clone().insert_axis(Axis(1));
        TsallisTransferEntropy::<1, 1, 1, 1, 1, 3, 2, 1, 2>::new(
            &source_2d,
            &destination_2d,
            k,
            q,
            noise_level,
        )
    }

    /// Create a multi-dimensional Tsallis-based transfer entropy estimator.
    pub fn nd_tsallis<
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
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisTransferEntropy<
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
        TsallisTransferEntropy::<
            SRC_HIST,
            DEST_HIST,
            STEP_SIZE,
            D_SOURCE,
            D_TARGET,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >::new(source, destination, k, q, noise_level)
    }

    /// Create a Kozachenko-Leonenko (KL) based transfer entropy estimator.
    pub fn new_kl(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoTransferEntropy<1, 1, 1, 1, 1, 3, 2, 1, 2> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let destination_2d = destination.clone().insert_axis(Axis(1));
        KozachenkoLeonenkoTransferEntropy::<1, 1, 1, 1, 1, 3, 2, 1, 2>::new(
            &source_2d,
            &destination_2d,
            k,
            noise_level,
        )
    }

    /// Create a multi-dimensional Kozachenko-Leonenko (KL) based transfer entropy estimator.
    pub fn nd_kl<
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
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoTransferEntropy<
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
        KozachenkoLeonenkoTransferEntropy::<
            SRC_HIST,
            DEST_HIST,
            STEP_SIZE,
            D_SOURCE,
            D_TARGET,
            D_JOINT,
            D_XP_YP,
            D_YP,
            D_YF_YP,
        >::new(source, destination, k, noise_level)
    }

    /// Create a Rényi-based conditional transfer entropy estimator.
    pub fn new_cte_renyi(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        condition: &Array1<f64>,
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiConditionalTransferEntropy<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let destination_2d = destination.clone().insert_axis(Axis(1));
        let condition_2d = condition.clone().insert_axis(Axis(1));
        RenyiConditionalTransferEntropy::<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3>::new(
            &source_2d,
            &destination_2d,
            &condition_2d,
            k,
            alpha,
            noise_level,
        )
    }

    /// Create a multi-dimensional Rényi-based conditional transfer entropy estimator.
    pub fn nd_cte_renyi<
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
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiConditionalTransferEntropy<
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
        RenyiConditionalTransferEntropy::<
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
        >::new(source, destination, condition, k, alpha, noise_level)
    }

    /// Create a Tsallis-based conditional transfer entropy estimator.
    pub fn new_cte_tsallis(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        condition: &Array1<f64>,
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisConditionalTransferEntropy<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let destination_2d = destination.clone().insert_axis(Axis(1));
        let condition_2d = condition.clone().insert_axis(Axis(1));
        TsallisConditionalTransferEntropy::<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3>::new(
            &source_2d,
            &destination_2d,
            &condition_2d,
            k,
            q,
            noise_level,
        )
    }

    /// Create a multi-dimensional Tsallis-based conditional transfer entropy estimator.
    pub fn nd_cte_tsallis<
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
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisConditionalTransferEntropy<
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
        TsallisConditionalTransferEntropy::<
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
        >::new(source, destination, condition, k, q, noise_level)
    }

    /// Create a Kozachenko-Leonenko (KL) based conditional transfer entropy estimator.
    pub fn new_cte_kl(
        source: &Array1<f64>,
        destination: &Array1<f64>,
        condition: &Array1<f64>,
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoConditionalTransferEntropy<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3> {
        let source_2d = source.clone().insert_axis(Axis(1));
        let destination_2d = destination.clone().insert_axis(Axis(1));
        let condition_2d = condition.clone().insert_axis(Axis(1));
        KozachenkoLeonenkoConditionalTransferEntropy::<1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 3>::new(
            &source_2d,
            &destination_2d,
            &condition_2d,
            k,
            noise_level,
        )
    }

    /// Create a multi-dimensional Kozachenko-Leonenko (KL) based conditional transfer entropy estimator.
    pub fn nd_cte_kl<
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
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoConditionalTransferEntropy<
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
        KozachenkoLeonenkoConditionalTransferEntropy::<
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
        >::new(source, destination, condition, k, noise_level)
    }
}

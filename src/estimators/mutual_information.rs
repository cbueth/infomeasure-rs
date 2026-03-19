// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Mutual Information Estimators
//!
//! This module provides estimators for computing mutual information $I(X;Y)$ and
//! conditional mutual information $I(X;Y|Z)$.
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
//! ### Mutual Information (MI)
//!
//! $I(X_1; X_2; \dots; X_n) = \log p(X_1, \dots, X_n) - \sum \log p(X_i)$
//!
//! The estimators `KernelMutualInformation2` through `KernelMutualInformation6` support 2 to 6
//! random variables respectively. They use:
//! - `D_JOINT`: $\sum_{i=1}^n D_i$ - Total dimensionality of the joint space
//! - `D1`, `D2`, ..., `Dn`: Dimensions of individual random variables
//!
//! **Example for 2 variables**: If $X$ has dimension $D_1$ and $Y$ has dimension $D_2$, then
//! `D_JOINT = D_1 + D_2`.
//!
//! ### Conditional Mutual Information (CMI)
//!
//! $I(X; Y | Z) = \log \frac{p(X,Y,Z)p(Z)}{p(X,Z)p(Y,Z)}$
//!
//! The `KernelConditionalMutualInformation` and `KsgConditionalMutualInformation` structs use:
//! - `D1`, `D2`: Dimensions of the input variables X and Y
//! - `D_COND`: Dimension of the conditioning variable Z
//! - `D_JOINT`: $D_1 + D_2 + D_{cond}$ - Total dimensionality including condition
//! - `D1_COND`: $D_1 + D_{cond}$ - Dimension of X joined with condition
//! - `D2_COND`: $D_2 + D_{cond}$ - Dimension of Y joined with condition
//!
//! ### Helper Macros
//!
//! To simplify instantiation and automatically calculate these dimensions, use the following macros:
//! - `new_kernel_mi!` - Creates a `KernelMutualInformation` estimator (supports 2-6 variables)
//! - `new_kernel_cmi!` - Creates a `KernelConditionalMutualInformation` estimator
//! - `new_ksg_mi!` - Creates a `KsgMutualInformation` estimator
//! - `new_ksg_cmi!` - Creates a `KsgConditionalMutualInformation` estimator
//! - `new_renyi_mi!` - Creates a `RenyiMutualInformation` estimator
//! - `new_renyi_cmi!` - Creates a `RenyiConditionalMutualInformation` estimator
//! - `new_tsallis_mi!` - Creates a `TsallisMutualInformation` estimator
//! - `new_tsallis_cmi!` - Creates a `TsallisConditionalMutualInformation` estimator
//! - `new_kl_mi!` - Creates a KL-divergence based MI estimator
//! - `new_jsd_mi!` - Creates a JSD-based MI estimator
//! - `new_ordinal_mi!` - Creates an `OrdinalMutualInformation` estimator
//!
//! These macros handle the dimension calculations automatically based on the input
//! dimensionalities you provide.

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
use ndarray::{Array1, Array2, Axis};

use crate::estimators::approaches::kernel::{
    KernelConditionalMutualInformation, KernelMutualInformation2, KernelMutualInformation3,
    KernelMutualInformation4,
};

use crate::estimators::approaches::expfam::kozachenko_leonenko::{
    KozachenkoLeonenkoConditionalMutualInformation, KozachenkoLeonenkoMutualInformation2,
};
use crate::estimators::approaches::expfam::ksg::{
    KsgConditionalMutualInformation, KsgMutualInformation2,
};
use crate::estimators::approaches::expfam::renyi::{
    RenyiConditionalMutualInformation, RenyiMutualInformation2,
};
use crate::estimators::approaches::expfam::tsallis::{
    TsallisConditionalMutualInformation, TsallisMutualInformation2,
};
use crate::estimators::approaches::ordinal::ordinal_estimator::{
    OrdinalConditionalMutualInformation, OrdinalMutualInformation,
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

/// Macro for creating a new `KsgMutualInformation` estimator.
#[macro_export]
macro_rules! new_ksg_mi {
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr) => {{
        const D_JOINT: usize = $d1 + $d2;
        $crate::estimators::approaches::expfam::ksg::KsgMutualInformation2::<D_JOINT, $d1, $d2>::new($series, $k, $noise)
    }};
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3;
        $crate::estimators::approaches::expfam::ksg::KsgMutualInformation3::<D_JOINT, $d1, $d2, $d3>::new($series, $k, $noise)
    }};
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4;
        $crate::estimators::approaches::expfam::ksg::KsgMutualInformation4::<D_JOINT, $d1, $d2, $d3, $d4>::new($series, $k, $noise)
    }};
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5;
        $crate::estimators::approaches::expfam::ksg::KsgMutualInformation5::<D_JOINT, $d1, $d2, $d3, $d4, $d5>::new($series, $k, $noise)
    }};
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr, $d6:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5 + $d6;
        $crate::estimators::approaches::expfam::ksg::KsgMutualInformation6::<D_JOINT, $d1, $d2, $d3, $d4, $d5, $d6>::new($series, $k, $noise)
    }};
}

/// Macro for creating a new `KsgConditionalMutualInformation` estimator.
#[macro_export]
macro_rules! new_ksg_cmi {
    ($series:expr, $cond:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d_cond:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d_cond;
        const D1_COND: usize = $d1 + $d_cond;
        const D2_COND: usize = $d2 + $d_cond;
        $crate::estimators::approaches::expfam::ksg::KsgConditionalMutualInformation::<
            $d1,
            $d2,
            $d_cond,
            D_JOINT,
            D1_COND,
            D2_COND,
        >::new($series, $cond, $k, $noise)
    }};
}

/// Macro for creating a new `RenyiMutualInformation` estimator.
#[macro_export]
macro_rules! new_renyi_mi {
    ($series:expr, $k:expr, $alpha:expr, $noise:expr, $d1:expr, $d2:expr) => {{
        const D_JOINT: usize = $d1 + $d2;
        $crate::estimators::approaches::expfam::renyi::RenyiMutualInformation2::<D_JOINT, $d1, $d2>::new($series, $k, $alpha, $noise)
    }};
    ($series:expr, $k:expr, $alpha:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3;
        $crate::estimators::approaches::expfam::renyi::RenyiMutualInformation3::<D_JOINT, $d1, $d2, $d3>::new($series, $k, $alpha, $noise)
    }};
    ($series:expr, $k:expr, $alpha:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4;
        $crate::estimators::approaches::expfam::renyi::RenyiMutualInformation4::<D_JOINT, $d1, $d2, $d3, $d4>::new($series, $k, $alpha, $noise)
    }};
    ($series:expr, $k:expr, $alpha:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5;
        $crate::estimators::approaches::expfam::renyi::RenyiMutualInformation5::<D_JOINT, $d1, $d2, $d3, $d4, $d5>::new($series, $k, $alpha, $noise)
    }};
    ($series:expr, $k:expr, $alpha:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr, $d6:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5 + $d6;
        $crate::estimators::approaches::expfam::renyi::RenyiMutualInformation6::<D_JOINT, $d1, $d2, $d3, $d4, $d5, $d6>::new($series, $k, $alpha, $noise)
    }};
}

/// Macro for creating a new `RenyiConditionalMutualInformation` estimator.
#[macro_export]
macro_rules! new_renyi_cmi {
    ($series:expr, $cond:expr, $k:expr, $alpha:expr, $noise:expr, $d1:expr, $d2:expr, $d_cond:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d_cond;
        const D1_COND: usize = $d1 + $d_cond;
        const D2_COND: usize = $d2 + $d_cond;
        $crate::estimators::approaches::expfam::renyi::RenyiConditionalMutualInformation::<
            $d1,
            $d2,
            $d_cond,
            D_JOINT,
            D1_COND,
            D2_COND,
        >::new($series, $cond, $k, $alpha, $noise)
    }};
}

/// Macro for creating a new `TsallisMutualInformation` estimator.
#[macro_export]
macro_rules! new_tsallis_mi {
    ($series:expr, $k:expr, $q:expr, $noise:expr, $d1:expr, $d2:expr) => {{
        const D_JOINT: usize = $d1 + $d2;
        $crate::estimators::approaches::expfam::tsallis::TsallisMutualInformation2::<
            D_JOINT,
            $d1,
            $d2,
        >::new($series, $k, $q, $noise)
    }};
    ($series:expr, $k:expr, $q:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3;
        $crate::estimators::approaches::expfam::tsallis::TsallisMutualInformation3::<
            D_JOINT,
            $d1,
            $d2,
            $d3,
        >::new($series, $k, $q, $noise)
    }};
    ($series:expr, $k:expr, $q:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4;
        $crate::estimators::approaches::expfam::tsallis::TsallisMutualInformation4::<
            D_JOINT,
            $d1,
            $d2,
            $d3,
            $d4,
        >::new($series, $k, $q, $noise)
    }};
    ($series:expr, $k:expr, $q:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5;
        $crate::estimators::approaches::expfam::tsallis::TsallisMutualInformation5::<
            D_JOINT,
            $d1,
            $d2,
            $d3,
            $d4,
            $d5,
        >::new($series, $k, $q, $noise)
    }};
    ($series:expr, $k:expr, $q:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr, $d6:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5 + $d6;
        $crate::estimators::approaches::expfam::tsallis::TsallisMutualInformation6::<
            D_JOINT,
            $d1,
            $d2,
            $d3,
            $d4,
            $d5,
            $d6,
        >::new($series, $k, $q, $noise)
    }};
}

/// Macro for creating a new `TsallisConditionalMutualInformation` estimator.
#[macro_export]
macro_rules! new_tsallis_cmi {
    ($series:expr, $cond:expr, $k:expr, $q:expr, $noise:expr, $d1:expr, $d2:expr, $d_cond:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d_cond;
        const D1_COND: usize = $d1 + $d_cond;
        const D2_COND: usize = $d2 + $d_cond;
        $crate::estimators::approaches::expfam::tsallis::TsallisConditionalMutualInformation::<
            $d1,
            $d2,
            $d_cond,
            D_JOINT,
            D1_COND,
            D2_COND,
        >::new($series, $cond, $k, $q, $noise)
    }};
}

/// Macro for creating a new `KozachenkoLeonenkoMutualInformation` estimator.
#[macro_export]
macro_rules! new_kl_mi {
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr) => {{
        const D_JOINT: usize = $d1 + $d2;
        $crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoMutualInformation2::<D_JOINT, $d1, $d2>::new($series, $k, $noise)
    }};
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3;
        $crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoMutualInformation3::<D_JOINT, $d1, $d2, $d3>::new($series, $k, $noise)
    }};
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4;
        $crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoMutualInformation4::<D_JOINT, $d1, $d2, $d3, $d4>::new($series, $k, $noise)
    }};
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5;
        $crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoMutualInformation5::<D_JOINT, $d1, $d2, $d3, $d4, $d5>::new($series, $k, $noise)
    }};
    ($series:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d3:expr, $d4:expr, $d5:expr, $d6:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d3 + $d4 + $d5 + $d6;
        $crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoMutualInformation6::<D_JOINT, $d1, $d2, $d3, $d4, $d5, $d6>::new($series, $k, $noise)
    }};
}

/// Macro for creating a new `KozachenkoLeonenkoConditionalMutualInformation` estimator.
#[macro_export]
macro_rules! new_kl_cmi {
    ($series:expr, $cond:expr, $k:expr, $noise:expr, $d1:expr, $d2:expr, $d_cond:expr) => {{
        const D_JOINT: usize = $d1 + $d2 + $d_cond;
        const D1_COND: usize = $d1 + $d_cond;
        const D2_COND: usize = $d2 + $d_cond;
        $crate::estimators::approaches::expfam::kozachenko_leonenko::KozachenkoLeonenkoConditionalMutualInformation::<
            $d1,
            $d2,
            $d_cond,
            D_JOINT,
            D1_COND,
            D2_COND,
        >::new($series, $cond, $k, $noise)
    }};
}

pub struct MutualInformation;

/// Facade for creating mutual information (MI) and conditional mutual information (CMI) estimators.
///
/// This struct provides a unified interface for all MI/CMI estimation techniques supported
/// by the library. It includes methods for discrete, kernel-based, ordinal, and
/// exponential family (k-NN) estimators.
///
/// Each estimator can be used to compute the global MI value or local MI values
/// (if supported) using the [`GlobalValue`](crate::estimators::traits::GlobalValue) and [`LocalValues`](crate::estimators::traits::LocalValues) traits.
///
/// # Relationship to Other Measures
///
/// Mutual information is related to several other information-theoretic measures:
///
/// - **Entropy**: $I(X;X) = H(X)$ (MI with itself is entropy)
/// - **Conditional MI**: $I(X;Y|Z)$ - MI with a conditioning variable
/// - **Transfer Entropy**: $T_{X \to Y} = I(X^{(k)}; Y_{t+1} | Y^{(l)})$ - directed MI for time series
/// - **Conditional TE**: $T_{X \to Y|Z}$ - TE with conditioning
///
/// For a detailed conceptual guide, see the [Mutual Information Guide](crate::guide::mutual_information).
///
/// # Examples
///
/// This section provides examples for all MI/CMI estimators available through the `MutualInformation` facade.
///
/// ## Discrete MI Estimators
///
/// ### Maximum Likelihood (MLE)
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let y = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let mi = MutualInformation::new_discrete_mle(&[x, y]).global_value();
/// assert!(mi > 0.0); // correlated data has positive MI
/// ```
///
/// ### Miller–Madow (bias correction)
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let y = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let mi_mm = MutualInformation::new_discrete_miller_madow(&[x, y]).global_value();
/// assert!(mi_mm >= 0.0);
/// ```
///
/// ### Shrinkage (James–Stein)
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let y = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let mi_shrink = MutualInformation::new_discrete_shrink(&[x, y]).global_value();
/// assert!(mi_shrink >= 0.0);
/// ```
///
/// ### Grassberger
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let y = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let mi = MutualInformation::new_discrete_grassberger(&[x, y]).global_value();
/// assert!(mi >= 0.0);
/// ```
///
/// ### Chao–Shen
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let y = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let mi = MutualInformation::new_discrete_chao_shen(&[x, y]).global_value();
/// assert!(mi >= 0.0);
/// ```
///
/// ### NSB
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let y = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let mi = MutualInformation::new_discrete_nsb(&[x, y]).global_value();
/// assert!(mi >= 0.0);
/// ```
///
/// ### Bayes (Jeffrey prior)
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let y = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let mi = MutualInformation::new_discrete_bayes(&[x, y]).global_value();
/// assert!(mi >= 0.0);
/// ```
///
/// ## Continuous MI Estimators
///
/// **When to use**: Continuous estimators work best with real-valued data. Use:
/// - **Kernel**: When you need explicit control over bandwidth
/// - **KSG**: Generally preferred for most continuous use cases (more robust)
/// - **Rényi/Tsallis**: For generalized entropy approaches
///
/// ### Kernel MI (1D)
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let mi = MutualInformation::new_kernel(&[x, y], 1.0).global_value();
/// assert!(mi >= 0.0);
/// ```
///
/// ### Kernel MI (multi-dimensional)
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
/// let y = array![[0.0], [1.0], [2.0]];
/// let mi = MutualInformation::nd_kernel::<3, 2, 1>(&[x, y], 1.0).global_value();
/// assert!(mi >= 0.0);
/// ```
///
/// ### KSG (Kraskov–Stögbauer–Grassberger) MI
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let mi = MutualInformation::new_ksg(&[x, y], 3, 1e-10).global_value();
/// assert!(mi >= 0.0);
/// ```
///
/// ### Rényi MI
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let mi = MutualInformation::new_renyi(&[x, y], 3, 2.0, 1e-10).global_value();
/// assert!(mi >= 0.0);
/// ```
///
/// ## Ordinal MI Estimators
///
/// **When to use**: Ordinal estimators are ideal for time series analysis because they:
/// - Are robust to amplitude variations and monotonic transformations
/// - Capture temporal ordering through permutation patterns
/// - Work well with noisy data
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
/// let y = array![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
/// let mi = MutualInformation::new_ordinal(&[x, y], 3, 1, true).global_value();
/// assert!(mi >= 0.0);
/// ```
///
/// ## Conditional MI (CMI) Estimators
///
/// **When to use**: Use CMI when you need to:
/// - Measure dependence between X and Y while controlling for Z
/// - Detect direct vs indirect dependencies in networks
/// - Identify synergy (information only in combination) vs redundancy (shared info)
///
/// ### Discrete CMI
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let y = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let z = array![0, 0, 1, 1, 0, 0, 1, 1];
/// let cmi = MutualInformation::new_cmi_discrete_mle(&[x, y], &z).global_value();
/// assert!(cmi >= 0.0);
/// ```
///
/// ### Kernel CMI
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
/// let z = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
/// let cmi = MutualInformation::new_cmi_kernel(&[x, y], &z, 1.0).global_value();
/// assert!(cmi >= 0.0);
/// ```
///
/// ### KSG CMI
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
/// let z = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
/// let cmi = MutualInformation::new_cmi_ksg(&[x, y], &z, 3, 1e-10).global_value();
/// assert!(cmi >= 0.0);
/// ```
///
/// ### Ordinal CMI
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// let x = array![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
/// let y = array![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
/// let z = array![0.5, 1.5, 1.0, 2.5, 2.0, 3.5, 3.0, 4.5];
/// let cmi = MutualInformation::new_cmi_ordinal(&[x, y], &z, 3, 1, true).global_value();
/// assert!(cmi >= 0.0);
/// ```
///
/// ## Multivariate MI Example
///
/// For more than two variables:
///
/// ```rust
/// use infomeasure::estimators::mutual_information::MutualInformation;
/// use infomeasure::estimators::traits::GlobalValue;
/// use ndarray::array;
///
/// // Three variables: X, Y, Z
/// let x = array![0, 0, 0, 0, 1, 1, 1, 1];
/// let y = array![0, 0, 1, 1, 0, 0, 1, 1];
/// let z = array![0, 1, 0, 1, 0, 1, 0, 1];
///
/// // Pairwise MI between all pairs
/// let mi_xy = MutualInformation::new_discrete_mle(&[x.clone(), y.clone()]).global_value();
/// let mi_xz = MutualInformation::new_discrete_mle(&[x.clone(), z.clone()]).global_value();
/// let mi_yz = MutualInformation::new_discrete_mle(&[y.clone(), z.clone()]).global_value();
///
/// // All non-negative
/// assert!(mi_xy >= 0.0);
/// assert!(mi_xz >= 0.0);
/// assert!(mi_yz >= 0.0);
/// ```
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

    /// Create an Ordinal-based mutual information estimator.
    pub fn new_ordinal(
        series: &[Array1<f64>],
        order: usize,
        step_size: usize,
        stable: bool,
    ) -> OrdinalMutualInformation {
        OrdinalMutualInformation::new(series, order, step_size, stable)
    }

    /// Create an Ordinal-based conditional mutual information estimator.
    pub fn new_cmi_ordinal(
        series: &[Array1<f64>],
        cond: &Array1<f64>,
        order: usize,
        step_size: usize,
        stable: bool,
    ) -> OrdinalConditionalMutualInformation {
        OrdinalConditionalMutualInformation::new(series, cond, order, step_size, stable)
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
    /// Create a 3-variable Multi-dimensional Kernel-based mutual information estimator.
    pub fn nd_kernel3<const D_JOINT: usize, const D1: usize, const D2: usize, const D3: usize>(
        series: &[Array2<f64>],
        bandwidth: f64,
    ) -> KernelMutualInformation3<D_JOINT, D1, D2, D3> {
        KernelMutualInformation3::new(series, "box".to_string(), bandwidth)
    }

    /// Create a 3-variable Multi-dimensional Kernel-based mutual information estimator with specific kernel type.
    pub fn nd_kernel3_with_type<
        const D_JOINT: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
    >(
        series: &[Array2<f64>],
        kernel_type: String,
        bandwidth: f64,
    ) -> KernelMutualInformation3<D_JOINT, D1, D2, D3> {
        KernelMutualInformation3::new(series, kernel_type, bandwidth)
    }

    /// Create a 4-variable Multi-dimensional Kernel-based mutual information estimator.
    pub fn nd_kernel4<
        const D_JOINT: usize,
        const D1: usize,
        const D2: usize,
        const D3: usize,
        const D4: usize,
    >(
        series: &[Array2<f64>],
        bandwidth: f64,
    ) -> KernelMutualInformation4<D_JOINT, D1, D2, D3, D4> {
        KernelMutualInformation4::new(series, "box".to_string(), bandwidth)
    }

    /// Create a KSG (kNN-based) mutual information estimator.
    pub fn new_ksg(
        series: &[Array1<f64>],
        k: usize,
        noise_level: f64,
    ) -> KsgMutualInformation2<2, 1, 1> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        KsgMutualInformation2::new(&series_2d, k, noise_level)
    }

    /// Create a multi-dimensional KSG (kNN-based) mutual information estimator.
    pub fn nd_ksg<const D_JOINT: usize, const D1: usize, const D2: usize>(
        series: &[Array2<f64>],
        k: usize,
        noise_level: f64,
    ) -> KsgMutualInformation2<D_JOINT, D1, D2> {
        KsgMutualInformation2::new(series, k, noise_level)
    }

    /// Create a KSG (kNN-based) conditional mutual information estimator.
    pub fn new_cmi_ksg(
        series: &[Array1<f64>],
        cond: &Array1<f64>,
        k: usize,
        noise_level: f64,
    ) -> KsgConditionalMutualInformation<1, 1, 1, 3, 2, 2> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        let cond_2d = cond.clone().insert_axis(Axis(1));
        KsgConditionalMutualInformation::new(&series_2d, &cond_2d, k, noise_level)
    }

    /// Create a multi-dimensional KSG (kNN-based) conditional mutual information estimator.
    pub fn nd_cmi_ksg<
        const D1: usize,
        const D2: usize,
        const D_COND: usize,
        const D_JOINT: usize,
        const D1_COND: usize,
        const D2_COND: usize,
    >(
        series: &[Array2<f64>],
        cond: &Array2<f64>,
        k: usize,
        noise_level: f64,
    ) -> KsgConditionalMutualInformation<D1, D2, D_COND, D_JOINT, D1_COND, D2_COND> {
        KsgConditionalMutualInformation::new(series, cond, k, noise_level)
    }

    /// Create a Rényi mutual information estimator.
    pub fn new_renyi(
        series: &[Array1<f64>],
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiMutualInformation2<2, 1, 1> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        RenyiMutualInformation2::new(&series_2d, k, alpha, noise_level)
    }

    /// Create a multi-dimensional Rényi mutual information estimator.
    pub fn nd_renyi<const D_JOINT: usize, const D1: usize, const D2: usize>(
        series: &[Array2<f64>],
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiMutualInformation2<D_JOINT, D1, D2> {
        RenyiMutualInformation2::new(series, k, alpha, noise_level)
    }

    /// Create a Rényi conditional mutual information estimator.
    pub fn new_cmi_renyi(
        series: &[Array1<f64>],
        cond: &Array1<f64>,
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiConditionalMutualInformation<1, 1, 1, 3, 2, 2> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        let cond_2d = cond.clone().insert_axis(Axis(1));
        RenyiConditionalMutualInformation::new(&series_2d, &cond_2d, k, alpha, noise_level)
    }

    /// Create a multi-dimensional Rényi conditional mutual information estimator.
    pub fn nd_cmi_renyi<
        const D1: usize,
        const D2: usize,
        const DZ: usize,
        const D_JOINT: usize,
        const D1Z: usize,
        const D2Z: usize,
    >(
        series: &[Array2<f64>],
        cond: &Array2<f64>,
        k: usize,
        alpha: f64,
        noise_level: f64,
    ) -> RenyiConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z> {
        RenyiConditionalMutualInformation::new(series, cond, k, alpha, noise_level)
    }

    /// Create a Tsallis mutual information estimator.
    pub fn new_tsallis(
        series: &[Array1<f64>],
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisMutualInformation2<2, 1, 1> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        TsallisMutualInformation2::new(&series_2d, k, q, noise_level)
    }

    /// Create a multi-dimensional Tsallis mutual information estimator.
    pub fn nd_tsallis<const D_JOINT: usize, const D1: usize, const D2: usize>(
        series: &[Array2<f64>],
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisMutualInformation2<D_JOINT, D1, D2> {
        TsallisMutualInformation2::new(series, k, q, noise_level)
    }

    /// Create a Tsallis conditional mutual information estimator.
    pub fn new_cmi_tsallis(
        series: &[Array1<f64>],
        cond: &Array1<f64>,
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisConditionalMutualInformation<1, 1, 1, 3, 2, 2> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        let cond_2d = cond.clone().insert_axis(Axis(1));
        TsallisConditionalMutualInformation::new(&series_2d, &cond_2d, k, q, noise_level)
    }

    /// Create a multi-dimensional Tsallis conditional mutual information estimator.
    pub fn nd_cmi_tsallis<
        const D1: usize,
        const D2: usize,
        const DZ: usize,
        const D_JOINT: usize,
        const D1Z: usize,
        const D2Z: usize,
    >(
        series: &[Array2<f64>],
        cond: &Array2<f64>,
        k: usize,
        q: f64,
        noise_level: f64,
    ) -> TsallisConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z> {
        TsallisConditionalMutualInformation::new(series, cond, k, q, noise_level)
    }

    /// Create a Kozachenko-Leonenko (KL) mutual information estimator.
    pub fn new_kl(
        series: &[Array1<f64>],
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoMutualInformation2<2, 1, 1> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        KozachenkoLeonenkoMutualInformation2::new(&series_2d, k, noise_level)
    }

    /// Create a multi-dimensional Kozachenko-Leonenko (KL) mutual information estimator.
    pub fn nd_kl<const D_JOINT: usize, const D1: usize, const D2: usize>(
        series: &[Array2<f64>],
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoMutualInformation2<D_JOINT, D1, D2> {
        KozachenkoLeonenkoMutualInformation2::new(series, k, noise_level)
    }

    /// Create a Kozachenko-Leonenko (KL) conditional mutual information estimator.
    pub fn new_cmi_kl(
        series: &[Array1<f64>],
        cond: &Array1<f64>,
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoConditionalMutualInformation<1, 1, 1, 3, 2, 2> {
        let series_2d: Vec<Array2<f64>> = series
            .iter()
            .map(|s| s.clone().insert_axis(Axis(1)))
            .collect();
        let cond_2d = cond.clone().insert_axis(Axis(1));
        KozachenkoLeonenkoConditionalMutualInformation::new(&series_2d, &cond_2d, k, noise_level)
    }

    /// Create a multi-dimensional Kozachenko-Leonenko (KL) conditional mutual information estimator.
    pub fn nd_cmi_kl<
        const D1: usize,
        const D2: usize,
        const DZ: usize,
        const D_JOINT: usize,
        const D1Z: usize,
        const D2Z: usize,
    >(
        series: &[Array2<f64>],
        cond: &Array2<f64>,
        k: usize,
        noise_level: f64,
    ) -> KozachenkoLeonenkoConditionalMutualInformation<D1, D2, DZ, D_JOINT, D1Z, D2Z> {
        KozachenkoLeonenkoConditionalMutualInformation::new(series, cond, k, noise_level)
    }
}

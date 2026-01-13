
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
    
    // pub fn new_discrete() -> discrete
    // 
    // pub fn new_discrete(data_x: Vec<i32>, data_y: Vec<i32>) -> discrete::DiscreteMutualInformation {
    //     discrete::DiscreteMutualInformation::new(data_x, data_y)
    // }
    // 
    // pub fn new_kernel(data_x: Vec<f64>, data_y: Vec<f64>) -> kernel::KernelMutualInformation {
    //     kernel::KernelMutualInformation::new(data_x, data_y)
    // }
    // 
    // pub fn new_ordinal(data_x: Vec<i32>, data_y: Vec<i32>) -> ordinal::OrdinalMutualInformation {
    //     ordinal::OrdinalMutualInformation::new(data_x, data_y)
    // }
}
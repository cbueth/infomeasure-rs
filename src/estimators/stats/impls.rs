// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Resample` glue for the continuous estimator families (kernel and KSG).
//!
//! Only the source variable (the first input for mutual information, the
//! source series for transfer entropy) is resampled; the target and condition
//! are held fixed, matching the Python package's semantics.

use rand::rngs::SmallRng;

use crate::estimators::approaches::expfam::ksg::{
    KsgMutualInformation2, KsgMutualInformation3, KsgMutualInformation4,
};
use crate::estimators::approaches::kernel::{
    KernelMutualInformation2, KernelMutualInformation3, KernelMutualInformation4,
    KernelMutualInformation5, KernelMutualInformation6,
};

use super::config::TestMethod;
use super::driver::resample_rows;
use super::traits::Resample;

macro_rules! impl_kernel_mi_resample {
    ($name:ident, $($d:ident),+) => {
        impl<const D_JOINT: usize, $(const $d: usize),+> Resample for $name<D_JOINT, $($d),+> {
            fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
                let mut data = self.data.clone();
                data[0] = resample_rows(&data[0], method, rng);
                let mut est = Self::new(&data, self.kernel_type.clone(), self.bandwidth);
                est.force_cpu = self.force_cpu;
                est
            }
        }
    };
}

impl_kernel_mi_resample!(KernelMutualInformation2, D1, D2);
impl_kernel_mi_resample!(KernelMutualInformation3, D1, D2, D3);
impl_kernel_mi_resample!(KernelMutualInformation4, D1, D2, D3, D4);
impl_kernel_mi_resample!(KernelMutualInformation5, D1, D2, D3, D4, D5);
impl_kernel_mi_resample!(KernelMutualInformation6, D1, D2, D3, D4, D5, D6);

macro_rules! impl_ksg_mi_resample {
    ($name:ident, $($d:ident),+) => {
        impl<const D_JOINT: usize, $(const $d: usize),+> Resample for $name<D_JOINT, $($d),+> {
            fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
                let mut data = self.data.clone();
                data[0] = resample_rows(&data[0], method, rng);
                // Rebuild without re-adding noise: the stored data is already
                // noised, so `noise_level = 0.0` leaves it untouched.
                let mut est = Self::new(&data, self.k, 0.0);
                est.ksg_type = self.ksg_type;
                est.base = self.base;
                est.noise_level = self.noise_level;
                est.use_chebyshev = self.use_chebyshev;
                est
            }
        }
    };
}

impl_ksg_mi_resample!(KsgMutualInformation2, D1, D2);
impl_ksg_mi_resample!(KsgMutualInformation3, D1, D2, D3);
impl_ksg_mi_resample!(KsgMutualInformation4, D1, D2, D3, D4);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimators::mutual_information::MutualInformation;
    use crate::estimators::stats::{StatisticalTest, StatisticalTestConfig};
    use ndarray::Array1;

    fn ramp(n: usize) -> Array1<f64> {
        Array1::from((0..n).map(|i| i as f64).collect::<Vec<_>>())
    }

    #[test]
    fn dependent_kernel_mi_is_significant() {
        // A smoothing bandwidth is required: at tiny bandwidths the kernel MI
        // degenerates to ln(N) regardless of the pairing.
        let mi = MutualInformation::new_kernel(&[ramp(40), ramp(40)], 2.0);
        let res = mi
            .statistical_test(&StatisticalTestConfig::permutation(50).seed(1))
            .unwrap();
        assert!(
            res.p_value <= 0.05,
            "expected a significant result, got p = {}",
            res.p_value
        );
        assert!(res.observed_value > res.null_mean);
        assert!(res.t_score > 3.0);
    }

    #[test]
    fn dependent_ksg_mi_is_significant() {
        let mi = MutualInformation::new_ksg(&[ramp(40), ramp(40)], 4, 1e-10);
        let res = mi
            .statistical_test(&StatisticalTestConfig::permutation(30).seed(1))
            .unwrap();
        assert!(
            res.p_value <= 0.1,
            "expected a significant result, got p = {}",
            res.p_value
        );
    }

    #[test]
    fn result_is_reproducible_with_a_seed() {
        let mi = MutualInformation::new_kernel(&[ramp(40), ramp(40)], 2.0);
        let cfg = StatisticalTestConfig::permutation(20).seed(99);
        let a = mi.statistical_test(&cfg).unwrap();
        let b = mi.statistical_test(&cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn bootstrap_also_runs() {
        let mi = MutualInformation::new_ksg(&[ramp(30), ramp(30)], 4, 1e-10);
        let res = mi
            .statistical_test(&StatisticalTestConfig::bootstrap(20).seed(3))
            .unwrap();
        assert_eq!(res.method, TestMethod::Bootstrap);
        assert!((0.0..=1.0).contains(&res.p_value));
    }
}

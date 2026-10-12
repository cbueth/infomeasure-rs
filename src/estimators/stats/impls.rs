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
    KsgConditionalMutualInformation, KsgConditionalTransferEntropy, KsgMutualInformation2,
    KsgMutualInformation3, KsgMutualInformation4, KsgTransferEntropy,
};
use crate::estimators::approaches::kernel::{
    KernelConditionalMutualInformation, KernelConditionalTransferEntropy, KernelMutualInformation2,
    KernelMutualInformation3, KernelMutualInformation4, KernelMutualInformation5,
    KernelMutualInformation6, KernelTransferEntropy,
};
use crate::estimators::approaches::ordinal::ordinal_estimator::{
    OrdinalConditionalMutualInformation, OrdinalConditionalTransferEntropy,
    OrdinalMutualInformation, OrdinalTransferEntropy,
};

use super::config::TestMethod;
use super::driver::{resample_1d, resample_rows};
use super::traits::{EffectiveValueMeasure, Resample};

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

macro_rules! impl_kernel_cmi_resample {
    ($name:ident, $($d:ident),+) => {
        impl<const D1: usize, $(const $d: usize),+> Resample for $name<D1, $($d),+> {
            fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
                let mut series = self.series.clone();
                series[0] = resample_rows(&series[0], method, rng);
                let mut est = Self::new(
                    &series,
                    &self.cond,
                    self.kernel_type.clone(),
                    self.bandwidth,
                );
                est.force_cpu = self.force_cpu;
                est
            }
        }
    };
}

macro_rules! impl_ksg_cmi_resample {
    ($name:ident, $($d:ident),+) => {
        impl<const D1: usize, $(const $d: usize),+> Resample for $name<D1, $($d),+> {
            fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
                let mut data = self.data.clone();
                data[0] = resample_rows(&data[0], method, rng);
                let mut est = Self::new(&data, &self.cond, self.k, 0.0);
                est.ksg_type = self.ksg_type;
                est.base = self.base;
                est.noise_level = self.noise_level;
                est.use_chebyshev = self.use_chebyshev;
                est
            }
        }
    };
}

impl_kernel_cmi_resample!(
    KernelConditionalMutualInformation,
    D2,
    D_COND,
    D_JOINT,
    D1_COND,
    D2_COND
);
impl_ksg_cmi_resample!(
    KsgConditionalMutualInformation,
    D2,
    D_COND,
    D_JOINT,
    D1_COND,
    D2_COND
);

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
    fn rebuild(&self, source: &ndarray::Array2<f64>) -> Self {
        Self::new(
            source,
            &self.dest,
            SRC_HIST,
            DEST_HIST,
            STEP_SIZE,
            self.kernel_type.clone(),
            self.bandwidth,
        )
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
    fn rebuild(&self, source: &ndarray::Array2<f64>) -> Self {
        Self::new(
            source,
            &self.dest,
            &self.cond,
            SRC_HIST,
            DEST_HIST,
            COND_HIST,
            STEP_SIZE,
            self.kernel_type.clone(),
            self.bandwidth,
        )
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
> Resample
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
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
        let source = resample_rows(&self.source, method, rng);
        let mut est = self.rebuild(&source);
        est.force_cpu = self.force_cpu;
        est
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
> EffectiveValueMeasure
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
    const COND_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D_XP_YP_ZP: usize,
    const D_YP_ZP: usize,
    const D_YF_YP_ZP: usize,
> Resample
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
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
        let source = resample_rows(&self.source, method, rng);
        let mut est = self.rebuild(&source);
        est.force_cpu = self.force_cpu;
        est
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
> EffectiveValueMeasure
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
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_JOINT: usize,
    const D_XP_YP: usize,
    const D_YP: usize,
    const D_YF_YP: usize,
> Resample
    for KsgTransferEntropy<
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
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
        let source = resample_rows(&self.source, method, rng);
        Self::new(
            &source,
            &self.dest,
            self.internal_cmi.k,
            self.internal_cmi.noise_level,
        )
        .with_type(self.internal_cmi.ksg_type)
        .with_base(self.internal_cmi.base)
        .with_chebyshev(self.internal_cmi.use_chebyshev)
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
> EffectiveValueMeasure
    for KsgTransferEntropy<
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
    const COND_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D_XP_YP_ZP: usize,
    const D_YP_ZP: usize,
    const D_YF_YP_ZP: usize,
> Resample
    for KsgConditionalTransferEntropy<
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
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
        let source = resample_rows(&self.source, method, rng);
        Self::new(
            &source,
            &self.dest,
            &self.cond,
            self.internal_cmi.k,
            self.internal_cmi.noise_level,
        )
        .with_type(self.internal_cmi.ksg_type)
        .with_base(self.internal_cmi.base)
        .with_chebyshev(self.internal_cmi.use_chebyshev)
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
> EffectiveValueMeasure
    for KsgConditionalTransferEntropy<
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

impl Resample for OrdinalMutualInformation {
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
        let mut series = self.series.clone();
        series[0] = resample_1d(&series[0], method, rng);
        Self::new(&series, self.order, self.step_size, self.stable)
    }
}

impl Resample for OrdinalConditionalMutualInformation {
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
        let mut series = self.series.clone();
        series[0] = resample_1d(&series[0], method, rng);
        Self::new(&series, &self.cond, self.order, self.step_size, self.stable)
    }
}

impl Resample for OrdinalTransferEntropy {
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
        let source = resample_1d(&self.source, method, rng);
        Self::new(
            &source,
            &self.dest,
            self.order,
            self.src_hist_len,
            self.dest_hist_len,
            self.step_size,
            self.stable,
        )
    }
}

impl EffectiveValueMeasure for OrdinalTransferEntropy {}

impl Resample for OrdinalConditionalTransferEntropy {
    fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
        let source = resample_1d(&self.source, method, rng);
        Self::new(
            &source,
            &self.dest,
            &self.cond,
            self.order,
            self.src_hist_len,
            self.dest_hist_len,
            self.cond_hist_len,
            self.step_size,
            self.stable,
        )
    }
}

impl EffectiveValueMeasure for OrdinalConditionalTransferEntropy {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimators::mutual_information::MutualInformation;
    use crate::estimators::stats::{EffectiveValue, StatisticalTest, StatisticalTestConfig};
    use crate::estimators::transfer_entropy::TransferEntropy;
    use ndarray::{Array1, Axis};

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

    fn wave(n: usize) -> Array1<f64> {
        Array1::from((0..n).map(|i| (i as f64 * 0.5).sin()).collect::<Vec<_>>())
    }

    #[test]
    fn kernel_te_effective_value_is_reproducible() {
        let te = TransferEntropy::new_kernel(&ramp(30), &wave(30), 1, 1, 1, 0.5);
        let cfg = StatisticalTestConfig::permutation(1).seed(5);
        let a = te.effective_value(&cfg).unwrap();
        let b = te.effective_value(&cfg).unwrap();
        assert!(a.is_finite());
        assert_eq!(a, b);
    }

    #[test]
    fn kernel_te_statistical_test_runs() {
        let te = TransferEntropy::new_kernel(&ramp(30), &wave(30), 1, 1, 1, 0.5);
        let res = te
            .statistical_test(&StatisticalTestConfig::permutation(20).seed(1))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
        assert!(res.observed_value.is_finite());
    }

    #[test]
    fn dependent_kernel_cmi_is_significant() {
        let cmi = MutualInformation::new_cmi_kernel(&[ramp(40), ramp(40)], &wave(40), 2.0);
        let res = cmi
            .statistical_test(&StatisticalTestConfig::permutation(30).seed(1))
            .unwrap();
        assert!(
            res.p_value <= 0.1,
            "expected a significant result, got p = {}",
            res.p_value
        );
    }

    #[test]
    fn ksg_cmi_resamples_and_runs() {
        let series = [ramp(30).insert_axis(Axis(1)), ramp(30).insert_axis(Axis(1))];
        let cond = wave(30).insert_axis(Axis(1));
        let cmi =
            KsgConditionalMutualInformation::<1, 1, 1, 3, 2, 2>::new(&series, &cond, 4, 1e-10);
        let res = cmi
            .statistical_test(&StatisticalTestConfig::permutation(20).seed(1))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
    }

    #[test]
    fn ksg_te_statistical_test_and_effective_value() {
        let te = TransferEntropy::new_ksg(&ramp(30), &wave(30), 1, 1, 1, 4, 1e-10);
        let res = te
            .statistical_test(&StatisticalTestConfig::permutation(20).seed(1))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
        let ete = te
            .effective_value(&StatisticalTestConfig::permutation(1).seed(1))
            .unwrap();
        assert!(ete.is_finite());
    }

    #[test]
    fn ksg_cte_resamples_and_runs() {
        let cte =
            TransferEntropy::new_cte_ksg(&ramp(30), &wave(30), &wave(30), 1, 1, 1, 1, 4, 1e-10);
        let res = cte
            .statistical_test(&StatisticalTestConfig::permutation(20).seed(1))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
    }

    #[test]
    fn ordinal_mi_statistical_test_runs() {
        let mi = MutualInformation::new_ordinal(&[ramp(40), wave(40)], 3, 1, false);
        let res = mi
            .statistical_test(&StatisticalTestConfig::permutation(30).seed(1))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
    }

    #[test]
    fn ordinal_te_statistical_test_and_effective_value() {
        let te = TransferEntropy::new_ordinal(&ramp(40), &wave(40), 3, 1, 1, 1, false);
        let res = te
            .statistical_test(&StatisticalTestConfig::permutation(30).seed(1))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
        let ete = te
            .effective_value(&StatisticalTestConfig::permutation(1).seed(1))
            .unwrap();
        assert!(ete.is_finite());
    }
}

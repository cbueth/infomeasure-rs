// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `Resample` glue for the continuous estimator families (kernel and KSG).
//!
//! Only the source variable (the first input for mutual information, the
//! source series for transfer entropy) is resampled; the target and condition
//! are held fixed, matching the Python package's semantics.

use rand::rngs::SmallRng;

use crate::estimators::approaches::expfam::kozachenko_leonenko::{
    KozachenkoLeonenkoConditionalMutualInformation, KozachenkoLeonenkoConditionalTransferEntropy,
    KozachenkoLeonenkoMutualInformation2, KozachenkoLeonenkoMutualInformation3,
    KozachenkoLeonenkoMutualInformation4, KozachenkoLeonenkoMutualInformation5,
    KozachenkoLeonenkoMutualInformation6, KozachenkoLeonenkoTransferEntropy,
};
use crate::estimators::approaches::expfam::ksg::{
    KsgConditionalMutualInformation, KsgConditionalTransferEntropy, KsgMutualInformation2,
    KsgMutualInformation3, KsgMutualInformation4, KsgTransferEntropy,
};
use crate::estimators::approaches::expfam::renyi::{
    RenyiConditionalMutualInformation, RenyiConditionalTransferEntropy, RenyiMutualInformation2,
    RenyiMutualInformation3, RenyiMutualInformation4, RenyiMutualInformation5,
    RenyiMutualInformation6, RenyiTransferEntropy,
};
use crate::estimators::approaches::expfam::tsallis::{
    TsallisConditionalMutualInformation, TsallisConditionalTransferEntropy,
    TsallisMutualInformation2, TsallisMutualInformation3, TsallisMutualInformation4,
    TsallisMutualInformation5, TsallisMutualInformation6, TsallisTransferEntropy,
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

macro_rules! impl_expfam_mi_resample {
    ($name:ident, $($d:ident),+) => {
        impl<const D_JOINT: usize, $(const $d: usize),+> Resample for $name<D_JOINT, $($d),+> {
            fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
                let mut out = self.clone();
                out.data[0] = resample_rows(&out.data[0], method, rng);
                out
            }
        }
    };
}

impl_expfam_mi_resample!(RenyiMutualInformation2, D1, D2);
impl_expfam_mi_resample!(RenyiMutualInformation3, D1, D2, D3);
impl_expfam_mi_resample!(RenyiMutualInformation4, D1, D2, D3, D4);
impl_expfam_mi_resample!(RenyiMutualInformation5, D1, D2, D3, D4, D5);
impl_expfam_mi_resample!(RenyiMutualInformation6, D1, D2, D3, D4, D5, D6);
impl_expfam_mi_resample!(TsallisMutualInformation2, D1, D2);
impl_expfam_mi_resample!(TsallisMutualInformation3, D1, D2, D3);
impl_expfam_mi_resample!(TsallisMutualInformation4, D1, D2, D3, D4);
impl_expfam_mi_resample!(TsallisMutualInformation5, D1, D2, D3, D4, D5);
impl_expfam_mi_resample!(TsallisMutualInformation6, D1, D2, D3, D4, D5, D6);
impl_expfam_mi_resample!(KozachenkoLeonenkoMutualInformation2, D1, D2);
impl_expfam_mi_resample!(KozachenkoLeonenkoMutualInformation3, D1, D2, D3);
impl_expfam_mi_resample!(KozachenkoLeonenkoMutualInformation4, D1, D2, D3, D4);
impl_expfam_mi_resample!(KozachenkoLeonenkoMutualInformation5, D1, D2, D3, D4, D5);
impl_expfam_mi_resample!(KozachenkoLeonenkoMutualInformation6, D1, D2, D3, D4, D5, D6);

macro_rules! impl_expfam_cmi_resample {
    ($name:ident) => {
        impl<
            const D1: usize,
            const D2: usize,
            const DZ: usize,
            const D_JOINT: usize,
            const D1Z: usize,
            const D2Z: usize,
        > Resample for $name<D1, D2, DZ, D_JOINT, D1Z, D2Z>
        {
            fn resampled(&self, method: TestMethod, rng: &mut SmallRng) -> Self {
                let mut out = self.clone();
                out.data[0] = resample_rows(&out.data[0], method, rng);
                out
            }
        }
    };
}

impl_expfam_cmi_resample!(RenyiConditionalMutualInformation);
impl_expfam_cmi_resample!(TsallisConditionalMutualInformation);
impl_expfam_cmi_resample!(KozachenkoLeonenkoConditionalMutualInformation);

macro_rules! impl_expfam_te_resample {
    ($name:ident) => {
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
            for $name<
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
                let mut out = self.clone();
                out.source = resample_rows(&out.source, method, rng);
                out
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
            for $name<
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
    };
}

impl_expfam_te_resample!(RenyiTransferEntropy);
impl_expfam_te_resample!(TsallisTransferEntropy);
impl_expfam_te_resample!(KozachenkoLeonenkoTransferEntropy);

macro_rules! impl_expfam_cte_resample {
    ($name:ident) => {
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
            for $name<
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
                let mut out = self.clone();
                out.source = resample_rows(&out.source, method, rng);
                out
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
            for $name<
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
    };
}

impl_expfam_cte_resample!(RenyiConditionalTransferEntropy);
impl_expfam_cte_resample!(TsallisConditionalTransferEntropy);
impl_expfam_cte_resample!(KozachenkoLeonenkoConditionalTransferEntropy);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimators::approaches::discrete::DiscreteMutualInformation;
    use crate::estimators::approaches::discrete::mle::DiscreteEntropy;
    use crate::estimators::mutual_information::MutualInformation;
    use crate::estimators::stats::driver::{make_rng, resample_1d_i32};
    use crate::estimators::stats::{
        EffectiveValue, Resample, StatisticalTest, StatisticalTestConfig, TestMethod,
    };
    use crate::estimators::traits::GlobalValue;
    use crate::estimators::transfer_entropy::TransferEntropy;
    use ndarray::{Array1, Axis};

    fn permuted_i32(series: &Array1<i32>, seed: u64) -> Array1<i32> {
        let mut rng = make_rng(Some(seed));
        resample_1d_i32(series, TestMethod::Permutation, &mut rng)
    }

    fn permuted_f64(series: &Array1<f64>, seed: u64) -> Array1<f64> {
        let mut rng = make_rng(Some(seed));
        resample_1d(series, TestMethod::Permutation, &mut rng)
    }

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
        let a = te.effective_value(TestMethod::Permutation, Some(5));
        let b = te.effective_value(TestMethod::Permutation, Some(5));
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
        let ete = te.effective_value(TestMethod::Permutation, Some(1));
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
        let ete = te.effective_value(TestMethod::Permutation, Some(1));
        assert!(ete.is_finite());
    }

    fn symbols(n: usize, alphabet: i32) -> Array1<i32> {
        Array1::from((0..n).map(|i| (i as i32) % alphabet).collect::<Vec<_>>())
    }

    #[test]
    fn discrete_mi_is_significant() {
        // y = x: the joint is perfectly dependent, so the observed MI is high and
        // permuting the source collapses it.
        let mi = MutualInformation::new_discrete_mle(&[symbols(40, 5), symbols(40, 5)]);
        let res = mi
            .statistical_test(&StatisticalTestConfig::permutation(50).seed(1))
            .unwrap();
        assert!(
            res.p_value <= 0.05,
            "expected a significant result, got p = {}",
            res.p_value
        );
        assert!(res.observed_value > res.null_mean);
    }

    #[test]
    fn discrete_bias_corrected_mi_runs() {
        let mi = MutualInformation::new_discrete_miller_madow(&[symbols(40, 5), symbols(40, 7)]);
        let res = mi
            .statistical_test(&StatisticalTestConfig::permutation(20).seed(2))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
    }

    #[test]
    fn discrete_cmi_is_significant() {
        // Y = X, Z independent: I(X; Y | Z) is large, and permuting the source
        // collapses it.
        let x = symbols(40, 5);
        let cmi = MutualInformation::new_cmi_discrete_mle(&[x.clone(), x], &symbols(40, 7));
        let res = cmi
            .statistical_test(&StatisticalTestConfig::permutation(40).seed(1))
            .unwrap();
        assert!(
            res.p_value <= 0.1,
            "expected a significant result, got p = {}",
            res.p_value
        );
    }

    #[test]
    fn discrete_bias_corrected_cmi_runs() {
        let cmi = MutualInformation::new_cmi_discrete_miller_madow(
            &[symbols(40, 5), symbols(40, 7)],
            &symbols(40, 3),
        );
        let res = cmi
            .statistical_test(&StatisticalTestConfig::permutation(20).seed(2))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
    }

    #[test]
    fn discrete_te_statistical_test_and_effective_value() {
        let te = TransferEntropy::new_discrete_mle(&symbols(60, 3), &symbols(60, 3), 1, 1, 1);
        let res = te
            .statistical_test(&StatisticalTestConfig::permutation(30).seed(1))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
        let ete = te.effective_value(TestMethod::Permutation, Some(1));
        assert!(ete.is_finite());
    }

    #[test]
    fn discrete_cte_resamples_and_runs() {
        let cte = TransferEntropy::new_cte_discrete_mle(
            &symbols(60, 3),
            &symbols(60, 3),
            &symbols(60, 5),
            1,
            1,
            1,
            1,
        );
        let res = cte
            .statistical_test(&StatisticalTestConfig::permutation(20).seed(1))
            .unwrap();
        assert!((0.0..=1.0).contains(&res.p_value));
    }

    macro_rules! expfam_smoke {
        ($mi:expr, $cmi:expr, $te:expr, $cte:expr $(,)?) => {{
            let cfg = StatisticalTestConfig::permutation(15).seed(1);
            let mi = $mi;
            assert!((0.0..=1.0).contains(&mi.statistical_test(&cfg).unwrap().p_value));
            let cmi = $cmi;
            assert!((0.0..=1.0).contains(&cmi.statistical_test(&cfg).unwrap().p_value));
            let te = $te;
            assert!((0.0..=1.0).contains(&te.statistical_test(&cfg).unwrap().p_value));
            assert!(
                te.effective_value(TestMethod::Permutation, Some(1))
                    .is_finite()
            );
            let cte = $cte;
            assert!((0.0..=1.0).contains(&cte.statistical_test(&cfg).unwrap().p_value));
            assert!(
                cte.effective_value(TestMethod::Permutation, Some(1))
                    .is_finite()
            );
        }};
    }

    #[test]
    fn renyi_family_significance_smoke() {
        expfam_smoke!(
            MutualInformation::new_renyi(&[wave(50), wave(50)], 4, 0.9, 1e-10),
            MutualInformation::new_cmi_renyi(&[wave(50), wave(50)], &ramp(50), 4, 0.9, 1e-10),
            TransferEntropy::new_renyi(&ramp(50), &wave(50), 4, 0.9, 1e-10),
            TransferEntropy::new_cte_renyi(&ramp(50), &wave(50), &ramp(50), 4, 0.9, 1e-10),
        );
    }

    #[test]
    fn tsallis_family_significance_smoke() {
        expfam_smoke!(
            MutualInformation::new_tsallis(&[wave(50), wave(50)], 4, 1.1, 1e-10),
            MutualInformation::new_cmi_tsallis(&[wave(50), wave(50)], &ramp(50), 4, 1.1, 1e-10),
            TransferEntropy::new_tsallis(&ramp(50), &wave(50), 4, 1.1, 1e-10),
            TransferEntropy::new_cte_tsallis(&ramp(50), &wave(50), &ramp(50), 4, 1.1, 1e-10),
        );
    }

    #[test]
    fn kl_family_significance_smoke() {
        expfam_smoke!(
            MutualInformation::new_kl(&[wave(50), wave(50)], 4, 1e-10),
            MutualInformation::new_cmi_kl(&[wave(50), wave(50)], &ramp(50), 4, 1e-10),
            TransferEntropy::new_kl(&ramp(50), &wave(50), 4, 1e-10),
            TransferEntropy::new_cte_kl(&ramp(50), &wave(50), &ramp(50), 4, 1e-10),
        );
    }

    macro_rules! assert_resample_matches_rebuild {
        ($seed:expr, $estimator:expr, $rebuilt:expr $(,)?) => {{
            let estimator = $estimator;
            let mut rng = make_rng(Some($seed));
            let got = estimator
                .resampled(TestMethod::Permutation, &mut rng)
                .global_value();
            let expected = $rebuilt;
            assert!(
                (got - expected).abs() < 1e-12,
                "resampled {got} != explicit rebuild {expected}"
            );
        }};
    }

    #[test]
    fn discrete_mi_resample_matches_explicit_rebuild() {
        let x = symbols(40, 5);
        let y = symbols(40, 7);
        assert_resample_matches_rebuild!(
            3,
            MutualInformation::new_discrete_mle(&[x.clone(), y.clone()]),
            MutualInformation::new_discrete_mle(&[permuted_i32(&x, 3), y]).global_value()
        );
    }

    #[test]
    fn discrete_cmi_resample_matches_explicit_rebuild() {
        let x = symbols(40, 5);
        let y = symbols(40, 7);
        let z = symbols(40, 3);
        assert_resample_matches_rebuild!(
            4,
            MutualInformation::new_cmi_discrete_mle(&[x.clone(), y.clone()], &z),
            MutualInformation::new_cmi_discrete_mle(&[permuted_i32(&x, 4), y], &z).global_value()
        );
    }

    #[test]
    fn discrete_te_resample_matches_explicit_rebuild() {
        let src = symbols(60, 3);
        let dst = symbols(60, 5);
        assert_resample_matches_rebuild!(
            5,
            TransferEntropy::new_discrete_mle(&src.clone(), &dst.clone(), 1, 1, 1),
            TransferEntropy::new_discrete_mle(&permuted_i32(&src, 5), &dst, 1, 1, 1).global_value()
        );
    }

    #[test]
    fn discrete_cte_resample_matches_explicit_rebuild() {
        let src = symbols(60, 3);
        let dst = symbols(60, 5);
        let cond = symbols(60, 2);
        assert_resample_matches_rebuild!(
            6,
            TransferEntropy::new_cte_discrete_mle(
                &src.clone(),
                &dst.clone(),
                &cond.clone(),
                1,
                1,
                1,
                1
            ),
            TransferEntropy::new_cte_discrete_mle(&permuted_i32(&src, 6), &dst, &cond, 1, 1, 1, 1)
                .global_value()
        );
    }

    #[test]
    fn kernel_mi_resample_matches_explicit_rebuild() {
        let x = ramp(40);
        let y = wave(40);
        assert_resample_matches_rebuild!(
            7,
            MutualInformation::new_kernel(&[x.clone(), y.clone()], 0.5),
            MutualInformation::new_kernel(&[permuted_f64(&x, 7), y], 0.5).global_value()
        );
    }

    #[test]
    fn ordinal_mi_resample_matches_explicit_rebuild() {
        let x = ramp(40);
        let y = wave(40);
        assert_resample_matches_rebuild!(
            8,
            MutualInformation::new_ordinal(&[x.clone(), y.clone()], 3, 1, false),
            MutualInformation::new_ordinal(&[permuted_f64(&x, 8), y], 3, 1, false).global_value()
        );
    }

    #[test]
    fn ordinal_te_resample_matches_explicit_rebuild() {
        let src = ramp(40);
        let dst = wave(40);
        assert_resample_matches_rebuild!(
            9,
            TransferEntropy::new_ordinal(&src.clone(), &dst.clone(), 3, 1, 1, 1, false),
            TransferEntropy::new_ordinal(&permuted_f64(&src, 9), &dst, 3, 1, 1, 1, false)
                .global_value()
        );
    }

    #[test]
    fn discrete_mi_prepared_matches_resampled() {
        // Force the generic `Spaces` representation so the cached-marginal fast
        // path is exercised.
        let mi =
            DiscreteMutualInformation::new(&[symbols(40, 5), symbols(40, 7)], DiscreteEntropy::new);
        let prepared = mi
            .prepared(TestMethod::Permutation)
            .expect("spaces MI caches its marginals");
        let mut rng = make_rng(Some(11));
        let got = prepared(0, &mut rng);
        let mut rng = make_rng(Some(11));
        let expected = mi
            .resampled(TestMethod::Permutation, &mut rng)
            .global_value();
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }
}

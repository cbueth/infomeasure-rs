// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! GPU host side for the KSG marginal and conditional neighbour counts.
//!
//! Tier T2: the same dense fixed-radius count as the box kernel, but with
//! per-query radii (the joint k-th-neighbour distances the estimator already
//! computed on the CPU). All spaces of one measure go into a single submit.
//!
//! Counts are integers, but the shader evaluates f32 distances, so a candidate
//! exactly at the radius boundary could be classified differently than on the
//! CPU. Parity is checked by the estimator suites; the ambiguous case is
//! handled there if it ever shows up.

use crate::estimators::gpu::{BatchJob, GpuContext, ShaderKind};
use ndarray::ArrayView2;

const KSG_COUNT_WGSL: &str = include_str!("ksg_count.wgsl");

const METRIC_SQUARED_EUCLIDEAN: u32 = 0;
const METRIC_CHEBYSHEV: u32 = 1;

/// Relative band around the radius used to detect f32 boundary ambiguity. Well
/// above the f32 coordinate error (~1e-7), so a count that is stable across it
/// matches the exact f64 comparison.
const AMBIGUITY_MARGIN: f32 = 1e-5;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct KsgCountConfig {
    point_count: u32,
    dim: u32,
    inclusive: u32,
    chebyshev: u32,
    margin: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Whether a space may take the GPU count path (size gate and `u32` shape).
fn count_eligible(n: usize) -> bool {
    n > 0 && n <= u32::MAX as usize && n >= crate::estimators::gpu::gpu_min_points_ksg()
}

/// Builds one count job for an `n x D` space with per-query epsilons.
///
/// `exclusive` selects the KSG Type1 strict comparison (`d < eps`). It returns
/// `None` when the space is below the gate or the shapes do not line up, so the
/// caller can fall back to the CPU path.
pub(crate) fn count_job<const D: usize>(
    points: ArrayView2<'_, f64>,
    epsilons: &[f64],
    use_chebyshev: bool,
    exclusive: bool,
) -> Option<BatchJob> {
    let n = points.nrows();
    if points.ncols() != D || epsilons.len() != n || !count_eligible(n) {
        return None;
    }

    let config = KsgCountConfig {
        point_count: n as u32,
        dim: D as u32,
        inclusive: u32::from(!exclusive),
        chebyshev: if use_chebyshev {
            METRIC_CHEBYSHEV
        } else {
            METRIC_SQUARED_EUCLIDEAN
        },
        margin: AMBIGUITY_MARGIN,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let eps_f32: Vec<f32> = epsilons.iter().map(|&e| e as f32).collect();

    Some(BatchJob {
        kind: ShaderKind::Count,
        wgsl: KSG_COUNT_WGSL,
        points: super::expfam_gpu::pack_cloud::<D>(points),
        extra_storage: Some(bytemuck::cast_slice(&eps_f32).to_vec()),
        config: bytemuck::bytes_of(&config).to_vec(),
        // Two counts per query: at the radius and at the tightened radius.
        n_items: (2 * n) as u32,
    })
}

/// One space's raw count at the radius and at the tightened radius, both
/// including the query itself.
pub(crate) struct RawCounts {
    pub at_radius: Vec<f64>,
    pub tightened: Vec<f64>,
}

impl RawCounts {
    /// True when a candidate lies within the ambiguity band, so the f32 count
    /// may differ from the exact f64 one and the space must be recomputed.
    pub fn ambiguous(&self) -> bool {
        self.at_radius != self.tightened
    }
}

/// Runs every space of one measure in a **single** submit. `None` when the GPU
/// is unavailable or the dispatch fails, in which case the caller uses the CPU
/// path for all spaces.
pub(crate) fn count_raw_batch(jobs: &[BatchJob]) -> Option<Vec<RawCounts>> {
    if jobs.is_empty() {
        return None;
    }
    let ctx = GpuContext::get()?;
    let raw = ctx.run_compute_batch(jobs)?;
    Some(
        raw.into_iter()
            .map(|values| {
                let n = values.len() / 2;
                let at_radius = values[..n].iter().map(|&v| f64::from(v)).collect();
                let tightened = values[n..].iter().map(|&v| f64::from(v)).collect();
                RawCounts {
                    at_radius,
                    tightened,
                }
            })
            .collect(),
    )
}

/// Applies the KSG self-exclusion to a raw count array, mirroring the CPU
/// `SortedSpace` path: Type1 drops the query itself whenever `eps > 0` and
/// yields zero for `eps == 0`, Type2 keeps the inclusive raw count.
pub(crate) fn apply_self_exclusion(raw: &[f64], epsilons: &[f64], type1: bool) -> Vec<f64> {
    if type1 {
        raw.iter()
            .zip(epsilons)
            .map(|(&c, &eps)| if eps > 0.0 { c - 1.0 } else { 0.0 })
            .collect()
    } else {
        raw.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn type1_drops_self_and_zero_radius_rows() {
        let raw = [4.0, 3.0, 1.0];
        let eps = [0.5, 0.0, 0.25];
        assert_eq!(apply_self_exclusion(&raw, &eps, true), vec![3.0, 0.0, 0.0]);
    }

    #[test]
    fn type2_keeps_the_raw_inclusive_count() {
        let raw = [4.0, 3.0, 1.0];
        let eps = [0.5, 0.0, 0.25];
        assert_eq!(apply_self_exclusion(&raw, &eps, false), vec![4.0, 3.0, 1.0]);
    }

    #[test]
    fn ineligible_shapes_are_rejected_without_a_device() {
        let data = array![[0.0], [1.0]];
        assert!(count_job::<1>(data.view(), &[0.0, 0.0], false, true).is_none());
        assert!(count_job::<1>(data.view(), &[0.0], false, true).is_none());
        assert!(count_job::<2>(data.view(), &[0.0, 0.0], false, true).is_none());
    }
}

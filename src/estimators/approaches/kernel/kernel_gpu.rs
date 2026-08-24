// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// GPU-accelerated implementation of kernel entropy calculation
// This module is only included when the `gpu` feature flag is enabled

use crate::estimators::approaches::kernel::KernelEntropy;
use crate::estimators::gpu::{GpuContext, gpu_min_points_box, gpu_min_points_gaussian};
use bytemuck::{Pod, Zeroable};
use ndarray::Array1;

// Define a struct for the bandwidth that can be sent to the GPU (for Box kernel)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuBandwidth {
    value: f32,         // Single bandwidth value for all dimensions
    dim_count: u32,     // Actual number of dimensions
    _padding: [u32; 2], // Padding to ensure 16-byte alignment
}

// Define a struct for the configuration parameters
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuConfig {
    point_count: u32,
    dim_count: u32,
    normalization: f32,
    adaptive_radius: f32,
}

impl<const K: usize> KernelEntropy<K> {
    /// Packs this estimator's state into a [`BatchJob`] for its kernel type,
    /// applying the same eligibility gates as [`KernelEntropy::kde_probability_density`].
    ///
    /// Returns `None` when the CPU path should be used instead (forced, below
    /// the size crossover, over the dimension limit) or when packing fails.
    #[cfg(feature = "gpu")]
    pub(crate) fn try_density_job(&self) -> Option<crate::estimators::gpu::BatchJob> {
        if self.force_cpu {
            return None;
        }
        // Size crossovers mirror kde_probability_density, resolving through
        // PR A's layered gate constants (env / programmatic / default).
        let min_points = match self.kernel_type.as_str() {
            "box" => gpu_min_points_box(),
            "gaussian" => gpu_min_points_gaussian(),
            _ => return None,
        };
        if K > 32 || self.n_samples < min_points {
            return None;
        }

        match self.kernel_type.as_str() {
            "gaussian" => self.gaussian_batch_job().ok(),
            "box" => self.box_batch_job().ok(),
            _ => None,
        }
    }

    #[cfg(feature = "gpu")]
    fn gaussian_batch_job(
        &self,
    ) -> Result<crate::estimators::gpu::BatchJob, Box<dyn std::error::Error>> {
        use crate::estimators::gpu::{BatchJob, ShaderKind};

        let n = self.points.len();

        // Calculate normalization factor: N * sqrt(det(2πΣ_scaled))
        let det_scaled_cov = if let Some(ref l) = self.cholesky_factor {
            let mut diag_prod = 1.0;
            for i in 0..K {
                diag_prod *= l[i * (K + 1)];
            }
            diag_prod * diag_prod
        } else {
            self.std_devs
                .iter()
                .map(|&s| (self.bandwidth * s).powi(2))
                .product()
        };
        let normalization =
            (n as f64) * (2.0 * std::f64::consts::PI).powf(K as f64 / 2.0) * det_scaled_cov.sqrt();
        // Whitened space ⇒ covariance identity ⇒ data-independent truncation
        // radius (matches gaussian_kernel_density_cpu_whitened).
        let adaptive_radius = if self.n_samples > 5000 { 36.0 } else { 64.0 };

        let wpoints = self
            .whitened_points
            .as_deref()
            .ok_or("whitened points unavailable for Gaussian GPU path")?;

        // Compact flat layout: N * K f32 values, row-major (points[i*K + d]).
        let mut points = Vec::with_capacity(wpoints.len() * K);
        for point in wpoints {
            for &val in point.iter() {
                points.push(val as f32);
            }
        }

        let config = GpuConfig {
            point_count: n as u32,
            dim_count: K as u32,
            normalization: normalization as f32,
            adaptive_radius: adaptive_radius as f32,
        };

        Ok(BatchJob {
            kind: ShaderKind::Gaussian,
            wgsl: include_str!("gaussian_kernel.wgsl"),
            points: bytemuck::cast_slice(&points).to_vec(),
            extra_storage: None,
            config: bytemuck::bytes_of(&config).to_vec(),
            n_items: n as u32,
        })
    }

    #[cfg(feature = "gpu")]
    fn box_batch_job(
        &self,
    ) -> Result<crate::estimators::gpu::BatchJob, Box<dyn std::error::Error>> {
        use crate::estimators::gpu::{BatchJob, ShaderKind};

        // volume = bandwidth^K; denominator of the KDE formula.
        let volume = self.bandwidth.powi(K as i32);
        let normalization = self.n_samples as f64 * volume;

        // Compact flat layout: N * K f32 values, row-major (points[i*K + d]).
        let mut points = Vec::with_capacity(self.points.len() * K);
        for point in &self.points {
            for &val in point.iter() {
                points.push(val as f32);
            }
        }
        let bandwidth = GpuBandwidth {
            value: self.bandwidth as f32,
            dim_count: K as u32,
            _padding: [0; 2],
        };
        let config = GpuConfig {
            point_count: self.points.len() as u32,
            dim_count: K as u32,
            normalization: normalization as f32,
            adaptive_radius: 0.0, // Not used for box kernel
        };

        Ok(BatchJob {
            kind: ShaderKind::Box,
            wgsl: include_str!("box_kernel.wgsl"),
            points: bytemuck::cast_slice(&points).to_vec(),
            extra_storage: Some(bytemuck::bytes_of(&bandwidth).to_vec()),
            config: bytemuck::bytes_of(&config).to_vec(),
            n_items: self.points.len() as u32,
        })
    }

    /// Computes local probability density values using a Gaussian kernel with GPU acceleration
    pub fn gaussian_kernel_density_gpu(&self) -> Array1<f64> {
        // Check if dimensions are within supported range
        if K > 32 {
            return self.gaussian_kernel_density_cpu();
        }

        // The GPU shader returns density directly (no exp round-trip).
        match self.run_gaussian_gpu_calculation() {
            Ok(result) => result,
            Err(_) => self.gaussian_kernel_density_cpu(),
        }
    }

    /// Computes local probability density values using a box kernel with GPU acceleration
    pub fn box_kernel_density_gpu(&self) -> Array1<f64> {
        // Check if dimensions are within supported range
        if K > 32 {
            return self.box_kernel_density_cpu();
        }

        // The GPU shader returns density directly (no exp round-trip).
        match self.run_box_gpu_calculation() {
            Ok(result) => result,
            Err(_) => self.box_kernel_density_cpu(),
        }
    }

    /// Computes local entropy values using a Gaussian kernel with GPU acceleration
    ///
    /// This implementation uses the GPU via wgpu to accelerate the calculation of
    /// pairwise distances and Gaussian kernel values, which can provide significant
    /// performance improvements for large datasets and high-dimensional data.
    ///
    /// # Implementation Details
    ///
    /// 1. The data points and scale factors are transferred to the GPU
    /// 2. A compute shader calculates the Gaussian kernel contributions for all points in parallel
    /// 3. The results are transferred back to the CPU
    /// 4. The final entropy values are calculated by applying logarithm and dimension-dependent normalization
    ///
    /// # Performance Characteristics
    ///
    /// The GPU implementation provides dramatic speedups compared to the CPU implementation:
    /// - Around the dispatch gate: significant speedups begin to materialise
    /// - For 5000 data points: ~89-131x faster, with significant gains even for low dimensions
    /// - For 10000 data points: ~87-337x faster, with the most dramatic improvements for lower dimensions
    ///
    /// # Adaptive Radius
    ///
    /// The GPU implementation uses an enhanced adaptive radius calculation to better handle
    /// different data sizes and bandwidths:
    /// - For large datasets (> 5000 points) with small bandwidths (< 0.5): 4σ radius
    /// - For smaller datasets with small bandwidths (< 0.5): 5σ radius
    /// - For large datasets with normal bandwidths: 3σ radius
    /// - For smaller datasets with normal bandwidths: 4σ radius
    ///
    /// # Fallback Behavior
    ///
    /// This method automatically falls back to the CPU implementation in the following cases:
    /// - If the dataset has fewer than [`gpu_min_points_gaussian`] points (GPU overhead
    ///   outweighs benefits). The gate is adaptive: its built-in default lives in
    ///   `estimators::gpu`, software renderers are gated off entirely, and it can be
    ///   tuned per machine via `INFOMEASURE_GPU_MIN_GAUSSIAN`.
    /// - If the dimensionality exceeds 32 (current GPU implementation limitation)
    /// - If any step of the GPU calculation fails (ensures robustness)
    pub fn gaussian_kernel_local_values_gpu(&self) -> Array1<f64> {
        // Check if dimensions are within supported range
        if K > 32 {
            println!(
                "GPU implementation only supports up to 32 dimensions, falling back to CPU implementation"
            );
            return self.gaussian_kernel_local_values();
        }

        // Check if we have enough points to make GPU acceleration worthwhile
        // (adaptive per-machine gate, see `estimators::gpu`).
        if self.points.len() < gpu_min_points_gaussian() {
            return self.gaussian_kernel_local_values();
        }

        // Try to run the GPU implementation, fall back to CPU if it fails
        match self.run_gaussian_gpu_calculation() {
            Ok(density) => density.mapv(|d| if d > 0.0 { -d.ln() } else { 0.0 }),
            Err(e) => {
                println!("GPU calculation failed: {e}, falling back to CPU implementation",);
                self.gaussian_kernel_local_values()
            }
        }
    }

    /// Computes local entropy values using a box kernel with GPU acceleration
    ///
    /// This implementation uses the GPU via wgpu to accelerate the calculation of
    /// pairwise distances and neighbor counting, which can provide significant
    /// performance improvements for large datasets and high-dimensional data.
    ///
    /// # Implementation Details
    ///
    /// 1. The data points and bandwidth are transferred to the GPU
    /// 2. A compute shader counts neighbors within bandwidth/2 for all points in parallel
    /// 3. The results are transferred back to the CPU
    /// 4. The final entropy values are calculated by applying logarithm
    ///
    /// # Performance Characteristics
    ///
    /// The Box kernel GPU implementation shows a different performance profile compared to the Gaussian kernel:
    /// - For small datasets (100-1000 points), the CPU implementation is faster due to GPU setup overhead
    /// - For medium datasets (3200-5000 points), the GPU implementation shows moderate speedups
    /// - For large datasets (10000+ points), the GPU implementation provides dramatic speedups (9.5-37.1x)
    /// - For high dimensions with large datasets, the GPU implementation completes calculations that
    ///   would timeout on the CPU
    ///
    /// # Fallback Behavior
    ///
    /// This method automatically falls back to the CPU implementation in the following cases:
    /// - If the dataset has fewer than [`gpu_min_points_box`] points (GPU overhead
    ///   outweighs benefits). The gate is adaptive: its built-in default lives in
    ///   `estimators::gpu`, software renderers are gated off entirely, and it can be
    ///   tuned per machine via `INFOMEASURE_GPU_MIN_BOX`.
    /// - If the dimensionality exceeds 32 (current GPU implementation limitation)
    /// - If any step of the GPU calculation fails (ensures robustness)
    pub fn box_kernel_local_values_gpu(&self) -> Array1<f64> {
        // Check if dimensions are within supported range
        if K > 32 {
            println!(
                "GPU implementation only supports up to 32 dimensions, falling back to CPU implementation"
            );
            return self.box_kernel_local_values();
        }

        // Check if we have enough points to make GPU acceleration worthwhile
        // (adaptive per-machine gate, see `estimators::gpu`).
        if self.points.len() < gpu_min_points_box() {
            return self.box_kernel_local_values();
        }

        // Try to run the GPU implementation, fall back to CPU if it fails
        match self.run_box_gpu_calculation() {
            Ok(density) => density.mapv(|d| if d > 0.0 { -d.ln() } else { 0.0 }),
            Err(e) => {
                println!("GPU calculation failed: {e}, falling back to CPU implementation");
                self.box_kernel_local_values()
            }
        }
    }

    /// Main GPU calculation function for Gaussian kernel
    ///
    /// This method handles the actual GPU computation for the Gaussian kernel entropy calculation.
    /// It prepares the data for the GPU, runs the computation, and processes the results.
    ///
    /// # Implementation Details
    ///
    /// - Uses an adaptive radius calculation based on data size and bandwidth
    /// - Applies dimension-dependent normalization to the results
    /// - Uses a WGSL compute shader for parallel processing
    /// - Handles numerical stability with Kahan summation for higher dimensions
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<f64>)`: Array of local entropy values if the GPU calculation succeeds
    /// - `Err(Box<dyn std::error::Error>)`: Error if any step of the GPU calculation fails
    fn run_gaussian_gpu_calculation(&self) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let ctx = GpuContext::get().ok_or("Failed to obtain GPU context")?;
        let job = self.gaussian_batch_job()?;
        let mut results = ctx
            .run_compute_batch(&[job])
            .ok_or("GPU computation failed")?;
        let raw = results.pop().ok_or("GPU computation returned no result")?;
        Ok(Array1::from_iter(raw.into_iter().map(|v| v as f64)))
    }

    /// Main GPU calculation function for Box kernel.
    ///
    /// Prepares the packed job payload and runs it through the shared batch
    /// runner as a single-job batch; see [`GpuContext::run_compute_batch`].
    fn run_box_gpu_calculation(&self) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let ctx = GpuContext::get().ok_or("Failed to obtain GPU context")?;
        let job = self.box_batch_job()?;
        let mut results = ctx
            .run_compute_batch(&[job])
            .ok_or("GPU computation failed")?;
        let raw = results.pop().ok_or("GPU computation returned no result")?;
        Ok(Array1::from_iter(raw.into_iter().map(|v| v as f64)))
    }
}

/// Decodes a batch result vector (one f32 per point, in order) into density
/// values per space.
#[cfg(feature = "gpu")]
pub(crate) fn densities_from_batch(raw: Vec<Vec<f32>>) -> Vec<Array1<f64>> {
    raw.into_iter()
        .map(|r| Array1::from_iter(r.into_iter().map(|v| v as f64)))
        .collect()
}

/// Runs every GPU-eligible space in `sources` through one batched round-trip,
/// returning densities in the same order. `None` if any space falls back to
/// CPU or a GPU step fails (callers then evaluate each space individually).
#[cfg(feature = "gpu")]
pub(crate) fn try_density_batch(
    sources: &[Box<
        dyn crate::estimators::approaches::kernel::kernel_estimator::DensityJobSource,
    >],
) -> Option<Vec<Array1<f64>>> {
    let ctx = GpuContext::get()?;
    let mut jobs = Vec::with_capacity(sources.len());
    for s in sources {
        jobs.push(s.try_job()?);
    }
    let results = ctx.run_compute_batch(&jobs)?;
    Some(densities_from_batch(results))
}

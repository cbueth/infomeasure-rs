// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// GPU-accelerated implementation of kernel entropy calculation
// This module is only included when the `gpu` feature flag is enabled

use crate::estimators::approaches::kernel::KernelEntropy;
use crate::estimators::gpu::{
    ComputePass, GpuContext, ShaderKind, gpu_min_points_box, gpu_min_points_gaussian,
};
use bytemuck::{Pod, Zeroable};
use ndarray::Array1;
use wgpu::util::DeviceExt;

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

/// Bind-group layout for the (whitened) Gaussian shader:
///   binding 0: points (storage, read) — whitened points
///   binding 1: config (uniform)
///   binding 2: output (storage, read_write)
fn gaussian_bind_group_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
    ctx.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gaussian Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
}

/// Bind-group layout for the box shader:
///   binding 0: points    (storage, read)
///   binding 1: bandwidth (storage, read)
///   binding 2: config    (uniform)
///   binding 3: output    (storage, read_write)
fn box_bind_group_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
    ctx.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Box Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
}

impl<const K: usize> KernelEntropy<K> {
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
        let n = self.points.len();

        // Calculate normalization factor: N * sqrt(det(2πΣ_scaled))
        // det(2πΣ_scaled) = (2π)^K * det(Σ_scaled)
        // det(Σ_scaled) = det(L * L^T) = det(L)^2
        // Since L is lower triangular, det(L) is the product of its diagonal elements.
        let det_scaled_cov = if let Some(ref l) = self.cholesky_factor {
            let mut diag_prod = 1.0;
            for i in 0..K {
                diag_prod *= l[i * (K + 1)];
            }
            diag_prod * diag_prod
        } else {
            // Fallback to diagonal covariance if cholesky_factor is None
            self.std_devs
                .iter()
                .map(|&s| (self.bandwidth * s).powi(2))
                .product()
        };

        // Normalization: N * (2π)^(K/2) * sqrt(det(Σ_scaled))
        let normalization =
            (n as f64) * (2.0 * std::f64::consts::PI).powf(K as f64 / 2.0) * det_scaled_cov.sqrt();

        // In whitened space the covariance is the identity, so the truncation
        // radius is a fixed, data-independent constant (same as the CPU
        // `gaussian_kernel_density_cpu_whitened` path).
        let adaptive_radius = if self.n_samples > 5000 { 36.0 } else { 64.0 };

        let ctx = GpuContext::get().ok_or("Failed to obtain GPU context")?;

        // Pack the WHITENED points for the GPU. In whitened space d_M² = ‖y_p-y_q‖²
        // is a plain Euclidean distance, so the shader only needs these points.
        let wpoints = self
            .whitened_points
            .as_deref()
            .ok_or("whitened points unavailable for Gaussian GPU path")?;

        // Compact flat layout: N * K f32 values, row-major (points[i*K + d]).
        let mut gpu_points = Vec::with_capacity(wpoints.len() * K);
        for point in wpoints {
            for &val in point.iter() {
                gpu_points.push(val as f32);
            }
        }

        let gpu_config = GpuConfig {
            point_count: n as u32,
            dim_count: K as u32,
            normalization: normalization as f32,
            adaptive_radius: adaptive_radius as f32,
        };

        // Create buffers
        let points_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Points Buffer"),
                contents: bytemuck::cast_slice(&gpu_points),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_bytes = (n as u64) * std::mem::size_of::<f32>() as u64;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let layout = gaussian_bind_group_layout(ctx);
        let config_buffer = ctx.config_buffer(&gpu_config);
        let bindings = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: config_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffer.as_entire_binding(),
            },
        ];

        let raw = ctx
            .run_compute(
                ShaderKind::Gaussian,
                include_str!("gaussian_kernel.wgsl"),
                ComputePass {
                    layout: &layout,
                    bindings: &bindings,
                    output: &output_buffer,
                    out_bytes,
                    n_items: n as u32,
                },
            )
            .ok_or("GPU computation failed")?;
        let result: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();

        // Convert the results to f64 and return
        let mut local_values = Array1::<f64>::zeros(n);
        for (i, &val) in result.iter().enumerate() {
            local_values[i] = val as f64;
        }

        Ok(local_values)
    }

    /// Main GPU calculation function for Box kernel
    ///
    /// This method handles the actual GPU computation for the Box kernel entropy calculation.
    /// It prepares the data for the GPU, runs the computation, and processes the results.
    ///
    /// # Implementation Details
    ///
    /// - Uses Manhattan distance to count neighbors within bandwidth/2
    /// - Normalizes by the volume of the hypercube (bandwidth^d) and the number of samples
    /// - Uses a WGSL compute shader for parallel processing
    /// - Optimized for high-dimensional data and large datasets
    ///
    /// # Returns
    ///
    /// - `Ok(Array1<f64>)`: Array of local entropy values if the GPU calculation succeeds
    /// - `Err(Box<dyn std::error::Error>)`: Error if any step of the GPU calculation fails
    fn run_box_gpu_calculation(&self) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        // Calculate volume = bandwidth^d (where d = K)
        // This is the volume of the hypercube with side length = bandwidth
        let volume = self.bandwidth.powi(K as i32);

        // Normalization factor: N * volume
        // This is the denominator in the KDE formula: f̂(x) = (1/Nh^d) ∑ K((x - x_i)/h)
        // where K is the box kernel (uniform within the bandwidth)
        let normalization = self.n_samples as f64 * volume;

        let ctx = GpuContext::get().ok_or("Failed to obtain GPU context")?;

        // Prepare data for GPU — compact flat layout: N * K f32 values,
        // row-major (points[i*K + d]).
        let mut gpu_points = Vec::with_capacity(self.points.len() * K);
        for point in &self.points {
            for &val in point.iter() {
                gpu_points.push(val as f32);
            }
        }

        // Prepare bandwidth for GPU
        let gpu_bandwidth = GpuBandwidth {
            value: self.bandwidth as f32,
            dim_count: K as u32,
            _padding: [0; 2],
        };

        let gpu_config = GpuConfig {
            point_count: self.points.len() as u32,
            dim_count: K as u32,
            normalization: normalization as f32,
            adaptive_radius: 0.0, // Not used for box kernel
        };

        // Create buffers
        let points_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Points Buffer"),
                contents: bytemuck::cast_slice(&gpu_points),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let bandwidth_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Bandwidth Buffer"),
                contents: bytemuck::bytes_of(&gpu_bandwidth),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_bytes = (self.points.len() as u64) * std::mem::size_of::<f32>() as u64;
        let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let layout = box_bind_group_layout(ctx);
        let config_buffer = ctx.config_buffer(&gpu_config);
        let bindings = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bandwidth_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: config_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
        ];

        let raw = ctx
            .run_compute(
                ShaderKind::Box,
                include_str!("box_kernel.wgsl"),
                ComputePass {
                    layout: &layout,
                    bindings: &bindings,
                    output: &output_buffer,
                    out_bytes,
                    n_items: self.points.len() as u32,
                },
            )
            .ok_or("GPU computation failed")?;
        let result: Vec<f32> = bytemuck::cast_slice(&raw).to_vec();

        // Convert the results to f64 and return
        let mut local_values = Array1::<f64>::zeros(self.points.len());
        for (i, &val) in result.iter().enumerate() {
            local_values[i] = val as f64;
        }

        Ok(local_values)
    }
}

// GPU-accelerated implementation of kernel entropy calculation
// This module is only included when the `gpu_support` feature flag is enabled

use ndarray::Array1;
use crate::estimators::approaches::kernel::KernelEntropy;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use futures_intrusive::channel::shared::oneshot_channel;
use pollster::block_on;

// Define a struct for the point data that can be sent to the GPU
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuPoint {
    values: [f32; 32], // Support up to 32 dimensions
    _padding: [f32; 0], // No padding needed
}

// Define a struct for the scale factors that can be sent to the GPU (for Gaussian kernel)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuScaleFactors {
    values: [f32; 32], // Support up to 32 dimensions
    dim_count: u32,    // Actual number of dimensions
    _padding: [u32; 3], // Padding to ensure 16-byte alignment
}

// Define a struct for the bandwidth that can be sent to the GPU (for Box kernel)
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuBandwidth {
    value: f32,        // Single bandwidth value for all dimensions
    dim_count: u32,    // Actual number of dimensions
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
    /// - For 1000 data points: ~4-17x faster, with higher speedups for higher dimensions
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
    /// - If the dataset has fewer than 500 points (GPU overhead outweighs benefits)
    /// - If the dimensionality exceeds 32 (current GPU implementation limitation)
    /// - If any step of the GPU calculation fails (ensures robustness)
    pub fn gaussian_kernel_local_values_gpu(&self) -> Array1<f64> {
        // Check if dimensions are within supported range
        if K > 32 {
            println!("GPU implementation only supports up to 32 dimensions, falling back to CPU implementation");
            return self.gaussian_kernel_local_values();
        }

        // Check if we have enough points to make GPU acceleration worthwhile
        // Based on benchmark analysis, GPU is beneficial for Gaussian kernel when dataset size >= 500
        if self.points.len() < 500 {
            return self.gaussian_kernel_local_values();
        }

        // Try to run the GPU implementation, fall back to CPU if it fails
        match self.run_gaussian_gpu_calculation() {
            Ok(result) => result,
            Err(e) => {
                println!("GPU calculation failed: {}, falling back to CPU implementation", e);
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
    /// - For medium datasets (5000 points), the GPU implementation shows significant speedups (1.7-12.6x)
    /// - For large datasets (10000+ points), the GPU implementation provides dramatic speedups (9.5-37.1x)
    /// - For high dimensions with large datasets, the GPU implementation completes calculations that
    ///   would timeout on the CPU
    ///
    /// # Fallback Behavior
    ///
    /// This method automatically falls back to the CPU implementation in the following cases:
    /// - If the dataset has fewer than 5000 points (GPU overhead outweighs benefits)
    /// - If the dimensionality exceeds 32 (current GPU implementation limitation)
    /// - If any step of the GPU calculation fails (ensures robustness)
    pub fn box_kernel_local_values_gpu(&self) -> Array1<f64> {
        // Check if dimensions are within supported range
        if K > 32 {
            println!("GPU implementation only supports up to 32 dimensions, falling back to CPU implementation");
            return self.box_kernel_local_values();
        }

        // Check if we have enough points to make GPU acceleration worthwhile
        // Based on benchmark analysis, GPU is beneficial for Box kernel when dataset size >= 5000
        if self.points.len() < 5000 {
            return self.box_kernel_local_values();
        }

        // Try to run the GPU implementation, fall back to CPU if it fails
        match self.run_box_gpu_calculation() {
            Ok(result) => result,
            Err(e) => {
                println!("GPU calculation failed: {}, falling back to CPU implementation", e);
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
        // Pre-compute scale factors once for all query points
        let scale_factors: Vec<f64> = (0..K).map(|dim| self.bandwidth * self.std_devs[dim]).collect();

        // Calculate the product of scale factors for normalization
        let scaled_bandwidth_product = scale_factors.iter().fold(1.0, |product, &factor| product * factor);

        // Normalization factor: N * (h*σ)^d
        let normalization = (self.points.len() as f64) * scaled_bandwidth_product;

        // Calculate max scaled bandwidth for search radius
        let max_scaled_bandwidth = scale_factors.iter().fold(0.0f64, |max_val, &val| max_val.max(val));

        // Determine adaptive radius based on data density and bandwidth
        // For small bandwidths, we need a larger radius to ensure enough neighbors are included
        let adaptive_radius = if self.bandwidth < 0.5 {
            // For small bandwidths, use a larger radius to ensure enough neighbors
            if self.n_samples > 5000 {
                16.0 * max_scaled_bandwidth.powi(2) // 4σ | (4*h)^2 for large datasets with small bandwidth
            } else {
                25.0 * max_scaled_bandwidth.powi(2) // 5σ | (5*h)^2 for smaller datasets with small bandwidth
            }
        } else {
            // For normal bandwidths, use the standard radius
            if self.n_samples > 5000 {
                9.0 * max_scaled_bandwidth.powi(2) // 3σ | (3*h)^2 for large datasets
            } else {
                16.0 * max_scaled_bandwidth.powi(2) // 4σ | (4*h)^2 for smaller datasets
            }
        };

        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        // Request an adapter
        let adapter = match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })) {
            Ok(adapter) => adapter,
            Err(_) => return Err("Failed to find an appropriate adapter".into()),
        };

        // Create device and queue
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Gaussian Kernel Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::default(),
            },
        ))?;

        // Prepare data for GPU
        let mut gpu_points = Vec::with_capacity(self.points.len());
        for point in &self.points {
            let mut gpu_point = GpuPoint {
                values: [0.0; 32],
                _padding: [],
            };

            // Copy point values to GPU point
            for (i, &val) in point.iter().enumerate() {
                gpu_point.values[i] = val as f32;
            }

            gpu_points.push(gpu_point);
        }

        // Prepare scale factors for GPU
        let mut gpu_scale_factors = GpuScaleFactors {
            values: [0.0; 32],
            dim_count: K as u32,
            _padding: [0; 3],
        };

        // Copy scale factors to GPU scale factors
        for (i, &factor) in scale_factors.iter().enumerate() {
            gpu_scale_factors.values[i] = factor as f32;
        }


        
        let gpu_config = GpuConfig {
            point_count: self.points.len() as u32,
            dim_count: K as u32,
            normalization: normalization as f32,
            adaptive_radius: adaptive_radius as f32,
        };

        // Create buffers
        let points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Points Buffer"),
            contents: bytemuck::cast_slice(&gpu_points),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let scale_factors_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scale Factors Buffer"),
            contents: bytemuck::bytes_of(&gpu_scale_factors),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&gpu_config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (self.points.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.points.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gaussian Kernel Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gaussian_kernel.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gaussian Kernel Bind Group Layout"),
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
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Gaussian Kernel Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Gaussian Kernel Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Gaussian Kernel Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scale_factors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Gaussian Kernel Command Encoder"),
        });

        // Compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Gaussian Kernel Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups
            // Use 256 threads per workgroup
            let workgroup_size = 256;
            let workgroup_count = (self.points.len() as u32 + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (self.points.len() * std::mem::size_of::<f32>()) as u64,
        );

        // Submit command buffer
        queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        // Wait for the GPU to finish
        device.poll(wgpu::PollType::Wait).expect("Failed to poll device");

        // Get the mapped data
        if let Some(Ok(())) = block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();

            // Convert the results to f64 and return
            let mut local_values = Array1::<f64>::zeros(self.points.len());
            for (i, &val) in result.iter().enumerate() {
                local_values[i] = val as f64;
            }

            // Apply dimension-dependent normalization factor
            let dim_factor = (K as f64 / 2.0) * (2.0 * std::f64::consts::PI).ln();
            local_values.mapv_inplace(|x| x + dim_factor);

            Ok(local_values)
        } else {
            Err("Failed to read back results from GPU".into())
        }
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

        // Initialize wgpu
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        // Request an adapter
        let adapter = match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })) {
            Ok(adapter) => adapter,
            Err(_) => return Err("Failed to find an appropriate adapter".into()),
        };

        // Create device and queue
        let (device, queue) = block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Box Kernel Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::default(),
            },
        ))?;

        // Prepare data for GPU
        let mut gpu_points = Vec::with_capacity(self.points.len());
        for point in &self.points {
            let mut gpu_point = GpuPoint {
                values: [0.0; 32],
                _padding: [],
            };

            // Copy point values to GPU point
            for (i, &val) in point.iter().enumerate() {
                gpu_point.values[i] = val as f32;
            }

            gpu_points.push(gpu_point);
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
        let points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Points Buffer"),
            contents: bytemuck::cast_slice(&gpu_points),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bandwidth_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bandwidth Buffer"),
            contents: bytemuck::bytes_of(&gpu_bandwidth),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::bytes_of(&gpu_config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (self.points.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.points.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Box Kernel Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("box_kernel.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Box Kernel Bind Group Layout"),
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
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Box Kernel Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Box Kernel Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Box Kernel Bind Group"),
            layout: &bind_group_layout,
            entries: &[
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
            ],
        });

        // Create command encoder
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Box Kernel Command Encoder"),
        });

        // Compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Box Kernel Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups
            // Use 256 threads per workgroup
            let workgroup_size = 256;
            let workgroup_count = (self.points.len() as u32 + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (self.points.len() * std::mem::size_of::<f32>()) as u64,
        );

        // Submit command buffer
        queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        // Wait for the GPU to finish
        device.poll(wgpu::PollType::Wait).expect("Failed to poll device");

        // Get the mapped data
        if let Some(Ok(())) = block_on(receiver.receive()) {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();

            // Convert the results to f64 and return
            let mut local_values = Array1::<f64>::zeros(self.points.len());
            for (i, &val) in result.iter().enumerate() {
                local_values[i] = val as f64;
            }

            Ok(local_values)
        } else {
            Err("Failed to read back results from GPU".into())
        }
    }
}
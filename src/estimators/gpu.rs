// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared wgpu context for all GPU-accelerated estimator paths.
//!
//! This module is only compiled when the `gpu` feature is enabled. It provides a
//! lazily-initialised singleton [`GpuContext`] that owns the wgpu instance,
//! adapter, device and queue, plus a cache of compute pipelines (one per
//! [`ShaderKind`]).
//!
//! Centralising wgpu setup here prevents repeated per-call
//! `Instance`, `request_adapter`, `request_device`, pipeline construction.

use futures_intrusive::channel::shared::oneshot_channel;
use pollster::block_on;
use rustc_hash::FxHashMap;
use std::sync::{LazyLock, Mutex};
use wgpu::util::DeviceExt;

/// Identifies one of the compute shaders, used as the pipeline-cache key.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderKind {
    Gaussian,
    Box,
    Histogram,
}

/// Bundles the per-invocation parameters of a [`GpuContext::run_compute`] call.
pub struct ComputePass<'a> {
    /// Bind-group layout describing the shader's bindings.
    pub layout: &'a wgpu::BindGroupLayout,
    /// Resource bindings for the compute pass.
    pub bindings: &'a [wgpu::BindGroupEntry<'a>],
    /// Storage buffer the shader writes its results into.
    pub output: &'a wgpu::Buffer,
    /// Number of bytes to read back from `output`.
    pub out_bytes: u64,
    /// Number of work-items (one thread per item, 256 threads per workgroup).
    pub n_items: u32,
}

/// A lazily-initialised, process-wide wgpu context.
///
/// Creation happens exactly once on first use. If no hardware adapter can be
/// found, [`GpuContext::get`] caches `None` so every later call cheaply falls
/// back to CPU, preserving the estimator-level fallback behaviour.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipelines: Mutex<FxHashMap<ShaderKind, wgpu::ComputePipeline>>,
}

impl GpuContext {
    /// Returns the process-wide context, or `None` if no hardware adapter was
    /// found (callers must fall back to CPU).
    pub fn get() -> Option<&'static GpuContext> {
        static CTX: LazyLock<Option<GpuContext>> = LazyLock::new(GpuContext::init);
        CTX.as_ref()
    }

    fn init() -> Option<GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("infomeasure GPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .ok()?;

        Some(GpuContext {
            device,
            queue,
            pipelines: Mutex::new(FxHashMap::default()),
        })
    }

    /// Returns the cached compute pipeline for `kind`, compiling `wgsl` on first
    /// use. `None` if the shader cannot be compiled.
    ///
    /// The pipeline is built from the given `layout`, so the bind groups created
    /// by [`run_compute`](Self::run_compute) with the same layout are compatible.
    /// A `wgpu::ComputePipeline` is a cheap reference-counted handle, so returning
    /// an owned clone is fine and avoids exposing a borrow across the lock.
    pub fn pipeline(
        &self,
        kind: ShaderKind,
        wgsl: &'static str,
        layout: &wgpu::BindGroupLayout,
    ) -> Option<wgpu::ComputePipeline> {
        // Fast path: return an already-compiled pipeline without taking a long
        // lock. (Re-entrant lookups from concurrent calls are cheap.)
        {
            let pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(p) = pipelines.get(&kind) {
                return Some(p.clone());
            }
        }

        // Compile outside the lock so a shader-validation failure cannot poison
        // the cache for other threads.
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("infomeasure shader"),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("infomeasure pipeline layout"),
                bind_group_layouts: &[layout],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("infomeasure pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        Some(pipelines.entry(kind).or_insert(pipeline).clone())
    }

    /// Creates a uniform buffer from a `Pod` value (e.g. a config struct).
    pub fn config_buffer<T: bytemuck::Pod>(&self, value: &T) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("infomeasure config buffer"),
                contents: bytemuck::bytes_of(value),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Creates a uniform buffer from raw bytes.
    pub fn uniform_buffer(&self, bytes: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("infomeasure uniform buffer"),
                contents: bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    /// Runs a compute pass and reads back `out_bytes` bytes into a CPU `Vec<u8>`.
    ///
    /// `params.layout` and `params.bindings` fully describe the bind group.
    /// The caller is responsible for allocating its own storage buffers (points,
    /// parameters, output) and constructing the layout to match the shader.
    /// `params.n_items` is the number of work-items (one thread per item, 256
    /// threads per workgroup).
    ///
    /// Returns `Some(bytes)` on success, `None` on any GPU error (the caller
    /// should fall back to CPU).
    pub fn run_compute(
        &self,
        kind: ShaderKind,
        wgsl: &'static str,
        params: ComputePass<'_>,
    ) -> Option<Vec<u8>> {
        let pipeline = self.pipeline(kind, wgsl, params.layout)?;

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("infomeasure bind group"),
            layout: params.layout,
            entries: params.bindings,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("infomeasure staging buffer"),
            size: params.out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("infomeasure encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("infomeasure compute pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size = 256u32;
            let workgroup_count = params.n_items.div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        encoder.copy_buffer_to_buffer(params.output, 0, &staging_buffer, 0, params.out_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (sender, receiver) = oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).ok();
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
        let _ = block_on(receiver.receive())?;

        let view = slice.get_mapped_range();
        let result = view.to_vec();
        drop(view);
        staging_buffer.unmap();

        Some(result)
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::estimators::approaches::discrete::mle_gpu::gpu_histogram_rows_dense;
    use ndarray::array;
    use rstest::rstest;

    /// The shared context is a process-wide singleton: repeated calls return the
    /// same instance (no per-call wgpu setup).
    #[test]
    fn context_is_shared_singleton() {
        let a = GpuContext::get().expect("a hardware GPU adapter should be available");
        let b = GpuContext::get().expect("a hardware GPU adapter should be available");
        assert!(std::ptr::eq(a, b));
    }

    /// `pipeline()` caches by kind: a second request for the same `ShaderKind`
    /// returns the identical (reference-equal) pipeline instead of recompiling.
    #[test]
    fn pipeline_is_cached_per_kind() {
        let ctx = GpuContext::get().expect("a hardware GPU adapter should be available");

        // Layout must match the box shader: 0/1 storage-read, 2 uniform, 3
        // storage-read-write, or pipeline validation fails.
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test kernel layout"),
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
        let wgsl: &'static str = include_str!("approaches/kernel/box_kernel.wgsl");
        let first = ctx
            .pipeline(ShaderKind::Box, wgsl, &layout)
            .expect("box pipeline compiles");
        let second = ctx
            .pipeline(ShaderKind::Box, wgsl, &layout)
            .expect("box pipeline compiles");
        assert_eq!(
            first, second,
            "repeated requests must return the cached pipeline"
        );
    }

    /// Runs the histogram shader through `run_compute` and verifies the GPU
    /// per-row dense histogram matches a CPU reference for small integer ranges.
    #[rstest]
    #[case(array![[0, 1, 2, 0, 1], [1, 1, 2, 3, 3]])]
    #[case(array![[5, 5, 6, 7, 7, 7], [0, 0, 0, 1, 1, 2]])]
    #[case(array![[9, 9, 9, 9], [0, 1, 2, 3], [4, 4, 4, 4]])]
    fn histogram_matches_cpu_reference(#[case] data: ndarray::Array2<i32>) {
        let gpu = gpu_histogram_rows_dense(&data).expect("GPU histogram should succeed");
        assert_eq!(gpu.len(), data.nrows());

        for (r, row) in data.rows().into_iter().enumerate() {
            let mut expected = std::collections::HashMap::new();
            for &v in row {
                *expected.entry(v).or_insert(0) += 1;
            }
            let got = &gpu[r];
            assert_eq!(got.len(), expected.len(), "row {r} histogram size mismatch");
            for (&k, &count) in &expected {
                assert_eq!(
                    got.get(&k).copied(),
                    Some(count),
                    "row {r} count for value {k} mismatch"
                );
            }
        }
    }
}

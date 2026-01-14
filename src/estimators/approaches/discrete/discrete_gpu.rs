// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// GPU-accelerated utilities for discrete (histogram-based) estimators.
// This module is compiled only when the `gpu_support` feature is enabled.

#![cfg(feature = "gpu_support")]

use futures_intrusive::channel::shared::oneshot_channel;
use ndarray::Array2;
use pollster::block_on;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Try to compute per-row dense histograms using the GPU.
///
/// Preconditions for using the GPU path:
/// - The input is a 2D array of i32 values (row-major contiguous assumed by ndarray)
/// - The global value range (max - min) across the entire matrix is small (<= MAX_BINS)
///
/// If any condition fails or a GPU error occurs, returns None and callers should fall back to CPU.
pub fn gpu_histogram_rows_dense(data: &Array2<i32>) -> Option<Vec<HashMap<i32, usize>>> {
    const MAX_BINS: i32 = 4096; // keep in sync with CPU dense threshold

    let (rows, cols) = data.dim();
    if rows == 0 || cols == 0 {
        return Some(Vec::new());
    }

    // Compute global min/max on CPU (cheap and necessary for binning)
    let mut min_v = i32::MAX;
    let mut max_v = i32::MIN;
    for v in data.iter() {
        if *v < min_v {
            min_v = *v;
        }
        if *v > max_v {
            max_v = *v;
        }
    }
    let range = max_v.saturating_sub(min_v);
    if range > MAX_BINS {
        return None;
    }

    // Flatten data
    let flat: Vec<i32> = data.iter().cloned().collect();
    let bins = (range as u32) + 1;
    let total = (rows as u32) * (cols as u32);

    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = match block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })) {
        Ok(adapter) => adapter,
        Err(_) => return None,
    };

    let (device, queue) = match block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("Discrete Histogram Device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::default(),
    })) {
        Ok(pair) => pair,
        Err(_) => return None,
    };

    // Buffers
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Histogram Input Buffer"),
        contents: bytemuck::cast_slice(&flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Output buffer holds rows * bins u32 counters
    let out_elems = (rows as u64) * (bins as u64);
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Histogram Output Buffer"),
        size: out_elems * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Histogram Staging Buffer"),
        size: out_elems * std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Uniforms
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Config {
        rows: u32,
        cols: u32,
        min_v: i32,
        bins: u32,
    }
    let cfg = Config {
        rows: rows as u32,
        cols: cols as u32,
        min_v,
        bins,
    };
    let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Histogram Config Buffer"),
        contents: bytemuck::bytes_of(&cfg),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Shader
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Discrete Histogram Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("histogram.wgsl").into()),
    });

    // Bind group layout
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Histogram BGL"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
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
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Histogram Pipeline Layout"),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Histogram Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Histogram BG"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: config_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Histogram Encoder"),
    });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Histogram Compute Pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg_size = 256u32;
        let wg_count = (total + wg_size - 1) / wg_size;
        cpass.dispatch_workgroups(wg_count, 1, 1);
    }

    // Copy results
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &staging_buffer,
        0,
        out_elems * std::mem::size_of::<u32>() as u64,
    );
    queue.submit(std::iter::once(encoder.finish()));

    // Read back
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |v| {
        sender.send(v).ok();
    });
    device.poll(wgpu::PollType::Wait).ok()?;
    if block_on(receiver.receive()).is_none() {
        return None;
    }
    let view = slice.get_mapped_range();
    let counts_u32: Vec<u32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging_buffer.unmap();

    // Convert to Vec<HashMap<i32, usize>> per row
    let mut result: Vec<HashMap<i32, usize>> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut map = HashMap::new();
        let base = r as usize * bins as usize;
        for b in 0..(bins as usize) {
            let c = counts_u32[base + b] as usize;
            if c != 0 {
                let val = min_v + (b as i32);
                map.insert(val, c);
            }
        }
        result.push(map);
    }

    Some(result)
}

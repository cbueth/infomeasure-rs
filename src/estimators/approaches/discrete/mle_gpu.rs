// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

// GPU-accelerated utilities for discrete (histogram-based) estimators.
// This module is compiled only when the `gpu` feature is enabled.

use ndarray::Array2;
use rustc_hash::FxHashMap;
use wgpu::util::DeviceExt;

use crate::estimators::gpu::{ComputePass, GpuContext, ShaderKind};

/// Try to compute per-row dense histograms using the GPU.
///
/// Preconditions for using the GPU path:
/// - The input is a 2D array of i32 values (row-major contiguous assumed by ndarray)
/// - The global value range (max - min) across the entire matrix is small (<= MAX_BINS)
///
/// If any condition fails or a GPU error occurs, returns None and callers should fall back to CPU.
pub fn gpu_histogram_rows_dense(data: &Array2<i32>) -> Option<Vec<FxHashMap<i32, usize>>> {
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

    let ctx = GpuContext::get()?;

    // Flatten data
    let flat: Vec<i32> = data.iter().cloned().collect();
    let bins = (range as u32) + 1;
    let total = (rows as u32) * (cols as u32);

    // Buffers
    let input_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Histogram Input Buffer"),
            contents: bytemuck::cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Output buffer holds rows * bins u32 counters
    let out_elems = (rows as u64) * (bins as u64);
    let out_bytes = out_elems * std::mem::size_of::<u32>() as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Histogram Output Buffer"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
    let config_buffer = ctx.config_buffer(&cfg);

    // Bind group layout (matches histogram.wgsl):
    //   binding 0: input  (storage, read)
    //   binding 1: output (storage, read_write)
    //   binding 2: config (uniform)
    let layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Histogram Bind Group Layout"),
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

    let bindings = [
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
    ];

    let raw = ctx.run_compute(
        ShaderKind::Histogram,
        include_str!("histogram.wgsl"),
        ComputePass {
            layout: &layout,
            bindings: &bindings,
            output: &output_buffer,
            out_bytes,
            n_items: total,
        },
    )?;
    let counts_u32: Vec<u32> = bytemuck::cast_slice(&raw).to_vec();

    // Convert to Vec<FxHashMap<i32, usize>> per row
    let mut result: Vec<FxHashMap<i32, usize>> = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut map = FxHashMap::default();
        let base = r * bins as usize;
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

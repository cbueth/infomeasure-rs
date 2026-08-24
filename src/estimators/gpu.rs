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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};
use wgpu::util::DeviceExt;

/// Threads per workgroup for all compute dispatches.
const BATCH_WORKGROUP_SIZE: u32 = 256;

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

/// Minimum dataset size (in points) at which the Gaussian-kernel GPU path
/// begins to win over the optimised CPU hot path.
///
/// Provisional snapshot from M4 Pro crossover measurements (tie at ~800,
/// clear GPU win by 1200), kept conservative for discrete cards that pay
/// transfer costs. Crossovers are machine-relative and shift with either
/// side's optimisations: tune per machine via `INFOMEASURE_GPU_MIN_GAUSSIAN`
/// and consult the `gpu_crossover` Bencher history instead of retuning here.
pub const GAUSSIAN_GPU_MIN_POINTS: usize = 1200;

/// Same as [`GAUSSIAN_GPU_MIN_POINTS`] for the box kernel, whose GPU win only
/// materialises at larger sizes (neighbour counting vs. exp evaluation).
/// M4 Pro tie at ~3200, clear win by 4000.
pub const BOX_GPU_MIN_POINTS: usize = 4000;

/// Per-kernel minimum points below which the CPU path is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMinPoints {
    pub gaussian: usize,
    pub r#box: usize,
}

impl GpuMinPoints {
    /// Shipped constants, derived from benchmark analysis. Deliberately kept
    /// for every *hardware* adapter: per-machine crossovers are tuned through
    /// the overrides below rather than a hardware-family table.
    const HARDWARE_DEFAULTS: Self = Self {
        gaussian: GAUSSIAN_GPU_MIN_POINTS,
        r#box: BOX_GPU_MIN_POINTS,
    };

    /// Software renderers (llvmpipe/lavapipe, WARP, virtio) execute WGSL on
    /// the CPU and lose to the native kernel at any size, so by default they
    /// never get dispatched. wgpu ranks them last but does not exclude them,
    /// so on GPU-less machines they would otherwise be selected silently.
    const SOFTWARE_NEVER: Self = Self {
        gaussian: usize::MAX,
        r#box: usize::MAX,
    };

    fn defaults(is_software_renderer: bool) -> Self {
        if is_software_renderer {
            Self::SOFTWARE_NEVER
        } else {
            Self::HARDWARE_DEFAULTS
        }
    }
}

/// Partial override of [`GpuMinPoints`]; `None` fields fall through to the
/// next layer (programmatic → environment → built-in default).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuMinPointsOverride {
    pub gaussian: Option<usize>,
    pub r#box: Option<usize>,
}

/// A `DeviceType::Cpu` or virtual adapter executes shaders in software, see
/// [`GpuMinPoints::SOFTWARE_NEVER`]. Detection relies on the backend-reported
/// device type alone — adapter name strings are unreliable (e.g. Mesa reports
/// "llvmpipe" for OpenGL while Vulkan uses real hardware).
fn is_software_renderer(info: &wgpu::AdapterInfo) -> bool {
    matches!(
        info.device_type,
        wgpu::DeviceType::Cpu | wgpu::DeviceType::VirtualGpu
    )
}

/// Parses one env-var value into a point threshold. Invalid values are treated
/// as unset rather than errors (`""`, `"fast"`, negatives for [`usize`]...).
fn parse_env_value(raw: Option<&str>) -> Option<usize> {
    raw.map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(|s| s.parse().ok())
}

/// Parses the env-var pair into an override. Both variables are independent,
/// so a single valid value yields a partial override.
fn parse_env_min_points(gaussian: Option<&str>, r#box: Option<&str>) -> GpuMinPointsOverride {
    GpuMinPointsOverride {
        gaussian: parse_env_value(gaussian),
        r#box: parse_env_value(r#box),
    }
}

/// Layers overrides onto defaults: programmatic beats environment beats the
/// built-in constants. Uniform for every adapter, including software ones —
/// explicit requests may always force a path (e.g. to benchmark it).
fn resolve_min_points(
    defaults: GpuMinPoints,
    env: GpuMinPointsOverride,
    programmatic: GpuMinPointsOverride,
) -> GpuMinPoints {
    GpuMinPoints {
        gaussian: programmatic
            .gaussian
            .or(env.gaussian)
            .unwrap_or(defaults.gaussian),
        r#box: programmatic.r#box.or(env.r#box).unwrap_or(defaults.r#box),
    }
}

/// Adapter information captured once per process without creating a device,
/// so gate resolution never forces full wgpu initialisation for small-N calls
/// that stay on the CPU anyway. `None` if no adapter exists at all — then the
/// GPU paths fall back to CPU regardless and the thresholds never matter.
static ADAPTER_INFO: LazyLock<Option<wgpu::AdapterInfo>> = LazyLock::new(|| {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()
    .map(|adapter| adapter.get_info())
});

/// Environment overrides, read once at first use:
/// `INFOMEASURE_GPU_MIN_GAUSSIAN` / `INFOMEASURE_GPU_MIN_BOX`.
static ENV_OVERRIDE: LazyLock<GpuMinPointsOverride> = LazyLock::new(|| {
    parse_env_min_points(
        std::env::var("INFOMEASURE_GPU_MIN_GAUSSIAN")
            .ok()
            .as_deref(),
        std::env::var("INFOMEASURE_GPU_MIN_BOX").ok().as_deref(),
    )
});

/// Programmatic override packed into a single atomic (two `u32` lanes,
/// `u32::MAX` = unset), so gate reads stay lock-free on the hot path.
const UNSET_LANE: u64 = u32::MAX as u64;
static PROGRAMMATIC_OVERRIDE: AtomicU64 = AtomicU64::new((UNSET_LANE << 32) | UNSET_LANE);

fn encode_lane(value: Option<usize>) -> u64 {
    // Clamp instead of colliding with the unset sentinel. Sizes near
    // u32::MAX points are not representable datasets anyway.
    let v = value.map_or(u32::MAX, |v| {
        u32::try_from(v).unwrap_or(u32::MAX - 1).min(u32::MAX - 1)
    });
    v as u64
}

fn decode_lane(lane: u64) -> Option<usize> {
    (lane != UNSET_LANE).then_some(lane as usize)
}

fn programmatic_override() -> GpuMinPointsOverride {
    let bits = PROGRAMMATIC_OVERRIDE.load(Ordering::Relaxed);
    GpuMinPointsOverride {
        gaussian: decode_lane(bits >> 32),
        r#box: decode_lane(bits & 0xFFFF_FFFF),
    }
}

/// Overrides the dispatch gates for this process (highest precedence).
///
/// Intended for benchmarks and tests: hidden from docs because production
/// code should rely on the environment variables instead.
///
/// - `Some(0)` forces the corresponding kernel onto the GPU at any size.
/// - `None` clears that lane, falling back to env/default resolution.
#[doc(hidden)]
pub fn set_gpu_min_points_override(gaussian: Option<usize>, r#box: Option<usize>) {
    PROGRAMMATIC_OVERRIDE.store(
        (encode_lane(gaussian) << 32) | encode_lane(r#box),
        Ordering::Relaxed,
    );
}

/// Built-in minimum points for the currently detected adapter, before any
/// override is applied (software renderers resolve to `usize::MAX`).
#[doc(hidden)]
pub fn gpu_min_points_gaussian_default() -> usize {
    GpuMinPoints::defaults(ADAPTER_INFO.as_ref().is_some_and(is_software_renderer)).gaussian
}

/// See [`gpu_min_points_gaussian_default`].
#[doc(hidden)]
pub fn gpu_min_points_box_default() -> usize {
    GpuMinPoints::defaults(ADAPTER_INFO.as_ref().is_some_and(is_software_renderer)).r#box
}

/// Effective Gaussian-kernel gate: points below this stay on the CPU.
pub fn gpu_min_points_gaussian() -> usize {
    resolve_min_points(
        GpuMinPoints::defaults(ADAPTER_INFO.as_ref().is_some_and(is_software_renderer)),
        *ENV_OVERRIDE,
        programmatic_override(),
    )
    .gaussian
}

/// Effective box-kernel gate: points below this stay on the CPU.
pub fn gpu_min_points_box() -> usize {
    resolve_min_points(
        GpuMinPoints::defaults(ADAPTER_INFO.as_ref().is_some_and(is_software_renderer)),
        *ENV_OVERRIDE,
        programmatic_override(),
    )
    .r#box
}

/// The adapter selected by the same request wgpu performs for the compute
/// context, captured without device creation. Useful for logging which
/// hardware (or software renderer) a run dispatched to.
#[doc(hidden)]
pub fn gpu_adapter_info() -> Option<&'static wgpu::AdapterInfo> {
    ADAPTER_INFO.as_ref()
}

/// A self-contained compute job for [`GpuContext::run_compute_batch`].
///
/// Payloads are raw bytes for the shader's bindings; the batch runner creates
/// the device buffers, encodes every dispatch into a single command encoder,
/// and reads all results back after one submit-and-wait cycle.
pub struct BatchJob {
    /// Which cached pipeline/layout to use.
    pub kind: ShaderKind,
    /// WGSL source (only read on first compile for `kind`).
    pub wgsl: &'static str,
    /// Read-only storage payload bound at slot 0 (the point cloud).
    pub points: Vec<u8>,
    /// Optional second read-only storage payload at slot 1 (box bandwidth).
    pub extra_storage: Option<Vec<u8>>,
    /// Uniform payload.
    pub config: Vec<u8>,
    /// Work-items; output is `n_items` f32 values.
    pub n_items: u32,
}

/// A lazily-initialised, process-wide wgpu context.
///
/// Creation happens exactly once on first use. If no hardware adapter can be
/// found, [`GpuContext::get`] caches `None` so every later call cheaply falls
/// back to CPU, preserving the estimator-level fallback behaviour.
pub struct GpuContext {
    /// Adapter the context was created on, for diagnostics (which hardware or
    /// software renderer a run actually dispatched to).
    pub info: wgpu::AdapterInfo,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pipelines: Mutex<FxHashMap<ShaderKind, wgpu::ComputePipeline>>,
    bind_group_layouts: Mutex<FxHashMap<ShaderKind, wgpu::BindGroupLayout>>,
    /// Whether the adapter exposes timestamp queries; gates GPU pass timing.
    timestamps_enabled: bool,
    /// Whether timestamps can be written *inside* a pass, enabling per-job
    /// attribution within a batched pass rather than one duration for the
    /// whole pass.
    timestamps_inside: bool,
    /// Flipped off when an adapter reports timestamps but produces only zero
    /// deltas (observed on Metal, whose pass-boundary writes are not wired
    /// through), so downstream consumers stop receiving meaningless data.
    timings_usable: std::sync::atomic::AtomicBool,
    /// Per-job GPU pass durations (milliseconds) of the most recent batch.
    /// `None` while a batch is in flight or when timing is unsupported. This
    /// lives at the dispatch chokepoint so every estimator that routes through
    /// [`GpuContext::run_compute_batch`] is profileable without per-estimator
    /// plumbing.
    last_batch_gpu_ms: Mutex<Option<Vec<f64>>>,
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

        let info = adapter.get_info();
        eprintln!(
            "infomeasure GPU: dispatching to '{}' ({:?} via {:?}){}",
            info.name,
            info.device_type,
            info.backend,
            if is_software_renderer(&info) {
                " — software renderer, gated off unless overridden"
            } else {
                ""
            }
        );

        // Timestamp queries are the machine-readable window onto GPU pass
        // durations. Not every adapter exposes them (some software renderers
        // do not), so they are requested only when available and everything
        // downstream degrades to phase wall-clock timing.
        let wanted = wgpu::Features::TIMESTAMP_QUERY;
        let wanted_inside = wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        let available = adapter.features();
        let timestamps_enabled = available.contains(wanted);
        let timestamps_inside = available.contains(wanted_inside);
        let required = if timestamps_enabled && timestamps_inside {
            wanted | wanted_inside
        } else if timestamps_enabled {
            wanted
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("infomeasure GPU Device"),
            required_features: required,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
        }))
        .ok()?;

        Some(GpuContext {
            info,
            device,
            queue,
            pipelines: Mutex::new(FxHashMap::default()),
            bind_group_layouts: Mutex::new(FxHashMap::default()),
            timestamps_enabled,
            timestamps_inside,
            timings_usable: std::sync::atomic::AtomicBool::new(true),
            last_batch_gpu_ms: Mutex::new(None),
        })
    }

    /// Returns the cached bind-group layout for `kind`, creating it on first
    /// use. Layouts are cheap handles, so cloning out of the cache is fine.
    pub fn bind_group_layout(&self, kind: ShaderKind) -> wgpu::BindGroupLayout {
        let mut layouts = self
            .bind_group_layouts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(l) = layouts.get(&kind) {
            return l.clone();
        }
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_read = |b: u32| entry(b, wgpu::BufferBindingType::Storage { read_only: true });
        let storage_rw = |b: u32| entry(b, wgpu::BufferBindingType::Storage { read_only: false });
        let uniform = |b: u32| entry(b, wgpu::BufferBindingType::Uniform);
        // Gaussian shape: points ro / config uniform / output rw.
        // Box shape: points ro / extra storage ro / config uniform / output rw.
        // Histogram shape: data ro / atomic counts rw / config uniform.
        let entries: &[wgpu::BindGroupLayoutEntry] = match kind {
            ShaderKind::Gaussian => &[storage_read(0), uniform(1), storage_rw(2)],
            ShaderKind::Box => &[storage_read(0), storage_read(1), uniform(2), storage_rw(3)],
            ShaderKind::Histogram => &[storage_read(0), storage_rw(1), uniform(2)],
        };
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("infomeasure bind group layout"),
                entries,
            });
        layouts.insert(kind, layout.clone());
        layout
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

    /// Runs several compute jobs with a **single** submit-and-wait cycle.
    ///
    /// Every job's dispatch is encoded into one command encoder and submitted
    /// together; results are read back after one `poll`. This amortises the
    /// per-invocation round-trip stall (measured at a ~2.5 ms fixed floor on
    /// Metal, dwarfing everything else) across jobs — multi-space estimators
    /// (MI/CMI/TE/CTE density evaluations) stop paying it per space.
    ///
    /// Results are returned in job order as `n_items` f32 values each.
    /// `None` on any GPU error (callers should fall back to CPU); an empty
    /// job list returns an empty result vector.
    pub fn run_compute_batch(&self, jobs: &[BatchJob]) -> Option<Vec<Vec<f32>>> {
        if jobs.is_empty() {
            return Some(Vec::new());
        }

        struct Prepared {
            bind_group: wgpu::BindGroup,
            pipeline: wgpu::ComputePipeline,
            output: wgpu::Buffer,
            staging: wgpu::Buffer,
            out_bytes: u64,
            n_items: u32,
        }
        let mut prepared = Vec::with_capacity(jobs.len());

        for job in jobs.iter() {
            let points_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("batch points buffer"),
                    contents: &job.points,
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let extra_buffer = job.extra_storage.as_ref().map(|bytes| {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("batch aux storage buffer"),
                        contents: bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            });
            let config_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("batch config buffer"),
                    contents: &job.config,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let out_bytes = job.n_items as u64 * std::mem::size_of::<f32>() as u64;
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch output buffer"),
                size: out_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let layout = self.bind_group_layout(job.kind);
            // Binding shapes are fixed per shader kind; only box carries an
            // extra read-only storage payload.
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: points_buffer.as_entire_binding(),
            }];
            match job.kind {
                ShaderKind::Gaussian => {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 1,
                        resource: config_buffer.as_entire_binding(),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    });
                }
                ShaderKind::Box => {
                    let extra = extra_buffer.as_ref()?;
                    entries.push(wgpu::BindGroupEntry {
                        binding: 1,
                        resource: extra.as_entire_binding(),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2,
                        resource: config_buffer.as_entire_binding(),
                    });
                    entries.push(wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    });
                }
                // The histogram shader is dispatched through its own path.
                ShaderKind::Histogram => return None,
            }

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("batch bind group"),
                layout: &layout,
                entries: &entries,
            });
            let pipeline = self.pipeline(job.kind, job.wgsl, &layout)?;
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch staging buffer"),
                size: out_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            prepared.push(Prepared {
                bind_group,
                pipeline,
                output: output_buffer,
                staging,
                out_bytes,
                n_items: job.n_items,
            });
        }

        // Timestamp layouts: with inside-pass support, index 0 is the pass
        // begin (= job 0 start); for each job i, index 2i+1 after its dispatch
        // (= end) and index 2i before it (= start, automatic for i = 0); the
        // pass-end marker lands at index 2 * n_jobs. Without it, only the two
        // pass boundaries are recorded, yielding one duration for the whole
        // batch.
        let n_jobs = prepared.len();
        let per_job = self.timestamps_inside;
        let (query_set, resolve_buffer, ts_staging) = if self.timestamps_enabled {
            let count = if per_job { 2 * n_jobs as u32 + 1 } else { 2 };
            let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("batch timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count,
            });
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch timestamp resolve"),
                size: 8 * count as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("batch timestamp staging"),
                size: 8 * count as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (Some(qs), Some(resolve), Some(staging))
        } else {
            (None, None, None)
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("batch encoder"),
            });
        let last_index = if per_job { 2 * n_jobs as u32 } else { 1 };
        let timestamp_writes = query_set
            .as_ref()
            .map(|qs| wgpu::ComputePassTimestampWrites {
                query_set: qs,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(last_index),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("batch compute pass"),
                timestamp_writes,
            });
            for (i, p) in prepared.iter().enumerate() {
                if i > 0
                    && per_job
                    && let Some(qs) = &query_set
                {
                    compute_pass.write_timestamp(qs, (2 * i) as u32);
                }

                compute_pass.set_pipeline(&p.pipeline);
                compute_pass.set_bind_group(0, &p.bind_group, &[]);
                compute_pass.dispatch_workgroups(p.n_items.div_ceil(BATCH_WORKGROUP_SIZE), 1, 1);
                if per_job && let Some(qs) = &query_set {
                    compute_pass.write_timestamp(qs, (2 * i + 1) as u32);
                }
            }
        }
        for p in &prepared {
            encoder.copy_buffer_to_buffer(&p.output, 0, &p.staging, 0, p.out_bytes);
        }
        if let (Some(qs), Some(resolve), Some(ts_staging)) =
            (&query_set, &resolve_buffer, &ts_staging)
        {
            let count = if per_job { 2 * n_jobs as u32 + 1 } else { 2 };
            encoder.resolve_query_set(qs, 0..count, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, ts_staging, 0, 8 * u64::from(count));
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // One poll wakes every mapped slice; collect in job order.
        let mut receivers = Vec::with_capacity(prepared.len());
        for p in &prepared {
            let slice = p.staging.slice(..);
            let (sender, receiver) = oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| {
                sender.send(v).ok();
            });
            receivers.push(receiver);
        }
        let ts_receiver = ts_staging.as_ref().map(|staging| {
            let slice = staging.slice(..);
            let (sender, receiver) = oneshot_channel();
            slice.map_async(wgpu::MapMode::Read, move |v| {
                sender.send(v).ok();
            });
            (staging, receiver)
        });
        self.device.poll(wgpu::PollType::wait_indefinitely()).ok()?;

        let mut results = Vec::with_capacity(prepared.len());
        for (p, receiver) in prepared.iter().zip(receivers) {
            let _ = block_on(receiver.receive())?;
            let slice = p.staging.slice(..);
            let view = slice.get_mapped_range();
            let floats: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
            drop(view);
            p.staging.unmap();
            results.push(floats);
        }

        // Convert raw ticks to milliseconds per job and publish them for the
        // profiler harness (see [`GpuContext::last_batch_gpu_ms`]).
        if let (Some((ts_staging, ts_receiver)), Some(period_ns)) =
            (ts_receiver, Some(self.queue.get_timestamp_period()))
        {
            let _ = block_on(ts_receiver.receive())?;
            let slice = ts_staging.slice(..);
            let view = slice.get_mapped_range();
            let ticks: Vec<u64> = bytemuck::cast_slice(&view).to_vec();
            drop(view);
            ts_staging.unmap();
            use std::sync::atomic::Ordering;
            let period_f = f64::from(period_ns);
            let mut gpu_ms = Vec::with_capacity(n_jobs);
            if per_job {
                for i in 0..n_jobs {
                    let start = ticks[2 * i];
                    let end = ticks[2 * i + 1];
                    gpu_ms.push(end.saturating_sub(start) as f64 * period_f / 1e6);
                }
            } else {
                let total = ticks[1].saturating_sub(ticks[0]) as f64 * period_f / 1e6;
                gpu_ms.push(total);
            }
            // Some backends accept timestamp queries yet record no usable
            // delta; publish nothing rather than streams of zeros.
            if gpu_ms.iter().all(|v| *v == 0.0) {
                self.timings_usable.store(false, Ordering::Relaxed);
            } else if self.timings_usable.load(Ordering::Relaxed) {
                *self
                    .last_batch_gpu_ms
                    .lock()
                    .unwrap_or_else(|e| e.into_inner()) = Some(gpu_ms);
            }
        }
        Some(results)
    }

    /// Whether the adapter exposes timestamp queries and GPU pass timing is
    /// active.
    pub fn timestamps_enabled(&self) -> bool {
        self.timestamps_enabled
    }

    /// Per-job GPU pass durations in milliseconds of the most recently
    /// completed batch, `None` when no batch has run yet, the adapter lacks
    /// timestamp support, or the adapter's timestamps proved unusable. Because this is populated inside
    /// [`GpuContext::run_compute_batch`], every estimator that dispatches on
    /// the GPU is profileable through it without per-estimator plumbing.
    pub fn last_batch_gpu_ms(&self) -> Option<Vec<f64>> {
        self.last_batch_gpu_ms
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;
    use crate::estimators::approaches::discrete::mle_gpu::gpu_histogram_rows_dense;
    use ndarray::array;
    use rstest::rstest;
    use wgpu::util::DeviceExt;

    /// Synthetic [`wgpu::AdapterInfo`] for classification tests.
    fn info_fixture(name: &str, vendor: u32, device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 32,
            subgroup_max_size: 1024,
            transient_saves_memory: false,
        }
    }

    #[rstest]
    #[case("llvmpipe (LLVM 18.1.3)", wgpu::DeviceType::Cpu, true)]
    #[case("WARP Software Adapter", wgpu::DeviceType::Cpu, true)]
    #[case("virtio_gpu", wgpu::DeviceType::VirtualGpu, true)]
    #[case("NVIDIA GeForce GTX 1060", wgpu::DeviceType::DiscreteGpu, false)]
    #[case("Apple M4 Pro", wgpu::DeviceType::IntegratedGpu, false)]
    #[case("Mystery Renderer", wgpu::DeviceType::Other, false)]
    fn detects_software_renderers_by_device_type(
        #[case] name: &str,
        #[case] device_type: wgpu::DeviceType,
        #[case] expected: bool,
    ) {
        // Vendor IDs are deliberately bogus: detection must rely on the backend
        // reported device type alone (names lie, e.g. GL llvmpipe strings).
        let info = info_fixture(name, 0xdead, device_type);
        assert_eq!(is_software_renderer(&info), expected);
    }

    #[rstest]
    #[case(Some("1600"), Some("5000"), Some(1600), Some(5000))]
    #[case(Some(" 42 "), None, Some(42), None)]
    #[case(None, Some("3000"), None, Some(3000))]
    fn parse_env_min_points_accepts_valid_values(
        #[case] gaussian: Option<&str>,
        #[case] r#box: Option<&str>,
        #[case] want_gaussian: Option<usize>,
        #[case] want_box: Option<usize>,
    ) {
        let parsed = parse_env_min_points(gaussian, r#box);
        assert_eq!(parsed.gaussian, want_gaussian);
        assert_eq!(parsed.r#box, want_box);
    }

    #[test]
    fn parse_env_min_points_rejects_invalid_values() {
        let parsed = parse_env_min_points(Some("fast"), Some("-5"));
        assert_eq!(parsed, GpuMinPointsOverride::default());
        let parsed_partial = parse_env_min_points(Some("900"), Some("nope"));
        assert_eq!(parsed_partial.gaussian, Some(900));
        assert_eq!(parsed_partial.r#box, None);
    }

    #[test]
    fn resolution_precedence_programmatic_over_env_over_defaults() {
        let hardware = GpuMinPoints {
            gaussian: 1600,
            r#box: 5000,
        };
        let env = GpuMinPointsOverride {
            gaussian: Some(1000),
            r#box: None,
        };

        // No overrides at all → shipped constants.
        let resolved = resolve_min_points(
            hardware,
            GpuMinPointsOverride::default(),
            GpuMinPointsOverride::default(),
        );
        assert_eq!(resolved, hardware);

        // Env fills gaps, programmatic wins everywhere.
        let resolved = resolve_min_points(
            hardware,
            env,
            GpuMinPointsOverride {
                gaussian: None,
                r#box: Some(7),
            },
        );
        assert_eq!(
            resolved,
            GpuMinPoints {
                gaussian: 1000,
                r#box: 7
            }
        );
    }

    #[test]
    fn hardware_defaults_match_the_benchmarked_crossover() {
        assert_eq!(
            GpuMinPoints::defaults(false),
            GpuMinPoints {
                gaussian: GAUSSIAN_GPU_MIN_POINTS,
                r#box: BOX_GPU_MIN_POINTS
            },
            "defaults are provisional crossover snapshots (see the constants' \
             docs). Retune only with gpu_crossover data, never by hand"
        );
    }

    #[test]
    fn software_renderers_never_get_dispatched_by_default() {
        let never = GpuMinPoints::defaults(true);
        assert_eq!(never.gaussian, usize::MAX);
        assert_eq!(never.r#box, usize::MAX);
        // Overrides stay uniform (prog > env > default): an explicit request
        // may still force the software path, e.g. for benchmarking it.
        let resolved = resolve_min_points(
            GpuMinPoints::defaults(true),
            GpuMinPointsOverride {
                gaussian: Some(1),
                r#box: None,
            },
            GpuMinPointsOverride::default(),
        );
        assert_eq!(resolved.gaussian, 1);
        assert_eq!(resolved.r#box, usize::MAX);
    }

    /// The programmatic override is a global atomic, so this test restores it.
    /// A concurrent estimator call observing a transient override value merely
    /// routes differently for one call — parity between CPU and GPU paths is
    /// enforced independently, so no assertion elsewhere depends on routing.
    #[test]
    fn programmatic_override_roundtrip_and_clear() {
        set_gpu_min_points_override(Some(42), None);
        assert_eq!(gpu_min_points_gaussian(), 42);
        assert_eq!(gpu_min_points_box(), gpu_min_points_box_default());

        set_gpu_min_points_override(None, Some(7));
        assert_eq!(gpu_min_points_gaussian(), gpu_min_points_gaussian_default());
        assert_eq!(gpu_min_points_box(), 7);

        set_gpu_min_points_override(None, None);
        assert_eq!(gpu_min_points_gaussian(), gpu_min_points_gaussian_default());
        assert_eq!(gpu_min_points_box(), gpu_min_points_box_default());
    }

    /// Gate resolution must work without panicking on any machine — including
    /// ones with no adapter at all (defaults then simply never matter because
    /// the GPU paths fall back to CPU anyway).
    #[test]
    fn gate_getters_work_without_hardware_assumptions() {
        assert!(gpu_min_points_gaussian() > 0);
        assert!(gpu_min_points_box() > 0);
        let _ = gpu_adapter_info(); // probe must be side-effect free for callers
    }

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

    fn box_job(n: usize, seed: u64) -> BatchJob {
        // Deterministic pseudo-random points in [0, 1)^2, packed flat f32.
        let mut state = seed;
        let mut points = Vec::with_capacity(n * 2 * 4);
        for _ in 0..n * 2 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let v = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32;
            points.extend_from_slice(&v.to_ne_bytes());
        }
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Bandwidth {
            value: f32,
            dim_count: u32,
            _padding: [u32; 2],
        }
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Config {
            point_count: u32,
            dim_count: u32,
            normalization: f32,
            _padding: u32,
        }
        let bandwidth = Bandwidth {
            value: 0.9,
            dim_count: 2,
            _padding: [0; 2],
        };
        let config = Config {
            point_count: n as u32,
            dim_count: 2,
            normalization: (n as f32) * (0.9f32 * 0.45).powi(2),
            _padding: 0,
        };
        BatchJob {
            kind: ShaderKind::Box,
            wgsl: include_str!("approaches/kernel/box_kernel.wgsl"),
            points,
            extra_storage: Some(bytemuck::bytes_of(&bandwidth).to_vec()),
            config: bytemuck::bytes_of(&config).to_vec(),
            n_items: n as u32,
        }
    }

    /// When the adapter's timestamps are usable, a batched run publishes one
    /// positive duration per job; adapters whose timestamps are unsupported or
    /// unusable publish `None` instead of zeros.
    #[test]
    fn batch_publishes_gpu_timings_when_usable() {
        let ctx = GpuContext::get().expect("a hardware GPU adapter should be available");
        let jobs = [box_job(64, 7), box_job(96, 11)];
        ctx.run_compute_batch(&jobs).expect("batch should succeed");
        if let Some(ms) = ctx.last_batch_gpu_ms() {
            assert_eq!(ms.len(), 2, "one duration per job");
            assert!(
                ms.iter().all(|v| v.is_finite() && *v > 0.0),
                "durations must be positive, got {ms:?}"
            );
        }
        // else: timestamps unsupported/unusable on this adapter — fine.
    }

    /// Batched execution must produce results bit-for-bit identical to running
    /// the same jobs sequentially through `run_compute` — batching amortises
    /// submission overhead, never changes arithmetic.
    #[test]
    fn batch_matches_sequential_runs() {
        let ctx = GpuContext::get().expect("a hardware GPU adapter should be available");
        let jobs = [box_job(64, 7), box_job(96, 11)];

        let mut sequential = Vec::new();
        for job in &jobs {
            let out_bytes = job.n_items as u64 * 4;
            let points_buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test seq points"),
                    contents: &job.points,
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let extra_buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test seq bandwidth"),
                    contents: job.extra_storage.as_deref().unwrap(),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let config_buffer = ctx
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test seq config"),
                    contents: &job.config,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test batch seq output"),
                size: out_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let layout = ctx.bind_group_layout(job.kind);
            let bindings = [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: points_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: extra_buffer.as_entire_binding(),
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
                    job.kind,
                    job.wgsl,
                    ComputePass {
                        layout: &layout,
                        bindings: &bindings,
                        output: &output_buffer,
                        out_bytes,
                        n_items: job.n_items,
                    },
                )
                .expect("sequential run should succeed");
            sequential.push(bytemuck::cast_slice::<u8, f32>(&raw).to_vec());
        }

        let batched = ctx.run_compute_batch(&jobs).expect("batch should succeed");
        assert_eq!(batched.len(), jobs.len());
        for (seq, bat) in sequential.iter().zip(&batched) {
            assert_eq!(seq, bat, "batched result must match sequential result");
        }
    }
}

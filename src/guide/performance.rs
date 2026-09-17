// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: Apache-2.0

//! # Acceleration: GPU (wgpu) and CPU parallelism (rayon)
//!
//! The crate ships two **opt-in** accelerators. Both are off by default, both are
//! bit-for-bit compatible with the plain CPU path (they change *when*, not
//! *what*, is computed), and each owns the jobs it is genuinely good at.
//!
//! | Feature | Engine | Best at | Requires |
//! |---------|--------|---------|----------|
//! | `gpu` | wgpu (Vulkan / Metal / DX12 / WebGPU) | Dense, regular, large-N work: fixed-radius neighbour counts, weighted Gaussian density, discrete histograms, dense k-NN distances for the exponential-family estimators | A hardware GPU adapter |
//! | `parallel` | rayon (CPU) | Query-parallel CPU loops with independent per-item results — above all, kernels on machines without a GPU | Nothing beyond the feature flag |
//!
//! ## Choosing between them
//!
//! The rule of thumb is **prefer `gpu` where it applies**. wgpu wins decisively
//! on the dense O(N²) tiers (Gaussian density, box counts) once a call is large
//! enough to amortise a device dispatch, and it does so without occupying your
//! CPU cores.
//!
//! The exponential-family ($k$-nearest-neighbour) estimators also have a dense
//! GPU tier: all pairwise distances plus a per-row $k$-selection. Unlike the
//! kernels it only overtakes the kd-tree for **high-dimensional, large-$N$**
//! data, because the exact search is irregular and cheap at low $D$ — see
//! [GPU sizing gates](#gpu-sizing-gates).
//!
//! Use `parallel` instead when the GPU tier does not apply:
//!
//! - the `gpu` feature is not compiled in,
//! - no hardware adapter is present (the library declines software adapters),
//! - the call is **below** the GPU size gate, or
//! - the estimator has no GPU implementation at all.
//!
//! In those situations `parallel` spreads the per-query CPU loop across cores.
//! For the Gaussian kernel this is typically a **6–8×** speed-up at
//! $N \gtrsim 2000$ on an 8–12 core machine; the box kernel benefits mainly at
//! larger $N$.
//!
//! > **They do not run on the same call.** When both features are enabled, the
//! > existing GPU gate decides first: above the gate the work goes to wgpu, and
//! > `parallel` only applies to the CPU fallback the gate declined. There is no
//! > configuration in which one computation is both dispatched to the GPU and
//! > fanned out over rayon.
//!
//! ## Enabling
//!
//! ```toml
//! # GPU-accelerated builds
//! infomeasure = { version = "0.4", features = ["gpu"] }
//!
//! # GPU-less / CPU-only builds that want multi-core density loops
//! infomeasure = { version = "0.4", features = ["parallel"] }
//!
//! # Both: wgpu owns the large-N tiers, rayon covers the CPU fallback
//! infomeasure = { version = "0.4", features = ["gpu", "parallel"] }
//! ```
//!
//! Or from the command line:
//!
//! ```bash
//! cargo build --release --features gpu
//! cargo build --release --features parallel
//! ```
//!
//! ## Controlling the CPU thread count
//!
//! `parallel` builds use rayon's global pool (all available cores by default).
//! Set [`RAYON_NUM_THREADS`](https://docs.rs/rayon/latest/rayon/#environment-variables)
//! to bound it, for example to keep a machine responsive or to reproduce a
//! single-threaded measurement:
//!
//! ```bash
//! RAYON_NUM_THREADS=1 cargo run --release --features parallel
//! ```
//!
//! The published benchmarks run this way on purpose: their CPU track is
//! single-threaded so that every toolkit is compared on equal footing.
//!
//! ## GPU sizing gates
//!
//! The GPU tier only engages above a machine-relative size threshold (defaults
//! for a Gaussian kernel around 1200 points and a box kernel around 4000). Below
//! it the CPU path is faster. Tune the thresholds per machine with
//! `INFOMEASURE_GPU_MIN_GAUSSIAN` and `INFOMEASURE_GPU_MIN_BOX` when you know
//! your hardware's crossover better than the shipped defaults.
//!
//! The exponential-family dense tier is gated on **both** size and
//! dimensionality: by default at least about 4000 points **and** $D \ge 8$ on
//! integrated GPUs. The reason is the fixed dispatch/readback cost (roughly
//! 2.5 ms on Metal): a scalar distance scan cannot amortise it while the kd-tree
//! is still efficient, and the tree degenerates with dimensionality. Measured
//! crossover on an Apple M4 Pro is 1–4D never (up to $N = 6000$), 6D around
//! $N = 5000$, 8D around $N = 3000$ and 16D below $N = 2000$. Override with
//! `INFOMEASURE_GPU_MIN_EXPFAM` and `INFOMEASURE_GPU_MIN_EXPFAM_DIM`; below
//! either gate the estimators use the exact kd-tree path unchanged.
//!
//! The gates are **device-aware**. A dedicated card is relatively stronger than
//! the host CPU, so the tiers engage earlier there: on a GTX 1060 the expfam
//! crossover is around $D = 4$ (4D wins 1.4× even at $N = 2000$, 16D up to
//! 11×), and the kernels win much earlier too (Gaussian by about 16× already at
//! 1000 points, box by about 1.5×). Discrete adapters therefore use
//! $D \ge 4$ with about 2000 points for the expfam tier and about 1000 points
//! for both kernels. Integrated GPUs and unknown device types keep the
//! conservative profile. Explicit overrides always win over the profile.
//!
//! These thresholds are set against the **single-thread** CPU baseline (the fair
//! published track). Since the GPU gate decides before `parallel`, a discrete
//! box call around 1000 points preempts the multi-thread fallback even though
//! four CPU threads are still a little faster at that size. That is deliberate:
//! the single-thread win is larger and grows with $N$, and `parallel` remains the
//! fallback for GPU-less machines and calls below the gate.
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::Array1;
//!
//! // Identical code with or without the `gpu` / `parallel` features: the
//! // accelerators are selected internally, the result does not change.
//! let data = Array1::from((0..5000).map(|i| (i as f64 * 0.001).sin()).collect::<Vec<_>>());
//! let gaussian = Entropy::new_kernel_with_type(data, "gaussian".to_string(), 0.5);
//! let value = gaussian.global_value();
//! assert!(value.is_finite());
//! ```

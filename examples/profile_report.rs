// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Machine-readable hotspot profiler for estimator hot paths.
//!
//! Runs one estimator family in a loop under the [`pprof`] sampling profiler
//! and prints a ranked self-time table (`text`) or structured JSON (`json`)
//! to stdout. Side files (`flamegraph.svg`, `profile.pb`) are written under
//! `target/profiling/`.
//!
//! Configuration via environment variables:
//!
//! ```text
//! PROFILE_ESTIMATOR  one of the workloads below        (required)
//!   discrete_entropy     discrete MLE entropy
//!   mi_discrete          discrete MLE mutual information
//!   ordinal              ordinal-pattern entropy
//!   renyi                KSG-style Rényi entropy
//!   tsallis              KSG-style Tsallis entropy
//!   kl                   Kozachenko–Leonenko entropy
//!   ksg_mi               KSG mutual information
//!   ksg_cmi              KSG conditional mutual information
//!   kernel_gaussian_cpu  Gaussian KDE mutual information (CPU path)
//!   kernel_box_cpu       box KDE mutual information (CPU path)
//! PROFILE_N          dataset size            (default 10000)
//! PROFILE_K          neighbour count         (default 3)
//! PROFILE_BW         kernel bandwidth        (default 0.9)
//! PROFILE_ORDER      ordinal pattern order   (default 3)
//! PROFILE_ALPHA      Rényi alpha             (default 1.2)
//! PROFILE_Q          Tsallis q               (default 1.2)
//! PROFILE_SECONDS    sampling duration       (default 5)
//! PROFILE_TOP        table rows              (default 25)
//! PROFILE_FORMAT     text | json             (default text)
//! ```
//!
//! Run with:
//!
//! ```sh
//! RUSTFLAGS="-Cforce-frame-pointers" OPENBLAS_NUM_THREADS=1 \
//!     cargo run --profile profiling --features profiling --example profile_report
//! ```
//!
//! Both prefixes matter:
//! - `RUSTFLAGS="-Cforce-frame-pointers"` keeps frame pointers so the sampler
//!   can walk stacks without libunwind, whose DWARF unwinder is not
//!   async-signal-safe on Apple silicon and traps intermittently.
//! - `OPENBLAS_NUM_THREADS=1` stops OpenBLAS (configured pre-main) from
//!   busy-spinning its worker pool through ~20% of the samples.

use std::collections::HashMap;
use std::fs::{File, create_dir_all};
use std::time::{Duration, Instant};

use ndarray::Array1;
use pprof::ProfilerGuardBuilder;

const FREQ_HZ: i32 = 1000;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Correlated Gaussian pair, mirroring `generate_correlated` in the benches.
fn correlated_pair(n: usize, correlation: f64, seed: u64) -> (Array1<f64>, Array1<f64>) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let z: f64 = rng.sample(normal);
        let w: f64 = rng.sample(normal);
        x.push(z);
        y.push(correlation * z + (1.0 - correlation.powi(2)).sqrt() * w);
    }
    (Array1::from(x), Array1::from(y))
}

fn uniform_ints(n: usize, states: i32, seed: u64) -> Array1<i32> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from((0..n).map(|_| rng.gen_range(0..states)).collect::<Vec<_>>())
}

fn gaussians(n: usize, seed: u64) -> Vec<f64> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..n).map(|_| rng.sample(normal)).collect()
}

enum Workload {
    DiscreteEntropy,
    MiDiscrete,
    Ordinal,
    Renyi,
    Tsallis,
    Kl,
    KsgMi,
    KsgCmi,
    KernelGaussianCpu,
    KernelBoxCpu,
}

impl Workload {
    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "discrete_entropy" => Self::DiscreteEntropy,
            "mi_discrete" => Self::MiDiscrete,
            "ordinal" => Self::Ordinal,
            "renyi" => Self::Renyi,
            "tsallis" => Self::Tsallis,
            "kl" => Self::Kl,
            "ksg_mi" => Self::KsgMi,
            "ksg_cmi" => Self::KsgCmi,
            "kernel_gaussian_cpu" => Self::KernelGaussianCpu,
            "kernel_box_cpu" => Self::KernelBoxCpu,
            _ => return None,
        })
    }

    fn describe(&self) -> &'static str {
        match self {
            Self::DiscreteEntropy => "discrete_entropy",
            Self::MiDiscrete => "mi_discrete",
            Self::Ordinal => "ordinal",
            Self::Renyi => "renyi",
            Self::Tsallis => "tsallis",
            Self::Kl => "kl",
            Self::KsgMi => "ksg_mi",
            Self::KsgCmi => "ksg_cmi",
            Self::KernelGaussianCpu => "kernel_gaussian_cpu",
            Self::KernelBoxCpu => "kernel_box_cpu",
        }
    }
}

fn main() {
    let estimator = env_or("PROFILE_ESTIMATOR", "");
    let workload = Workload::parse(&estimator).unwrap_or_else(|| {
        eprintln!(
            "PROFILE_ESTIMATOR must be one of: discrete_entropy, mi_discrete, \
             ordinal, renyi, tsallis, kl, ksg_mi, ksg_cmi, \
             kernel_gaussian_cpu, kernel_box_cpu"
        );
        std::process::exit(2);
    });
    let n: usize = env_or("PROFILE_N", "10000").parse().expect("PROFILE_N");
    let k: usize = env_or("PROFILE_K", "3").parse().expect("PROFILE_K");
    let bw: f64 = env_or("PROFILE_BW", "0.9").parse().expect("PROFILE_BW");
    let order: usize = env_or("PROFILE_ORDER", "3").parse().expect("PROFILE_ORDER");
    let alpha: f64 = env_or("PROFILE_ALPHA", "1.2")
        .parse()
        .expect("PROFILE_ALPHA");
    let q: f64 = env_or("PROFILE_Q", "1.2").parse().expect("PROFILE_Q");
    let seconds: u64 = env_or("PROFILE_SECONDS", "5")
        .parse()
        .expect("PROFILE_SECONDS");
    let top: usize = env_or("PROFILE_TOP", "25").parse().expect("PROFILE_TOP");
    let format = env_or("PROFILE_FORMAT", "text");
    let seed: u64 = env_or("PROFILE_SEED", "42").parse().expect("PROFILE_SEED");
    let params = Params {
        n,
        k,
        bw,
        order,
        alpha,
        q,
        seed,
    };

    // One untimed iteration first so allocator warm-up and lazy statics do not
    // pollute the samples.
    run_once(&workload, &params);
    let deadline = Instant::now() + Duration::from_secs(seconds);

    let guard = ProfilerGuardBuilder::default()
        .frequency(FREQ_HZ)
        .build()
        .unwrap();
    let mut iterations = 0usize;
    while Instant::now() < deadline {
        run_once(&workload, &params);
        iterations += 1;
    }
    let elapsed = deadline.saturating_duration_since(Instant::now() - Duration::ZERO);

    let report = guard.report().build().expect("profile report");

    // Aggregate self time (innermost symbol) and total time (anywhere on
    // stack). Frames are innermost-PC first, and within one PC the inlined
    // symbols run caller-to-callee, so reversing each group puts the true
    // leaf symbol first.
    struct Entry {
        file: String,
        line: u32,
        self_samples: usize,
        total_samples: usize,
    }
    let mut entries: HashMap<String, Entry> = HashMap::new();
    let mut total_samples = 0usize;
    for (frames, count) in report.data.iter() {
        let count = (*count).max(0) as usize;
        if count == 0 || frames.frames.is_empty() {
            continue;
        }
        total_samples += count;

        let mut ordered: Vec<&pprof::Symbol> = Vec::new();
        for frame_group in &frames.frames {
            ordered.extend(frame_group.iter().rev());
        }
        for (idx, sym) in ordered.iter().enumerate() {
            let key = sym.name();
            let file = sym
                .filename
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            let line = sym.lineno.unwrap_or_default();
            let slot = entries.entry(key).or_insert_with(|| Entry {
                file,
                line,
                self_samples: 0,
                total_samples: 0,
            });
            slot.total_samples += count;
            if idx == 0 {
                slot.self_samples += count;
            }
        }
    }

    let mut ranked: Vec<(&String, &Entry)> = entries.iter().collect();
    ranked.sort_by_key(|(_, e)| std::cmp::Reverse(e.self_samples));
    ranked.truncate(top);

    let out_dir = "target/profiling";
    create_dir_all(out_dir).ok();

    // Side artifact for human inspection: a flamegraph rooted at the profiled
    // function. Stacks are emitted in collapsed form and pruned below
    // `run_once`, so runtime startup frames and background threads (BLAS pool,
    // allocator service calls) do not bury the interesting layers.
    let svg_path = format!("{out_dir}/flamegraph_{}.svg", workload.describe());
    let mut collapsed = String::new();
    for (frames, count) in report.data.iter() {
        let count = (*count).max(0) as usize;
        if count == 0 || frames.frames.is_empty() {
            continue;
        }
        let mut ordered: Vec<&pprof::Symbol> = Vec::new();
        for frame_group in &frames.frames {
            ordered.extend(frame_group.iter().rev());
        }
        let Some(cut) = ordered
            .iter()
            .position(|sym| sym.name().contains("profile_report::run_once"))
        else {
            continue;
        };
        ordered.truncate(cut + 1);
        let line = ordered
            .iter()
            .rev()
            .map(|sym| sym.name())
            .collect::<Vec<_>>()
            .join(";");
        collapsed.push_str(&line);
        collapsed.push(' ');
        collapsed.push_str(&count.to_string());
        collapsed.push('\n');
    }
    if let Ok(mut f) = File::create(&svg_path) {
        pprof::flamegraph::from_lines(
            &mut pprof::flamegraph::Options::default(),
            collapsed.lines(),
            &mut f,
        )
        .ok();
    }

    if format == "json" {
        let rows: Vec<_> = ranked
            .iter()
            .map(|(sym, e)| {
                serde_json::json!({
                    "symbol": sym,
                    "file": e.file,
                    "line": e.line,
                    "self_samples": e.self_samples,
                    "total_samples": e.total_samples,
                    "self_pct": 100.0 * e.self_samples as f64 / total_samples.max(1) as f64,
                    "total_pct": 100.0 * e.total_samples as f64 / total_samples.max(1) as f64,
                })
            })
            .collect();
        let doc = serde_json::json!({
            "estimator": workload.describe(),
            "n": n,
            "k": k,
            "bandwidth": bw,
            "order": order,
            "alpha": alpha,
            "q": q,
            "seconds": seconds,
            "iterations": iterations,
            "frequency_hz": FREQ_HZ,
            "total_samples": total_samples,
            "flamegraph": svg_path,
            "entries": rows,
        });
        println!("{doc}");
    } else {
        println!(
            "estimator={est} n={n} k={k} bw={bw} seconds={seconds} iterations={iterations} samples={total_samples}",
            est = workload.describe(),
        );
        println!("flamegraph -> {svg_path}");
        println!("   #   self%  total%  function");
        for (rank, (sym, e)) in ranked.iter().enumerate() {
            let sp = 100.0 * e.self_samples as f64 / total_samples.max(1) as f64;
            let tp = 100.0 * e.total_samples as f64 / total_samples.max(1) as f64;
            println!(
                "{:>4} {:>6.2} {:>6.2}  {} ({}:{})",
                rank + 1,
                sp,
                tp,
                sym,
                e.file.rsplit('/').next().unwrap_or(&e.file),
                e.line
            );
        }
    }
    let _ = elapsed;
}

struct Params {
    n: usize,
    k: usize,
    bw: f64,
    order: usize,
    alpha: f64,
    q: f64,
    seed: u64,
}

fn run_once(workload: &Workload, p: &Params) {
    use infomeasure::estimators::entropy::{Entropy, GlobalValue};
    use infomeasure::estimators::mutual_information::MutualInformation;
    use std::hint::black_box;

    match workload {
        Workload::DiscreteEntropy => {
            let data = uniform_ints(p.n, 10, p.seed);
            black_box(Entropy::new_discrete(data).global_value());
        }
        Workload::MiDiscrete => {
            let x = uniform_ints(p.n, 10, p.seed);
            let y = uniform_ints(p.n, 10, p.seed.wrapping_add(1));
            black_box(MutualInformation::new_discrete_mle(&[x, y]).global_value());
        }
        Workload::Ordinal => {
            let data = Array1::from(gaussians(p.n, p.seed));
            black_box(Entropy::new_ordinal(data, p.order).global_value());
        }
        Workload::Renyi => {
            const NOISE: f64 = 1e-10;
            let data = Array1::from(gaussians(p.n, p.seed));
            black_box(Entropy::new_renyi_1d(data, p.k, p.alpha, NOISE).global_value());
        }
        Workload::Tsallis => {
            const NOISE: f64 = 1e-10;
            let data = Array1::from(gaussians(p.n, p.seed));
            black_box(Entropy::new_tsallis_1d(data, p.k, p.q, NOISE).global_value());
        }
        Workload::Kl => {
            const NOISE: f64 = 1e-10;
            let data = Array1::from(gaussians(p.n, p.seed));
            black_box(Entropy::new_kl_1d(data, p.k, NOISE).global_value());
        }
        Workload::KsgMi => {
            const NOISE: f64 = 1e-10;
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            black_box(MutualInformation::new_ksg(&[x, y], p.k, NOISE).global_value());
        }
        Workload::KsgCmi => {
            const NOISE: f64 = 1e-10;
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let z = Array1::from(gaussians(p.n, p.seed.wrapping_add(2)));
            black_box(MutualInformation::new_cmi_ksg(&[x, y], &z, p.k, NOISE).global_value());
        }
        Workload::KernelGaussianCpu => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let mut mi = MutualInformation::new_kernel_with_type(&[x, y], "gaussian".into(), p.bw);
            mi.set_force_cpu(true);
            black_box(mi.global_value());
        }
        Workload::KernelBoxCpu => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let mut mi = MutualInformation::new_kernel_with_type(&[x, y], "box".into(), p.bw);
            mi.set_force_cpu(true);
            black_box(mi.global_value());
        }
    }
}

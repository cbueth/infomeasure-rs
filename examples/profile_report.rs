// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Machine-readable hotspot profiler for estimator hot paths.
//!
//! Runs one estimator family in a loop under the [`pprof`] sampling profiler
//! and prints a ranked self-time table (`text`) or structured JSON (`json`)
//! to stdout. Side files (`flamegraph.svg`) are written under
//! `target/profiling/`.
//!
//! Configuration via environment variables:
//!
//! ```text
//! PROFILE_ESTIMATOR  one of the workloads below        (required)
//!   discrete_entropy     discrete MLE entropy
//!   mi_discrete          discrete MLE mutual information
//!   discrete_cmi         discrete MLE conditional MI
//!   discrete_te          discrete MLE transfer entropy
//!   discrete_cte         discrete MLE conditional TE
//!   ordinal              ordinal-pattern entropy
//!   {ordinal,renyi_entropy,tsallis_entropy,kl_entropy}_mi/_cmi/_te/_cte
//!                        Rényi/Tsallis/KL/KSG-family measures
//!   ksg_mi/_cmi/_te/_cte KSG information measures
//!   kernel_{gaussian,box}_{mi,cmi,te,cte}_cpu       kernel measures (CPU path)

//! Running any PROFILE_ESTIMATOR without a match prints the full list.
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

/// Lagged source-target pair plus an independent conditioning series,
/// mirroring the TE bench generators.
fn lagged_triple(n: usize, coupling: f64, seed: u64) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_distr::Normal;

    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut source = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);
    for _ in 0..n {
        source.push(rng.sample(normal));
    }
    for &src in &source {
        let noise: f64 = rng.sample(normal);
        target.push(coupling * src + (1.0 - coupling * coupling).sqrt() * noise);
    }
    let cond = gaussians(n, seed.wrapping_add(7));
    (
        Array1::from(source),
        Array1::from(target),
        Array1::from(cond),
    )
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

#[derive(Clone, Copy)]
enum Measure {
    Entropy,
    Mi,
    Cmi,
    Te,
    Cte,
}

#[derive(Clone, Copy)]
enum Family {
    Discrete,
    Ordinal,
    Renyi,
    Tsallis,
    Kl,
    Ksg,
    KernelGaussian,
    KernelBox,
}

struct Workload {
    family: Family,
    measure: Measure,
}

impl Workload {
    fn key(&self) -> String {
        match (self.family, self.measure) {
            (Family::Discrete, Measure::Entropy) => "discrete_entropy".into(),
            // Historic bench-group spellings kept for these two.
            (Family::Discrete, Measure::Mi) => "mi_discrete".into(),
            (Family::Discrete, m) => format!("discrete_{}", measure_tag(m)),
            (Family::Ordinal, Measure::Entropy) => "ordinal".into(),
            (f, m) => {
                let fam = match f {
                    Family::Ordinal => "ordinal",
                    Family::Renyi => "renyi",
                    Family::Tsallis => "tsallis",
                    Family::Kl => "kl",
                    Family::Ksg => "ksg",
                    Family::KernelGaussian => "kernel_gaussian",
                    Family::KernelBox => "kernel_box",
                    Family::Discrete => unreachable!(),
                };
                let meas = if matches!(f, Family::KernelGaussian | Family::KernelBox) {
                    // kernel_gaussian_cmi_cpu style
                    format!("{}_cpu", measure_tag(m))
                } else {
                    measure_tag(m).to_string()
                };
                format!("{fam}_{meas}")
            }
        }
    }

    fn all() -> Vec<Self> {
        use Family as F;
        use Measure as M;
        let mut v = Vec::new();
        for (fam, has_entropy) in [
            (F::Discrete, true),
            (F::Ordinal, true),
            (F::Renyi, true),
            (F::Tsallis, true),
            (F::Kl, true),
            (F::Ksg, false),
            (F::KernelGaussian, false),
            (F::KernelBox, false),
        ] {
            if has_entropy {
                v.push(Self {
                    family: fam,
                    measure: M::Entropy,
                });
            }
            v.push(Self {
                family: fam,
                measure: M::Mi,
            });
            v.push(Self {
                family: fam,
                measure: M::Cmi,
            });
            v.push(Self {
                family: fam,
                measure: M::Te,
            });
            v.push(Self {
                family: fam,
                measure: M::Cte,
            });
        }
        v
    }

    fn parse(s: &str) -> Option<Self> {
        Self::all().into_iter().find(|w| w.key() == s)
    }
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

fn main() {
    let estimator = env_or("PROFILE_ESTIMATOR", "");
    let workload = Workload::parse(&estimator).unwrap_or_else(|| {
        eprintln!(
            "PROFILE_ESTIMATOR must be one of:\n{}",
            Workload::all()
                .iter()
                .map(|w| "  ".to_string() + &w.key())
                .collect::<Vec<_>>()
                .join("\n")
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
    let svg_path = format!("{out_dir}/flamegraph_{}.svg", workload.key());
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
            "estimator": workload.key(),
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
            "estimator={} n={n} k={k} bw={bw} seconds={seconds} iterations={iterations} samples={total_samples}",
            workload.key()
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
}

const NOISE: f64 = 1e-10;

fn run_once(workload: &Workload, p: &Params) {
    use infomeasure::estimators::entropy::{Entropy, GlobalValue};
    use infomeasure::estimators::mutual_information::MutualInformation;
    use infomeasure::estimators::transfer_entropy::TransferEntropy;
    use std::hint::black_box;

    macro_rules! gv {
        ($e:expr) => {
            black_box($e.global_value())
        };
    }

    match (workload.family, workload.measure) {
        (Family::Discrete, Measure::Entropy) => {
            gv!(Entropy::new_discrete(uniform_ints(p.n, 10, p.seed)))
        }
        (Family::Discrete, Measure::Mi) => {
            let x = uniform_ints(p.n, 10, p.seed);
            let y = uniform_ints(p.n, 10, p.seed.wrapping_add(1));
            gv!(MutualInformation::new_discrete_mle(&[x, y]))
        }
        (Family::Discrete, Measure::Cmi) => {
            let x = uniform_ints(p.n, 10, p.seed);
            let y = uniform_ints(p.n, 10, p.seed.wrapping_add(1));
            let z = uniform_ints(p.n, 10, p.seed.wrapping_add(2));
            gv!(MutualInformation::new_cmi_discrete_mle(&[x, y], &z))
        }
        (Family::Discrete, Measure::Te) => {
            let s = uniform_ints(p.n, 10, p.seed);
            let t = uniform_ints(p.n, 10, p.seed.wrapping_add(1));
            gv!(TransferEntropy::new_discrete_mle(&s, &t, 1, 1, 1))
        }
        (Family::Discrete, Measure::Cte) => {
            let s = uniform_ints(p.n, 10, p.seed);
            let t = uniform_ints(p.n, 10, p.seed.wrapping_add(1));
            let c = uniform_ints(p.n, 10, p.seed.wrapping_add(2));
            gv!(TransferEntropy::new_cte_discrete_mle(
                &s, &t, &c, 1, 1, 1, 1
            ))
        }
        (Family::Ordinal, Measure::Entropy) => {
            gv!(Entropy::new_ordinal(
                Array1::from(gaussians(p.n, p.seed)),
                p.order
            ))
        }
        (Family::Ordinal, Measure::Mi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            gv!(MutualInformation::new_ordinal(&[x, y], p.order, 1, false))
        }
        (Family::Ordinal, Measure::Cmi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let z = Array1::from(gaussians(p.n, p.seed.wrapping_add(2)));
            gv!(MutualInformation::new_cmi_ordinal(
                &[x, y],
                &z,
                p.order,
                1,
                false
            ))
        }
        (Family::Ordinal, Measure::Te) => {
            let (s, t, _) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_ordinal(
                &s, &t, p.order, 1, 1, 1, false
            ))
        }
        (Family::Ordinal, Measure::Cte) => {
            let (s, t, c) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_cte_ordinal(
                &s, &t, &c, p.order, 1, 1, 1, 1, false
            ))
        }
        (Family::Renyi, Measure::Entropy) => {
            gv!(Entropy::new_renyi_1d(
                Array1::from(gaussians(p.n, p.seed)),
                p.k,
                p.alpha,
                NOISE
            ))
        }
        (Family::Renyi, Measure::Mi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            gv!(MutualInformation::new_renyi(&[x, y], p.k, p.alpha, NOISE))
        }
        (Family::Renyi, Measure::Cmi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let z = Array1::from(gaussians(p.n, p.seed.wrapping_add(2)));
            gv!(MutualInformation::new_cmi_renyi(
                &[x, y],
                &z,
                p.k,
                p.alpha,
                NOISE
            ))
        }
        (Family::Renyi, Measure::Te) => {
            let (s, t, _) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_renyi(&s, &t, p.k, p.alpha, NOISE))
        }
        (Family::Renyi, Measure::Cte) => {
            let (s, t, c) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_cte_renyi(
                &s, &t, &c, p.k, p.alpha, NOISE
            ))
        }
        (Family::Tsallis, Measure::Entropy) => {
            gv!(Entropy::new_tsallis_1d(
                Array1::from(gaussians(p.n, p.seed)),
                p.k,
                p.q,
                NOISE
            ))
        }
        (Family::Tsallis, Measure::Mi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            gv!(MutualInformation::new_tsallis(&[x, y], p.k, p.q, NOISE))
        }
        (Family::Tsallis, Measure::Cmi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let z = Array1::from(gaussians(p.n, p.seed.wrapping_add(2)));
            gv!(MutualInformation::new_cmi_tsallis(
                &[x, y],
                &z,
                p.k,
                p.q,
                NOISE
            ))
        }
        (Family::Tsallis, Measure::Te) => {
            let (s, t, _) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_tsallis(&s, &t, p.k, p.q, NOISE))
        }
        (Family::Tsallis, Measure::Cte) => {
            let (s, t, c) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_cte_tsallis(
                &s, &t, &c, p.k, p.q, NOISE
            ))
        }
        (Family::Kl, Measure::Entropy) => {
            gv!(Entropy::new_kl_1d(
                Array1::from(gaussians(p.n, p.seed)),
                p.k,
                NOISE
            ))
        }
        (Family::Kl, Measure::Mi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            gv!(MutualInformation::new_kl(&[x, y], p.k, NOISE))
        }
        (Family::Kl, Measure::Cmi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let z = Array1::from(gaussians(p.n, p.seed.wrapping_add(2)));
            gv!(MutualInformation::new_cmi_kl(&[x, y], &z, p.k, NOISE))
        }
        (Family::Kl, Measure::Te) => {
            let (s, t, _) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_kl(&s, &t, p.k, NOISE))
        }
        (Family::Kl, Measure::Cte) => {
            let (s, t, c) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_cte_kl(&s, &t, &c, p.k, NOISE))
        }
        (Family::Ksg, Measure::Mi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            gv!(MutualInformation::new_ksg(&[x, y], p.k, NOISE))
        }
        (Family::Ksg, Measure::Cmi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let z = Array1::from(gaussians(p.n, p.seed.wrapping_add(2)));
            gv!(MutualInformation::new_cmi_ksg(&[x, y], &z, p.k, NOISE))
        }
        (Family::Ksg, Measure::Te) => {
            let (s, t, _) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_ksg(&s, &t, 1, 1, 1, p.k, NOISE))
        }
        (Family::Ksg, Measure::Cte) => {
            let (s, t, c) = lagged_triple(p.n, 0.5, p.seed);
            gv!(TransferEntropy::new_cte_ksg(
                &s, &t, &c, 1, 1, 1, 1, p.k, NOISE
            ))
        }
        (family @ (Family::KernelGaussian | Family::KernelBox), Measure::Mi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let kt = kernel_type(&family);
            let mut est = MutualInformation::new_kernel_with_type(&[x, y], kt, p.bw);
            est.set_force_cpu(true);
            gv!(est)
        }
        (family @ (Family::KernelGaussian | Family::KernelBox), Measure::Cmi) => {
            let (x, y) = correlated_pair(p.n, 0.5, p.seed);
            let z = Array1::from(gaussians(p.n, p.seed.wrapping_add(2)));
            let kt = kernel_type(&family);
            let mut est = MutualInformation::new_cmi_kernel_with_type(&[x, y], &z, kt, p.bw);
            est.set_force_cpu(true);
            gv!(est)
        }
        (family @ (Family::KernelGaussian | Family::KernelBox), Measure::Te) => {
            let (s, t, _) = lagged_triple(p.n, 0.5, p.seed);
            let kt = kernel_type(&family);
            let mut est = TransferEntropy::new_kernel_with_type(&s, &t, 1, 1, 1, kt, p.bw);
            est.set_force_cpu(true);
            gv!(est)
        }
        (family @ (Family::KernelGaussian | Family::KernelBox), Measure::Cte) => {
            let (s, t, c) = lagged_triple(p.n, 0.5, p.seed);
            let kt = kernel_type(&family);
            let mut est =
                TransferEntropy::new_cte_kernel_with_type(&s, &t, &c, 1, 1, 1, 1, kt, p.bw);
            est.set_force_cpu(true);
            gv!(est)
        }
        _ => unreachable!("entropy-only families have no MI/CMI/TE/CTE workloads"),
    };
}

fn kernel_type(family: &Family) -> String {
    match family {
        Family::KernelGaussian => "gaussian".to_string(),
        Family::KernelBox => "box".to_string(),
        _ => unreachable!(),
    }
}

fn measure_tag(m: Measure) -> &'static str {
    match m {
        Measure::Entropy => "entropy",
        Measure::Mi => "mi",
        Measure::Cmi => "cmi",
        Measure::Te => "te",
        Measure::Cte => "cte",
    }
}

// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fair cross-package runtime collector (infomeasure-rs side).
//!
//! Run with: `cargo bench --bench collect_cross_package`
//!
//! Times **only the estimator call** on pre-generated, identical datasets
//! (`target/bench-data/`). Warm-up calls are discarded; fixed rounds, no
//! dynamic timing. Output is the schema-v2 watch JSON.
//!
//! Env:
//!   BENCH_DATA_DIR     dataset dir (default `target/bench-data`)
//!   BENCH_OUT          output JSON (default `<data>/cross_package.json`)
//!   BENCH_SHORT=1      warm-up 1 + 3 iterations (local iteration)
//!   BENCH_WARMUP / BENCH_ITERATIONS   explicit overrides
//!   BENCH_SIZES        comma-separated input lengths

#![allow(unused_imports)]

use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod utils;

use utils::datasets::*;
use utils::hardware::detect_hardware;

#[derive(Clone, Copy, Debug)]
enum Measure {
    Entropy,
    Mi,
    Cmi,
    Te,
    Cte,
}

impl Measure {
    fn as_str(self) -> &'static str {
        match self {
            Self::Entropy => "entropy",
            Self::Mi => "mi",
            Self::Cmi => "cmi",
            Self::Te => "te",
            Self::Cte => "cte",
        }
    }
    fn n_cols(self) -> usize {
        n_cols(self.as_str())
    }
}

#[derive(Clone, Copy, Debug)]
enum Approach {
    Discrete,
    Ksg,
    KernelBox,
    KernelGaussian,
}

impl Approach {
    fn as_str(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Ksg => "ksg",
            Self::KernelBox => "kernel_box",
            Self::KernelGaussian => "kernel_gaussian",
        }
    }
    fn kind(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            _ => "continuous",
        }
    }
    fn kernel_name(self) -> &'static str {
        match self {
            Self::KernelGaussian => "gaussian",
            _ => "box",
        }
    }
    fn is_discrete(self) -> bool {
        matches!(self, Self::Discrete)
    }
    fn is_kernel(self) -> bool {
        matches!(self, Self::KernelBox | Self::KernelGaussian)
    }
}

fn function_name(measure: Measure, approach: Approach) -> &'static str {
    use Approach::*;
    use Measure::*;
    match (measure, approach) {
        (Entropy, Discrete) => "Entropy::new_discrete",
        (Entropy, Ksg) => "Entropy::new_kl_1d",
        (Entropy, KernelBox | KernelGaussian) => "Entropy::new_kernel_with_type",
        (Mi, Discrete) => "MutualInformation::new_discrete_mle",
        (Mi, Ksg) => "MutualInformation::new_ksg",
        (Mi, KernelBox | KernelGaussian) => "MutualInformation::new_kernel_with_type",
        (Cmi, Discrete) => "MutualInformation::new_cmi_discrete_mle",
        (Cmi, Ksg) => "MutualInformation::new_cmi_ksg",
        (Cmi, KernelBox | KernelGaussian) => "MutualInformation::new_cmi_kernel_with_type",
        (Te, Discrete) => "TransferEntropy::new_discrete_mle",
        (Te, Ksg) => "TransferEntropy::new_ksg",
        (Te, KernelBox | KernelGaussian) => "TransferEntropy::new_kernel_with_type",
        (Cte, Discrete) => "TransferEntropy::new_cte_discrete_mle",
        (Cte, Ksg) => "TransferEntropy::new_cte_ksg",
        (Cte, KernelBox | KernelGaussian) => "TransferEntropy::new_cte_kernel_with_type",
    }
}

/// Construct + compute the estimator on one dataset. This is the timed region.
fn run_case(measure: Measure, approach: Approach, n: usize, seed: u64, dir: &Path) -> f64 {
    let id = dataset_id(measure.as_str(), approach.kind(), seed, n);
    let path = dataset_path(dir, &id);
    if approach.is_discrete() {
        let flat = read_i32(&path).expect("read discrete dataset");
        let cols = deinterleave_i32(&flat, measure.n_cols());
        let a = |i: usize| Array1::from(cols[i].clone());
        match measure {
            Measure::Entropy => Entropy::new_discrete(a(0)).global_value(),
            Measure::Mi => MutualInformation::new_discrete_mle(&[a(0), a(1)]).global_value(),
            Measure::Cmi => {
                MutualInformation::new_cmi_discrete_mle(&[a(0), a(1)], &a(2)).global_value()
            }
            Measure::Te => TransferEntropy::new_discrete_mle(&a(0), &a(1), 1, 1, 1).global_value(),
            Measure::Cte => TransferEntropy::new_cte_discrete_mle(&a(0), &a(1), &a(2), 1, 1, 1, 1)
                .global_value(),
        }
    } else {
        let flat = read_f64(&path).expect("read continuous dataset");
        let cols = deinterleave_f64(&flat, measure.n_cols());
        let a = |i: usize| Array1::from(cols[i].clone());
        let kt = approach.kernel_name().to_string();
        match measure {
            Measure::Entropy => match approach {
                Approach::Ksg => Entropy::new_kl_1d(a(0), K, NOISE_LEVEL).global_value(),
                _ => Entropy::new_kernel_with_type(a(0), kt, BANDWIDTH).global_value(),
            },
            Measure::Mi => match approach {
                Approach::Ksg => {
                    MutualInformation::new_ksg(&[a(0), a(1)], K, NOISE_LEVEL).global_value()
                }
                _ => MutualInformation::new_kernel_with_type(&[a(0), a(1)], kt, BANDWIDTH)
                    .global_value(),
            },
            Measure::Cmi => match approach {
                Approach::Ksg => {
                    MutualInformation::new_cmi_ksg(&[a(0), a(1)], &a(2), K, NOISE_LEVEL)
                        .global_value()
                }
                _ => {
                    MutualInformation::new_cmi_kernel_with_type(&[a(0), a(1)], &a(2), kt, BANDWIDTH)
                        .global_value()
                }
            },
            Measure::Te => match approach {
                Approach::Ksg => {
                    TransferEntropy::new_ksg(&a(0), &a(1), 1, 1, 1, K, NOISE_LEVEL).global_value()
                }
                _ => TransferEntropy::new_kernel_with_type(&a(0), &a(1), 1, 1, 1, kt, BANDWIDTH)
                    .global_value(),
            },
            Measure::Cte => match approach {
                Approach::Ksg => {
                    TransferEntropy::new_cte_ksg(&a(0), &a(1), &a(2), 1, 1, 1, 1, K, NOISE_LEVEL)
                        .global_value()
                }
                _ => TransferEntropy::new_cte_kernel_with_type(
                    &a(0),
                    &a(1),
                    &a(2),
                    1,
                    1,
                    1,
                    1,
                    kt,
                    BANDWIDTH,
                )
                .global_value(),
            },
        }
    }
}

fn stats(times: &[f64]) -> Value {
    let n = times.len();
    let mean = times.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
    } else {
        0.0
    };
    let stddev = var.sqrt();
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let half = 1.96 * stddev / (n as f64).sqrt();
    json!({
        "mean": mean,
        "stddev": stddev,
        "min": sorted[0],
        "max": sorted[n - 1],
        "median": median,
        "samples": n,
        "ci_lower": mean - half,
        "ci_upper": mean + half,
    })
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn main() {
    let dir = data_dir();
    let out = std::env::var("BENCH_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.join("cross_package.json"));

    let short = std::env::var("BENCH_SHORT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let warmup = env_usize("BENCH_WARMUP", if short { 1 } else { 3 });
    let iterations = env_usize("BENCH_ITERATIONS", if short { 3 } else { 10 });

    let sizes = sizes();
    let measures = [
        Measure::Entropy,
        Measure::Mi,
        Measure::Cmi,
        Measure::Te,
        Measure::Cte,
    ];
    let approaches = [
        Approach::Discrete,
        Approach::Ksg,
        Approach::KernelBox,
        Approach::KernelGaussian,
    ];

    let mut benchmarks = Vec::new();
    for measure in measures {
        for approach in approaches {
            for &n in &sizes {
                let mut times = Vec::with_capacity(SEEDS.len() * iterations);
                let mut value = 0.0f64;
                for &seed in &SEEDS {
                    for _ in 0..warmup {
                        std::hint::black_box(run_case(measure, approach, n, seed, &dir));
                    }
                    for _ in 0..iterations {
                        let t0 = Instant::now();
                        let v = run_case(measure, approach, n, seed, &dir);
                        times.push(t0.elapsed().as_secs_f64());
                        value = v;
                    }
                }
                let st = stats(&times);
                println!(
                    "  {:>7} {:<16} n={:<6} {:>9.3} ms",
                    measure.as_str(),
                    approach.as_str(),
                    n,
                    st["mean"].as_f64().unwrap() * 1e3
                );

                let id = format!(
                    "{}/{}/n{}/{}",
                    measure.as_str(),
                    approach.as_str(),
                    n,
                    "infomeasure-rs"
                );
                benchmarks.push(json!({
                    "id": id,
                    "package": "infomeasure-rs",
                    "language": "rust",
                    "measure": measure.as_str(),
                    "approach": approach.as_str(),
                    "function": function_name(measure, approach),
                    "params": {
                        "n": n,
                        "k": K,
                        "bandwidth": if approach.is_kernel() { json!(BANDWIDTH) } else { Value::Null },
                        "order": Value::Null,
                        "delay": LAG,
                        "alpha": Value::Null,
                        "q": Value::Null,
                        "dims": 1,
                        "method": if approach.is_discrete() { json!("mle") } else { Value::Null },
                        "kernel_type": if approach.is_kernel() { json!(approach.kernel_name()) } else { Value::Null },
                    },
                    "statistics": st,
                    "value": value,
                }));
            }
        }
    }

    let hardware = detect_hardware();
    let output = json!({
        "meta": {
            "schema": 2,
            "generated": epoch_secs(),
            "run_id": format!("cross_package_{}", epoch_secs()),
            "hardware": {
                "cpu": hardware.cpu_model,
                "cores": hardware.cpu_cores,
                "memory_gb": hardware.memory_gb,
                "os": hardware.os,
                "gpu": Value::Null,
            },
            "runtime": { "threads": 1, "warmup": warmup, "iterations": iterations, "short": short },
            "seeds": SEEDS,
            "packages": [{
                "id": "infomeasure-rs",
                "language": "rust",
                "version": env!("CARGO_PKG_VERSION"),
                "commit": option_env!("GIT_COMMIT").unwrap_or(""),
                "released": option_env!("GIT_RELEASED").unwrap_or(""),
            }],
        },
        "benchmarks": benchmarks,
    });

    std::fs::write(&out, serde_json::to_string_pretty(&output).unwrap()).expect("write output");
    println!(
        "wrote {} entries to {}",
        output["benchmarks"].as_array().unwrap().len(),
        out.display()
    );
}

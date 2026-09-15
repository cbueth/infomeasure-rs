// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Alphabet-scaling collector for **infomeasure-rs**.
//!
//! Times the discrete MLE estimators across state counts at the sizes in
//! `detailed_grid.json`'s `alphabet` block, skipping larger N once a cell's mean
//! exceeds the configured budget. Output is a separate fragment family
//! (`<data>/results/infomeasure-rs_alphabet.json`) so the cross/detailed
//! fragments stay untouched.
//!
//! Run with: `cargo bench --bench collect_alphabet`
//!
//! Env:
//!   BENCH_DATA_DIR   dataset dir (default `target/bench-data`)
//!   BENCH_OUT        output JSON (default `<data>/results/infomeasure-rs_alphabet.json`)
//!   BENCH_RESUME=1   keep an existing fragment when the fingerprint matches
//!   BENCH_SHORT=1    warm-up 1 + up to 3 iterations

#![allow(unused_imports)]

use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

mod utils;

use utils::datasets::*;
use utils::grid::AlphabetConfig;
use utils::hardware::{HardwareInfo, detect_hardware};

const PKG: &str = "infomeasure-rs";
const LANGUAGE: &str = "rust";

struct Rounds {
    warmup_max: usize,
    min_iters: usize,
    max_iters: usize,
    iter_budget: f64,
}

fn rounds() -> Rounds {
    if std::env::var("BENCH_SHORT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return Rounds {
            warmup_max: 1,
            min_iters: 1,
            max_iters: 3,
            iter_budget: 0.0,
        };
    }
    Rounds {
        warmup_max: env_usize("BENCH_WARMUP_MAX", 3),
        min_iters: env_usize("BENCH_MIN_ITERS", 3),
        max_iters: env_usize("BENCH_MAX_ITERS", 10),
        iter_budget: env_f64("BENCH_ITER_BUDGET_S", 1.5),
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Sizes for the alphabet sweep. `BENCH_ALPHABET_SIZES`, else the generic
/// `BENCH_SIZES`, else the grid's alphabet sizes.
fn alphabet_sizes(ab: &AlphabetConfig) -> Vec<usize> {
    let raw = std::env::var("BENCH_ALPHABET_SIZES")
        .ok()
        .or_else(|| std::env::var("BENCH_SIZES").ok());
    if let Some(raw) = raw {
        let sizes: Vec<usize> = raw
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        if !sizes.is_empty() {
            return sizes;
        }
    }
    ab.sizes.clone()
}

fn load_cols(measure: &str, states: i32, seed: u64, n: usize, dir: &Path) -> Vec<Vec<i32>> {
    let id = alphabet_dataset_id(measure, states, seed, n);
    let flat = read_i32(&dataset_path(dir, &id)).unwrap_or_else(|e| panic!("read {id}: {e}"));
    deinterleave_i32(&flat, n_cols(measure))
}

/// Timed region: build + evaluate the discrete MLE estimator.
fn run(measure: &str, states: usize, cols: &[Vec<i32>]) -> f64 {
    match measure {
        "entropy" => {
            Entropy::new_discrete_from_slice_with_alphabet(&cols[0], states).global_value()
        }
        "mi" => MutualInformation::mi_discrete_mle(&[&cols[0], &cols[1]])
            .with_alphabet(states)
            .global_only()
            .global_value(),
        "cmi" => MutualInformation::cmi_discrete_mle(&[&cols[0], &cols[1]], &cols[2])
            .with_alphabet(states)
            .global_only()
            .global_value(),
        "te" => TransferEntropy::te_discrete_mle(&cols[0], &cols[1], 1, 1, 1)
            .with_alphabet(states)
            .global_only()
            .global_value(),
        "cte" => TransferEntropy::cte_discrete_mle(&cols[0], &cols[1], &cols[2], 1, 1, 1, 1)
            .with_alphabet(states)
            .global_only()
            .global_value(),
        other => unreachable!("unknown alphabet measure {other}"),
    }
}

fn function_name(measure: &str) -> &'static str {
    match measure {
        "entropy" => "Entropy::new_discrete_from_slice",
        "mi" => "MutualInformation::mi_discrete_mle",
        "cmi" => "MutualInformation::cmi_discrete_mle",
        "te" => "TransferEntropy::te_discrete_mle",
        "cte" => "TransferEntropy::cte_discrete_mle",
        _ => "?",
    }
}

fn stats(times: &[f64]) -> Value {
    let n = times.len();
    let mean = times.iter().sum::<f64>() / n as f64;
    let sd = if n > 1 {
        (times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt()
    } else {
        0.0
    };
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    };
    let half = 1.96 * sd / (n as f64).sqrt();
    json!({
        "mean": mean, "stddev": sd, "min": sorted[0], "max": sorted[n - 1],
        "median": median, "samples": n, "ci_lower": mean - half, "ci_upper": mean + half,
    })
}

fn fingerprint(hw: &HardwareInfo) -> String {
    use std::hash::{Hash, Hasher};

    let mut h = std::collections::hash_map::DefaultHasher::new();
    env!("CARGO_PKG_VERSION").hash(&mut h);
    option_env!("GIT_COMMIT").unwrap_or("").hash(&mut h);
    std::env::var("BENCH_COMMIT")
        .unwrap_or_default()
        .hash(&mut h);
    DATA_VERSION.hash(&mut h);
    hw.cpu_model.hash(&mut h);
    utils::grid::source_hash().hash(&mut h);
    let r = rounds();
    r.warmup_max.hash(&mut h);
    r.min_iters.hash(&mut h);
    r.max_iters.hash(&mut h);
    r.iter_budget.to_bits().hash(&mut h);
    format!("{:016x}", h.finish())
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn data_dir() -> PathBuf {
    std::env::var("BENCH_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/bench-data"))
}

fn main() {
    let dir = data_dir();
    let out = std::env::var("BENCH_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dir.join("results").join("infomeasure-rs_alphabet.json"));

    let ab: AlphabetConfig = utils::grid::alphabet();
    let budget = ab.budget_s;
    let sizes = alphabet_sizes(&ab);
    let hw = detect_hardware();
    let run_id = format!("infomeasure-rs_alphabet_{}", epoch_secs());
    let fp = fingerprint(&hw);
    let r = rounds();

    let mut benchmarks: Vec<Value> = Vec::new();
    let mut done: HashSet<String> = HashSet::new();
    if std::env::var("BENCH_RESUME")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
        && let Ok(text) = std::fs::read_to_string(&out)
        && let Ok(obj) = serde_json::from_str::<Value>(&text)
    {
        let stored = obj
            .get("meta")
            .and_then(|m| m.get("fingerprint"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if stored == fp {
            if let Some(arr) = obj.get("benchmarks").and_then(Value::as_array) {
                for b in arr {
                    if let Some(id) = b.get("id").and_then(Value::as_str) {
                        done.insert(id.to_string());
                    }
                    benchmarks.push(b.clone());
                }
            }
            println!("resume: {} existing entries kept", benchmarks.len());
        } else {
            println!("resume: fingerprint changed; starting fresh");
        }
    }

    let mut coverage: Vec<Value> = Vec::new();
    for measure in ALPHABET_MEASURES {
        let states_list = ab.states_for(measure);
        if states_list.is_empty() {
            continue;
        }
        coverage.push(json!([measure, "discrete"]));
        for states in states_list {
            let mut stopped = false;
            for &n in &sizes {
                if stopped {
                    break;
                }
                let id = format!("{measure}/discrete/mle/b{states}/n{n}/{PKG}");
                if done.contains(&id) {
                    continue;
                }
                let mut times = Vec::new();
                let mut value = 0.0_f64;
                for &seed in &SEEDS {
                    let cols = load_cols(measure, states as i32, seed, n, &dir);
                    let w0 = Instant::now();
                    let mut w = 0;
                    loop {
                        std::hint::black_box(run(measure, states, &cols));
                        w += 1;
                        if w >= r.warmup_max
                            || (r.iter_budget > 0.0 && w0.elapsed().as_secs_f64() >= 0.4)
                        {
                            break;
                        }
                    }
                    let t0 = Instant::now();
                    let mut it = 0;
                    loop {
                        let s = Instant::now();
                        value = run(measure, states, &cols);
                        times.push(s.elapsed().as_secs_f64());
                        it += 1;
                        if it >= r.max_iters
                            || (it >= r.min_iters
                                && (r.iter_budget <= 0.0
                                    || t0.elapsed().as_secs_f64() >= r.iter_budget))
                        {
                            break;
                        }
                    }
                }
                let st = stats(&times);
                let cell_mean = st["mean"].as_f64().unwrap_or(0.0);
                let delay = if measure == "te" || measure == "cte" {
                    json!(1)
                } else {
                    Value::Null
                };
                benchmarks.push(json!({
                    "id": id, "package": PKG, "language": LANGUAGE,
                    "measure": measure, "approach": "discrete",
                    "function": function_name(measure),
                    "params": {
                        "n": n, "states": states, "method": "mle", "delay": delay,
                        "k": Value::Null, "bandwidth": Value::Null, "order": Value::Null,
                        "alpha": Value::Null, "q": Value::Null, "dims": 1,
                        "kernel_type": Value::Null,
                    },
                    "representative": true,
                    "statistics": st,
                    "value": value,
                }));
                done.insert(id);
                println!(
                    "  {measure:8} b{states:<4} n={n:<6} {:.4} ms",
                    cell_mean * 1e3
                );
                if cell_mean > budget {
                    println!("  -> b{states} n={n} exceeded {budget}s; skipping larger N");
                    stopped = true;
                }
            }
        }
        // Flush per measure so a cancelled run keeps progress.
        write_fragment(&out, &benchmarks, &hw, &run_id, &fp, &coverage);
    }

    write_fragment(&out, &benchmarks, &hw, &run_id, &fp, &coverage);
    println!("wrote {} entries to {}", benchmarks.len(), out.display());
}

fn write_fragment(
    out: &Path,
    benchmarks: &[Value],
    hw: &HardwareInfo,
    run_id: &str,
    fp: &str,
    coverage: &[Value],
) {
    if let Some(parent) = out.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let output = json!({
        "meta": {
            "schema": 2,
            "generated": epoch_secs(),
            "run_id": run_id,
            "fingerprint": fp,
            "hardware": {
                "cpu": hw.cpu_model.clone(), "cores": hw.cpu_cores,
                "memory_gb": hw.memory_gb, "os": hw.os.clone(), "gpu": Value::Null,
            },
            "runtime": { "threads": 1, "adaptive": true, "family": "alphabet" },
            "seeds": SEEDS,
            "packages": [{ "id": PKG, "language": LANGUAGE, "version": env!("CARGO_PKG_VERSION") }],
            "coverage": coverage,
        },
        "benchmarks": benchmarks,
    });
    std::fs::write(out, serde_json::to_string_pretty(&output).unwrap()).expect("write output");
}

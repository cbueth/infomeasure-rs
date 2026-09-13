// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fair runtime collector for the **infomeasure-rs** package.
//!
//! Run with: `cargo bench --bench collect_cross_package`
//!
//! Times **only the estimator call** on pre-generated, identical datasets
//! (`target/bench-data/`). Warm-up calls are discarded; adaptive rounds bound
//! wall time. The full grid from `benches/detailed_grid.json` is collected:
//! representative variants (flagged `representative: true`) use all seeds and
//! the cross-package size set, detailed-only variants use the first seed and a
//! tighter budget. Output is a schema-v2 fragment.
//!
//! Env:
//!   BENCH_DATA_DIR     dataset dir (default `target/bench-data`)
//!   BENCH_OUT          output JSON (default `<data>/results/infomeasure-rs.json`)
//!   BENCH_SHORT=1      warm-up 1 + up to 3 iterations (local iteration)
//!   BENCH_WARMUP / BENCH_ITERATIONS   explicit overrides

#![allow(unused_imports)]

use infomeasure::estimators::approaches::discrete::bayes::AlphaParam;
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::Instant;

mod utils;

use utils::datasets::*;
use utils::grid::{Grid, Variant};
use utils::hardware::detect_hardware;

/// Pre-loaded dataset columns (file I/O happens outside the timed region).
enum Loaded {
    I32(Vec<Vec<i32>>),
    F64(Vec<Vec<f64>>),
}

fn is_discrete(approach: &str) -> bool {
    approach == "discrete"
}

fn load_cols(measure: &str, approach: &str, n: usize, seed: u64, dir: &Path) -> Loaded {
    let kind = if is_discrete(approach) {
        "discrete"
    } else {
        "continuous"
    };
    let id = dataset_id(measure, kind, seed, n);
    let path = dataset_path(dir, &id);
    if is_discrete(approach) {
        let flat = read_i32(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        Loaded::I32(deinterleave_i32(&flat, n_cols(measure)))
    } else {
        let flat = read_f64(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        Loaded::F64(deinterleave_f64(&flat, n_cols(measure)))
    }
}

/// Construct + compute an estimator on pre-loaded data. This is the timed region.
fn run_variant(v: &Variant, data: &Loaded) -> f64 {
    let measure = v.measure.as_str();
    let method = v.method.as_deref().unwrap_or("mle");

    if is_discrete(v.approach.as_str()) {
        let cols = match data {
            Loaded::I32(c) => c,
            Loaded::F64(_) => unreachable!("discrete approach with continuous data"),
        };
        let a = |i: usize| Array1::from(cols[i].clone());
        return match measure {
            "entropy" => {
                let x = a(0);
                match method {
                    "miller_madow" => Entropy::new_miller_madow(x).global_value(),
                    "shrink" => Entropy::new_shrink(x).global_value(),
                    "grassberger" => Entropy::new_grassberger(x).global_value(),
                    "zhang" => Entropy::new_zhang(x).global_value(),
                    "bayes" => Entropy::new_bayes(x, AlphaParam::Laplace, None).global_value(),
                    "bonachela" => Entropy::new_bonachela(x).global_value(),
                    "chao_shen" => Entropy::new_chao_shen(x).global_value(),
                    "chao_wang_jost" => Entropy::new_chao_wang_jost(x).global_value(),
                    "ansb" => Entropy::new_ansb(x, None).global_value(),
                    "nsb" => Entropy::new_nsb(x, None).global_value(),
                    _ => Entropy::new_discrete(x).global_value(),
                }
            }
            "mi" => {
                let s = [a(0), a(1)];
                match method {
                    "miller_madow" => {
                        MutualInformation::new_discrete_miller_madow(&s).global_value()
                    }
                    "shrink" => MutualInformation::new_discrete_shrink(&s).global_value(),
                    "chao_shen" => MutualInformation::new_discrete_chao_shen(&s).global_value(),
                    "chao_wang_jost" => {
                        MutualInformation::new_discrete_chao_wang_jost(&s).global_value()
                    }
                    "nsb" => MutualInformation::new_discrete_nsb(&s).global_value(),
                    "ansb" => MutualInformation::new_discrete_ansb(&s).global_value(),
                    "bonachela" => MutualInformation::new_discrete_bonachela(&s).global_value(),
                    "grassberger" => MutualInformation::new_discrete_grassberger(&s).global_value(),
                    "zhang" => MutualInformation::new_discrete_zhang(&s).global_value(),
                    "bayes" => MutualInformation::new_discrete_bayes(&s).global_value(),
                    _ => MutualInformation::new_discrete_mle(&s).global_value(),
                }
            }
            "cmi" => {
                let z = a(2);
                let s = [a(0), a(1)];
                match method {
                    "miller_madow" => {
                        MutualInformation::new_cmi_discrete_miller_madow(&s, &z).global_value()
                    }
                    "shrink" => MutualInformation::new_cmi_discrete_shrink(&s, &z).global_value(),
                    "chao_shen" => {
                        MutualInformation::new_cmi_discrete_chao_shen(&s, &z).global_value()
                    }
                    "chao_wang_jost" => {
                        MutualInformation::new_cmi_discrete_chao_wang_jost(&s, &z).global_value()
                    }
                    "nsb" => MutualInformation::new_cmi_discrete_nsb(&s, &z).global_value(),
                    "ansb" => MutualInformation::new_cmi_discrete_ansb(&s, &z).global_value(),
                    "bonachela" => {
                        MutualInformation::new_cmi_discrete_bonachela(&s, &z).global_value()
                    }
                    "grassberger" => {
                        MutualInformation::new_cmi_discrete_grassberger(&s, &z).global_value()
                    }
                    "zhang" => MutualInformation::new_cmi_discrete_zhang(&s, &z).global_value(),
                    "bayes" => MutualInformation::new_cmi_discrete_bayes(&s, &z).global_value(),
                    _ => MutualInformation::new_cmi_discrete_mle(&s, &z).global_value(),
                }
            }
            "te" => {
                let (x, y) = (a(0), a(1));
                match method {
                    "miller_madow" => {
                        TransferEntropy::new_discrete_miller_madow(&x, &y, 1, 1, 1).global_value()
                    }
                    "shrink" => {
                        TransferEntropy::new_discrete_shrink(&x, &y, 1, 1, 1).global_value()
                    }
                    "chao_shen" => {
                        TransferEntropy::new_discrete_chao_shen(&x, &y, 1, 1, 1).global_value()
                    }
                    "chao_wang_jost" => {
                        TransferEntropy::new_discrete_chao_wang_jost(&x, &y, 1, 1, 1).global_value()
                    }
                    "nsb" => TransferEntropy::new_discrete_nsb(&x, &y, 1, 1, 1).global_value(),
                    "ansb" => TransferEntropy::new_discrete_ansb(&x, &y, 1, 1, 1).global_value(),
                    "bonachela" => {
                        TransferEntropy::new_discrete_bonachela(&x, &y, 1, 1, 1).global_value()
                    }
                    "grassberger" => {
                        TransferEntropy::new_discrete_grassberger(&x, &y, 1, 1, 1).global_value()
                    }
                    "zhang" => TransferEntropy::new_discrete_zhang(&x, &y, 1, 1, 1).global_value(),
                    "bayes" => TransferEntropy::new_discrete_bayes(&x, &y, 1, 1, 1).global_value(),
                    _ => TransferEntropy::new_discrete_mle(&x, &y, 1, 1, 1).global_value(),
                }
            }
            "cte" => {
                let (x, y, z) = (a(0), a(1), a(2));
                match method {
                    "miller_madow" => {
                        TransferEntropy::new_cte_discrete_miller_madow(&x, &y, &z, 1, 1, 1, 1)
                            .global_value()
                    }
                    "shrink" => TransferEntropy::new_cte_discrete_shrink(&x, &y, &z, 1, 1, 1, 1)
                        .global_value(),
                    "chao_shen" => {
                        TransferEntropy::new_cte_discrete_chao_shen(&x, &y, &z, 1, 1, 1, 1)
                            .global_value()
                    }
                    "chao_wang_jost" => {
                        TransferEntropy::new_cte_discrete_chao_wang_jost(&x, &y, &z, 1, 1, 1, 1)
                            .global_value()
                    }
                    "nsb" => {
                        TransferEntropy::new_cte_discrete_nsb(&x, &y, &z, 1, 1, 1, 1).global_value()
                    }
                    "ansb" => TransferEntropy::new_cte_discrete_ansb(&x, &y, &z, 1, 1, 1, 1)
                        .global_value(),
                    "bonachela" => {
                        TransferEntropy::new_cte_discrete_bonachela(&x, &y, &z, 1, 1, 1, 1)
                            .global_value()
                    }
                    "grassberger" => {
                        TransferEntropy::new_cte_discrete_grassberger(&x, &y, &z, 1, 1, 1, 1)
                            .global_value()
                    }
                    "zhang" => TransferEntropy::new_cte_discrete_zhang(&x, &y, &z, 1, 1, 1, 1)
                        .global_value(),
                    "bayes" => TransferEntropy::new_cte_discrete_bayes(&x, &y, &z, 1, 1, 1, 1)
                        .global_value(),
                    _ => {
                        TransferEntropy::new_cte_discrete_mle(&x, &y, &z, 1, 1, 1, 1).global_value()
                    }
                }
            }
            other => unreachable!("unknown discrete measure {other}"),
        };
    }

    let cols = match data {
        Loaded::F64(c) => c,
        Loaded::I32(_) => unreachable!("continuous approach with discrete data"),
    };
    let a = |i: usize| Array1::from(cols[i].clone());
    let kt = v.kernel.clone().unwrap_or_else(|| "box".into());
    let k = v.k.unwrap_or(K);
    let bw = v.bandwidth.unwrap_or(BANDWIDTH);
    let order = v.order.unwrap_or(2);
    let alpha = v.alpha.unwrap_or(1.0);
    let q = v.q.unwrap_or(1.0);

    match measure {
        "entropy" => {
            let x = a(0);
            match v.approach.as_str() {
                "ordinal" => Entropy::new_ordinal(x, order).global_value(),
                "kernel" => Entropy::new_kernel_with_type(x, kt, bw).global_value(),
                "kl_cheb" => Entropy::new_kl_1d(x, k, NOISE_LEVEL)
                    .with_chebyshev(true)
                    .global_value(),
                "kl_k" => Entropy::new_kl_1d(x, k, NOISE_LEVEL).global_value(),
                "renyi" => Entropy::new_renyi_1d(x, k, alpha, NOISE_LEVEL).global_value(),
                "tsallis" => Entropy::new_tsallis_1d(x, k, q, NOISE_LEVEL).global_value(),
                _ => Entropy::new_kl_1d(x, k, NOISE_LEVEL).global_value(),
            }
        }
        "mi" => {
            let (x, y) = (a(0), a(1));
            match v.approach.as_str() {
                "ordinal" => {
                    MutualInformation::new_ordinal(&[x, y], order, 1, false).global_value()
                }
                "kernel" => MutualInformation::new_kernel_with_type(&[x, y], kt, bw).global_value(),
                "kl" => MutualInformation::new_kl(&[x, y], k, NOISE_LEVEL).global_value(),
                "renyi" => {
                    MutualInformation::new_renyi(&[x, y], k, alpha, NOISE_LEVEL).global_value()
                }
                "tsallis" => {
                    MutualInformation::new_tsallis(&[x, y], k, q, NOISE_LEVEL).global_value()
                }
                _ => MutualInformation::new_ksg(&[x, y], k, NOISE_LEVEL).global_value(),
            }
        }
        "cmi" => {
            let (x, y, z) = (a(0), a(1), a(2));
            match v.approach.as_str() {
                "ordinal" => {
                    MutualInformation::new_cmi_ordinal(&[x, y], &z, order, 1, false).global_value()
                }
                "kernel" => {
                    MutualInformation::new_cmi_kernel_with_type(&[x, y], &z, kt, bw).global_value()
                }
                "kl" => MutualInformation::new_cmi_kl(&[x, y], &z, k, NOISE_LEVEL).global_value(),
                "renyi" => MutualInformation::new_cmi_renyi(&[x, y], &z, k, alpha, NOISE_LEVEL)
                    .global_value(),
                "tsallis" => MutualInformation::new_cmi_tsallis(&[x, y], &z, k, q, NOISE_LEVEL)
                    .global_value(),
                _ => MutualInformation::new_cmi_ksg(&[x, y], &z, k, NOISE_LEVEL).global_value(),
            }
        }
        "te" => {
            let (x, y) = (a(0), a(1));
            match v.approach.as_str() {
                "ordinal" => {
                    TransferEntropy::new_ordinal(&x, &y, order, 1, 1, 1, false).global_value()
                }
                "kernel" => {
                    TransferEntropy::new_kernel_with_type(&x, &y, 1, 1, 1, kt, bw).global_value()
                }
                "renyi" => TransferEntropy::new_renyi(&x, &y, k, alpha, NOISE_LEVEL).global_value(),
                "tsallis" => TransferEntropy::new_tsallis(&x, &y, k, q, NOISE_LEVEL).global_value(),
                _ => TransferEntropy::new_ksg(&x, &y, 1, 1, 1, k, NOISE_LEVEL).global_value(),
            }
        }
        "cte" => {
            let (x, y, z) = (a(0), a(1), a(2));
            match v.approach.as_str() {
                "ordinal" => TransferEntropy::new_cte_ordinal(&x, &y, &z, order, 1, 1, 1, 1, false)
                    .global_value(),
                "kernel" => {
                    TransferEntropy::new_cte_kernel_with_type(&x, &y, &z, 1, 1, 1, 1, kt, bw)
                        .global_value()
                }
                "kl" => TransferEntropy::new_cte_kl(&x, &y, &z, k, NOISE_LEVEL).global_value(),
                "renyi" => {
                    TransferEntropy::new_cte_renyi(&x, &y, &z, k, alpha, NOISE_LEVEL).global_value()
                }
                "tsallis" => {
                    TransferEntropy::new_cte_tsallis(&x, &y, &z, k, q, NOISE_LEVEL).global_value()
                }
                _ => TransferEntropy::new_cte_ksg(&x, &y, &z, 1, 1, 1, 1, k, NOISE_LEVEL)
                    .global_value(),
            }
        }
        other => unreachable!("unknown continuous measure {other}"),
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

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Round configuration. Representative variants use the standard adaptive
/// budget; detailed-only variants use a tight budget so the full grid stays
/// inside the per-package wall-time target.
struct Rounds {
    warmup_max: usize,
    warmup_budget: f64,
    min_iters: usize,
    max_iters: usize,
    iter_budget: f64,
}

fn rounds_config(detail_only: bool) -> Rounds {
    let short = std::env::var("BENCH_SHORT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if short {
        return Rounds {
            warmup_max: 1,
            warmup_budget: 0.0,
            min_iters: 1,
            max_iters: 2,
            iter_budget: 0.0,
        };
    }
    if detail_only {
        // Profile A: guarantees >=5 samples (10 for fast calls) so the
        // reported stddev/CI are meaningful. Fall back to profile B by setting
        // BENCH_DETAIL_MIN_ITERS=3 if the wall-time budget is tight.
        Rounds {
            warmup_max: env_usize("BENCH_DETAIL_WARMUP_MAX", 1),
            warmup_budget: env_f64("BENCH_DETAIL_WARMUP_BUDGET_S", 0.1),
            min_iters: env_usize("BENCH_DETAIL_MIN_ITERS", 5),
            max_iters: env_usize("BENCH_DETAIL_MAX_ITERS", 10),
            iter_budget: env_f64("BENCH_DETAIL_ITER_BUDGET_S", 0.3),
        }
    } else {
        Rounds {
            warmup_max: env_usize("BENCH_WARMUP_MAX", 3),
            warmup_budget: env_f64("BENCH_WARMUP_BUDGET_S", 0.4),
            min_iters: env_usize("BENCH_MIN_ITERS", 3),
            max_iters: env_usize("BENCH_MAX_ITERS", 10),
            iter_budget: env_f64("BENCH_ITER_BUDGET_S", 1.5),
        }
    }
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
        .unwrap_or_else(|_| dir.join("results").join("infomeasure-rs.json"));

    let grid: Grid = utils::grid::load("rust");
    // Optional size override for fast local iteration (also forwarded by CI).
    let (cross_sizes, detailed_sizes) = match std::env::var("BENCH_SIZES") {
        Ok(s) => {
            let overridden: Vec<usize> =
                s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            if overridden.is_empty() {
                (grid.cross_sizes.clone(), grid.detailed_sizes.clone())
            } else {
                (overridden.clone(), overridden)
            }
        }
        Err(_) => (grid.cross_sizes.clone(), grid.detailed_sizes.clone()),
    };
    let mut benchmarks = Vec::new();
    let mut cross_rounds = None;
    let mut detail_rounds = None;

    for v in &grid.variants {
        for &n in &detailed_sizes {
            // Representative entries: a representative variant at a cross size.
            // They use all seeds + the standard adaptive budget; every other
            // entry uses one seed + the tight detailed budget.
            let is_rep = v.cross && cross_sizes.contains(&n);
            let seeds: &[u64] = if is_rep { &SEEDS } else { &SEEDS[..1] };
            let rounds = if is_rep {
                cross_rounds.get_or_insert_with(|| rounds_config(false))
            } else {
                detail_rounds.get_or_insert_with(|| rounds_config(true))
            };

            let mut times = Vec::new();
            let mut value = 0.0f64;
            for &seed in seeds {
                // File I/O is deliberately outside the timed region.
                let data = load_cols(&v.measure, &v.approach, n, seed, &dir);
                let w0 = Instant::now();
                let mut w = 0;
                loop {
                    std::hint::black_box(run_variant(v, &data));
                    w += 1;
                    if w >= rounds.warmup_max {
                        break;
                    }
                    if rounds.warmup_budget > 0.0
                        && w0.elapsed().as_secs_f64() >= rounds.warmup_budget
                    {
                        break;
                    }
                }
                let t0 = Instant::now();
                let mut it = 0;
                loop {
                    let s = Instant::now();
                    value = run_variant(v, &data);
                    times.push(s.elapsed().as_secs_f64());
                    it += 1;
                    if it >= rounds.max_iters {
                        break;
                    }
                    if it >= rounds.min_iters
                        && (rounds.iter_budget <= 0.0
                            || t0.elapsed().as_secs_f64() >= rounds.iter_budget)
                    {
                        break;
                    }
                }
            }
            let st = stats(&times);
            println!(
                "  {:>7} {:<16} {:<22} n={:<6} {:>9.3} ms{}",
                v.measure,
                v.approach,
                v.slug(),
                n,
                st["mean"].as_f64().unwrap() * 1e3,
                if is_rep { "  [cross]" } else { "" }
            );

            let id = format!(
                "{}/{}/{}/n{}/{}",
                v.measure,
                v.approach,
                v.slug(),
                n,
                "infomeasure-rs"
            );
            benchmarks.push(json!({
                "id": id,
                "package": "infomeasure-rs",
                "language": "rust",
                "measure": v.measure,
                "approach": v.approach,
                "function": v.function(),
                "representative": is_rep,
                "params": v.params(n),
                "statistics": st,
                "value": value,
            }));
        }
    }

    let hardware = detect_hardware();
    let output = json!({
        "meta": {
            "schema": 2,
            "generated": epoch_secs(),
            "run_id": format!("infomeasure-rs_{}", epoch_secs()),
            "hardware": {
                "cpu": hardware.cpu_model,
                "cores": hardware.cpu_cores,
                "memory_gb": hardware.memory_gb,
                "os": hardware.os,
                "gpu": Value::Null,
            },
            "runtime": {
                "threads": 1,
                "adaptive": true,
                "representative": {
                    "warmup_max": rounds_config(false).warmup_max,
                    "warmup_budget_s": rounds_config(false).warmup_budget,
                    "min_iters": rounds_config(false).min_iters,
                    "max_iters": rounds_config(false).max_iters,
                    "iter_budget_s": rounds_config(false).iter_budget,
                },
                "detailed": {
                    "warmup_max": rounds_config(true).warmup_max,
                    "warmup_budget_s": rounds_config(true).warmup_budget,
                    "min_iters": rounds_config(true).min_iters,
                    "max_iters": rounds_config(true).max_iters,
                    "iter_budget_s": rounds_config(true).iter_budget,
                },
            },
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

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    std::fs::write(&out, serde_json::to_string_pretty(&output).unwrap()).expect("write output");
    println!(
        "wrote {} entries to {}",
        output["benchmarks"].as_array().unwrap().len(),
        out.display()
    );
}

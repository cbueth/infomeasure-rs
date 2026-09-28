// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Schema-v2 fragment writer, mirroring `scripts/bench_packages/_common.py`.

use std::path::PathBuf;

use serde_json::{json, Value};

use crate::datasets::data_dir;
use crate::timing::Rounds;

pub fn stats(times: &[f64]) -> Value {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let var = if times.len() > 1 {
        times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let stddev = var.sqrt();
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    };
    let half = if n > 0.0 {
        1.96 * stddev / n.sqrt()
    } else {
        0.0
    };
    json!({
        "mean": mean,
        "stddev": stddev,
        "min": times.iter().cloned().fold(f64::INFINITY, f64::min),
        "max": times.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "median": median,
        "samples": times.len(),
        "ci_lower": mean - half,
        "ci_upper": mean + half,
    })
}

pub fn params(measure: &str, approach: &str, n: usize, k: Option<usize>) -> Value {
    json!({
        "n": n,
        "k": k,
        "bandwidth": null,
        "order": null,
        "delay": if measure == "te" || measure == "cte" { json!(1) } else { Value::Null },
        "alpha": null,
        "q": null,
        "dims": 1,
        "method": if approach == "discrete" { json!("mle") } else { Value::Null },
        "kernel_type": null,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn entry(
    package: &str,
    measure: &str,
    approach: &str,
    slug: &str,
    function: &str,
    n: usize,
    params: Value,
    st: Value,
    value: f64,
) -> Value {
    json!({
        "id": format!("{measure}/{approach}/{slug}/n{n}/{package}"),
        "package": package,
        "language": "rust",
        "measure": measure,
        "approach": approach,
        "function": function,
        "representative": true,
        "params": params,
        "statistics": st,
        "value": value,
        "notes": null,
    })
}

/// Write `<data>/results/<package>.json` (override with `BENCH_OUT`).
#[allow(clippy::too_many_arguments)]
pub fn write_fragment(
    package: &str,
    language: &str,
    version: &str,
    benchmarks: &[Value],
    seeds: &[u64],
    r: &Rounds,
    extra: Value,
    limitations: &str,
) {
    use std::collections::BTreeSet;

    let coverage: Vec<Vec<String>> = {
        let set: BTreeSet<(String, String)> = benchmarks
            .iter()
            .map(|b| {
                (
                    b["measure"].as_str().unwrap().to_string(),
                    b["approach"].as_str().unwrap().to_string(),
                )
            })
            .collect();
        set.into_iter().map(|(m, a)| vec![m, a]).collect()
    };

    let mut pkg = json!({
        "id": package,
        "language": language,
        "version": version,
        "limitations": limitations,
    });
    if let (Some(dst), Some(src)) = (pkg.as_object_mut(), extra.as_object()) {
        for (k, v) in src {
            dst.insert(k.clone(), v.clone());
        }
    }

    let output = json!({
        "meta": {
            "schema": 2,
            "run_id": format!("fragment_{}", epoch_secs()),
            "fingerprint": Value::Null,
            "hardware": Value::Null,
            "runtime": {
                "threads": 1,
                "adaptive": !r.short,
                "family": Value::Null,
                "warmup_max": r.warmup_max,
                "warmup_budget_s": r.warmup_budget,
                "min_iters": r.min_iters,
                "max_iters": r.max_iters,
                "iter_budget_s": r.iter_budget,
            },
            "seeds": seeds,
            "packages": [pkg],
            "coverage": coverage,
        },
        "benchmarks": benchmarks,
    });

    let out = std::env::var("BENCH_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir().join("results").join(format!("{package}.json")));
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    std::fs::write(&out, serde_json::to_string_pretty(&output).unwrap()).expect("write fragment");
    println!("wrote {} entries to {}", benchmarks.len(), out.display());
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

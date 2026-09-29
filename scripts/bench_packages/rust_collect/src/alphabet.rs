// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Alphabet-scaling family support for the third-party Rust collectors.
//!
//! Mirrors `benches/collect_alphabet.rs` (infomeasure-rs) and
//! `scripts/bench_packages/collect_alphabet.py`: discrete MLE timed across state
//! counts at the cross sizes, written as `<package>_alphabet.json` so the
//! viewer's alphabet mode picks it up.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde_json::{json, Value};

use crate::datasets::{data_dir, n_cols};
use crate::timing::Rounds;

/// Fixed dataset seeds, matching `benches/utils/datasets.rs`.
pub const SEEDS: [u64; 4] = [610418971, 2086847849, 627358495, 1501472984];

/// The alphabet grid from `benches/detailed_grid.json`.
pub struct Config {
    pub states: Vec<usize>,
    pub sizes: Vec<usize>,
    /// Largest state count attempted per measure (memory guard).
    pub caps: BTreeMap<String, usize>,
    /// Per-cell mean (seconds) above which larger N is skipped.
    pub budget_s: f64,
}

impl Config {
    /// States to collect for a measure, capped by the memory guard.
    pub fn states_for(&self, measure: &str) -> Vec<usize> {
        let cap = self.caps.get(measure).copied().unwrap_or(0);
        self.states.iter().copied().filter(|&s| s <= cap).collect()
    }

    /// Sizes for the sweep: `BENCH_ALPHABET_SIZES`, else `BENCH_SIZES`, else grid.
    pub fn sizes(&self) -> Vec<usize> {
        let raw = std::env::var("BENCH_ALPHABET_SIZES")
            .ok()
            .or_else(|| std::env::var("BENCH_SIZES").ok());
        if let Some(raw) = raw {
            let parsed: Vec<usize> = raw
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            if !parsed.is_empty() {
                return parsed;
            }
        }
        self.sizes.clone()
    }
}

pub fn config() -> Config {
    let path =
        std::env::var("BENCH_GRID").unwrap_or_else(|_| "benches/detailed_grid.json".to_string());
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let root: Value = serde_json::from_str(&text).expect("parse detailed_grid.json");
    let a = &root["alphabet"];
    let caps = a["caps"]
        .as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_u64().map(|x| (k.clone(), x as usize)))
                .collect()
        })
        .unwrap_or_default();
    Config {
        states: usizes(&a["states"]),
        sizes: usizes(&a["sizes"]),
        caps,
        budget_s: a["budget_s"].as_f64().unwrap_or(2.0),
    }
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|y| y as usize))
                .collect()
        })
        .unwrap_or_default()
}

fn read_bytes(name: &str) -> Vec<u8> {
    let path: PathBuf = data_dir().join(name);
    fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// One alphabet dataset as `i32` columns (`<measure>_discrete_b<states>_s<seed>_n<n>.bin`).
pub fn load_discrete(measure: &str, states: usize, seed: u64, n: usize) -> Vec<Vec<i32>> {
    let id = format!("{measure}_discrete_b{states}_s{seed}_n{n}");
    let flat = read_bytes(&format!("{id}.bin"));
    let cols = n_cols(measure);
    let len = flat.len() / 4 / cols;
    let mut out = vec![Vec::with_capacity(len); cols];
    for i in 0..len {
        for (c, col) in out.iter_mut().enumerate() {
            let off = (i * cols + c) * 4;
            col.push(i32::from_le_bytes([
                flat[off],
                flat[off + 1],
                flat[off + 2],
                flat[off + 3],
            ]));
        }
    }
    out
}

/// One alphabet-fragment benchmark entry.
#[allow(clippy::too_many_arguments)]
pub fn entry(
    package: &str,
    language: &str,
    measure: &str,
    states: usize,
    n: usize,
    function: &str,
    st: Value,
    value: f64,
) -> Value {
    json!({
        "id": format!("{measure}/discrete/mle/b{states}/n{n}/{package}"),
        "package": package,
        "language": language,
        "measure": measure,
        "approach": "discrete",
        "function": function,
        "params": {
            "n": n,
            "states": states,
            "method": "mle",
            "delay": Value::Null,
            "k": Value::Null,
            "bandwidth": Value::Null,
            "order": Value::Null,
            "alpha": Value::Null,
            "q": Value::Null,
            "dims": 1,
            "kernel_type": Value::Null,
        },
        "representative": true,
        "statistics": st,
        "value": value,
    })
}

/// Write `<data>/results/<package>_alphabet.json` (override with `BENCH_OUT`).
#[allow(clippy::too_many_arguments)]
pub fn write_fragment(
    package: &str,
    language: &str,
    version: &str,
    benchmarks: &[Value],
    coverage: &[Vec<String>],
    r: &Rounds,
    extra: Value,
    limitations: &str,
) {
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
            "run_id": format!("fragment_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)),
            "fingerprint": Value::Null,
            "hardware": Value::Null,
            "runtime": { "threads": 1, "adaptive": !r.short, "family": "alphabet" },
            "seeds": SEEDS,
            "packages": [pkg],
            "coverage": coverage,
        },
        "benchmarks": benchmarks,
    });
    let out = std::env::var("BENCH_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            data_dir()
                .join("results")
                .join(format!("{package}_alphabet.json"))
        });
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).expect("create output dir");
    }
    fs::write(&out, serde_json::to_string_pretty(&output).unwrap()).expect("write fragment");
    println!("wrote {} entries to {}", benchmarks.len(), out.display());
}

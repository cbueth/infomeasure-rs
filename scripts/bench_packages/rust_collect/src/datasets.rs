// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reader for the canonical cross-package datasets
//! (`benches/gen_datasets.rs`). Mirrors `scripts/bench_packages/_common.py`.

use std::fs;
use std::path::PathBuf;

/// Columns per measure, matching `_common.COLS`.
pub fn n_cols(measure: &str) -> usize {
    match measure {
        "entropy" => 1,
        "mi" | "te" => 2,
        "cmi" | "cte" => 3,
        other => panic!("unknown measure {other}"),
    }
}

pub fn data_dir() -> PathBuf {
    std::env::var("BENCH_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/bench-data"))
}

fn read_bytes(name: &str) -> Vec<u8> {
    let path = data_dir().join(name);
    fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

/// Sizes (cross set) and seeds from `manifest.json`, with `BENCH_SIZES` override.
pub fn sizes_and_seeds() -> (Vec<usize>, Vec<u64>) {
    let text = fs::read_to_string(data_dir().join("manifest.json")).expect("read manifest.json");
    let m: serde_json::Value = serde_json::from_str(&text).expect("parse manifest.json");
    let mut sizes: Vec<usize> = m["sizes"]
        .as_array()
        .expect("manifest sizes")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    if let Ok(raw) = std::env::var("BENCH_SIZES") {
        let parsed: Vec<usize> = raw
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        if !parsed.is_empty() {
            sizes = parsed;
        }
    }
    let seeds: Vec<u64> = m["seeds"]
        .as_array()
        .expect("manifest seeds")
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    (sizes, seeds)
}

fn deinterleave_i32(flat: &[u8], cols: usize) -> Vec<Vec<i32>> {
    let n = flat.len() / 4 / cols;
    let mut out = vec![Vec::with_capacity(n); cols];
    for i in 0..n {
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

fn deinterleave_f64(flat: &[u8], cols: usize) -> Vec<Vec<f64>> {
    let n = flat.len() / 8 / cols;
    let mut out = vec![Vec::with_capacity(n); cols];
    for i in 0..n {
        for (c, col) in out.iter_mut().enumerate() {
            let off = (i * cols + c) * 8;
            let mut b = [0u8; 8];
            b.copy_from_slice(&flat[off..off + 8]);
            col.push(f64::from_le_bytes(b));
        }
    }
    out
}

/// One discrete dataset as columns (`i32`).
pub fn load_discrete(measure: &str, seed: u64, n: usize) -> Vec<Vec<i32>> {
    let name = format!("{measure}_discrete_s{seed}_n{n}.bin");
    deinterleave_i32(&read_bytes(&name), n_cols(measure))
}

/// One continuous dataset as columns (`f64`).
pub fn load_continuous(measure: &str, seed: u64, n: usize) -> Vec<Vec<f64>> {
    let name = format!("{measure}_continuous_s{seed}_n{n}.bin");
    deinterleave_f64(&read_bytes(&name), n_cols(measure))
}

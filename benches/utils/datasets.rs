// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Shared canonical-dataset format for the cross-package benchmark.
//
// Layout under `target/bench-data/` (override with `BENCH_DATA_DIR`):
//   manifest.json          describes every dataset (consumed by Java/Python/R)
//   <dataset_id>.bin       raw little-endian array, row-major [n, cols]
//
// One dataset per (measure, kind, seed, size). `kind` is `discrete` (i32) or
// `continuous` (f64). Consumers read the identical bytes, so no per-language
// RNG is involved.

#![allow(dead_code)]

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

pub const DATA_VERSION: u32 = 1;

/// Fixed dataset seeds (randomly generated for this project, not hand-picked).
pub const SEEDS: [u64; 5] = [610418971, 2086847849, 627358495, 1501472984, 1400190726];

pub const NUM_STATES_DISCRETE: i32 = 10;
pub const NUM_STATES_TE: i32 = 5;
pub const K: usize = 4;
pub const BANDWIDTH: f64 = 0.5;
pub const NOISE_LEVEL: f64 = 1e-10;
pub const CORRELATION: f64 = 0.5;
pub const COUPLING: f64 = 0.5;
pub const LAG: usize = 1;

/// Representative input lengths (override with `BENCH_SIZES=…`).
pub fn sizes() -> Vec<usize> {
    parse_sizes("BENCH_SIZES").unwrap_or_else(|| vec![100, 400, 1600, 6400, 12500])
}

fn parse_sizes(env: &str) -> Option<Vec<usize>> {
    let parsed: Vec<usize> = std::env::var(env)
        .ok()?
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    (!parsed.is_empty()).then_some(parsed)
}

pub fn data_dir() -> PathBuf {
    std::env::var("BENCH_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/bench-data"))
}

pub fn dataset_id(measure: &str, kind: &str, seed: u64, n: usize) -> String {
    format!("{measure}_{kind}_s{seed}_n{n}")
}

pub fn dataset_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.bin"))
}

/// Number of columns a dataset has for a given measure.
pub fn n_cols(measure: &str) -> usize {
    match measure {
        "entropy" => 1,
        "mi" | "te" => 2,
        "cmi" | "cte" => 3,
        _ => panic!("unknown measure {measure}"),
    }
}

pub fn ensure_dir(dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)
}

pub fn write_i32(path: &Path, data: &[i32]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()
}

pub fn write_f64(path: &Path, data: &[f64]) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    w.flush()
}

pub fn read_i32(path: &Path) -> std::io::Result<Vec<i32>> {
    let mut buf = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut buf)?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn read_f64(path: &Path) -> std::io::Result<Vec<f64>> {
    let mut buf = Vec::new();
    BufReader::new(File::open(path)?).read_to_end(&mut buf)?;
    Ok(buf
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Row-major interleave of columns into one contiguous buffer.
pub fn interleave_i32(cols: &[Vec<i32>]) -> Vec<i32> {
    let n = cols[0].len();
    let mut out = Vec::with_capacity(n * cols.len());
    for i in 0..n {
        for c in cols {
            out.push(c[i]);
        }
    }
    out
}

pub fn interleave_f64(cols: &[Vec<f64>]) -> Vec<f64> {
    let n = cols[0].len();
    let mut out = Vec::with_capacity(n * cols.len());
    for i in 0..n {
        for c in cols {
            out.push(c[i]);
        }
    }
    out
}

/// Split a row-major buffer back into columns.
pub fn deinterleave_i32(flat: &[i32], cols: usize) -> Vec<Vec<i32>> {
    let n = flat.len() / cols;
    (0..cols)
        .map(|c| (0..n).map(|i| flat[i * cols + c]).collect())
        .collect()
}

pub fn deinterleave_f64(flat: &[f64], cols: usize) -> Vec<Vec<f64>> {
    let n = flat.len() / cols;
    (0..cols)
        .map(|c| (0..n).map(|i| flat[i * cols + c]).collect())
        .collect()
}

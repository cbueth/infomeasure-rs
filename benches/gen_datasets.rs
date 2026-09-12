// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generate the canonical cross-package datasets into `target/bench-data/`.
//!
//! Run with: `cargo bench --bench gen_datasets`
//!
//! Deterministic: fixed seeds (`utils::datasets::SEEDS`), one file per
//! (measure, kind, seed, size), plus a `manifest.json` that every language
//! reads. Regenerating never changes the bytes.

#![allow(unused_imports)]

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde_json::json;
use std::fs::File;
use std::io::Write;
use std::path::Path;

mod utils;

use utils::datasets::*;

fn normal_series(rng: &mut SmallRng, n: usize) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..n).map(|_| normal.sample(rng)).collect()
}

fn correlated_pair(rng: &mut SmallRng, n: usize, rho: f64) -> (Vec<f64>, Vec<f64>) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let s = (1.0 - rho * rho).sqrt();
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let z: f64 = normal.sample(rng);
        let w: f64 = normal.sample(rng);
        x.push(z);
        y.push(rho * z + s * w);
    }
    (x, y)
}

fn lagged_pair(rng: &mut SmallRng, n: usize, coupling: f64, lag: usize) -> (Vec<f64>, Vec<f64>) {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let s = (1.0 - coupling * coupling).sqrt();
    let raw: Vec<f64> = (0..n + lag).map(|_| normal.sample(rng)).collect();
    let source = raw[..n].to_vec();
    let target: Vec<f64> = (0..n)
        .map(|i| coupling * raw[i + lag] + s * normal.sample(rng))
        .collect();
    (source, target)
}

fn discrete_series(rng: &mut SmallRng, n: usize, states: i32) -> Vec<i32> {
    (0..n).map(|_| rng.gen_range(0..states)).collect()
}

// Target follows the source with lag 1 (5-state), mirroring the existing TE benches.
fn discrete_te(rng: &mut SmallRng, n: usize) -> (Vec<i32>, Vec<i32>) {
    let source = discrete_series(rng, n, NUM_STATES_TE);
    let mut target = Vec::with_capacity(n);
    target.push(source[0]);
    for i in 1..n {
        target.push(if source[i - 1] == source[i] {
            source[i]
        } else {
            rng.gen_range(0..NUM_STATES_TE)
        });
    }
    (source, target)
}

#[allow(clippy::too_many_arguments)]
fn emit_i32(
    dir: &Path,
    entries: &mut Vec<serde_json::Value>,
    measure: &str,
    seed: u64,
    n: usize,
    cols: &[Vec<i32>],
) -> std::io::Result<()> {
    let id = dataset_id(measure, "discrete", seed, n);
    write_i32(&dataset_path(dir, &id), &interleave_i32(cols))?;
    entries.push(json!({
        "id": id, "measure": measure, "kind": "discrete", "dtype": "i32le",
        "shape": [n, cols.len()], "seed": seed, "file": format!("{id}.bin")
    }));
    Ok(())
}

fn emit_f64(
    dir: &Path,
    entries: &mut Vec<serde_json::Value>,
    measure: &str,
    seed: u64,
    n: usize,
    cols: &[Vec<f64>],
) -> std::io::Result<()> {
    let id = dataset_id(measure, "continuous", seed, n);
    write_f64(&dataset_path(dir, &id), &interleave_f64(cols))?;
    entries.push(json!({
        "id": id, "measure": measure, "kind": "continuous", "dtype": "f64le",
        "shape": [n, cols.len()], "seed": seed, "file": format!("{id}.bin")
    }));
    Ok(())
}

fn main() -> std::io::Result<()> {
    let dir = data_dir();
    ensure_dir(&dir)?;
    // Drop stale dataset files from a previous run (e.g. a changed seed set).
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "bin") {
                let _ = std::fs::remove_file(path);
            }
        }
    }
    let sizes = sizes();
    let mut entries = Vec::new();

    for &seed in &SEEDS {
        for &n in &sizes {
            let mut rng = SmallRng::seed_from_u64(seed);

            // Discrete (i32)
            emit_i32(
                &dir,
                &mut entries,
                "entropy",
                seed,
                n,
                &[discrete_series(&mut rng, n, NUM_STATES_DISCRETE)],
            )?;
            emit_i32(
                &dir,
                &mut entries,
                "mi",
                seed,
                n,
                &[
                    discrete_series(&mut rng, n, NUM_STATES_DISCRETE),
                    discrete_series(&mut rng, n, NUM_STATES_DISCRETE),
                ],
            )?;
            emit_i32(
                &dir,
                &mut entries,
                "cmi",
                seed,
                n,
                &[
                    discrete_series(&mut rng, n, NUM_STATES_DISCRETE),
                    discrete_series(&mut rng, n, NUM_STATES_DISCRETE),
                    discrete_series(&mut rng, n, NUM_STATES_DISCRETE),
                ],
            )?;
            let (te_src, te_tgt) = discrete_te(&mut rng, n);
            emit_i32(&dir, &mut entries, "te", seed, n, &[te_src, te_tgt])?;
            let (cte_src, cte_tgt) = discrete_te(&mut rng, n);
            emit_i32(
                &dir,
                &mut entries,
                "cte",
                seed,
                n,
                &[
                    cte_src,
                    cte_tgt,
                    discrete_series(&mut rng, n, NUM_STATES_TE),
                ],
            )?;

            // Continuous (f64)
            emit_f64(
                &dir,
                &mut entries,
                "entropy",
                seed,
                n,
                &[normal_series(&mut rng, n)],
            )?;
            let (mi_x, mi_y) = correlated_pair(&mut rng, n, CORRELATION);
            emit_f64(&dir, &mut entries, "mi", seed, n, &[mi_x, mi_y])?;
            let (cmi_x, cmi_y) = correlated_pair(&mut rng, n, CORRELATION);
            emit_f64(
                &dir,
                &mut entries,
                "cmi",
                seed,
                n,
                &[cmi_x, cmi_y, normal_series(&mut rng, n)],
            )?;
            let (cont_te_src, cont_te_tgt) = lagged_pair(&mut rng, n, COUPLING, LAG);
            emit_f64(
                &dir,
                &mut entries,
                "te",
                seed,
                n,
                &[cont_te_src, cont_te_tgt],
            )?;
            let (cont_cte_src, cont_cte_tgt) = lagged_pair(&mut rng, n, COUPLING, LAG);
            emit_f64(
                &dir,
                &mut entries,
                "cte",
                seed,
                n,
                &[cont_cte_src, cont_cte_tgt, normal_series(&mut rng, n)],
            )?;
        }
    }

    let manifest = json!({
        "version": DATA_VERSION,
        "seeds": SEEDS,
        "sizes": sizes,
        "parameters": {
            "k": K,
            "bandwidth": BANDWIDTH,
            "noise_level": NOISE_LEVEL,
            "correlation": CORRELATION,
            "coupling": COUPLING,
            "lag": LAG,
            "num_states_discrete": NUM_STATES_DISCRETE,
            "num_states_te": NUM_STATES_TE,
        },
        "datasets": entries,
    });
    let mut f = File::create(dir.join("manifest.json"))?;
    f.write_all(serde_json::to_string_pretty(&manifest).unwrap().as_bytes())?;

    println!(
        "wrote {} dataset files + manifest.json to {}",
        entries.len(),
        dir.display()
    );
    Ok(())
}

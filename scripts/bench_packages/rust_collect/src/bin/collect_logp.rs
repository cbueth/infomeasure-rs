// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-package collector: `logp` (Rust).
//!
//! Times only the estimator call on the canonical datasets and writes
//! `results/logp.json`. Coverage of the cross-package grid:
//! - discrete (MLE): Shannon entropy and MI (plug-in from empirical counts);
//! - ksg: MI via [`logp::mutual_information_ksg`] (Algorithm 1, matching the
//!   infomeasure/scipy default).
//!
//! `logp` reports nats; values are converted to bits so all rows share a unit.
//! It has no native CMI (only conditional entropy), no TE/CTE and no KL entropy
//! estimator, so those cells stay N/A — a measure assembled from several package
//! calls is not timed (see the benchmark plan's equalisation rule).

use bench_rust_collect::{datasets, filter, fragment, timing};
use logp::KsgVariant;
use serde_json::{json, Value};

const PACKAGE: &str = "logp";
const VERSION: &str = "0.2.3";
const TOL: f64 = 1e-9;
const K: usize = 4;
const LN2: f64 = std::f64::consts::LN_2;

fn n_states(col: &[i32]) -> usize {
    col.iter().copied().max().unwrap_or(0) as usize + 1
}

fn probs(col: &[i32]) -> Vec<f64> {
    let mut c = vec![0.0f64; n_states(col)];
    for &v in col {
        c[v as usize] += 1.0;
    }
    let len = col.len() as f64;
    c.iter().map(|v| v / len).collect()
}

fn joint(a: &[i32], b: &[i32]) -> (Vec<f64>, usize, usize) {
    let (nx, ny) = (n_states(a), n_states(b));
    let mut c = vec![0.0f64; nx * ny];
    for i in 0..a.len() {
        c[a[i] as usize * ny + b[i] as usize] += 1.0;
    }
    let len = a.len() as f64;
    for v in &mut c {
        *v /= len;
    }
    (c, nx, ny)
}

fn main() {
    let (sizes, seeds) = datasets::sizes_and_seeds();
    let rounds = timing::config();
    let mut benchmarks: Vec<Value> = Vec::new();

    // --- discrete, method = mle -------------------------------------------
    // Native measures only: logp has no CMI function (only conditional
    // entropy), and composing one from several calls is out of scope for the
    // fair harness, so the CMI cells stay N/A.
    for (measure, function) in [
        ("entropy", "logp::entropy_bits"),
        ("mi", "logp::mutual_information"),
    ] {
        if !filter::want(measure, "discrete") {
            continue;
        }
        for &n in &sizes {
            let mut times = Vec::new();
            let mut value = f64::NAN;
            for &seed in &seeds {
                let cols = datasets::load_discrete(measure, seed, n);
                let (t, v) = timing::time_call(
                    || match measure {
                        "entropy" => logp::entropy_bits(&probs(&cols[0]), TOL).unwrap(),
                        "mi" => {
                            let (p, nx, ny) = joint(&cols[0], &cols[1]);
                            logp::mutual_information(&p, nx, ny, TOL).unwrap() / LN2
                        }
                        _ => unreachable!(),
                    },
                    &rounds,
                );
                times.extend(t);
                value = v;
            }
            let st = fragment::stats(&times);
            println!(
                "  {measure:>7} discrete  mle  n={n:<6} {:>9.3} ms",
                st["mean"].as_f64().unwrap() * 1e3
            );
            benchmarks.push(fragment::entry(
                PACKAGE,
                measure,
                "discrete",
                "mle",
                function,
                n,
                fragment::params(measure, "discrete", n, None),
                st,
                value,
            ));
        }
    }

    // --- ksg (MI only) -----------------------------------------------------
    let ksg_sizes: &[usize] = if filter::want("mi", "ksg") {
        &sizes
    } else {
        &[]
    };
    for &n in ksg_sizes {
        let mut times = Vec::new();
        let mut value = f64::NAN;
        for &seed in &seeds {
            let cols = datasets::load_continuous("mi", seed, n);
            // 1-D samples; marshalling stays outside the timed region.
            let x: Vec<Vec<f64>> = cols[0].iter().map(|v| vec![*v]).collect();
            let y: Vec<Vec<f64>> = cols[1].iter().map(|v| vec![*v]).collect();
            let (t, v) = timing::time_call(
                || logp::mutual_information_ksg(&x, &y, K, KsgVariant::Alg1).unwrap() / LN2,
                &rounds,
            );
            times.extend(t);
            value = v;
        }
        let st = fragment::stats(&times);
        println!(
            "       mi ksg      k4   n={n:<6} {:>9.3} ms",
            st["mean"].as_f64().unwrap() * 1e3
        );
        benchmarks.push(fragment::entry(
            PACKAGE,
            "mi",
            "ksg",
            "k4",
            "logp::mutual_information_ksg",
            n,
            fragment::params("mi", "ksg", n, Some(K)),
            st,
            value,
        ));
    }

    fragment::write_fragment(
        PACKAGE,
        "rust",
        VERSION,
        &benchmarks,
        &seeds,
        &rounds,
        json!({
            "log_base": 2,
            "library": "logp",
            "ksg": {
                "algorithm": "1",
                "metric": "max-norm (chebyshev)",
                "normalisation": "none",
                "added_noise": "none",
                "theiler_window": "none",
                "neighbour_index": "brute force (O(N^2), no spatial index)",
                "units": "nats",
                "noise_note": "Adds no random jitter. Jitter (infomeasure 1e-10, JIDT 1e-8) helps on duplicate/degenerate samples but costs time inside the timed call.",
            },
        }),
        "Discrete plug-in MLE (Shannon entropy, MI) and KSG MI. KSG: Algorithm 1 \
         (strict marginal counts), Chebyshev/max-norm metric, no normalisation, \
         no added noise (infomeasure adds 1e-10 jitter), no Theiler window, \
         O(N^2) brute force (no spatial index). Values converted from nats to \
         bits. No native CMI (conditional entropy only), no TE/CTE and no KL \
         entropy estimator; its plug-in Renyi/Tsallis are not the grid's kNN \
         variants. Unsupported cells are N/A.",
    );
}

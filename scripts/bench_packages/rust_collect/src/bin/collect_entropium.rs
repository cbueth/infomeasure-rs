// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-package collector: `entropium` (Rust).
//!
//! Times only the estimator call on the canonical datasets and writes
//! `results/entropium.json`. Coverage of the cross-package grid (discrete,
//! MLE): Shannon entropy, MI, and CMI composed as `H(X|Z) - H(X|YZ)`.
//!
//! All `entropium` values are in bits. It has no TE/CTE, so those cells stay
//! N/A. `conditional_entropy` couples paired observations, so conditioning on
//! `(Y, Z)` encodes the pair into a single symbol inside the timed call.

use bench_rust_collect::{datasets, fragment, timing};
use serde_json::{json, Value};

const PACKAGE: &str = "entropium";
const VERSION: &str = "0.2.0";

fn n_states(col: &[i32]) -> i64 {
    col.iter().copied().max().unwrap_or(0) as i64 + 1
}

fn as_i64(col: &[i32]) -> Vec<i64> {
    col.iter().map(|v| *v as i64).collect()
}

/// Encode `(b, c)` into one symbol so `conditional_entropy(a, ·)` conditions on
/// both.
fn pair(a: &[i32], b: &[i32]) -> Vec<i64> {
    let base = n_states(b);
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x as i64 * base + y as i64)
        .collect()
}

fn main() {
    let (sizes, seeds) = datasets::sizes_and_seeds();
    let rounds = timing::config();
    let mut benchmarks: Vec<Value> = Vec::new();

    for (measure, function) in [
        ("entropy", "entropium::entropy"),
        ("mi", "entropium::mutual_information"),
        ("cmi", "entropium::conditional_entropy"),
    ] {
        for &n in &sizes {
            let mut times = Vec::new();
            let mut value = f64::NAN;
            for &seed in &seeds {
                let cols = datasets::load_discrete(measure, seed, n);
                let (t, v) = timing::time_call(
                    || match measure {
                        "entropy" => entropium::entropy(&cols[0]).unwrap(),
                        "mi" => entropium::mutual_information(&cols[0], &cols[1]).unwrap(),
                        "cmi" => {
                            // I(X;Y|Z) = H(X|Z) - H(X|YZ)
                            let x = as_i64(&cols[0]);
                            let z = as_i64(&cols[2]);
                            let yz = pair(&cols[1], &cols[2]);
                            let h_x_given_z = entropium::conditional_entropy(&x, &z).unwrap();
                            let h_x_given_yz = entropium::conditional_entropy(&x, &yz).unwrap();
                            h_x_given_z - h_x_given_yz
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

    fragment::write_fragment(
        PACKAGE,
        "rust",
        VERSION,
        &benchmarks,
        &seeds,
        &rounds,
        json!({ "base": 2, "library": "entropium" }),
        "Discrete MLE only (Shannon entropy, MI, CMI); values in bits; no TE/CTE. \
         CMI composed as H(X|Z) - H(X|YZ); the (Y,Z) pair is encoded inside the \
         timed call.",
    );
}

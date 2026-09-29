// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-package collector: `entropium` (Rust).
//!
//! Times only the estimator call on the canonical datasets and writes
//! `results/entropium.json`. Coverage of the cross-package grid (discrete,
//! MLE): Shannon entropy and MI.
//!
//! All `entropium` values are in bits. It has no native CMI (only conditional
//! entropy), no TE/CTE, so those cells stay N/A — a measure assembled from
//! several package calls is not timed (see the benchmark plan's equalisation
//! rule).

use bench_rust_collect::{datasets, filter, fragment, timing};
use serde_json::{json, Value};

const PACKAGE: &str = "entropium";
const VERSION: &str = "0.2.0";

fn main() {
    let (sizes, seeds) = datasets::sizes_and_seeds();
    let rounds = timing::config();
    let mut benchmarks: Vec<Value> = Vec::new();

    for (measure, function) in [
        ("entropy", "entropium::entropy"),
        ("mi", "entropium::mutual_information"),
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
                        "entropy" => entropium::entropy(&cols[0]).unwrap(),
                        "mi" => entropium::mutual_information(&cols[0], &cols[1]).unwrap(),
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
        json!({ "log_base": 2, "library": "entropium" }),
        "Discrete MLE only (Shannon entropy, MI); values in bits. No native CMI \
         (conditional entropy only), no TE/CTE; unsupported cells are N/A.",
    );
}

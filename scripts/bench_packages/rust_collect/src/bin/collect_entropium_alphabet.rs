// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Alphabet-scaling collector: `entropium` (Rust).
//!
//! Times discrete MLE Shannon entropy and MI across state counts and the cross
//! sizes, writing `results/entropium_alphabet.json`. Values are in bits.

use bench_rust_collect::{alphabet, filter, fragment, timing};
use serde_json::{json, Value};

const PACKAGE: &str = "entropium";
const VERSION: &str = "0.2.0";

fn run(measure: &str, cols: &[Vec<i32>]) -> f64 {
    match measure {
        "entropy" => entropium::entropy(&cols[0]).unwrap(),
        "mi" => entropium::mutual_information(&cols[0], &cols[1]).unwrap(),
        other => unreachable!("unknown alphabet measure {other}"),
    }
}

fn function_name(measure: &str) -> &'static str {
    match measure {
        "entropy" => "entropium::entropy",
        "mi" => "entropium::mutual_information",
        _ => "?",
    }
}

fn main() {
    let ab = alphabet::config();
    let sizes = ab.sizes();
    let rounds = timing::config();
    let mut benchmarks: Vec<Value> = Vec::new();
    let mut coverage: Vec<Vec<String>> = Vec::new();

    for measure in ["entropy", "mi"] {
        if !filter::want(measure, "discrete") {
            continue;
        }
        coverage.push(vec![measure.to_string(), "discrete".to_string()]);
        for states in ab.states_for(measure) {
            let mut stopped = false;
            for &n in &sizes {
                if stopped {
                    break;
                }
                let mut times = Vec::new();
                let mut value = f64::NAN;
                for &seed in &alphabet::SEEDS {
                    let cols = alphabet::load_discrete(measure, states, seed, n);
                    let (t, v) = timing::time_call(|| run(measure, &cols), &rounds);
                    times.extend(t);
                    value = v;
                }
                let st = fragment::stats(&times);
                println!(
                    "  {measure:>7} discrete  b{states:<4} n={n:<6} {:>9.4} ms",
                    st["mean"].as_f64().unwrap() * 1e3
                );
                benchmarks.push(alphabet::entry(
                    PACKAGE,
                    "rust",
                    measure,
                    states,
                    n,
                    function_name(measure),
                    st.clone(),
                    value,
                ));
                if st["mean"].as_f64().unwrap_or(0.0) > ab.budget_s {
                    println!(
                        "  -> b{states} n={n} exceeded {}s; skipping larger N",
                        ab.budget_s
                    );
                    stopped = true;
                }
            }
            alphabet::write_fragment(
                PACKAGE,
                "rust",
                VERSION,
                &benchmarks,
                &coverage,
                &rounds,
                json!({ "log_base": 2, "library": "entropium" }),
                "Discrete MLE only (Shannon entropy, MI); values in bits. No CMI/TE/CTE.",
            );
        }
    }
}

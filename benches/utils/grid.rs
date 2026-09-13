// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Loader and expander for the shared detailed benchmark grid
//! (`benches/detailed_grid.json`). The same file drives the Python
//! infomeasure collector via `scripts/bench_packages/_grid.py`, so both
//! packages always measure an identical set of variants.

#![allow(dead_code)]

use serde_json::{Value, json};
use std::collections::BTreeMap;

/// One concrete benchmark variant: a measure plus the estimator parameters
/// that select it. `cross` marks the representative subset used for the
/// cross-package comparison page.
#[derive(Clone, Debug)]
pub struct Variant {
    pub measure: String,
    pub approach: String,
    pub method: Option<String>,
    pub k: Option<usize>,
    pub bandwidth: Option<f64>,
    pub kernel: Option<String>,
    pub order: Option<usize>,
    pub alpha: Option<f64>,
    pub q: Option<f64>,
    pub cross: bool,
    /// Seeds to use; populated by the collector (representative uses all,
    /// detailed-only uses the first seed only).
    pub detail_only: bool,
}

impl Variant {
    /// Stable, human-readable suffix for the entry id.
    pub fn slug(&self) -> String {
        match self.approach.as_str() {
            "discrete" => self.method.clone().unwrap_or_else(|| "mle".into()),
            "kernel" => format!(
                "{}_{}",
                self.kernel.clone().unwrap_or_else(|| "box".into()),
                bw_slug(self.bandwidth.unwrap_or(0.5))
            ),
            "ordinal" => format!("order{}", self.order.unwrap_or(0)),
            "renyi" => format!("alpha{}_k{}", num_slug(self.alpha), self.k.unwrap_or(0)),
            "tsallis" => format!("q{}_k{}", num_slug(self.q), self.k.unwrap_or(0)),
            _ => format!("k{}", self.k.unwrap_or(0)),
        }
    }

    /// Function name shown on the page (infomeasure-rs API symbol).
    pub fn function(&self) -> String {
        let m = self.api_measure();
        match self.approach.as_str() {
            "discrete" => {
                let method = self.method.as_deref().unwrap_or("mle");
                if method == "mle" {
                    format!("{m}::new_discrete_mle")
                } else {
                    format!("{m}::new_discrete_{method}")
                }
            }
            "kernel" => format!("{m}::new_kernel_with_type"),
            "ordinal" => format!("{m}::new_ordinal"),
            "ksg" => {
                if self.measure == "entropy" {
                    "Entropy::new_kl_1d".into()
                } else {
                    format!("{m}::new_ksg")
                }
            }
            "kl" => format!("{m}::new_kl"),
            "kl_cheb" => "Entropy::new_kl_1d.with_chebyshev".into(),
            "kl_k" => "Entropy::new_kl_1d".into(),
            "renyi" => format!("{m}::new_renyi"),
            "tsallis" => format!("{m}::new_tsallis"),
            other => other.to_string(),
        }
    }

    fn api_measure(&self) -> &'static str {
        match self.measure.as_str() {
            "entropy" => "Entropy",
            "mi" | "cmi" => "MutualInformation",
            "te" | "cte" => "TransferEntropy",
            _ => "?",
        }
    }

    /// Parameter object written to the fragment (schema v2).
    pub fn params(&self, n: usize) -> Value {
        serde_json::json!({
            "n": n,
            "k": self.k,
            "bandwidth": self.bandwidth,
            "order": self.order,
            "delay": if self.measure == "te" || self.measure == "cte" { json!(1) } else { Value::Null },
            "alpha": self.alpha,
            "q": self.q,
            "dims": 1,
            "method": self.method,
            "kernel_type": self.kernel,
        })
    }
}

fn bw_slug(bw: f64) -> String {
    format!("bw{}", num_slug(Some(bw)))
}

fn num_slug(v: Option<f64>) -> String {
    let v = v.unwrap_or(0.0);
    let s = format!("{v}");
    s.replace('.', "_")
}

/// The expanded grid for one language (`"rust"` or `"python"`).
pub struct Grid {
    pub cross_sizes: Vec<usize>,
    pub detailed_sizes: Vec<usize>,
    pub variants: Vec<Variant>,
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("int") as usize)
        .collect()
}

pub fn load(lang: &str) -> Grid {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benches/detailed_grid.json");
    let text = std::fs::read_to_string(path).expect("read detailed_grid.json");
    let root: Value = serde_json::from_str(&text).expect("parse detailed_grid.json");

    let cross_sizes = usizes(&root["sizes"]["cross"]);
    let detailed_sizes = usizes(&root["sizes"]["detailed"]);
    let values = &root["values"];
    let cross_map = &root["cross"];

    let mut variants = Vec::new();
    for (measure, entries) in root["approaches"].as_object().expect("approaches") {
        for entry in entries.as_array().expect("measure list") {
            let approach = entry["approach"].as_str().expect("approach").to_string();

            let langs = entry.get("langs").and_then(Value::as_array);
            let lang_ok = match langs {
                Some(l) => l.iter().any(|x| x.as_str() == Some(lang)),
                None => true,
            };
            if !lang_ok {
                continue;
            }

            let axes = entry["axes"].as_array().expect("axes");
            let mut combos: Vec<BTreeMap<String, Value>> = vec![BTreeMap::new()];
            for axis in axes {
                let axis = axis.as_str().expect("axis");
                let axis_values = values[axis].as_array().expect("axis values").clone();
                let mut next = Vec::new();
                for base in &combos {
                    for val in &axis_values {
                        let mut combo = base.clone();
                        combo.insert(axis.to_string(), val.clone());
                        next.push(combo);
                    }
                }
                combos = next;
            }

            for combo in combos {
                let mut var = Variant {
                    measure: measure.clone(),
                    approach: approach.clone(),
                    method: None,
                    k: None,
                    bandwidth: None,
                    kernel: None,
                    order: None,
                    alpha: None,
                    q: None,
                    cross: false,
                    detail_only: false,
                };
                for (axis, val) in &combo {
                    match axis.as_str() {
                        "method" => var.method = val.as_str().map(str::to_string),
                        "k" => var.k = val.as_u64().map(|x| x as usize),
                        "kl_k" => var.k = val.as_u64().map(|x| x as usize),
                        "bandwidth" => var.bandwidth = val.as_f64(),
                        "kernel" => var.kernel = val.as_str().map(str::to_string),
                        "order" => var.order = val.as_u64().map(|x| x as usize),
                        "alpha" => var.alpha = val.as_f64(),
                        "q" => var.q = val.as_f64(),
                        _ => {}
                    }
                }

                if let Some(spec) = cross_map.get(&approach).and_then(Value::as_object) {
                    var.cross = spec
                        .iter()
                        .all(|(axis, want)| combo.get(axis).is_some_and(|got| got == want));
                }
                var.detail_only = !var.cross;
                variants.push(var);
            }
        }
    }

    Grid {
        cross_sizes,
        detailed_sizes,
        variants,
    }
}

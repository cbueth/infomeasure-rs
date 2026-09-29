// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Optional `BENCH_MEASURES` / `BENCH_APPROACHES` allow-list filters.
//!
//! Both are comma-separated and case-insensitive; unset (or empty) means "no
//! filter". They mirror the filters `benches/collect_cross_package.rs` already
//! honours, so a single cell (e.g. discrete MI) can be collected without timing
//! the package's whole grid.

fn allows(var: &str, value: &str) -> bool {
    match std::env::var(var) {
        Ok(raw) if !raw.trim().is_empty() => raw
            .split(',')
            .map(str::trim)
            .any(|item| item.eq_ignore_ascii_case(value)),
        _ => true,
    }
}

/// Whether the `(measure, approach)` cell passes both environment filters.
pub fn want(measure: &str, approach: &str) -> bool {
    allows("BENCH_MEASURES", measure) && allows("BENCH_APPROACHES", approach)
}

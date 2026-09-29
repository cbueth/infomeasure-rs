// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared helpers for the third-party Rust comparison collectors.
//!
//! These mirror the schema-v2 contract of `scripts/bench_packages/_common.py`
//! so `merge_results.py` and the viewer treat the fragments identically: same
//! id shape, params object, statistics, and `meta.packages` entry.

pub mod datasets;
pub mod filter;
pub mod fragment;
pub mod timing;

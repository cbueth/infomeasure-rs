#![allow(unused_imports, dead_code)]

use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

pub mod data;
pub mod hardware;

pub use data::*;
pub use hardware::*;
/// Parse a comma-separated env var into a Vec<usize>.
/// Falls back to `default` if the env var is not set or parse fails.
fn parse_sizes(env_var: &str, default: &[usize]) -> Vec<usize> {
    std::env::var(env_var)
        .ok()
        .and_then(|val| {
            let parsed: Vec<usize> = val
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if parsed.is_empty() {
                None
            } else {
                Some(parsed)
            }
        })
        .unwrap_or_else(|| default.to_vec())
}

/// Benchmark sizes for slow approaches (kernel, KL/KSG, renyi, tsallis).
/// Override via the `BENCH_SIZES` environment variable.
pub fn bench_sizes() -> Vec<usize> {
    parse_sizes("BENCH_SIZES", &[100, 500, 1000, 5000, 10000])
}

/// Benchmark sizes for fast approaches (discrete, ordinal).
/// Override via the `BENCH_SIZES_EXTENDED` environment variable.
pub fn bench_sizes_extended() -> Vec<usize> {
    parse_sizes(
        "BENCH_SIZES_EXTENDED",
        &[100, 500, 1000, 5000, 10000, 50000, 100000],
    )
}

/// Parse a comma-separated env var into a Vec<f64>.
fn parse_f64s(env_var: &str, default: &[f64]) -> Vec<f64> {
    std::env::var(env_var)
        .ok()
        .and_then(|val| {
            let parsed: Vec<f64> = val
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if parsed.is_empty() {
                None
            } else {
                Some(parsed)
            }
        })
        .unwrap_or_else(|| default.to_vec())
}

/// k-nearest neighbors values for KL/KSG/Rényi/Tsallis estimators.
/// Override via the `BENCH_K_VALUES` environment variable.
pub fn bench_k_values() -> Vec<usize> {
    parse_sizes("BENCH_K_VALUES", &[3, 5])
}

/// Bandwidth values for kernel estimators.
/// Override via the `BENCH_BANDWIDTHS` environment variable.
pub fn bench_bandwidths() -> Vec<f64> {
    parse_f64s("BENCH_BANDWIDTHS", &[0.3, 0.5, 1.0, 1.5])
}

/// Order values for ordinal estimators.
/// Override via the `BENCH_ORDERS` environment variable.
pub fn bench_orders() -> Vec<usize> {
    parse_sizes("BENCH_ORDERS", &[2, 3, 4])
}

/// Alpha values for Rényi estimators.
/// Override via the `BENCH_ALPHAS` environment variable.
pub fn bench_alphas() -> Vec<f64> {
    parse_f64s("BENCH_ALPHAS", &[0.6, 1.2, 1.8])
}

/// Q values for Tsallis estimators.
/// Override via the `BENCH_Q_VALUES` environment variable.
pub fn bench_q_values() -> Vec<f64> {
    parse_f64s("BENCH_Q_VALUES", &[0.6, 1.2, 1.8])
}

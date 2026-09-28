// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adaptive warm-up/timed-iteration loop, mirroring `_common.timing_config`
//! and `_common.time_call`. File I/O stays outside the timed region.

use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Rounds {
    pub short: bool,
    pub warmup_max: u32,
    pub warmup_budget: f64,
    pub min_iters: u32,
    pub max_iters: u32,
    pub iter_budget: f64,
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

pub fn config() -> Rounds {
    let short = matches!(
        std::env::var("BENCH_SHORT").as_deref(),
        Ok("1") | Ok("true")
    );
    if short {
        return Rounds {
            short,
            warmup_max: 1,
            warmup_budget: 0.0,
            min_iters: 1,
            max_iters: 2,
            iter_budget: 0.0,
        };
    }
    Rounds {
        short,
        warmup_max: env_u32("BENCH_WARMUP_MAX", 3),
        warmup_budget: env_f64("BENCH_WARMUP_BUDGET_S", 0.4),
        min_iters: env_u32("BENCH_MIN_ITERS", 3),
        max_iters: env_u32("BENCH_MAX_ITERS", 10),
        iter_budget: env_f64("BENCH_ITER_BUDGET_S", 1.5),
    }
}

/// Warm up, then time `f` until the budget is met. Returns per-iteration
/// seconds and the value from the last call.
#[allow(unused_assignments)]
pub fn time_call<F: FnMut() -> f64>(mut f: F, r: &Rounds) -> (Vec<f64>, f64) {
    let w0 = Instant::now();
    let mut warm = 0;
    loop {
        f();
        warm += 1;
        if warm >= r.warmup_max {
            break;
        }
        if r.warmup_budget > 0.0 && w0.elapsed().as_secs_f64() >= r.warmup_budget {
            break;
        }
    }
    let mut times = Vec::new();
    let mut value = f64::NAN;
    let t0 = Instant::now();
    let mut it = 0;
    loop {
        let s = Instant::now();
        value = f();
        times.push(s.elapsed().as_secs_f64());
        it += 1;
        if it >= r.max_iters {
            break;
        }
        if it >= r.min_iters
            && (r.iter_budget <= 0.0 || t0.elapsed().as_secs_f64() >= r.iter_budget)
        {
            break;
        }
    }
    (times, value)
}

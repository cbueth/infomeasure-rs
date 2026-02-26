// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::expfam::utils::unit_ball_volume;
use rstest::rstest;
use validation::python;

#[rstest]
fn test_unit_ball_volume_parity(
    #[values(1, 2, 3, 5, 10)] d: usize,
    #[values(1.0, 2.0, f64::INFINITY)] p: f64,
    #[values(1.0, 2.0)] r: f64,
) {
    let rust_volume = unit_ball_volume(d, p) * r.powi(d as i32);

    let script = format!(
        "from infomeasure.estimators.utils.unit_ball_volume import unit_ball_volume as ubv\n\
         import numpy as np\n\
         p = float('inf') if {p_inf} else {p}\n\
         print(ubv({d}, r={r}, p=p))",
        d = d,
        r = r,
        p = p,
        p_inf = if p.is_infinite() { "True" } else { "False" }
    );

    let output = python::run_in_environment(&["-c", &script]).expect("Failed to run Python");
    let python_volume: f64 = output
        .trim()
        .parse()
        .expect("Failed to parse Python output");

    assert_abs_diff_eq!(rust_volume, python_volume, epsilon = 1e-10);
}

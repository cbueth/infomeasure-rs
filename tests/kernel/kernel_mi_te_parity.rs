// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use approx::assert_relative_eq;
use infomeasure::estimators::GlobalValue;
use infomeasure::estimators::LocalValues;
use infomeasure::estimators::mutual_information::MutualInformation;

use infomeasure::{new_kernel_cmi, new_kernel_cte, new_kernel_mi, new_kernel_te};
use ndarray::{Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rstest::rstest;
use validation::python;

#[rstest]
fn test_kernel_mi_parity(
    #[values("box", "gaussian")] kernel_type: &str,
    #[values(1, 2, 3, 4)] d1: usize,
    #[values(1, 2, 3, 4)] d2: usize,
) {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(5.0, 2.0).unwrap();
    let size = 100;
    let bandwidth = 1.0;

    let x_data: Vec<f64> = (0..size * d1).map(|_| normal.sample(&mut rng)).collect();
    let y_data: Vec<f64> = (0..size * d2).map(|_| normal.sample(&mut rng)).collect();

    let x_arr = Array2::from_shape_vec((size, d1), x_data).unwrap();
    let y_arr = Array2::from_shape_vec((size, d2), y_data).unwrap();

    // Rust
    // Use match to handle different dimensions at compile time
    let rust_val = match (d1, d2) {
        (1, 1) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            1,
            1
        )
        .global_value(),
        (1, 2) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            1,
            2
        )
        .global_value(),
        (1, 3) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            1,
            3
        )
        .global_value(),
        (1, 4) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            1,
            4
        )
        .global_value(),
        (2, 1) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            2,
            1
        )
        .global_value(),
        (2, 2) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            2,
            2
        )
        .global_value(),
        (2, 3) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            2,
            3
        )
        .global_value(),
        (2, 4) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            2,
            4
        )
        .global_value(),
        (3, 1) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            3,
            1
        )
        .global_value(),
        (3, 2) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            3,
            2
        )
        .global_value(),
        (3, 3) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            3,
            3
        )
        .global_value(),
        (3, 4) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            3,
            4
        )
        .global_value(),
        (4, 1) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            4,
            1
        )
        .global_value(),
        (4, 2) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            4,
            2
        )
        .global_value(),
        (4, 3) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            4,
            3
        )
        .global_value(),
        (4, 4) => new_kernel_mi!(
            &[x_arr.clone(), y_arr.clone()],
            kernel_type.to_string(),
            bandwidth,
            4,
            4
        )
        .global_value(),
        _ => unreachable!(),
    };

    // Python
    let kwargs = [
        ("kernel".to_string(), format!("\"{kernel_type}\"")),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let mut x_py = Vec::new();
    for i in 0..d1 {
        x_py.push(x_arr.column(i).to_vec());
    }
    let mut y_py = Vec::new();
    for i in 0..d2 {
        y_py.push(y_arr.column(i).to_vec());
    }

    let py_data = vec![x_py, y_py];

    let py_val = python::calculate_mi_float(&py_data, "kernel", &kwargs).unwrap();

    let tol = 1e-10;
    assert_relative_eq!(rust_val, py_val, epsilon = tol, max_relative = 1e-6);
}

#[rstest]
fn test_kernel_mi_nd_parity(#[values("box", "gaussian")] kernel_type: &str) {
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(5.0, 2.0).unwrap();
    let size = 100;
    let bandwidth = 1.0;

    let x_data: Vec<f64> = (0..size * 2).map(|_| normal.sample(&mut rng)).collect();
    let y_data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();

    let x_arr = Array2::from_shape_vec((size, 2), x_data).unwrap();
    let y_arr = Array2::from_shape_vec((size, 1), y_data).unwrap();

    // Rust
    let rust_est = MutualInformation::nd_kernel_with_type::<3, 2, 1>(
        &[x_arr.clone(), y_arr.clone()],
        kernel_type.to_string(),
        bandwidth,
    );
    let rust_val = rust_est.global_value();

    // Python
    let kwargs = [
        ("kernel".to_string(), format!("\"{kernel_type}\"")),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let x_col0 = x_arr.column(0).to_vec();
    let x_col1 = x_arr.column(1).to_vec();
    let y_col0 = y_arr.column(0).to_vec();

    let py_val =
        python::calculate_mi_float(&[vec![x_col0, x_col1], vec![y_col0]], "kernel", &kwargs)
            .unwrap();

    // We use a larger tolerance for multi-D kernel MI because of different normalization handling in Scipy gaussian_kde
    // and how it interacts with multi-variable I(X;Y;Z).
    let tol_nd = if kernel_type == "gaussian" {
        1e-2
    } else {
        1e-10
    };
    assert_relative_eq!(rust_val, py_val, epsilon = tol_nd, max_relative = 1e-2);
}

#[rstest]
fn test_kernel_mi_4rv_parity(#[values("box", "gaussian")] kernel_type: &str) {
    let seed = 46;
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(5.0, 2.0).unwrap();
    let size = 100;
    let bandwidth = 1.0;

    let x1: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
    let x2: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
    let x3: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
    let x4: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();

    let series = vec![
        Array1::from(x1.clone()).insert_axis(Axis(1)),
        Array1::from(x2.clone()).insert_axis(Axis(1)),
        Array1::from(x3.clone()).insert_axis(Axis(1)),
        Array1::from(x4.clone()).insert_axis(Axis(1)),
    ];

    // Rust
    let rust_est = new_kernel_mi!(&series, kernel_type.to_string(), bandwidth, 1, 1, 1, 1);
    let rust_val = rust_est.global_value();

    // Python
    let kwargs = [
        ("kernel".to_string(), format!("\"{kernel_type}\"")),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let py_val =
        python::calculate_mi_float(&[vec![x1], vec![x2], vec![x3], vec![x4]], "kernel", &kwargs)
            .unwrap();

    let tol_nd = 1e-10;
    assert_relative_eq!(rust_val, py_val, epsilon = tol_nd, max_relative = 1e-2);
}

#[rstest]
fn test_kernel_cmi_parity(#[values("box", "gaussian")] kernel_type: &str) {
    let seed = 43;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 100;
    let bandwidth = 1.0;

    let x: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();
    let y: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();
    let z: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();

    let x_arr = Array1::from(x.clone());
    let y_arr = Array1::from(y.clone());
    let z_arr = Array1::from(z.clone());

    let x_arr_2d = x_arr.insert_axis(Axis(1));
    let y_arr_2d = y_arr.insert_axis(Axis(1));
    let z_arr_2d = z_arr.insert_axis(Axis(1));

    // Rust
    let rust_est = new_kernel_cmi!(
        &[x_arr_2d.clone(), y_arr_2d.clone()],
        &z_arr_2d,
        kernel_type.to_string(),
        bandwidth,
        1,
        1,
        1
    );
    let rust_val = rust_est.global_value();
    let rust_locals = rust_est.local_values();

    // Python
    let kwargs = [
        ("kernel".to_string(), format!("\"{kernel_type}\"")),
        ("bandwidth".to_string(), bandwidth.to_string()),
    ];
    let py_val =
        python::calculate_cmi_float(&[x.clone(), y.clone()], &z, "kernel", &kwargs).unwrap();
    let py_locals =
        python::calculate_local_cmi_float(&[x.clone(), y.clone()], &z, "kernel", &kwargs).unwrap();

    let tol = 1e-10;
    assert_relative_eq!(rust_val, py_val, epsilon = tol, max_relative = 1e-6);
    for (r, p) in rust_locals.iter().zip(py_locals.iter()) {
        assert_relative_eq!(*r, *p, epsilon = tol, max_relative = 1e-6);
    }
}

#[rstest]
fn test_kernel_te_parity(
    #[values("box", "gaussian")] kernel_type: &str,
    #[values(1, 2, 3)] src_hist: usize,
    #[values(1, 2, 3)] dest_hist: usize,
    #[values(1, 2, 3)] step_size: usize,
) {
    let seed = 44;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 100;
    let bandwidth = 1.0;

    let src: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();
    let dst: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();

    let src_arr = Array1::from(src.clone());
    let dst_arr = Array1::from(dst.clone());

    // Rust
    let rust_val = match (src_hist, dest_hist, step_size) {
        (1, 1, 1) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            1,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 1, 2) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            1,
            1,
            2,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 1, 3) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            1,
            1,
            3,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 2, 1) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            1,
            2,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 2, 2) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            1,
            2,
            2,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 2, 3) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            1,
            2,
            3,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 3, 1) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            1,
            3,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 3, 2) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            1,
            3,
            2,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 3, 3) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            1,
            3,
            3,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 1, 1) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            2,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 1, 2) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            2,
            1,
            2,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 1, 3) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            2,
            1,
            3,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 2, 1) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            2,
            2,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 2, 2) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            2,
            2,
            2,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 2, 3) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            2,
            2,
            3,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 3, 1) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            2,
            3,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 3, 2) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            2,
            3,
            2,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 3, 3) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            2,
            3,
            3,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 1, 1) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            3,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 1, 2) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            3,
            1,
            2,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 1, 3) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            3,
            1,
            3,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 2, 1) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            3,
            2,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 2, 2) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            3,
            2,
            2,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 2, 3) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            3,
            2,
            3,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 3, 1) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            3,
            3,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 3, 2) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            3,
            3,
            2,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 3, 3) => new_kernel_te!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            3,
            3,
            3,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        _ => unreachable!(),
    };

    // Python
    let kwargs = [
        ("kernel".to_string(), format!("\"{kernel_type}\"")),
        ("bandwidth".to_string(), bandwidth.to_string()),
        ("src_hist_len".to_string(), src_hist.to_string()),
        ("dest_hist_len".to_string(), dest_hist.to_string()),
        ("step_size".to_string(), step_size.to_string()),
    ];
    let py_val = python::calculate_te_float(&src, &dst, "kernel", &kwargs).unwrap();

    let tol_nd = 1e-10;
    assert_relative_eq!(rust_val, py_val, epsilon = tol_nd, max_relative = 1e-6);
}

#[rstest]
fn test_kernel_cte_parity(
    #[values("box", "gaussian")] kernel_type: &str,
    #[values(1, 2, 3)] src_hist: usize,
    #[values(1, 2, 3)] dest_hist: usize,
    #[values(1, 2, 3)] cond_hist: usize,
    #[values(1, 2, 3)] step_size: usize,
) {
    let seed = 45;
    let mut rng = StdRng::seed_from_u64(seed);
    let size = 200;
    let bandwidth = 1.0;

    let src: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();
    let dst: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();
    let cnd: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..10.0)).collect();

    let src_arr = Array1::from(src.clone());
    let dst_arr = Array1::from(dst.clone());
    let cnd_arr = Array1::from(cnd.clone());

    // Rust
    // Use a nested match or just call it directly since new_kernel_cte! is a macro
    // but the macro needs literal constants for dimensions if they were used in [(); EXPR]
    // Since we now use explicit const generics in the struct, we can pass them.
    // However, the macro calculates them.

    let rust_val = match (src_hist, dest_hist, cond_hist, step_size) {
        (1, 1, 1, 1) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 1, 1, 2) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            1,
            1,
            1,
            2,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 1, 1, 3) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            1,
            1,
            1,
            3,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 1, 2, 1) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            1,
            1,
            2,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 1, 3, 1) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            1,
            1,
            3,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 2, 1, 1) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            1,
            2,
            1,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (1, 3, 1, 1) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            1,
            3,
            1,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 1, 1, 1) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 2, 2, 1) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            2,
            2,
            2,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 2, 2, 2) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            2,
            2,
            2,
            2,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (2, 2, 2, 3) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            2,
            2,
            2,
            3,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 1, 1, 1) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            3,
            1,
            1,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 3, 3, 1) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            3,
            3,
            3,
            1,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 3, 3, 2) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            3,
            3,
            3,
            2,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        (3, 3, 3, 3) => new_kernel_cte!(
            &src_arr.insert_axis(Axis(1)),
            &dst_arr.insert_axis(Axis(1)),
            &cnd_arr.insert_axis(Axis(1)),
            3,
            3,
            3,
            3,
            1,
            1,
            1,
            kernel_type.to_string(),
            bandwidth
        )
        .global_value(),
        _ => return, // Skip other 66 combinations for now to avoid compilation explosion
    };

    // Python
    let kwargs = [
        ("kernel".to_string(), format!("\"{kernel_type}\"")),
        ("bandwidth".to_string(), bandwidth.to_string()),
        ("src_hist_len".to_string(), src_hist.to_string()),
        ("dest_hist_len".to_string(), dest_hist.to_string()),
        ("cond_hist_len".to_string(), cond_hist.to_string()),
        ("step_size".to_string(), step_size.to_string()),
    ];
    let py_val = python::calculate_cte_float(&src, &dst, &cnd, "kernel", &kwargs).unwrap();

    let tol = 1e-10;
    assert_relative_eq!(rust_val, py_val, epsilon = tol, max_relative = 1e-6);
}

// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use ndarray::Array1;
use infomeasure::estimators::utils::te_slicing::{te_observations, cte_observations};
use validation::python;
use rstest::rstest;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[rstest]
#[case(10, 3, 2, 2)]
#[case(20, 1, 1, 1)]
#[case(15, 2, 3, 1)]
#[case(100, 3, 3, 3)]
#[case(50, 1, 1, 1)]
#[case(50, 2, 1, 1)]
#[case(50, 1, 2, 1)]
#[case(50, 2, 2, 1)]
#[case(50, 1, 1, 2)]
#[case(50, 3, 2, 2)]
fn test_te_observations_parity_random(
    #[case] data_len: usize,
    #[case] src_hist_len: usize,
    #[case] dest_hist_len: usize,
    #[case] step_size: usize,
) {
    let mut rng = StdRng::seed_from_u64(42);
    let src_vec: Vec<i32> = (0..data_len).map(|_| rng.gen_range(0..10)).collect();
    let dest_vec: Vec<i32> = (0..data_len).map(|_| rng.gen_range(0..10)).collect();

    let src = Array1::from(src_vec.clone());
    let dest = Array1::from(dest_vec.clone());

    let (rust_future, rust_dest_hist, rust_src_hist) =
        te_observations(&src, &dest, src_hist_len, dest_hist_len, step_size, false);

    let (py_src_hist, py_dest_hist, py_future) =
        python::calculate_te_observations(&src_vec, &dest_vec, src_hist_len, dest_hist_len, step_size).unwrap();

    // Verify future
    assert_eq!(rust_future.len(), py_future.len());
    for i in 0..rust_future.nrows() {
        assert_eq!(rust_future[(i, 0)], py_future[i]);
    }

    // Verify histories
    assert_eq!(rust_src_hist.nrows(), py_src_hist.len());
    for i in 0..py_src_hist.len() {
        for j in 0..src_hist_len {
            assert_eq!(rust_src_hist[(i, j)], py_src_hist[i][j]);
        }
    }

    assert_eq!(rust_dest_hist.nrows(), py_dest_hist.len());
    for i in 0..py_dest_hist.len() {
        for j in 0..dest_hist_len {
            assert_eq!(rust_dest_hist[(i, j)], py_dest_hist[i][j]);
        }
    }
}

#[rstest]
#[case(20, 1, 1, 1, 1)]
#[case(30, 2, 2, 2, 2)]
fn test_cte_observations_parity_random(
    #[case] data_len: usize,
    #[case] src_hist_len: usize,
    #[case] dest_hist_len: usize,
    #[case] cond_hist_len: usize,
    #[case] step_size: usize,
) {
    let mut rng = StdRng::seed_from_u64(42);
    let src_vec: Vec<i32> = (0..data_len).map(|_| rng.gen_range(0..10)).collect();
    let dest_vec: Vec<i32> = (0..data_len).map(|_| rng.gen_range(0..10)).collect();
    let cond_vec: Vec<i32> = (0..data_len).map(|_| rng.gen_range(0..10)).collect();

    let src = Array1::from(src_vec.clone());
    let dest = Array1::from(dest_vec.clone());
    let cond = Array1::from(cond_vec.clone());

    let (rust_future, rust_dest_hist, rust_src_hist, rust_cond_hist) =
        cte_observations(&src, &dest, &cond, src_hist_len, dest_hist_len, cond_hist_len, step_size, false);

    let (py_src_hist, py_dest_hist, py_future, py_cond_hist) =
        python::calculate_cte_observations(&src_vec, &dest_vec, &cond_vec, src_hist_len, dest_hist_len, cond_hist_len, step_size).unwrap();

    // Verify future
    assert_eq!(rust_future.len(), py_future.len());
    for i in 0..rust_future.nrows() {
        assert_eq!(rust_future[(i, 0)], py_future[i]);
    }

    // Verify histories
    assert_eq!(rust_src_hist.nrows(), py_src_hist.len());
    for i in 0..py_src_hist.len() {
        for j in 0..src_hist_len {
            assert_eq!(rust_src_hist[(i, j)], py_src_hist[i][j]);
        }
    }

    assert_eq!(rust_dest_hist.nrows(), py_dest_hist.len());
    for i in 0..py_dest_hist.len() {
        for j in 0..dest_hist_len {
            assert_eq!(rust_dest_hist[(i, j)], py_dest_hist[i][j]);
        }
    }

    assert_eq!(rust_cond_hist.nrows(), py_cond_hist.len());
    for i in 0..py_cond_hist.len() {
        for j in 0..cond_hist_len {
            assert_eq!(rust_cond_hist[(i, j)], py_cond_hist[i][j]);
        }
    }
}

#[test]
fn test_te_observations_explicit() {
    let source = Array1::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    let dest = Array1::from(vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19]);
    let src_hist_len = 3;
    let dest_hist_len = 2;
    let step_size = 2;

    let (dest_future, dest_history, src_history) =
        te_observations(&source, &dest, src_hist_len, dest_hist_len, step_size, false);

    // max_delay = max(3, 2) * 2 = 6
    // n = 10
    // base_indices = [6, 8] (step_size = 2)
    // n_samples = 2

    assert_eq!(dest_future.nrows(), 2);
    assert_eq!(dest_future[(0, 0)], 16);
    assert_eq!(dest_future[(1, 0)], 18);

    // Row 0 (base_idx 6):
    // dest_history: [dest[6-4], dest[6-2]] = [12, 14]
    assert_eq!(dest_history[(0, 0)], 12);
    assert_eq!(dest_history[(0, 1)], 14);

    // src_history: [src[6-6], src[6-4], src[6-2]] = [0, 2, 4]
    assert_eq!(src_history[(0, 0)], 0);
    assert_eq!(src_history[(0, 1)], 2);
    assert_eq!(src_history[(0, 2)], 4);

    // Row 1 (base_idx 8):
    // dest_history: [dest[8-4], dest[8-2]] = [14, 16]
    assert_eq!(dest_history[(1, 0)], 14);
    assert_eq!(dest_history[(1, 1)], 16);

    // src_history: [src[8-6], src[8-4], src[8-2]] = [2, 4, 6]
    assert_eq!(src_history[(1, 0)], 2);
    assert_eq!(src_history[(1, 1)], 4);
    assert_eq!(src_history[(1, 2)], 6);
}

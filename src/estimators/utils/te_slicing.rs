use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::thread_rng;

pub fn te_observations_const<
    T: Clone + Default,
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_JOINT: usize,
    const D_XP_YP: usize,
    const D_YP: usize,
    const D_YF_YP: usize,
>(
    source: &Array2<T>,
    destination: &Array2<T>,
    permute_src: bool,
) -> (Array2<T>, Array2<T>, Array2<T>) {
    let max_delay = SRC_HIST.max(DEST_HIST) * STEP_SIZE;
    let n = destination.nrows();

    if max_delay >= n {
        return (
            Array2::default((0, D_TARGET)),
            Array2::default((0, DEST_HIST * D_TARGET)),
            Array2::default((0, SRC_HIST * D_SOURCE)),
        );
    }

    let base_indices: Vec<usize> = (max_delay..n).step_by(STEP_SIZE).collect();
    let n_samples = base_indices.len();

    let mut dest_future = Array2::default((n_samples, D_TARGET));
    let mut dest_history = Array2::default((n_samples, DEST_HIST * D_TARGET));
    let mut src_history = Array2::default((n_samples, SRC_HIST * D_SOURCE));

    for (idx, &base_idx) in base_indices.iter().enumerate() {
        for d in 0..D_TARGET {
            dest_future[(idx, d)] = destination[(base_idx, d)].clone();
        }

        for j in 0..DEST_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_TARGET {
                dest_history[(idx, (DEST_HIST - 1 - j) * D_TARGET + d)] =
                    destination[(base_idx - offset, d)].clone();
            }
        }

        for j in 0..SRC_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_SOURCE {
                src_history[(idx, (SRC_HIST - 1 - j) * D_SOURCE + d)] =
                    source[(base_idx - offset, d)].clone();
            }
        }
    }

    if permute_src {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let mut permuted_src_history = Array2::default((n_samples, SRC_HIST * D_SOURCE));
        for (i, &new_idx) in indices.iter().enumerate() {
            for j in 0..SRC_HIST * D_SOURCE {
                permuted_src_history[(i, j)] = src_history[(new_idx, j)].clone();
            }
        }
        src_history = permuted_src_history;
    }

    (dest_future, dest_history, src_history)
}

/// Slices the data for CTE and optionally permutes the source history (const generic version).
pub fn cte_observations_const<
    T: Clone + Default,
    const SRC_HIST: usize,
    const DEST_HIST: usize,
    const COND_HIST: usize,
    const STEP_SIZE: usize,
    const D_SOURCE: usize,
    const D_TARGET: usize,
    const D_COND: usize,
    const D_JOINT: usize,
    const D_XP_YP_ZP: usize,
    const D_YP_ZP: usize,
    const D_YF_YP_ZP: usize,
>(
    source: &Array2<T>,
    destination: &Array2<T>,
    condition: &Array2<T>,
    permute_src: bool,
) -> (Array2<T>, Array2<T>, Array2<T>, Array2<T>) {
    let max_delay = SRC_HIST.max(DEST_HIST).max(COND_HIST) * STEP_SIZE;
    let n = destination.nrows();

    if max_delay >= n {
        return (
            Array2::default((0, D_TARGET)),
            Array2::default((0, DEST_HIST * D_TARGET)),
            Array2::default((0, SRC_HIST * D_SOURCE)),
            Array2::default((0, COND_HIST * D_COND)),
        );
    }

    let base_indices: Vec<usize> = (max_delay..n).step_by(STEP_SIZE).collect();
    let n_samples = base_indices.len();

    let mut dest_future = Array2::default((n_samples, D_TARGET));
    let mut dest_history = Array2::default((n_samples, DEST_HIST * D_TARGET));
    let mut src_history = Array2::default((n_samples, SRC_HIST * D_SOURCE));
    let mut cond_history = Array2::default((n_samples, COND_HIST * D_COND));

    for (idx, &base_idx) in base_indices.iter().enumerate() {
        for d in 0..D_TARGET {
            dest_future[(idx, d)] = destination[(base_idx, d)].clone();
        }

        for j in 0..DEST_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_TARGET {
                dest_history[(idx, (DEST_HIST - 1 - j) * D_TARGET + d)] =
                    destination[(base_idx - offset, d)].clone();
            }
        }

        for j in 0..SRC_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_SOURCE {
                src_history[(idx, (SRC_HIST - 1 - j) * D_SOURCE + d)] =
                    source[(base_idx - offset, d)].clone();
            }
        }

        for j in 0..COND_HIST {
            let offset = (j + 1) * STEP_SIZE;
            for d in 0..D_COND {
                cond_history[(idx, (COND_HIST - 1 - j) * D_COND + d)] =
                    condition[(base_idx - offset, d)].clone();
            }
        }
    }

    if permute_src {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let mut permuted_src_history = Array2::default((n_samples, SRC_HIST * D_SOURCE));
        for (i, &new_idx) in indices.iter().enumerate() {
            for j in 0..SRC_HIST * D_SOURCE {
                permuted_src_history[(i, j)] = src_history[(new_idx, j)].clone();
            }
        }
        src_history = permuted_src_history;
    }

    (dest_future, dest_history, src_history, cond_history)
}

/// Slice source and destination data into future and history components for TE.
///
/// Returns (dest_future, dest_history, src_history).
/// - dest_future: Y_{t+1} (shape Nx1)
/// - dest_history: Y_{t}, Y_{t-tau}, ..., Y_{t-(k-1)tau} (shape N x dest_hist_len)
/// - src_history: X_{t}, X_{t-tau}, ..., X_{t-(l-1)tau} (shape N x src_hist_len)
pub fn te_slices<T: Clone + Default>(
    source: &Array1<T>,
    destination: &Array1<T>,
    src_hist_len: usize,
    dest_hist_len: usize,
    step_size: usize,
) -> (Array2<T>, Array2<T>, Array2<T>) {
    te_observations(
        source,
        destination,
        src_hist_len,
        dest_hist_len,
        step_size,
        false,
    )
}

/// Slices the data and optionally permutes the source history.
pub fn te_observations<T: Clone + Default>(
    source: &Array1<T>,
    destination: &Array1<T>,
    src_hist_len: usize,
    dest_hist_len: usize,
    step_size: usize,
    permute_src: bool,
) -> (Array2<T>, Array2<T>, Array2<T>) {
    let max_delay = src_hist_len.max(dest_hist_len) * step_size;
    let n = destination.len();

    if max_delay >= n {
        return (
            Array2::default((0, 1)),
            Array2::default((0, dest_hist_len)),
            Array2::default((0, src_hist_len)),
        );
    }

    let base_indices: Vec<usize> = (max_delay..n).step_by(step_size).collect();
    let n_samples = base_indices.len();

    let mut dest_future = Array2::default((n_samples, 1));
    let mut dest_history = Array2::default((n_samples, dest_hist_len));
    let mut src_history = Array2::default((n_samples, src_hist_len));

    for (idx, &base_idx) in base_indices.iter().enumerate() {
        dest_future[(idx, 0)] = destination[base_idx].clone();

        for j in 0..dest_hist_len {
            let offset = (j + 1) * step_size;
            dest_history[(idx, dest_hist_len - 1 - j)] = destination[base_idx - offset].clone();
        }

        for j in 0..src_hist_len {
            let offset = (j + 1) * step_size;
            src_history[(idx, src_hist_len - 1 - j)] = source[base_idx - offset].clone();
        }
    }

    if permute_src {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let mut permuted_src_history = Array2::default((n_samples, src_hist_len));
        for (i, &new_idx) in indices.iter().enumerate() {
            for j in 0..src_hist_len {
                permuted_src_history[(i, j)] = src_history[(new_idx, j)].clone();
            }
        }
        src_history = permuted_src_history;
    }

    (dest_future, dest_history, src_history)
}

/// Slice data into future, history and condition components for CTE.
pub fn cte_slices<T: Clone + Default>(
    source: &Array1<T>,
    destination: &Array1<T>,
    condition: &Array1<T>,
    src_hist_len: usize,
    dest_hist_len: usize,
    cond_hist_len: usize,
    step_size: usize,
) -> (Array2<T>, Array2<T>, Array2<T>, Array2<T>) {
    cte_observations(
        source,
        destination,
        condition,
        src_hist_len,
        dest_hist_len,
        cond_hist_len,
        step_size,
        false,
    )
}

/// Slices the data for CTE and optionally permutes the source history.
pub fn cte_observations<T: Clone + Default>(
    source: &Array1<T>,
    destination: &Array1<T>,
    condition: &Array1<T>,
    src_hist_len: usize,
    dest_hist_len: usize,
    cond_hist_len: usize,
    step_size: usize,
    permute_src: bool,
) -> (Array2<T>, Array2<T>, Array2<T>, Array2<T>) {
    let max_delay = src_hist_len.max(dest_hist_len).max(cond_hist_len) * step_size;
    let n = destination.len();

    if max_delay >= n {
        return (
            Array2::default((0, 1)),
            Array2::default((0, dest_hist_len)),
            Array2::default((0, src_hist_len)),
            Array2::default((0, cond_hist_len)),
        );
    }

    let base_indices: Vec<usize> = (max_delay..n).step_by(step_size).collect();
    let n_samples = base_indices.len();

    let mut dest_future = Array2::default((n_samples, 1));
    let mut dest_history = Array2::default((n_samples, dest_hist_len));
    let mut src_history = Array2::default((n_samples, src_hist_len));
    let mut cond_history = Array2::default((n_samples, cond_hist_len));

    for (idx, &base_idx) in base_indices.iter().enumerate() {
        dest_future[(idx, 0)] = destination[base_idx].clone();

        for j in 0..dest_hist_len {
            let offset = (j + 1) * step_size;
            dest_history[(idx, dest_hist_len - 1 - j)] = destination[base_idx - offset].clone();
        }

        for j in 0..src_hist_len {
            let offset = (j + 1) * step_size;
            src_history[(idx, src_hist_len - 1 - j)] = source[base_idx - offset].clone();
        }

        for j in 0..cond_hist_len {
            let offset = (j + 1) * step_size;
            cond_history[(idx, cond_hist_len - 1 - j)] = condition[base_idx - offset].clone();
        }
    }

    if permute_src {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let mut permuted_src_history = Array2::default((n_samples, src_hist_len));
        for (i, &new_idx) in indices.iter().enumerate() {
            for j in 0..src_hist_len {
                permuted_src_history[(i, j)] = src_history[(new_idx, j)].clone();
            }
        }
        src_history = permuted_src_history;
    }

    (dest_future, dest_history, src_history, cond_history)
}

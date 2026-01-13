#![cfg(feature = "gpu_support")]

use ndarray::{array, Array2};
use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::discrete::mle::DiscreteEntropy;
use infomeasure::estimators::{GlobalValue, LocalValues};

// This is a smoke test to exercise the GPU histogram path for DiscreteEntropy::from_rows.
// It is ignored by default as not all CI environments have a usable GPU.
#[test]
#[ignore]
fn discrete_gpu_batch_matches_cpu() {
    let data: Array2<i32> = array![
        [0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9],
        [5,5,5,5,5,6,6,6,7,7,7,8,8,9,9,9,9,9,9,9],
        [1,1,1,2,2,3,3,3,4,4,4,4,0,0,0,0,0,0,0,0],
    ];

    // CPU per-row baseline
    let cpu: Vec<f64> = data
        .rows()
        .into_iter()
        .map(|r| DiscreteEntropy::new(r.to_owned()).global_value())
        .collect();

    // Attempt GPU-accelerated batch
    let batch = DiscreteEntropy::from_rows(data.clone());
    for (i, est) in batch.iter().enumerate() {
        let g = est.global_value();
        assert_abs_diff_eq!(g, cpu[i], epsilon = 1e-12);
    }
}

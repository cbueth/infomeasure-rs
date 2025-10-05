use approx::assert_abs_diff_eq;
use ndarray::{array, Array1, Array2};
use std::panic;

use infomeasure::estimators::approaches::common_nd::dataset::NdDataset;
use infomeasure::estimators::approaches::expfam::utils::knn_radii;

fn brute_force_lp<const K: usize>(points: &[[f64; K]], p: f64, i: usize, j: usize) -> f64 {
    let mut acc = 0.0;
    for d in 0..K {
        acc += (points[i][d] - points[j][d]).abs().powf(p);
    }
    acc.powf(1.0 / p)
}

#[test]
fn nd_dataset_from_array_construction() {
    // 2D data
    let data: Array2<f64> = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ];
    let ds = NdDataset::<2>::from_array2(data.clone());
    assert_eq!(ds.n, 4);
    assert_eq!(ds.points.len(), 4);
    // spot-check point contents
    assert_abs_diff_eq!(ds.points[2][1], 1.0, epsilon = 1e-12);

    // 1D via Array1
    let x: Array1<f64> = Array1::from(vec![0.0, 2.0, 5.0]);
    let ds1 = NdDataset::<1>::from_array1(x.clone());
    assert_eq!(ds1.n, 3);
    assert_abs_diff_eq!(ds1.points[1][0], 2.0, epsilon = 1e-12);
}

#[test]
fn nd_dataset_kth_neighbor_radii_euclidean_matches_utils_knn_radii() {
    let data: Array2<f64> = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ];
    let ds = NdDataset::<2>::from_array2(data.clone());
    for &k in &[1usize, 2, 3] {
        let r_ds = ds.kth_neighbor_radii_euclidean(k);
        let r_utils = knn_radii::<2>(data.view(), k);
        assert_eq!(r_ds.len(), r_utils.len());
        for i in 0..r_ds.len() {
            assert_abs_diff_eq!(r_ds[i], r_utils[i], epsilon = 1e-12);
        }
    }
}

#[test]
fn nd_dataset_kth_neighbor_radii_manhattan_simple_cases() {
    // Points on a grid; L1 distances have easy values
    let data: Array2<f64> = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ];
    let ds = NdDataset::<2>::from_array2(data);

    // For [0,0], nearest is [1,0] or [0,1] at distance 1
    let r1 = ds.kth_neighbor_radii_manhattan(1);
    for &v in &r1 { assert_abs_diff_eq!(v, 1.0, epsilon = 1e-12); }

    // k=3 (farthest neighbor L1 distance = 2 for corners)
    let r3 = ds.kth_neighbor_radii_manhattan(3);
    // Two corners will have 2, centers would also yield 2 here (no centers present), so check values
    assert!(r3.iter().all(|&v| (v - 2.0).abs() < 1e-12));
}

#[test]
fn nd_dataset_kth_neighbor_radii_minkowski_fractional_p() {
    // Small set to validate fractional p with an independent brute-force in test
    let data: Array2<f64> = array![
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 3.0],
    ];
    let ds = NdDataset::<2>::from_array2(data.clone());
    let p = 1.5f64;
    let k = 1usize;

    // Dataset method (uses O(N^2) for non {1,2})
    let r = ds.kth_neighbor_radii_minkowski(p, k);

    // Independent brute-force in the test for expected radii
    let points = ds.points.clone();
    let mut exp = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let mut dists: Vec<f64> = Vec::new();
        for j in 0..points.len() { if i!=j { dists.push(brute_force_lp(&points, p, i, j)); } }
        dists.sort_by(|a,b| a.partial_cmp(b).unwrap());
        exp.push(dists[k-1]);
    }

    assert_eq!(r.len(), exp.len());
    for i in 0..r.len() {
        assert_abs_diff_eq!(r[i], exp[i], epsilon = 1e-12);
    }
}

#[test]
fn nd_dataset_edge_cases_and_panics() {
    // Empty data
    let empty = Array2::<f64>::zeros((0, 2));
    let ds = NdDataset::<2>::from_array2(empty);
    assert_eq!(ds.kth_neighbor_radii_euclidean(1).len(), 0);
    assert_eq!(ds.kth_neighbor_radii_manhattan(1).len(), 0);
    assert_eq!(ds.kth_neighbor_radii_minkowski(2.0, 1).len(), 0);

    // k too large
    let data: Array2<f64> = array![[0.0, 0.0],[1.0, 0.0]];
    let ds = NdDataset::<2>::from_array2(data);
    assert!(panic::catch_unwind(|| { let _ = ds.kth_neighbor_radii_euclidean(2); }).is_err());
    assert!(panic::catch_unwind(|| { let _ = ds.kth_neighbor_radii_manhattan(2); }).is_err());
    assert!(panic::catch_unwind(|| { let _ = ds.kth_neighbor_radii_minkowski(2.0, 2); }).is_err());

    // invalid p
    let data: Array2<f64> = array![[0.0, 0.0],[1.0, 0.0]];
    let ds = NdDataset::<2>::from_array2(data);
    assert!(panic::catch_unwind(|| { let _ = ds.kth_neighbor_radii_minkowski(0.0, 1); }).is_err());
}

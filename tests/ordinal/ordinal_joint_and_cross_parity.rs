use approx::assert_abs_diff_eq;
use ndarray::Array1;

use infomeasure::estimators::approaches::ordinal::ordinal::OrdinalEntropy;
use validation::python;
use rstest::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[rstest]
#[case(vec![1.0, 2.0, 3.0, 2.0, 1.0], vec![0.0, 1.0, 0.0, 1.0, 0.0])]
#[case(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5.0, 4.0, 3.0, 2.0, 1.0])]  // disjoint
#[case(vec![0.0, 0.0, 1.0, 1.0, 2.0], vec![2.0, 2.0, 1.0, 1.0, 0.0])]
#[case({
    let mut rng = StdRng::seed_from_u64(265473);
    (0..100).map(|_| rng.gen_range(0.0..10.0)).collect()
}, {
    let mut rng = StdRng::seed_from_u64(123);
    (0..100).map(|_| rng.gen_range(0.0..10.0)).collect()
})]
fn ordinal_joint_entropy_python_parity_basic(#[case] x_vec: Vec<f64>, #[case] y_vec: Vec<f64>) {
    // Two small series with simple patterns
    let x = Array1::from(x_vec);
    let y = Array1::from(y_vec);
    let order = 3usize;
    let step = 1usize;

    // Rust joint entropy
    let h_rust = OrdinalEntropy::joint_entropy(&[x.clone(), y.clone()], order, step, true);

    // Python joint entropy: reduce both to symbols then reduce_joint_space and compute discrete entropy
    let kwargs = vec![
        ("embedding_dim".to_string(), order.to_string()),
        ("stable".to_string(), "True".to_string()),
    ];

    // symbolize both using Python then compute discrete entropy of joint
    let h_py = python::calculate_ordinal_joint_entropy_two_float(x.as_slice().unwrap(), y.as_slice().unwrap(), &kwargs)
        .expect("python joint ordinal failed");

    assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-10);
    println!("Ordinal joint entropy (Rust vs Python): {:.10} vs {:.10}", h_rust, h_py);
}
#[test]
fn ordinal_cross_entropy_python_parity_overlap_and_disjoint() {
    // Overlapping supports
    let x = Array1::from(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    let y = Array1::from(vec![2.0, 3.0, 2.0, 1.0, 0.0]);
    let order = 2usize;
    let step = 1usize;

    let hcx_rust = OrdinalEntropy::cross_entropy(&x, &y, order, step, true);
    let kwargs = vec![
        ("embedding_dim".to_string(), order.to_string()),
        ("stable".to_string(), "True".to_string()),
    ];
    let hcx_py = python::calculate_ordinal_cross_entropy_two_float(x.as_slice().unwrap(), y.as_slice().unwrap(), &kwargs)
        .expect("python cross ordinal failed");
    assert_abs_diff_eq!(hcx_rust, hcx_py, epsilon = 1e-10);
    println!("Ordinal cross entropy (Rust vs Python): {:.10} vs {:.10}", hcx_rust, hcx_py);

    // Disjoint supports: craft sequences to create different single patterns
    // For order=3, monotonic increasing vs monotonic decreasing produce disjoint pattern sets of length 1 each.
    let a = Array1::from(vec![0.0, 1.0, 2.0, 3.0]); // strictly increasing -> only [0,1,2]
    let b = Array1::from(vec![3.0, 2.0, 1.0, 0.0]); // strictly decreasing -> only [2,1,0]
    let hcx_rust_disjoint = OrdinalEntropy::cross_entropy(&a, &b, 3, 1, true);
    let hcx_py_disjoint = python::calculate_ordinal_cross_entropy_two_float(a.as_slice().unwrap(), b.as_slice().unwrap(), &vec![
        ("embedding_dim".to_string(), 3.to_string()),
        ("stable".to_string(), "True".to_string()),
    ]).expect("python cross ordinal failed (disjoint)");
    assert_abs_diff_eq!(hcx_rust_disjoint, hcx_py_disjoint, epsilon = 1e-12);
    println!("Ordinal cross entropy (Rust vs Python, disjoint): {:.10} vs {:.10}", hcx_rust_disjoint, hcx_py_disjoint);
}

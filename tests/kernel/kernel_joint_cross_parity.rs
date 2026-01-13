use approx::assert_abs_diff_eq;
use infomeasure::estimators::approaches::kernel::KernelEntropy;
use infomeasure::estimators::{CrossEntropy, JointEntropy};
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rstest::rstest;
use validation::python;

fn generate_random_data(size: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(0.0..1.0)).collect()
}

fn flat_from_array2(a: &Array2<f64>) -> Vec<f64> {
    let mut v = Vec::with_capacity(a.len());
    for r in 0..a.nrows() {
        for c in 0..a.ncols() {
            v.push(a[(r, c)]);
        }
    }
    v
}

#[rstest]
#[case("box", 0.5, 100, 42)]
#[case("box", 1.0, 100, 43)]
#[case("gaussian", 0.5, 50, 44)]
#[case("gaussian", 1.0, 50, 45)]
fn kernel_joint_python_parity_2d(
    #[case] kernel: &str,
    #[case] bw: f64,
    #[case] size: usize,
    #[case] seed: u64,
) {
    let x_data = generate_random_data(size, seed);
    let y_data = generate_random_data(size, seed + 1000);
    let x = Array1::from(x_data);
    let y = Array1::from(y_data);

    let series = [x.clone(), y.clone()];
    let kernel_s = kernel.to_string();

    // Rust Joint Entropy
    let h_rust = KernelEntropy::<2>::joint_entropy(&series, (kernel_s.clone(), bw));

    // Python Joint Entropy
    let mut joined = Array2::zeros((x.len(), 2));
    for i in 0..x.len() {
        joined[[i, 0]] = x[i];
        joined[[i, 1]] = y[i];
    }
    let flat = flat_from_array2(&joined);
    let kwargs = vec![
        ("kernel".to_string(), format!("\"{}\"", kernel_s)),
        ("bandwidth".to_string(), format!("{}", bw)),
    ];
    let h_py = python::calculate_entropy_float_nd(&flat, 2, "kernel", &kwargs)
        .expect("python kernel failed");

    if kernel == "gaussian" {
        // Uncorrelated random data should have smaller discrepancy even with diagonal covariance
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 0.1);
    } else {
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
    }
}

#[rstest]
#[case("box", 0.5, 50, 46)]
#[case("box", 1.0, 50, 47)]
#[case("gaussian", 0.5, 30, 48)]
#[case("gaussian", 1.0, 30, 49)]
fn kernel_cross_python_parity_1d(
    #[case] kernel: &str,
    #[case] bw: f64,
    #[case] size: usize,
    #[case] seed: u64,
) {
    let p_vec = generate_random_data(size, seed);
    let q_vec = generate_random_data(size, seed + 1000);
    let p_data = Array1::from(p_vec);
    let q_data = Array1::from(q_vec);

    let kernel_s = kernel.to_string();

    let est_p = KernelEntropy::<1>::new_with_kernel_type(p_data.clone(), kernel_s.clone(), bw);
    let est_q = KernelEntropy::<1>::new_with_kernel_type(q_data.clone(), kernel_s.clone(), bw);

    let h_rust = est_p.cross_entropy(&est_q);

    let kwargs = vec![
        ("kernel".to_string(), format!("\"{}\"", kernel_s)),
        ("bandwidth".to_string(), format!("{}", bw)),
    ];
    let h_py = python::calculate_cross_entropy_float_nd(
        p_data.as_slice().unwrap(),
        q_data.as_slice().unwrap(),
        1,
        "kernel",
        &kwargs,
    )
    .expect("python cross kernel failed");

    if kernel == "gaussian" {
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 0.1);
    } else {
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
    }
}

#[rstest]
#[case("box", 0.5, 30, 50)]
#[case("box", 1.0, 30, 51)]
#[case("gaussian", 0.5, 20, 52)]
#[case("gaussian", 1.0, 20, 53)]
fn kernel_cross_python_parity_2d(
    #[case] kernel: &str,
    #[case] bw: f64,
    #[case] size: usize,
    #[case] seed: u64,
) {
    let p_x = generate_random_data(size, seed);
    let p_y = generate_random_data(size, seed + 1);
    let q_x = generate_random_data(size, seed + 1000);
    let q_y = generate_random_data(size, seed + 1001);

    let mut p_vec = Vec::new();
    for i in 0..size {
        p_vec.push([p_x[i], p_y[i]]);
    }
    let mut q_vec = Vec::new();
    for i in 0..size {
        q_vec.push([q_x[i], q_y[i]]);
    }

    let p_data =
        Array2::from_shape_vec((p_vec.len(), 2), p_vec.into_iter().flatten().collect()).unwrap();
    let q_data =
        Array2::from_shape_vec((q_vec.len(), 2), q_vec.into_iter().flatten().collect()).unwrap();

    let kernel_s = kernel.to_string();

    let est_p = KernelEntropy::<2>::new_with_kernel_type(p_data.clone(), kernel_s.clone(), bw);
    let est_q = KernelEntropy::<2>::new_with_kernel_type(q_data.clone(), kernel_s.clone(), bw);

    let h_rust = est_p.cross_entropy(&est_q);

    let flat_p = flat_from_array2(&p_data);
    let flat_q = flat_from_array2(&q_data);

    let kwargs = vec![
        ("kernel".to_string(), format!("\"{}\"", kernel_s)),
        ("bandwidth".to_string(), format!("{}", bw)),
    ];
    let h_py = python::calculate_cross_entropy_float_nd(&flat_p, &flat_q, 2, "kernel", &kwargs)
        .expect("python cross kernel failed");

    if kernel == "gaussian" {
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 0.5);
    } else {
        assert_abs_diff_eq!(h_rust, h_py, epsilon = 1e-8);
    }
}

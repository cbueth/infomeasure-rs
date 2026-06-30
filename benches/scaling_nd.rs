#![allow(unused_imports)]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::entropy::GlobalValue;
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Duration;

mod utils;

use utils::data;

const N: usize = 1000;
const SEED: u64 = 42;
const NOISE_LEVEL: f64 = 1e-10;

fn generate_nd_data(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let flat = data::generate_gaussian_nd(rows, cols, 0.0, 1.0, seed);
    data::to_array2(&flat, rows, cols)
}

fn generate_correlated_nd(
    rows: usize,
    cols1: usize,
    cols2: usize,
    correlation: f64,
    seed: u64,
) -> (Array2<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut x = Vec::with_capacity(rows * cols1);
    let mut y = Vec::with_capacity(rows * cols2);
    for _ in 0..rows {
        for _ in 0..cols1 {
            x.push(normal.sample(&mut rng));
        }
        for _ in 0..cols2 {
            let z: f64 = normal.sample(&mut rng);
            let w: f64 = normal.sample(&mut rng);
            y.push(correlation * z + (1.0 - correlation.powi(2)).sqrt() * w);
        }
    }
    (
        Array2::from_shape_vec((rows, cols1), x).unwrap(),
        Array2::from_shape_vec((rows, cols2), y).unwrap(),
    )
}

macro_rules! dim_scaling_bench_ksg_mi {
    ($name:ident, $d1:expr, $d2:expr) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group("scaling_ksg_mi");
            group.measurement_time(Duration::from_secs(3));
            let k = 3usize;
            let dim_label = format!("{}d", $d1 + $d2);
            let (x, y) = generate_correlated_nd(N, $d1, $d2, 0.5, SEED);
            let id = BenchmarkId::new(&dim_label, N);
            group.bench_with_input(id, &N, |b, _| {
                b.iter(|| {
                    let mi = MutualInformation::nd_ksg::<{ $d1 + $d2 }, $d1, $d2>(
                        &[x.clone(), y.clone()],
                        k,
                        NOISE_LEVEL,
                    );
                    black_box(mi.global_value())
                });
            });
            group.finish();
        }
    };
}

macro_rules! dim_scaling_bench_kernel_mi {
    ($name:ident, $d1:expr, $d2:expr) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group("scaling_kernel_mi");
            group.measurement_time(Duration::from_secs(3));
            let bw = 0.5;
            let dim_label = format!("{}d", $d1 + $d2);
            let (x, y) = generate_correlated_nd(N, $d1, $d2, 0.5, SEED);
            let id = BenchmarkId::new(&dim_label, N);
            group.bench_with_input(id, &N, |b, _| {
                b.iter(|| {
                    let mi = MutualInformation::nd_kernel::<{ $d1 + $d2 }, $d1, $d2>(
                        &[x.clone(), y.clone()],
                        bw,
                    );
                    black_box(mi.global_value())
                });
            });
            group.finish();
        }
    };
}

macro_rules! dim_scaling_bench_ksg_te {
    ($name:ident, $dsource:expr, $dtarget:expr) => {
        fn $name(c: &mut Criterion) {
            let mut group = c.benchmark_group("scaling_ksg_te");
            group.measurement_time(Duration::from_secs(3));
            let k = 3usize;
            let dim_label = format!("{}d_s_{}d_t", $dsource, $dtarget);
            let source = generate_nd_data(N, $dsource, SEED);
            let target = generate_nd_data(N, $dtarget, SEED + 1);
            let id = BenchmarkId::new(&dim_label, N);
            group.bench_with_input(id, &N, |b, _| {
                b.iter(|| {
                    let te = TransferEntropy::nd_ksg::<
                        1,
                        1,
                        1,
                        $dsource,
                        $dtarget,
                        { $dsource + $dtarget },
                        { $dsource + $dtarget },
                        $dtarget,
                        $dtarget,
                    >(&source, &target, k, NOISE_LEVEL);
                    black_box(te.global_value())
                });
            });
            group.finish();
        }
    };
}

fn black_box<T>(t: T) -> T {
    use std::hint::black_box;
    black_box(t)
}

// KSG MI: fixed N=1000, vary total D = 1,2,3,5,10 (split D1/D2)
dim_scaling_bench_ksg_mi!(bench_ksg_mi_d1, 1, 1);
dim_scaling_bench_ksg_mi!(bench_ksg_mi_d2, 1, 2);
dim_scaling_bench_ksg_mi!(bench_ksg_mi_d3, 2, 2);
dim_scaling_bench_ksg_mi!(bench_ksg_mi_d5, 3, 3);
dim_scaling_bench_ksg_mi!(bench_ksg_mi_d10, 5, 6);

// Kernel MI: fixed N=1000, vary total D = 1,2,3,5,10
dim_scaling_bench_kernel_mi!(bench_kernel_mi_d1, 1, 1);
dim_scaling_bench_kernel_mi!(bench_kernel_mi_d2, 1, 2);
dim_scaling_bench_kernel_mi!(bench_kernel_mi_d3, 2, 2);
dim_scaling_bench_kernel_mi!(bench_kernel_mi_d5, 3, 3);
dim_scaling_bench_kernel_mi!(bench_kernel_mi_d10, 5, 6);

// KSG TE: fixed N=1000, vary source/target dimensions
dim_scaling_bench_ksg_te!(bench_ksg_te_d1, 1, 1);
dim_scaling_bench_ksg_te!(bench_ksg_te_d2, 2, 2);
dim_scaling_bench_ksg_te!(bench_ksg_te_d3, 3, 3);

criterion_group!(
    benches,
    bench_ksg_mi_d1,
    bench_ksg_mi_d2,
    bench_ksg_mi_d3,
    bench_ksg_mi_d5,
    bench_ksg_mi_d10,
    bench_kernel_mi_d1,
    bench_kernel_mi_d2,
    bench_kernel_mi_d3,
    bench_kernel_mi_d5,
    bench_kernel_mi_d10,
    bench_ksg_te_d1,
    bench_ksg_te_d2,
    bench_ksg_te_d3,
);
criterion_main!(benches);

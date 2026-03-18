#![allow(unused_imports)]
#[cfg(feature = "gpu")]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(feature = "gpu")]
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
#[cfg(feature = "gpu")]
use ndarray::Array1;
#[cfg(feature = "gpu")]
use rand::Rng;
#[cfg(feature = "gpu")]
use rand::SeedableRng;
#[cfg(feature = "gpu")]
use rand::rngs::StdRng;
#[cfg(feature = "gpu")]
use rand_distr::{Distribution, Normal};
#[cfg(feature = "gpu")]
use std::time::Duration;

#[cfg(feature = "gpu")]
mod utils;

#[cfg(feature = "gpu")]
fn bench_kernel_entropy_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_kernel_gpu");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 5000];
    let bandwidths = [0.1, 0.5, 1.0];
    let seed = 42u64;

    for bw in bandwidths {
        for size in sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();

            let id = BenchmarkId::new(format!("bw{}", bw.to_string().replace('.', "_")), size);
            group.bench_with_input(id, &(bw, size), |b, _| {
                let arr = Array1::from(data.clone());
                b.iter(|| {
                    let entropy =
                        Entropy::new_kernel_with_type(arr.clone(), "gaussian".to_string(), bw);
                    black_box(entropy.global_value())
                });
            });
        }
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_kernel_mi_gpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_kernel_gpu");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 5000];
    let bandwidths = [0.1, 0.5, 1.0];
    let seed = 42u64;

    for bw in bandwidths {
        for size in sizes {
            let (x, y) = generate_correlated(size, 0.5, seed);
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);

            let id = BenchmarkId::new(format!("bw{}", bw.to_string().replace('.', "_")), size);
            group.bench_with_input(id, &(bw, size), |b, _| {
                b.iter(|| {
                    use infomeasure::estimators::mutual_information::MutualInformation;
                    let mi =
                        infomeasure::estimators::mutual_information::MutualInformation::new_kernel(
                            &[x_arr.clone(), y_arr.clone()],
                            bw,
                        );
                    black_box(mi.global_value())
                });
            });
        }
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn generate_correlated(size: usize, correlation: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    use rand_distr::Normal;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(size);
    let mut y = Vec::with_capacity(size);

    for _ in 0..size {
        let z: f64 = rng.sample(Normal::new(0.0, 1.0).unwrap());
        let w: f64 = rng.sample(Normal::new(0.0, 1.0).unwrap());
        let xi = z;
        let yi = correlation * z + (1.0 - correlation.powi(2)).sqrt() * w;
        x.push(xi);
        y.push(yi);
    }

    (x, y)
}

#[cfg(feature = "gpu")]
fn black_box<T>(t: T) -> T {
    use std::hint::black_box;
    black_box(t)
}

#[cfg(feature = "gpu")]
criterion_group!(benches, bench_kernel_entropy_gpu, bench_kernel_mi_gpu);

#[cfg(feature = "gpu")]
criterion_main!(benches);

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("GPU benchmarks require the 'gpu' feature to be enabled.");
    println!("Run with: cargo bench --bench kernel_gpu --features gpu");
}

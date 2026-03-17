use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::entropy::GlobalValue;
use infomeasure::estimators::mutual_information::MutualInformation;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Duration;

mod utils;

fn generate_correlated(size: usize, correlation: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
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

fn bench_mi_renyi(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_renyi");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 5000];
    let alphas: [f64; 3] = [0.5, 1.0, 2.0];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for alpha in alphas {
            for size in sizes {
                let (x, y) = generate_correlated(size, 0.5, seed);
                let x_arr = Array1::from(x);
                let y_arr = Array1::from(y);

                let id = BenchmarkId::new(
                    format!("k{}_alpha{}", k, alpha.to_string().replace('.', "_")),
                    size,
                );
                group.bench_with_input(id, &(k, alpha, size), |b, _| {
                    b.iter(|| {
                        let mi = MutualInformation::new_renyi(
                            &[x_arr.clone(), y_arr.clone()],
                            k,
                            alpha,
                            noise_level,
                        );
                        black_box(mi.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_mi_tsallis(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_tsallis");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 5000];
    let qs: [f64; 3] = [0.5, 1.5, 2.0];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for q in qs {
            for size in sizes {
                let (x, y) = generate_correlated(size, 0.5, seed);
                let x_arr = Array1::from(x);
                let y_arr = Array1::from(y);

                let id =
                    BenchmarkId::new(format!("k{}_q{}", k, q.to_string().replace('.', "_")), size);
                group.bench_with_input(id, &(k, q, size), |b, _| {
                    b.iter(|| {
                        let mi = MutualInformation::new_tsallis(
                            &[x_arr.clone(), y_arr.clone()],
                            k,
                            q,
                            noise_level,
                        );
                        black_box(mi.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_mi_kl(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_kl");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 5000];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for size in sizes {
            let (x, y) = generate_correlated(size, 0.5, seed);
            let x_arr = Array1::from(x);
            let y_arr = Array1::from(y);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let mi =
                        MutualInformation::new_kl(&[x_arr.clone(), y_arr.clone()], k, noise_level);
                    black_box(mi.global_value())
                });
            });
        }
    }

    group.finish();
}

fn black_box<T>(t: T) -> T {
    use std::hint::black_box;
    black_box(t)
}

criterion_group!(benches, bench_mi_renyi, bench_mi_tsallis, bench_mi_kl);
criterion_main!(benches);

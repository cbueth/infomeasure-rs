use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use ndarray::{Array1, Array2};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Duration;

mod utils;

fn bench_renyi_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_renyi");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let alphas: [f64; 3] = [0.5, 1.0, 2.0];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for alpha in alphas {
            for size in sizes {
                let mut rng = StdRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, 1.0).unwrap();
                let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
                let arr = Array1::from(data);

                let id = BenchmarkId::new(
                    format!("k{}_alpha{}", k, alpha.to_string().replace('.', "_")),
                    size,
                );
                group.bench_with_input(id, &(k, alpha, size), |b, _| {
                    b.iter(|| {
                        let entropy = Entropy::new_renyi_1d(arr.clone(), k, alpha, noise_level);
                        black_box(entropy.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_tsallis_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_tsallis");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let qs: [f64; 3] = [0.5, 1.5, 2.0];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for q in qs {
            for size in sizes {
                let mut rng = StdRng::seed_from_u64(seed);
                let normal = Normal::new(0.0, 1.0).unwrap();
                let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
                let arr = Array1::from(data);

                let id =
                    BenchmarkId::new(format!("k{}_q{}", k, q.to_string().replace('.', "_")), size);
                group.bench_with_input(id, &(k, q, size), |b, _| {
                    b.iter(|| {
                        let entropy = Entropy::new_tsallis_1d(arr.clone(), k, q, noise_level);
                        black_box(entropy.global_value())
                    });
                });
            }
        }
    }

    group.finish();
}

fn bench_kl_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_kl");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let ks = [1, 3, 5];
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for size in sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();
            let arr = Array1::from(data);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let entropy = Entropy::new_kl_1d(arr.clone(), k, noise_level);
                    black_box(entropy.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_kl_nd_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_kl_nd");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000];
    let dims = [2, 4, 8];
    let k = 3;
    let seed = 42u64;
    let noise_level = 1e-10;

    for dim in dims {
        for size in sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let normal = Normal::new(0.0, 1.0).unwrap();
            let data: Vec<f64> = (0..size * dim).map(|_| normal.sample(&mut rng)).collect();
            let arr = Array2::from_shape_vec((size, dim), data).unwrap();

            let id = BenchmarkId::new(format!("{}d", dim), size);
            group.bench_with_input(id, &(dim, size), |b, _| {
                b.iter(|| {
                    let entropy = Entropy::kl_nd::<2>(arr.clone(), k, noise_level);
                    black_box(entropy.global_value())
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

criterion_group!(
    benches,
    bench_renyi_entropy,
    bench_tsallis_entropy,
    bench_kl_entropy,
    bench_kl_nd_entropy
);
criterion_main!(benches);

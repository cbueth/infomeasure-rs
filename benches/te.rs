use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::entropy::GlobalValue;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::time::Duration;

mod utils;

fn generate_lagged_series(
    size: usize,
    coupling: f64,
    lag: usize,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut source = Vec::with_capacity(size + lag);
    let mut target = Vec::with_capacity(size);

    for _ in 0..size + lag {
        source.push(rng.sample(Normal::new(0.0, 1.0).unwrap()));
    }

    for i in 0..size {
        let noise = rng.sample(Normal::new(0.0, 1.0).unwrap());
        target.push(coupling * source[i + lag] + (1.0 - coupling.powi(2)).sqrt() * noise);
    }

    let source = source[..size].to_vec();
    (source, target)
}

fn bench_discrete_te(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_discrete");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let num_states = 5;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let source: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let mut target: Vec<i32> = Vec::with_capacity(size);
        for i in 1..size {
            target.push(if source[i - 1] == source[i] {
                source[i]
            } else {
                rng.gen_range(0..num_states)
            });
        }
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);

        let id = BenchmarkId::new("mle", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let te = TransferEntropy::new_discrete_mle(&source_arr, &target_arr, 1, 1, 1);
                black_box(te.global_value())
            });
        });
    }

    group.finish();
}

fn bench_kernel_te(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_kernel");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 5000];
    let bandwidths = [0.1, 0.5, 1.0];
    let lag = 1;
    let seed = 42u64;

    for bw in bandwidths {
        for size in sizes {
            let (source, target) = generate_lagged_series(size, 0.5, lag, seed);
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);

            let id = BenchmarkId::new(format!("bw_{}", bw.to_string().replace('.', "_")), size);
            group.bench_with_input(id, &(bw, size), |b, _| {
                b.iter(|| {
                    let te = TransferEntropy::new_kernel(&source_arr, &target_arr, 1, 1, 1, bw);
                    black_box(te.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_ksg_te(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_ksg");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 5000];
    let ks = [1, 3, 5];
    let lag = 1;
    let seed = 42u64;
    let noise_level = 1e-10;

    for k in ks {
        for size in sizes {
            let (source, target) = generate_lagged_series(size, 0.5, lag, seed);
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);

            let id = BenchmarkId::new(format!("k{}", k), size);
            group.bench_with_input(id, &(k, size), |b, _| {
                b.iter(|| {
                    let te =
                        TransferEntropy::new_ksg(&source_arr, &target_arr, 1, 1, 1, k, noise_level);
                    black_box(te.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_ordinal_te(c: &mut Criterion) {
    let mut group = c.benchmark_group("te_ordinal");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let orders = [2, 3, 4];
    let lag = 1;
    let seed = 42u64;

    for order in orders {
        for size in sizes {
            let (source, target) = generate_lagged_series(size, 0.5, lag, seed);
            let source_arr = Array1::from(source);
            let target_arr = Array1::from(target);

            let id = BenchmarkId::new(format!("order_{}", order), size);
            group.bench_with_input(id, &(order, size), |b, _| {
                b.iter(|| {
                    let te = TransferEntropy::new_ordinal(
                        &source_arr,
                        &target_arr,
                        order,
                        1,
                        1,
                        1,
                        false,
                    );
                    black_box(te.global_value())
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
    bench_discrete_te,
    bench_kernel_te,
    bench_ksg_te,
    bench_ordinal_te
);
criterion_main!(benches);

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Duration;

mod utils;

fn bench_ordinal_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_ordinal");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let orders = [2, 3, 4];
    let seed = 42u64;

    for order in orders {
        for size in sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let data: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..100.0)).collect();
            let arr = Array1::from(data);

            let id = BenchmarkId::new(format!("order_{}", order), size);
            group.bench_with_input(id, &(order, size), |b, _| {
                b.iter(|| {
                    let entropy = Entropy::new_ordinal(arr.clone(), order);
                    black_box(entropy.global_value())
                });
            });
        }
    }

    group.finish();
}

fn bench_ordinal_entropy_delay(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_ordinal_delay");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [1000, 10000];
    let delays = [1, 2, 3];
    let order = 3;
    let seed = 42u64;

    for delay in delays {
        for size in sizes {
            let mut rng = StdRng::seed_from_u64(seed);
            let data: Vec<f64> = (0..size).map(|_| rng.gen_range(0.0..100.0)).collect();
            let arr = Array1::from(data);

            let id = BenchmarkId::new(format!("delay_{}", delay), size);
            group.bench_with_input(id, &(delay, size), |b, _| {
                b.iter(|| {
                    let entropy = Entropy::new_ordinal_with_step(arr.clone(), order, delay);
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

criterion_group!(benches, bench_ordinal_entropy, bench_ordinal_entropy_delay);
criterion_main!(benches);

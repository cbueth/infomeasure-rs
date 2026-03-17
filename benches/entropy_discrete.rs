use criterion::{criterion_group, criterion_main};
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Duration;

mod utils;

fn bench_entropy_small(c: &mut criterion::Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_small");
    group.measurement_time(Duration::from_secs(3));

    let sizes = [100, 1000, 10000];
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..10)).collect();

        group.bench_with_input(
            criterion::BenchmarkId::new("discrete", size),
            &size,
            |b, &s| {
                let arr = Array1::from(data.clone());
                b.iter(|| {
                    let entropy = Entropy::new_discrete(arr.clone());
                    black_box(entropy.global_value())
                });
            },
        );
    }

    group.finish();
}

fn black_box<T>(t: T) -> T {
    use std::hint::black_box;
    black_box(t)
}

criterion_group!(benches, bench_entropy_small);
criterion_main!(benches);

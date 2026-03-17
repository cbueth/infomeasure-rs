use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::approaches::discrete::bayes::AlphaParam;
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Duration;

mod utils;

fn bench_discrete_miller_madow(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_miller_madow");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000, 100000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("miller_madow", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_miller_madow(arr.clone());
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_discrete_shrink(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_shrink");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000, 100000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("shrink", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_shrink(arr.clone());
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_discrete_chao_shen(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_chao_shen");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000, 100000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("chao_shen", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_chao_shen(arr.clone());
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_discrete_nsb(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_nsb");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("nsb", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_nsb(arr.clone(), None);
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_discrete_grassberger(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_grassberger");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000, 100000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("grassberger", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_grassberger(arr.clone());
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_discrete_zhang(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_zhang");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000, 100000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("zhang", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_zhang(arr.clone());
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_discrete_bayes(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_bayes");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000, 100000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("bayes", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_bayes(arr.clone(), AlphaParam::Laplace, None);
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_discrete_bonachela(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_bonachela");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("bonachela", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_bonachela(arr.clone());
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_discrete_ansb(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_ansb");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("ansb", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_ansb(arr.clone(), None);
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn bench_discrete_chao_wang_jost(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_discrete_chao_wang_jost");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000, 100000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let data: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let arr = Array1::from(data);

        let id = BenchmarkId::new("chao_wang_jost", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_chao_wang_jost(arr.clone());
                black_box(entropy.global_value())
            });
        });
    }

    group.finish();
}

fn black_box<T>(t: T) -> T {
    use std::hint::black_box;
    black_box(t)
}

criterion_group!(
    benches,
    bench_discrete_miller_madow,
    bench_discrete_shrink,
    bench_discrete_chao_shen,
    bench_discrete_nsb,
    bench_discrete_grassberger,
    bench_discrete_zhang,
    bench_discrete_bayes,
    bench_discrete_bonachela,
    bench_discrete_ansb,
    bench_discrete_chao_wang_jost
);
criterion_main!(benches);

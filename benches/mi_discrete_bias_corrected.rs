use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::entropy::GlobalValue;
use infomeasure::estimators::mutual_information::MutualInformation;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Duration;

mod utils;

fn bench_mi_discrete_miller_madow(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_miller_madow");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("miller_madow", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi =
                    MutualInformation::new_discrete_miller_madow(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_mi_discrete_shrink(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_shrink");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("shrink", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi = MutualInformation::new_discrete_shrink(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_mi_discrete_chao_shen(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_chao_shen");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("chao_shen", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi = MutualInformation::new_discrete_chao_shen(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_mi_discrete_nsb(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_nsb");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 5000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("nsb", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi = MutualInformation::new_discrete_nsb(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_mi_discrete_ansb(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_ansb");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 5000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("ansb", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi = MutualInformation::new_discrete_ansb(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_mi_discrete_bonachela(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_bonachela");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 5000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("bonachela", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi = MutualInformation::new_discrete_bonachela(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_mi_discrete_grassberger(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_grassberger");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("grassberger", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi =
                    MutualInformation::new_discrete_grassberger(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_mi_discrete_zhang(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_zhang");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("zhang", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi = MutualInformation::new_discrete_zhang(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_mi_discrete_bayes(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_bayes");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("bayes", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi = MutualInformation::new_discrete_bayes(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
            });
        });
    }

    group.finish();
}

fn bench_mi_discrete_chao_wang_jost(c: &mut Criterion) {
    let mut group = c.benchmark_group("mi_discrete_chao_wang_jost");
    group.measurement_time(Duration::from_secs(5));

    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42u64;

    for size in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let x: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let y: Vec<i32> = (0..size).map(|_| rng.gen_range(0..num_states)).collect();
        let x_arr = Array1::from(x);
        let y_arr = Array1::from(y);

        let id = BenchmarkId::new("chao_wang_jost", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let mi =
                    MutualInformation::new_discrete_chao_wang_jost(&[x_arr.clone(), y_arr.clone()]);
                black_box(mi.global_value())
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
    bench_mi_discrete_miller_madow,
    bench_mi_discrete_shrink,
    bench_mi_discrete_chao_shen,
    bench_mi_discrete_nsb,
    bench_mi_discrete_ansb,
    bench_mi_discrete_bonachela,
    bench_mi_discrete_grassberger,
    bench_mi_discrete_zhang,
    bench_mi_discrete_bayes,
    bench_mi_discrete_chao_wang_jost
);
criterion_main!(benches);

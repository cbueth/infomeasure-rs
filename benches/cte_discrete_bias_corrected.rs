use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use infomeasure::estimators::entropy::GlobalValue;
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Duration;

mod utils;

use utils::bench_sizes_extended;

fn bench_cte_discrete_miller_madow(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_miller_madow");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("miller_madow", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_miller_madow(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_cte_discrete_shrink(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_shrink");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("shrink", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_shrink(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_cte_discrete_chao_shen(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_chao_shen");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("chao_shen", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_chao_shen(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_cte_discrete_nsb(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_nsb");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("nsb", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_nsb(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_cte_discrete_ansb(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_ansb");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("ansb", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_ansb(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_cte_discrete_bonachela(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_bonachela");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("bonachela", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_bonachela(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_cte_discrete_grassberger(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_grassberger");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("grassberger", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_grassberger(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_cte_discrete_zhang(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_zhang");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("zhang", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_zhang(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_cte_discrete_bayes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_bayes");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("bayes", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_bayes(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
            });
        });
    }

    group.finish();
}

fn bench_cte_discrete_chao_wang_jost(c: &mut Criterion) {
    let mut group = c.benchmark_group("cte_discrete_chao_wang_jost");
    group.measurement_time(Duration::from_secs(5));

    let sizes = bench_sizes_extended();
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
        let cond: Vec<i32> = (0..size - 1)
            .map(|_| rng.gen_range(0..num_states))
            .collect();
        let source_arr = Array1::from(source[..size - 1].to_vec());
        let target_arr = Array1::from(target);
        let cond_arr = Array1::from(cond);

        let id = BenchmarkId::new("chao_wang_jost", size);
        group.bench_with_input(id, &size, |b, _| {
            b.iter(|| {
                let cte = TransferEntropy::new_cte_discrete_chao_wang_jost(
                    &source_arr,
                    &target_arr,
                    &cond_arr,
                    1,
                    1,
                    1,
                    1,
                );
                black_box(cte.global_value())
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
    bench_cte_discrete_miller_madow,
    bench_cte_discrete_shrink,
    bench_cte_discrete_chao_shen,
    bench_cte_discrete_nsb,
    bench_cte_discrete_ansb,
    bench_cte_discrete_bonachela,
    bench_cte_discrete_grassberger,
    bench_cte_discrete_zhang,
    bench_cte_discrete_bayes,
    bench_cte_discrete_chao_wang_jost
);
criterion_main!(benches);

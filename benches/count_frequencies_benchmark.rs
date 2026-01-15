use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::OpenOptions;
use std::io::Write;

use infomeasure::estimators::approaches::discrete::discrete_utils::count_frequencies_slice;

fn gen_data(size: usize, num_states: i32, offset: i32, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| rng.gen_range(0..num_states) + offset)
        .collect()
}

fn bench_count_frequencies(c: &mut Criterion) {
    // Configurations: various sizes and ranges (num_states)
    let sizes: &[usize] = &[1_000, 10_000, 100_000, 1_000_000];
    let states: &[i32] = &[16, 64, 256, 1024, 4096]; // up to MAX_DENSE_RANGE

    // CSV for results
    let mut csv = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("count_frequencies_benchmark_results.csv")
        .expect("open csv");
    writeln!(csv, "Size,NumStates,OffsetMode,TimeNsPerIter").unwrap();

    let mut group = c.benchmark_group("count_frequencies_slice dense vs hashmap");

    for &n in sizes {
        for &k in states {
            // Prepare two datasets with identical range but different min
            let data_dense = gen_data(n, k, 0, 12345); // min_v >= 0 -> dense path eligible
            let min_dense = *data_dense.iter().min().unwrap_or(&0);
            let max_dense = *data_dense.iter().max().unwrap_or(&0);
            let range = max_dense - min_dense;

            // For hashmap path, shift values so min < 0 while keeping same range
            // Choose offset so that min becomes negative by at least 1
            let shift_down = k.min(2048); // keep modest magnitude
            let data_hash = data_dense
                .iter()
                .map(|&v| v - shift_down)
                .collect::<Vec<i32>>();

            // Warmup
            let _ = count_frequencies_slice(&data_dense);
            let _ = count_frequencies_slice(&data_hash);

            // Benchmark dense-eligible (min >= 0)
            let id_dense = BenchmarkId::new(format!("N{n}_K{k}_R{range}"), "dense_min>=0");
            group.bench_with_input(id_dense, &n, |b, _| {
                b.iter(|| {
                    let map = count_frequencies_slice(black_box(&data_dense));
                    black_box(map.len())
                });
            });

            // Manual timing for CSV (dense)
            let iters = 20usize;
            let mut durations = Vec::with_capacity(iters);
            for _ in 0..iters {
                let start = std::time::Instant::now();
                let map = count_frequencies_slice(&data_dense);
                std::hint::black_box(map.len());
                durations.push(start.elapsed());
            }
            let avg_ns_dense: u128 =
                durations.iter().map(|d| d.as_nanos()).sum::<u128>() / (iters as u128);
            writeln!(csv, "{n},{k},dense,{avg_ns_dense}").unwrap();

            // Benchmark hashmap-forced (min < 0)
            let id_hash = BenchmarkId::new(format!("N{n}_K{k}_R{range}"), "hashmap_min<0");
            group.bench_with_input(id_hash, &n, |b, _| {
                b.iter(|| {
                    let map = count_frequencies_slice(black_box(&data_hash));
                    black_box(map.len())
                });
            });

            // Manual timing for CSV (hash)
            let mut durations_h = Vec::with_capacity(iters);
            for _ in 0..iters {
                let start = std::time::Instant::now();
                let map = count_frequencies_slice(&data_hash);
                std::hint::black_box(map.len());
                durations_h.push(start.elapsed());
            }
            let avg_ns_hash: u128 =
                durations_h.iter().map(|d| d.as_nanos()).sum::<u128>() / (iters as u128);
            writeln!(csv, "{n},{k},hashmap,{avg_ns_hash}").unwrap();
        }
    }

    group.finish();
}

criterion_group!(benches, bench_count_frequencies);
criterion_main!(benches);

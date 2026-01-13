use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::Write;
use std::time::Duration;
use validation::python;

/// Generate random data with specified size and number of possible states
fn generate_random_data(size: usize, num_states: i32, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(0..num_states)).collect()
}

/// Benchmark function for unified discrete entropy calculation
fn bench_unified_discrete_entropy(c: &mut Criterion) {
    // Define test parameters
    let sizes = [100, 1000, 10000, 100000, 1000000];
    let num_states = 10;
    let seed = 42;
    let python_num_runs = 10; // Number of runs for Python benchmarking

    // Create a CSV file to store the results
    let mut csv_file = File::create("unified_benchmark_results.csv").unwrap();
    writeln!(csv_file, "Test Type,Parameter,Rust Time (ns),Python Time (ns),Speedup Ratio,Rust Entropy,Python Entropy").unwrap();

    // Create a benchmark group for different data sizes
    let mut group = c.benchmark_group("Unified Discrete Entropy - Data Size");

    // Set measurement time to ensure accurate results
    group.measurement_time(Duration::from_secs(10));

    // Store Rust benchmark results for later comparison
    let mut rust_times = Vec::new();

    for &size in &sizes {
        // Generate random data
        let data = generate_random_data(size, num_states, seed);
        let data_array = Array1::from(data.clone());

        // Benchmark Rust implementation
        let rust_id = BenchmarkId::new("Rust", size);
        group.bench_with_input(rust_id.clone(), &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_discrete(black_box(data_array.clone()));
                black_box(entropy.global_value())
            });
        });

        // Get the Rust benchmark results from Criterion
        // For simplicity, we'll run our own measurement to get a comparable result
        let mut rust_durations = Vec::new();
        for _ in 0..100 {
            let start = std::time::Instant::now();
            let entropy = Entropy::new_discrete(data_array.clone());
            let _ = entropy.global_value();
            rust_durations.push(start.elapsed());
        }

        // Calculate average Rust time
        let rust_time_ns = rust_durations.iter().map(|d| d.as_nanos()).sum::<u128>()
            / rust_durations.len() as u128;

        rust_times.push(rust_time_ns);

        // Benchmark Python implementation
        let python_time = python::benchmark_entropy(&data, python_num_runs).unwrap();
        let python_time_ns = (python_time * 1_000_000_000.0) as u128;

        // Calculate entropy values for verification
        let rust_entropy = {
            let entropy = Entropy::new_discrete(data_array.clone());
            entropy.global_value()
        };
        let python_entropy = python::calculate_entropy(&data, "discrete", &[]).unwrap();

        // Calculate speedup ratio (how many times faster is Rust than Python)
        let speedup = python_time_ns as f64 / rust_time_ns as f64;

        // Print comparison for this data size
        println!("\n=== Data Size: {} elements ===", size);
        println!(
            "Rust execution time:   {:.9} seconds ({} ns)",
            rust_time_ns as f64 / 1_000_000_000.0,
            rust_time_ns
        );
        println!(
            "Python execution time: {:.9} seconds ({} ns)",
            python_time, python_time_ns
        );
        println!("Speedup (Python/Rust): {:.2}x", speedup);
        println!("Rust entropy value:   {}", rust_entropy);
        println!("Python entropy value: {}", python_entropy);

        // Write to CSV
        writeln!(
            csv_file,
            "Data Size,{},{},{},{:.2},{},{}",
            size, rust_time_ns, python_time_ns, speedup, rust_entropy, python_entropy
        )
        .unwrap();
    }
    group.finish();

    // Benchmark with different numbers of states
    let size = 1000;
    let states = [2, 5, 10, 20, 50, 100];

    let mut group = c.benchmark_group("Unified Discrete Entropy - Number of States");
    group.measurement_time(Duration::from_secs(10));

    for &num_states in &states {
        // Generate random data
        let data = generate_random_data(size, num_states, seed);
        let data_array = Array1::from(data.clone());

        // Benchmark Rust implementation
        let rust_id = BenchmarkId::new("Rust", num_states);
        group.bench_with_input(rust_id.clone(), &num_states, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_discrete(black_box(data_array.clone()));
                black_box(entropy.global_value())
            });
        });

        // Get the Rust benchmark results from Criterion
        // For simplicity, we'll run our own measurement to get a comparable result
        let mut rust_durations = Vec::new();
        for _ in 0..100 {
            let start = std::time::Instant::now();
            let entropy = Entropy::new_discrete(data_array.clone());
            let _ = entropy.global_value();
            rust_durations.push(start.elapsed());
        }

        // Calculate average Rust time
        let rust_time_ns = rust_durations.iter().map(|d| d.as_nanos()).sum::<u128>()
            / rust_durations.len() as u128;

        // Benchmark Python implementation
        let python_time = python::benchmark_entropy(&data, python_num_runs).unwrap();
        let python_time_ns = (python_time * 1_000_000_000.0) as u128;

        // Calculate entropy values for verification
        let rust_entropy = {
            let entropy = Entropy::new_discrete(data_array.clone());
            entropy.global_value()
        };
        let python_entropy = python::calculate_entropy(&data, "discrete", &[]).unwrap();

        // Calculate speedup ratio (how many times faster is Rust than Python)
        let speedup = python_time_ns as f64 / rust_time_ns as f64;

        // Print comparison for this number of states
        println!("\n=== Number of States: {} ===", num_states);
        println!(
            "Rust execution time:   {:.9} seconds ({} ns)",
            rust_time_ns as f64 / 1_000_000_000.0,
            rust_time_ns
        );
        println!(
            "Python execution time: {:.9} seconds ({} ns)",
            python_time, python_time_ns
        );
        println!("Speedup (Python/Rust): {:.2}x", speedup);
        println!("Rust entropy value:   {}", rust_entropy);
        println!("Python entropy value: {}", python_entropy);

        // Write to CSV
        writeln!(
            csv_file,
            "States,{},{},{},{:.2},{},{}",
            num_states, rust_time_ns, python_time_ns, speedup, rust_entropy, python_entropy
        )
        .unwrap();
    }
    group.finish();

    println!("\nBenchmark results have been saved to unified_benchmark_results.csv");
}

criterion_group!(benches, bench_unified_discrete_entropy);
criterion_main!(benches);

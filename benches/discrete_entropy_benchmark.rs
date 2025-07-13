use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use infomeasure::estimators::entropy::{Entropy, LocalValues};
use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Generate random data with specified size and number of possible states
fn generate_random_data(size: usize, num_states: i32, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| rng.gen_range(0..num_states))
        .collect()
}

/// Benchmark function for discrete entropy calculation
fn bench_discrete_entropy(c: &mut Criterion) {
    // Define test parameters
    let sizes = [100, 1000, 10000];
    let num_states = 10;
    let seed = 42;

    // Create a benchmark group for different data sizes
    let mut group = c.benchmark_group("Discrete Entropy - Data Size");
    
    for &size in &sizes {
        // Generate random data
        let data = generate_random_data(size, num_states, seed);
        let data_array = Array1::from(data.clone());
        
        // Benchmark with this data size
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_discrete(black_box(data_array.clone()));
                black_box(entropy.global_value())
            });
        });
    }
    group.finish();

    // Benchmark with different numbers of states
    let size = 1000;
    let states = [2, 5, 10, 20, 50, 100];
    
    let mut group = c.benchmark_group("Discrete Entropy - Number of States");
    
    for &num_states in &states {
        // Generate random data
        let data = generate_random_data(size, num_states, seed);
        let data_array = Array1::from(data.clone());
        
        // Benchmark with this number of states
        group.bench_with_input(BenchmarkId::from_parameter(num_states), &num_states, |b, _| {
            b.iter(|| {
                let entropy = Entropy::new_discrete(black_box(data_array.clone()));
                black_box(entropy.global_value())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_discrete_entropy);
criterion_main!(benches);
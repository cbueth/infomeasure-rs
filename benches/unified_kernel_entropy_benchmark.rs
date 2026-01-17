use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::Write;
use std::time::Duration;
use validation::python;

/// Generate random multi-dimensional data with specified size and dimensions from a normal distribution
fn generate_random_nd_data(size: usize, dims: usize, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..size * dims).map(|_| normal.sample(&mut rng)).collect()
}

/// Benchmark function for unified kernel entropy calculation
fn bench_unified_kernel_entropy(c: &mut Criterion) {
    // Define test parameters
    let sizes = [100, 2000, 10000];
    let dimensions = [1, 4, 8, 18];
    let kernel_types = ["box", "gaussian"];
    let bandwidth = 0.5;
    let seed = 385;
    let python_num_runs = 5; // Number of runs for Python benchmarking

    // Create a CSV file to store the results
    let mut csv_file = File::create("../internal/unified_kernel_benchmark_results.csv").unwrap();
    writeln!(csv_file, "Test Type,Parameter,Kernel,Dimensions,Rust Time (ns),Python Time (ns),Speedup Ratio,Rust Entropy,Python Entropy").unwrap();

    // Benchmark with different data sizes
    for &dims in &dimensions {
        for &kernel_type in &kernel_types {
            // Create a benchmark group for different data sizes
            let group_name =
                format!("Unified Kernel Entropy - {kernel_type} Kernel - {dims}D - Data Size");
            let mut group = c.benchmark_group(&group_name);

            // Set measurement time to ensure accurate results
            group.measurement_time(Duration::from_secs(3));

            for &size in &sizes {
                // Generate random data
                let data = generate_random_nd_data(size, dims, seed);

                // Create kwargs for Python
                let kernel_kwargs = vec![
                    ("kernel".to_string(), format!("\"{kernel_type}\"")),
                    ("bandwidth".to_string(), format!("{bandwidth}")),
                ];

                // Benchmark Rust implementation
                let rust_id = BenchmarkId::new("Rust", size);
                group.bench_with_input(rust_id.clone(), &size, |b, _| {
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    b.iter(|| {
                        let entropy = match dims {
                            1 => Entropy::nd_kernel_with_type::<1>(
                                black_box(data_array.clone()),
                                kernel_type.to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            2 => Entropy::nd_kernel_with_type::<2>(
                                black_box(data_array.clone()),
                                kernel_type.to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            3 => Entropy::nd_kernel_with_type::<3>(
                                black_box(data_array.clone()),
                                kernel_type.to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            4 => Entropy::nd_kernel_with_type::<4>(
                                black_box(data_array.clone()),
                                kernel_type.to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            8 => Entropy::nd_kernel_with_type::<8>(
                                black_box(data_array.clone()),
                                kernel_type.to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            18 => Entropy::nd_kernel_with_type::<18>(
                                black_box(data_array.clone()),
                                kernel_type.to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported number of dimensions: {dims}"),
                        };
                        black_box(entropy)
                    });
                });

                // Get the Rust benchmark results from Criterion
                // For simplicity, we'll run our own measurement to get a comparable result
                let mut rust_durations = Vec::new();
                for _ in 0..20 {
                    let start = std::time::Instant::now();
                    let _ = if dims == 1 {
                        // 1D case
                        let data_array = Array1::from_vec(data.clone());
                        match kernel_type {
                            "box" => Entropy::new_kernel(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::new_kernel_with_type(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 2 {
                        // 2D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<2>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<2>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 3 {
                        // 3D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<3>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<3>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 4 {
                        // 4D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<4>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<4>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 8 {
                        // 8D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<8>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<8>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 18 {
                        // 18D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<18>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<18>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else {
                        panic!("Unsupported number of dimensions: {dims}");
                    };
                    rust_durations.push(start.elapsed());
                }

                // Calculate average Rust time
                let rust_time_ns = rust_durations.iter().map(|d| d.as_nanos()).sum::<u128>()
                    / rust_durations.len() as u128;

                // Benchmark Python implementation
                let python_time = python::benchmark_entropy_float_nd(
                    &data,
                    dims,
                    "kernel",
                    &kernel_kwargs,
                    python_num_runs,
                )
                .unwrap();
                let python_time_ns = (python_time * 1_000_000_000.0) as u128;

                // Calculate entropy values for verification
                let rust_entropy = if dims == 1 {
                    // 1D case
                    let data_array = Array1::from_vec(data.clone());
                    match kernel_type {
                        "box" => Entropy::new_kernel(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::new_kernel_with_type(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 2 {
                    // 2D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<2>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<2>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 3 {
                    // 3D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<3>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<3>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 4 {
                    // 4D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<4>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<4>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 8 {
                    // 8D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<8>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<8>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 18 {
                    // 18D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<18>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<18>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else {
                    panic!("Unsupported number of dimensions: {dims}");
                };

                let python_entropy =
                    python::calculate_entropy_float_nd(&data, dims, "kernel", &kernel_kwargs)
                        .unwrap();

                // Calculate speedup ratio (how many times faster is Rust than Python)
                let speedup = python_time_ns as f64 / rust_time_ns as f64;

                // Print comparison for this data size
                println!(
                    "\n=== Kernel: {kernel_type}, Dimensions: {dims}, Data Size: {size} elements ==="
                );
                println!(
                    "Rust execution time:   {:.9} seconds ({rust_time_ns} ns)",
                    rust_time_ns as f64 / 1_000_000_000.0,
                    rust_time_ns = rust_time_ns
                );
                println!("Python execution time: {python_time:.9} seconds ({python_time_ns} ns)");
                println!("Speedup (Python/Rust): {speedup:.2}x");
                println!("Rust entropy value:   {rust_entropy}");
                println!("Python entropy value: {python_entropy}");

                // Write to CSV
                writeln!(
                    csv_file,
                    "Data Size,{size},{kernel_type},{dims},{rust_time_ns},{python_time_ns},{speedup:.2},{rust_entropy},{python_entropy}"
                )
                .unwrap();
            }
            group.finish();
        }
    }

    // Benchmark with different bandwidths
    let size = 5000;
    let bandwidths = [0.1, 0.5, 1.0, 2.0];

    for &dims in &dimensions {
        for &kernel_type in &kernel_types {
            // Create a benchmark group for different bandwidths
            let group_name =
                format!("Unified Kernel Entropy - {kernel_type} Kernel - {dims}D - Bandwidth");
            let mut group = c.benchmark_group(&group_name);
            group.measurement_time(Duration::from_secs(3));

            for &bandwidth in &bandwidths {
                // Generate random data
                let data = generate_random_nd_data(size, dims, seed);

                // Create kwargs for Python
                let kernel_kwargs = vec![
                    ("kernel".to_string(), format!("\"{kernel_type}\"")),
                    ("bandwidth".to_string(), format!("{bandwidth}")),
                ];

                // Benchmark Rust implementation
                let rust_id = BenchmarkId::new("Rust", bandwidth);
                group.bench_with_input(rust_id.clone(), &bandwidth, |b, &bw| {
                    if dims == 1 {
                        // 1D case
                        let data_array = Array1::from_vec(data.clone());
                        b.iter(|| {
                            let entropy = match kernel_type {
                                "box" => Entropy::new_kernel(black_box(data_array.clone()), bw),
                                "gaussian" => Entropy::new_kernel_with_type(
                                    black_box(data_array.clone()),
                                    "gaussian".to_string(),
                                    bw,
                                ),
                                _ => panic!("Unsupported kernel type"),
                            };
                            black_box(entropy.global_value())
                        });
                    } else if dims == 2 {
                        // 2D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        b.iter(|| {
                            let entropy = match kernel_type {
                                "box" => Entropy::nd_kernel::<2>(black_box(data_array.clone()), bw),
                                "gaussian" => Entropy::nd_kernel_with_type::<2>(
                                    black_box(data_array.clone()),
                                    "gaussian".to_string(),
                                    bw,
                                ),
                                _ => panic!("Unsupported kernel type"),
                            };
                            black_box(entropy.global_value())
                        });
                    } else if dims == 3 {
                        // 3D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        b.iter(|| {
                            let entropy = match kernel_type {
                                "box" => Entropy::nd_kernel::<3>(black_box(data_array.clone()), bw),
                                "gaussian" => Entropy::nd_kernel_with_type::<3>(
                                    black_box(data_array.clone()),
                                    "gaussian".to_string(),
                                    bw,
                                ),
                                _ => panic!("Unsupported kernel type"),
                            };
                            black_box(entropy.global_value())
                        });
                    } else if dims == 4 {
                        // 4D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        b.iter(|| {
                            let entropy = match kernel_type {
                                "box" => Entropy::nd_kernel::<4>(black_box(data_array.clone()), bw),
                                "gaussian" => Entropy::nd_kernel_with_type::<4>(
                                    black_box(data_array.clone()),
                                    "gaussian".to_string(),
                                    bw,
                                ),
                                _ => panic!("Unsupported kernel type"),
                            };
                            black_box(entropy.global_value())
                        });
                    } else if dims == 8 {
                        // 8D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        b.iter(|| {
                            let entropy = match kernel_type {
                                "box" => Entropy::nd_kernel::<8>(black_box(data_array.clone()), bw),
                                "gaussian" => Entropy::nd_kernel_with_type::<8>(
                                    black_box(data_array.clone()),
                                    "gaussian".to_string(),
                                    bw,
                                ),
                                _ => panic!("Unsupported kernel type"),
                            };
                            black_box(entropy.global_value())
                        });
                    } else if dims == 18 {
                        // 18D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        b.iter(|| {
                            let entropy = match kernel_type {
                                "box" => {
                                    Entropy::nd_kernel::<18>(black_box(data_array.clone()), bw)
                                }
                                "gaussian" => Entropy::nd_kernel_with_type::<18>(
                                    black_box(data_array.clone()),
                                    "gaussian".to_string(),
                                    bw,
                                ),
                                _ => panic!("Unsupported kernel type"),
                            };
                            black_box(entropy.global_value())
                        });
                    } else {
                        panic!("Unsupported number of dimensions: {dims}");
                    }
                });

                // Get the Rust benchmark results from Criterion
                // For simplicity, we'll run our own measurement to get a comparable result
                let mut rust_durations = Vec::new();
                for _ in 0..20 {
                    let start = std::time::Instant::now();
                    let _ = if dims == 1 {
                        // 1D case
                        let data_array = Array1::from_vec(data.clone());
                        match kernel_type {
                            "box" => Entropy::new_kernel(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::new_kernel_with_type(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 2 {
                        // 2D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<2>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<2>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 3 {
                        // 3D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<3>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<3>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 4 {
                        // 4D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<4>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<4>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 8 {
                        // 8D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<8>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<8>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else if dims == 18 {
                        // 18D case
                        let data_array =
                            Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                        match kernel_type {
                            "box" => Entropy::nd_kernel::<18>(data_array, bandwidth).global_value(),
                            "gaussian" => Entropy::nd_kernel_with_type::<18>(
                                data_array,
                                "gaussian".to_string(),
                                bandwidth,
                            )
                            .global_value(),
                            _ => panic!("Unsupported kernel type"),
                        }
                    } else {
                        panic!("Unsupported number of dimensions: {dims}");
                    };
                    rust_durations.push(start.elapsed());
                }

                // Calculate average Rust time
                let rust_time_ns = rust_durations.iter().map(|d| d.as_nanos()).sum::<u128>()
                    / rust_durations.len() as u128;

                // Benchmark Python implementation
                let python_time = python::benchmark_entropy_float_nd(
                    &data,
                    dims,
                    "kernel",
                    &kernel_kwargs,
                    python_num_runs,
                )
                .unwrap();
                let python_time_ns = (python_time * 1_000_000_000.0) as u128;

                // Calculate entropy values for verification
                let rust_entropy = if dims == 1 {
                    // 1D case
                    let data_array = Array1::from_vec(data.clone());
                    match kernel_type {
                        "box" => Entropy::new_kernel(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::new_kernel_with_type(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 2 {
                    // 2D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<2>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<2>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 3 {
                    // 3D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<3>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<3>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 4 {
                    // 4D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<4>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<4>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 8 {
                    // 8D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<8>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<8>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else if dims == 18 {
                    // 18D case
                    let data_array = Array2::from_shape_vec((size, dims), data.clone()).unwrap();
                    match kernel_type {
                        "box" => Entropy::nd_kernel::<18>(data_array, bandwidth).global_value(),
                        "gaussian" => Entropy::nd_kernel_with_type::<18>(
                            data_array,
                            "gaussian".to_string(),
                            bandwidth,
                        )
                        .global_value(),
                        _ => panic!("Unsupported kernel type"),
                    }
                } else {
                    panic!("Unsupported number of dimensions: {dims}");
                };

                let python_entropy =
                    python::calculate_entropy_float_nd(&data, dims, "kernel", &kernel_kwargs)
                        .unwrap();

                // Calculate speedup ratio (how many times faster is Rust than Python)
                let speedup = python_time_ns as f64 / rust_time_ns as f64;

                // Print comparison for this bandwidth
                println!(
                    "\n=== Kernel: {kernel_type}, Dimensions: {dims}, Bandwidth: {bandwidth} ==="
                );
                println!(
                    "Rust execution time:   {:.9} seconds ({rust_time_ns} ns)",
                    rust_time_ns as f64 / 1_000_000_000.0,
                    rust_time_ns = rust_time_ns
                );
                println!("Python execution time: {python_time:.9} seconds ({python_time_ns} ns)");
                println!("Speedup (Python/Rust): {speedup:.2}x");
                println!("Rust entropy value:   {rust_entropy}");
                println!("Python entropy value: {python_entropy}");

                // Write to CSV
                writeln!(
                    csv_file,
                    "Bandwidth,{bandwidth},{kernel_type},{dims},{rust_time_ns},{python_time_ns},{speedup:.2},{rust_entropy},{python_entropy}"
                )
                .unwrap();
            }
            group.finish();
        }
    }

    println!("\nBenchmark results have been saved to unified_kernel_benchmark_results.csv");
}

criterion_group!(benches, bench_unified_kernel_entropy);
criterion_main!(benches);

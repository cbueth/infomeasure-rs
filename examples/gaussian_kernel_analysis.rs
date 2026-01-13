use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use ndarray::Array2;
use plotters::prelude::*;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use std::fs::File;
use std::io::Write;
use validation::python;

fn generate_gaussian_data(
    size: usize,
    dims: usize,
    mean: f64,
    std_dev: f64,
    seed: u64,
) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(mean, std_dev).unwrap();

    // Generate data for each dimension
    let data: Vec<f64> = (0..size * dims).map(|_| normal.sample(&mut rng)).collect();

    // Reshape into a 2D array with shape (size, dims)
    Array2::from_shape_vec((size, dims), data).expect("Failed to reshape data")
}

fn calculate_1d_entropy(data: &Array2<f64>, bandwidth: f64) -> f64 {
    Entropy::new_kernel_with_type(data.column(0).to_owned(), "gaussian".to_string(), bandwidth)
        .global_value()
}

fn calculate_2d_entropy(data: &Array2<f64>, bandwidth: f64) -> f64 {
    Entropy::nd_kernel_with_type::<2>(data.clone(), "gaussian".to_string(), bandwidth)
        .global_value()
}

fn calculate_3d_entropy(data: &Array2<f64>, bandwidth: f64) -> f64 {
    Entropy::nd_kernel_with_type::<3>(data.clone(), "gaussian".to_string(), bandwidth)
        .global_value()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parameters
    let size = 1000;
    let dims = 2; // Testing with d=2
    let mean = 1.0;
    let std_dev = 0.5;
    let seed = 21354;

    // Generate multi-dimensional Gaussian data
    let data = generate_gaussian_data(size, dims, mean, std_dev, seed);

    // Create bandwidth range
    let bandwidths: Vec<f64> = (1..26).map(|x| x as f64 * 0.1).collect();

    // Store results
    let mut results = Vec::new();

    // Create CSV file for results
    let mut csv_file = File::create("gaussian_kernel_comparison.csv")?;
    writeln!(csv_file, "bandwidth,rust_entropy,python_entropy")?;

    // Calculate entropies for different bandwidths
    println!("Data for d={}, mean={:.1}, std={:.1}", dims, mean, std_dev);

    for &bandwidth in &bandwidths {
        let rust_entropy = match dims {
            1 => calculate_1d_entropy(&data, bandwidth),
            2 => calculate_2d_entropy(&data, bandwidth),
            3 => calculate_3d_entropy(&data, bandwidth),
            _ => panic!("Unsupported number of dimensions: {}", dims),
        };

        // Python implementation
        let kernel_kwargs = [
            ("kernel".to_string(), "\"gaussian\"".to_string()),
            ("bandwidth".to_string(), bandwidth.to_string()),
        ];
        let python_entropy = python::calculate_entropy_float_nd(
            &data.as_slice().unwrap(),
            dims,
            "kernel",
            &kernel_kwargs,
        )?;

        // Store results
        results.push((bandwidth, rust_entropy, python_entropy));

        // Write to CSV
        writeln!(
            csv_file,
            "{},{},{}",
            bandwidth, rust_entropy, python_entropy
        )?;

        // Print progress
        println!(
            "Bandwidth: {:.1}, Rust: {:.6}, Python: {:.6}",
            bandwidth, rust_entropy, python_entropy
        );
    }

    // Create plot
    let root = BitMapBackend::new("gaussian_kernel_comparison.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_entropy = results
        .iter()
        .flat_map(|(_, r, p)| vec![r, p])
        .fold(0f64, |acc, &x| acc.max(x));

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Gaussian Kernel Entropy Comparison",
            ("sans-serif", 30).into_font(),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..4f64, 0f64..max_entropy * 1.1)?;

    chart
        .configure_mesh()
        .x_desc("Bandwidth")
        .y_desc("Entropy")
        .draw()?;

    // Plot Rust implementation
    chart
        .draw_series(LineSeries::new(
            results.iter().map(|(b, r, _)| (*b, *r)),
            &RED,
        ))?
        .label("Rust")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // Plot Python implementation
    chart
        .draw_series(LineSeries::new(
            results.iter().map(|(b, _, p)| (*b, *p)),
            &BLUE,
        ))?
        .label("Python")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    println!("\nResults saved to:");
    println!("- gaussian_kernel_comparison.csv");
    println!("- gaussian_kernel_comparison.png");

    Ok(())
}

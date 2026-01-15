use infomeasure::estimators::entropy::{Entropy, GlobalValue, LocalValues};
use ndarray::array;
use std::collections::HashMap;

fn main() {
    // Example discrete data - a sequence of integers
    let data = array!(1, 2, 1, 3, 2, 1, 4, 2, 3, 1);

    // Instantiate the discrete entropy estimator
    let discrete_entropy = Entropy::new_discrete(data.clone());

    // Calculate entropy manually to verify
    let mut counts = HashMap::new();
    for &value in &data {
        *counts.entry(value).or_insert(0) += 1;
    }

    let n = data.len() as f64;
    let mut manual_entropy = 0.0;
    for (_, &count) in counts.iter() {
        let p = count as f64 / n;
        manual_entropy -= p * p.ln();
    }

    println!("Discrete data: {data:?}");
    println!("Manual entropy calculation: {manual_entropy}");

    // Calculate local and global entropy values using the library
    let local_values = discrete_entropy.local_values();
    let global_value = discrete_entropy.global_value();

    println!("Local Entropy Values: {local_values:?}");
    println!("Global Entropy Value: {global_value}");
    println!("Comparison: Manual calculation vs. Library implementation");
    println!("  Manual: {manual_entropy}");
    println!("  Library: {global_value}");
}

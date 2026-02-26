// Example demonstrating the new ANSB API with default and custom thresholds
use infomeasure::estimators::entropy::{Entropy, GlobalValue};
use ndarray::Array1;

fn main() {
    println!("=== ANSB API Example ===");
    // Generate data: N=1000 samples.
    // For ANSB to return a value (not NaN), we must have N > K (where K is the support size).
    // The ANSB formula uses coincidences Delta = N - K.
    // If K is not provided, it uses the observed unique values K_obs.

    let n = 1000;
    // Generate data with some repeats: using a distribution that yields around 800-900 unique values.
    let data: Vec<i32> = (0..n)
        .map(|i| {
            // Using floor(sqrt(i) * 31) gives a good mix of repeats and unique values
            ((i as f64).sqrt() * 31.0) as i32
        })
        .collect();
    let data = Array1::from(data);

    let k_obs =
        infomeasure::estimators::approaches::discrete::discrete_utils::count_frequencies(&data)
            .len();
    let ratio = n as f64 / k_obs as f64;

    println!("Generated data: N={n}, Observed K_obs={k_obs}, Ratio N/K_obs={ratio:.3}");
    println!("Coincidences (N - K_obs) = {}", n - k_obs);
    println!(
        "Sample Data (first 20): {:?}\n",
        data.slice(ndarray::s![..20])
    );

    // 1. Default threshold (0.1)
    // This will warn because Ratio (1.x) > 0.1, but it will return a valid value because N > K_obs.
    let est_default = Entropy::new_ansb(data.clone(), None);
    let h_default = est_default.global_value();
    println!("Default threshold (0.1): entropy = {h_default:.6} (Warning expected above)");

    // 2. Custom threshold (2.0)
    // No warning here because Ratio (1.x) < 2.0.
    let est_custom = Entropy::new_ansb_with_threshold(data.clone(), None, 2.0);
    let h_custom = est_custom.global_value();
    println!("Custom threshold (2.0): entropy = {h_custom:.6} (No warning expected)");

    // 3. Using k_override
    // Providing a theoretical K that is still less than N.
    let k_override = 950;
    let est_k = Entropy::new_ansb_with_threshold(data.clone(), Some(k_override), 2.0);
    let h_k = est_k.global_value();
    println!("Explicit K={k_override}, threshold 2.0: entropy = {h_k:.6}");

    // 4. Demonstrate batch processing
    println!("\nBatch processing (N=500 per row):");
    let row1 = data.slice(ndarray::s![0..500]).to_owned();
    let row2 = data.slice(ndarray::s![500..1000]).to_owned();
    let data_2d = ndarray::stack![ndarray::Axis(0), row1, row2];

    // Batch API with custom threshold
    let batch = Entropy::new_ansb_rows_with_threshold(data_2d, None, 5.0);
    println!("Row 1 entropy: {:.6}", batch[0].global_value());
    println!("Row 2 entropy: {:.6}", batch[1].global_value());

    println!("\nNote: ANSB is defined only when N > K. If N <= K, it returns NaN.");
    println!("The undersampled threshold warning triggers if the ratio N/K exceeds the parameter.");
    println!(
        "Since N/K is always > 1 for valid ANSB estimates, the default threshold (0.1) always warns."
    );
}

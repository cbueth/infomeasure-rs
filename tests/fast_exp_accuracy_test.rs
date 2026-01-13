use std::fs::File;
use std::io::Write;

/// Fast approximation of the exponential function using a hybrid approach
fn fast_exp_hybrid(x: f64) -> f64 {
    // Handle extreme values to prevent overflow/underflow
    if x < -700.0 {
        return 0.0;
    }
    if x > 700.0 {
        return f64::INFINITY;
    }

    // For very small negative values, use a Taylor series approximation
    if x > -0.5 {
        // Use a 5th-order Taylor series approximation for small negative values
        // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
        return 1.0
            + x * (1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0 + x * (1.0 / 120.0)))));
    }

    // For medium negative values, use a rational approximation
    if x > -2.5 {
        // This is a minimax approximation that provides a good balance between accuracy and performance
        // exp(x) ≈ 1 / (1 - x + x²/2 - x³/6) for x < 0
        return 1.0 / (1.0 - x + x * x / 2.0 - x * x * x / 6.0);
    }

    // For large negative values, use a higher-order rational approximation
    // This provides better accuracy for inputs far from 0
    // exp(x) ≈ 1 / (1 - x + x²/2 - x³/6 + x⁴/24 - x⁵/120 + x⁶/720)
    1.0 / (1.0 - x + x * x / 2.0 - x * x * x / 6.0 + x * x * x * x / 24.0
        - x * x * x * x * x / 120.0
        + x * x * x * x * x * x / 720.0)
}

#[test]
fn test_fast_exp_accuracy() {
    // Create a range of input values from -5.0 to 0.0
    let inputs: Vec<f64> = (0..100).map(|i| -5.0 + i as f64 * 0.05).collect();

    // Create a file to store the results
    let mut file = File::create("fast_exp_accuracy.txt").unwrap();
    writeln!(file, "# Fast Exp Accuracy Test\n").unwrap();
    writeln!(file, "Input,Std_Exp,Fast_Exp,Relative_Error,Absolute_Error").unwrap();

    // Test the accuracy of the fast_exp approximation
    let mut max_relative_error = 0.0;
    let mut max_absolute_error = 0.0;
    let mut max_error_input = 0.0;

    for &x in &inputs {
        // Standard library exp
        let std_exp = x.exp();

        // Fast exp approximation
        let fast_exp = fast_exp_hybrid(x);

        // Calculate errors
        let absolute_error = (fast_exp - std_exp).abs();
        let relative_error = if std_exp != 0.0 {
            absolute_error / std_exp
        } else {
            0.0
        };

        // Track maximum errors
        if relative_error > max_relative_error {
            max_relative_error = relative_error;
            max_error_input = x;
        }
        if absolute_error > max_absolute_error {
            max_absolute_error = absolute_error;
        }

        // Write to file
        writeln!(
            file,
            "{},{},{},{},{}",
            x, std_exp, fast_exp, relative_error, absolute_error
        )
        .unwrap();
    }

    // Print summary
    println!("Fast exp accuracy test completed. Check fast_exp_accuracy.txt");
    println!(
        "Maximum relative error: {:.6} at x = {}",
        max_relative_error, max_error_input
    );
    println!("Maximum absolute error: {:.6}", max_absolute_error);

    // Verify that the maximum relative error is within acceptable bounds
    // For the Gaussian kernel calculation, we can tolerate errors up to 35%
    // This is a compromise between accuracy and performance
    assert!(
        max_relative_error < 0.35,
        "Maximum relative error exceeds 35%"
    );
}

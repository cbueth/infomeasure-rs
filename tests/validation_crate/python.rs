//! Python validation module for comparing the Rust implementation with the Python infomeasure package.
//!
//! This module provides functionality for validating the Rust implementation against
//! the Python infomeasure package.

use ndarray::Array1;
use std::process::Command;
use std::path::Path;
use serde::Serialize;
use rand::{Rng, thread_rng};

/// Checks if the micromamba environment exists.
///
/// # Returns
///
/// `true` if the environment exists, `false` otherwise.
fn environment_exists() -> bool {
    let output = Command::new("micromamba")
        .args(["env", "list"])
        .output();

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.contains("infomeasure-rs-validation")
        },
        Err(_) => false,
    }
}

/// Creates the micromamba environment if it doesn't exist.
///
/// # Returns
///
/// `true` if the environment was created successfully or already exists, `false` otherwise.
fn ensure_environment() -> bool {
    if environment_exists() {
        return true;
    }

    // Get the path to the environment.yml file
    let env_file = Path::new("tests/validation_crate/environment.yml");
    if !env_file.exists() {
        eprintln!("environment.yml file not found at {:?}", env_file);
        return false;
    }

    println!("Creating micromamba environment 'infomeasure-rs-validation'...");
    let output = Command::new("micromamba")
        .args(["env", "create", "-f", env_file.to_str().unwrap()])
        .output();

    match output {
        Ok(output) => {
            if output.status.success() {
                println!("Environment created successfully.");
                true
            } else {
                eprintln!("Failed to create environment: {}", String::from_utf8_lossy(&output.stderr));
                false
            }
        },
        Err(e) => {
            eprintln!("Failed to execute micromamba: {}", e);
            false
        },
    }
}

/// Runs a command in the micromamba environment.
///
/// # Arguments
///
/// * `args` - The arguments to pass to the command
///
/// # Returns
///
/// The output of the command
fn run_in_environment(args: &[&str]) -> Result<String, String> {
    // Get the path to the micromamba executable
    let micromamba_path = which_micromamba().map_err(|e| e.to_string())?;

    // Run the command in the micromamba environment
    let output = Command::new(&micromamba_path)
        .args(["run", "-n", "infomeasure-rs-validation"])
        .args(args)
        .output()
        .map_err(|e| format!("Failed to execute command: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(format!("Command failed: {}", String::from_utf8_lossy(&output.stderr)))
    }
}

/// Gets the path to the micromamba executable.
///
/// # Returns
///
/// The path to the micromamba executable
fn which_micromamba() -> Result<String, String> {
    let output = Command::new("which")
        .arg("micromamba")
        .output()
        .map_err(|e| format!("Failed to execute 'which micromamba': {}", e))?;

    if output.status.success() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(path)
    } else {
        Err("micromamba not found in PATH".to_string())
    }
}

/// Saves data to a temporary file and returns the file path.
///
/// # Arguments
///
/// * `data` - The data to save
///
/// # Returns
///
/// The path to the temporary file
fn save_data_to_temp_file<T: Serialize>(data: &[T]) -> Result<std::path::PathBuf, String> {
    // Create a temporary file with a unique identifier
    let temp_dir = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let random_component = thread_rng().r#gen::<u64>();
    let unique_id = format!("{}_{}", timestamp, random_component);
    let file_path = temp_dir.join(format!("data_{}.npy", unique_id));

    // Serialize the data to JSON
    let json_data = serde_json::to_string(data)
        .map_err(|e| format!("Failed to serialize data to JSON: {}", e))?;

    // Write the data to the file
    std::fs::write(&file_path, json_data)
        .map_err(|e| format!("Failed to write data to temporary file: {}", e))?;

    Ok(file_path)
}


/// Calculates entropy using the Python infomeasure package with generic data type.
///
/// # Arguments
///
/// * `data` - The data to calculate entropy for (can be any serializable type)
/// * `approach` - The approach to use for entropy calculation (e.g., "discrete", "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
///   - For "discrete" approach: typically empty
///   - For "kernel" approach: typically includes "kernel" and "bandwidth" parameters
///     e.g., [("kernel".to_string(), "\"box\"".to_string()), ("bandwidth".to_string(), "0.5".to_string())]
///
/// # Returns
///
/// The entropy value calculated by the Python infomeasure package
///
/// # Examples
///
/// ```
/// // For discrete approach with integer data
/// let entropy = calculate_entropy_generic(&data, "discrete", &[]).unwrap();
///
/// // For kernel approach with float data
/// let kernel_kwargs = [
///     ("kernel".to_string(), "\"box\"".to_string()),
///     ("bandwidth".to_string(), "0.5".to_string())
/// ];
/// let entropy = calculate_entropy_generic(&data, "kernel", &kernel_kwargs).unwrap();
/// ```
pub fn calculate_entropy_generic<T: Serialize>(data: &[T], approach: &str, kwargs: &[(String, String)]) -> Result<f64, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".to_string());
    }

    // Save data to a temporary file
    let data_file_path = save_data_to_temp_file(data)?;

    // Create a temporary Python script with a unique identifier
    let temp_dir = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let random_component = thread_rng().r#gen::<u64>();
    let unique_id = format!("{}_{}", timestamp, random_component);
    let script_path = temp_dir.join(format!("calculate_entropy_{}.py", unique_id));

    // Build kwargs string
    let kwargs_str = kwargs.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");

    let script_content = format!(r#"
import infomeasure as im
import sys
import json
import numpy as np

# Load data from file
with open("{}", "r") as f:
    data = json.load(f)

est = im.estimator(data, measure="h", approach="{}", base="e", {})
result = est.result()
print(result)
"#, data_file_path.to_str().unwrap(), approach, kwargs_str);

    std::fs::write(&script_path, script_content)
        .map_err(|e| format!("Failed to write temporary script: {}", e))?;

    // Run the script in the micromamba environment
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])
        .map_err(|e| format!("Failed to run Python script: {}", e))?;

    // Parse the output as a float
    let result = output.trim().parse::<f64>()
        .map_err(|e| format!("Failed to parse output as float: {}", e))?;

    // Clean up the temporary files
    let _ = std::fs::remove_file(script_path);
    let _ = std::fs::remove_file(data_file_path);

    Ok(result)
}

/// Calculates entropy using the Python infomeasure package.
///
/// # Arguments
///
/// * `data` - The data to calculate entropy for
/// * `approach` - The approach to use for entropy calculation (e.g., "discrete", "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
///   - For "discrete" approach: typically empty
///   - For "kernel" approach: typically includes "kernel" and "bandwidth" parameters
///     e.g., [("kernel".to_string(), "\"box\"".to_string()), ("bandwidth".to_string(), "0.5".to_string())]
///
/// # Returns
///
/// The entropy value calculated by the Python infomeasure package
///
/// # Examples
///
/// ```
/// // For discrete approach
/// let entropy = calculate_entropy(&data, "discrete", &[]).unwrap();
///
/// // For kernel approach
/// let kernel_kwargs = [
///     ("kernel".to_string(), "\"box\"".to_string()),
///     ("bandwidth".to_string(), "0.5".to_string())
/// ];
/// let entropy = calculate_entropy(&data, "kernel", &kernel_kwargs).unwrap();
/// ```
pub fn calculate_entropy(data: &[i32], approach: &str, kwargs: &[(String, String)]) -> Result<f64, String> {
    calculate_entropy_generic(data, approach, kwargs)
}

/// Calculates entropy using the Python infomeasure package with float data.
///
/// # Arguments
///
/// * `data` - The float data to calculate entropy for
/// * `approach` - The approach to use for entropy calculation (e.g., "discrete", "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
///   - For "discrete" approach: typically empty
///   - For "kernel" approach: typically includes "kernel" and "bandwidth" parameters
///     e.g., [("kernel".to_string(), "\"box\"".to_string()), ("bandwidth".to_string(), "0.5".to_string())]
///
/// # Returns
///
/// The entropy value calculated by the Python infomeasure package
///
/// # Examples
///
/// ```
/// // For kernel approach with float data
/// let kernel_kwargs = [
///     ("kernel".to_string(), "\"box\"".to_string()),
///     ("bandwidth".to_string(), "0.5".to_string())
/// ];
/// let entropy = calculate_entropy_float(&data, "kernel", &kernel_kwargs).unwrap();
/// ```
pub fn calculate_entropy_float(data: &[f64], approach: &str, kwargs: &[(String, String)]) -> Result<f64, String> {
    calculate_entropy_generic(data, approach, kwargs)
}

/// Calculates local entropy values using the Python infomeasure package with generic data type.
///
/// # Arguments
///
/// * `data` - The data to calculate local entropy values for (can be any serializable type)
/// * `approach` - The approach to use for entropy calculation (e.g., "discrete", "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
///   - For "discrete" approach: typically empty
///   - For "kernel" approach: typically includes "kernel" and "bandwidth" parameters
///     e.g., [("kernel".to_string(), "\"box\"".to_string()), ("bandwidth".to_string(), "0.5".to_string())]
///
/// # Returns
///
/// The local entropy values calculated by the Python infomeasure package
///
/// # Examples
///
/// ```
/// // For discrete approach with integer data
/// let local_entropy = calculate_local_entropy_generic(&data, "discrete", &[]).unwrap();
///
/// // For kernel approach with float data
/// let kernel_kwargs = [
///     ("kernel".to_string(), "\"box\"".to_string()),
///     ("bandwidth".to_string(), "0.5".to_string())
/// ];
/// let local_entropy = calculate_local_entropy_generic(&data, "kernel", &kernel_kwargs).unwrap();
/// ```
pub fn calculate_local_entropy_generic<T: Serialize>(data: &[T], approach: &str, kwargs: &[(String, String)]) -> Result<Vec<f64>, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".to_string());
    }

    // Save data to a temporary file
    let data_file_path = save_data_to_temp_file(data)?;

    // Create a temporary Python script with a unique identifier
    let temp_dir = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let random_component = thread_rng().r#gen::<u64>();
    let unique_id = format!("{}_{}", timestamp, random_component);
    let script_path = temp_dir.join(format!("calculate_local_entropy_{}.py", unique_id));

    // Build kwargs string
    let kwargs_str = kwargs.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");

    let script_content = format!(r#"
import infomeasure as im
import sys
import json
import numpy as np

# Load data from file
with open("{}", "r") as f:
    data = json.load(f)

est = im.estimator(data, measure="h", approach="{}", base="e", {})
local_vals = est.local_vals()
print(json.dumps(local_vals.tolist()))
"#, data_file_path.to_str().unwrap(), approach, kwargs_str);

    std::fs::write(&script_path, script_content)
        .map_err(|e| format!("Failed to write temporary script: {}", e))?;

    // Run the script in the micromamba environment
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])
        .map_err(|e| format!("Failed to run Python script: {}", e))?;

    // Parse the output as a JSON array of floats
    let local_vals: Vec<f64> = serde_json::from_str(&output.trim())
        .map_err(|e| format!("Failed to parse output as JSON array: {}", e))?;

    // Clean up the temporary files
    let _ = std::fs::remove_file(script_path);
    let _ = std::fs::remove_file(data_file_path);

    Ok(local_vals)
}

/// Calculates local entropy values using the Python infomeasure package.
///
/// # Arguments
///
/// * `data` - The data to calculate local entropy values for
/// * `approach` - The approach to use for entropy calculation (e.g., "discrete", "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
///   - For "discrete" approach: typically empty
///   - For "kernel" approach: typically includes "kernel" and "bandwidth" parameters
///     e.g., [("kernel".to_string(), "\"box\"".to_string()), ("bandwidth".to_string(), "0.5".to_string())]
///
/// # Returns
///
/// The local entropy values calculated by the Python infomeasure package
///
/// # Examples
///
/// ```
/// // For discrete approach
/// let local_entropy = calculate_local_entropy(&data, "discrete", &[]).unwrap();
///
/// // For kernel approach
/// let kernel_kwargs = [
///     ("kernel".to_string(), "\"box\"".to_string()),
///     ("bandwidth".to_string(), "0.5".to_string())
/// ];
/// let local_entropy = calculate_local_entropy(&data, "kernel", &kernel_kwargs).unwrap();
/// ```
pub fn calculate_local_entropy(data: &[i32], approach: &str, kwargs: &[(String, String)]) -> Result<Vec<f64>, String> {
    calculate_local_entropy_generic(data, approach, kwargs)
}

/// Calculates local entropy values using the Python infomeasure package with float data.
///
/// # Arguments
///
/// * `data` - The float data to calculate local entropy values for
/// * `approach` - The approach to use for entropy calculation (e.g., "discrete", "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
///   - For "discrete" approach: typically empty
///   - For "kernel" approach: typically includes "kernel" and "bandwidth" parameters
///     e.g., [("kernel".to_string(), "\"box\"".to_string()), ("bandwidth".to_string(), "0.5".to_string())]
///
/// # Returns
///
/// The local entropy values calculated by the Python infomeasure package
///
/// # Examples
///
/// ```
/// // For kernel approach with float data
/// let kernel_kwargs = [
///     ("kernel".to_string(), "\"box\"".to_string()),
///     ("bandwidth".to_string(), "0.5".to_string())
/// ];
/// let local_entropy = calculate_local_entropy_float(&data, "kernel", &kernel_kwargs).unwrap();
/// ```
pub fn calculate_local_entropy_float(data: &[f64], approach: &str, kwargs: &[(String, String)]) -> Result<Vec<f64>, String> {
    calculate_local_entropy_generic(data, approach, kwargs)
}

/// Converts entropy values from natural log (base e) to base 2.
///
/// # Arguments
///
/// * `entropy` - The entropy value in base e
///
/// # Returns
///
/// The entropy value in base 2
pub fn convert_to_base2(entropy: f64) -> f64 {
    entropy / std::f64::consts::LN_2
}

/// Converts an array of entropy values from natural log (base e) to base 2.
///
/// # Arguments
///
/// * `entropy` - The array of entropy values in base e
///
/// # Returns
///
/// The array of entropy values in base 2
pub fn convert_array_to_base2(entropy: &Array1<f64>) -> Array1<f64> {
    entropy.mapv(|x| x / std::f64::consts::LN_2)
}

/// Calculates entropy using the Python infomeasure package with n-dimensional float data.
///
/// # Arguments
///
/// * `data` - The flat array of float data to calculate entropy for
/// * `dims` - The number of dimensions in the data
/// * `approach` - The approach to use for entropy calculation (e.g., "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
///   - For "kernel" approach: typically includes "kernel" and "bandwidth" parameters
///     e.g., [("kernel".to_string(), "\"box\"".to_string()), ("bandwidth".to_string(), "0.5".to_string())]
///
/// # Returns
///
/// The entropy value calculated by the Python infomeasure package
///
/// # Examples
///
/// ```
/// // For kernel approach with 2D float data
/// let kernel_kwargs = [
///     ("kernel".to_string(), "\"gaussian\"".to_string()),
///     ("bandwidth".to_string(), "0.5".to_string()),
/// ];
/// let entropy = calculate_entropy_float_nd(&data, 2, "kernel", &kernel_kwargs).unwrap();
/// ```
pub fn calculate_entropy_float_nd(data: &[f64], dims: usize, approach: &str, kwargs: &[(String, String)]) -> Result<f64, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".to_string());
    }

    // Check if data length is divisible by dims
    if data.len() % dims != 0 {
        return Err(format!("Data length ({}) is not divisible by the number of dimensions ({})", data.len(), dims));
    }

    // Calculate the number of samples
    let n_samples = data.len() / dims;

    // Reshape the flat array into a Vec<Vec<f64>> with shape (n_samples, dims)
    let data_vec: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| {
            (0..dims)
                .map(|j| data[i * dims + j])
                .collect()
        })
        .collect();

    // Use the generic function to calculate entropy
    calculate_entropy_generic(&data_vec, approach, kwargs)
}

/// Calculates local entropy values using the Python infomeasure package with n-dimensional float data.
///
/// # Arguments
///
/// * `data` - The flat array of float data to calculate local entropy values for
/// * `dims` - The number of dimensions in the data
/// * `approach` - The approach to use for entropy calculation (e.g., "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
///   - For "kernel" approach: typically includes "kernel" and "bandwidth" parameters
///     e.g., [("kernel".to_string(), "\"box\"".to_string()), ("bandwidth".to_string(), "0.5".to_string())]
///
/// # Returns
///
/// The local entropy values calculated by the Python infomeasure package
///
/// # Examples
///
/// ```
/// // For kernel approach with 2D float data
/// let kernel_kwargs = [
///     ("kernel".to_string(), "\"gaussian\"".to_string()),
///     ("bandwidth".to_string(), "0.5".to_string()),
/// ];
/// let local_entropy = calculate_local_entropy_float_nd(&data, 2, "kernel", &kernel_kwargs).unwrap();
/// ```
pub fn calculate_local_entropy_float_nd(data: &[f64], dims: usize, approach: &str, kwargs: &[(String, String)]) -> Result<Vec<f64>, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".to_string());
    }

    // Check if data length is divisible by dims
    if data.len() % dims != 0 {
        return Err(format!("Data length ({}) is not divisible by the number of dimensions ({})", data.len(), dims));
    }

    // Calculate the number of samples
    let n_samples = data.len() / dims;

    // Reshape the flat array into a Vec<Vec<f64>> with shape (n_samples, dims)
    let data_vec: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| {
            (0..dims)
                .map(|j| data[i * dims + j])
                .collect()
        })
        .collect();

    // Use the generic function to calculate local entropy values
    calculate_local_entropy_generic(&data_vec, approach, kwargs)
}

/// Benchmarks the Python infomeasure package's discrete entropy calculation.
///
/// # Arguments
///
/// * `data` - The data to calculate entropy for
/// * `num_runs` - The number of times to run the calculation for more accurate timing
///
/// # Returns
///
/// The average execution time in seconds
pub fn benchmark_entropy(data: &[i32], num_runs: usize) -> Result<f64, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".to_string());
    }

    // Create a temporary Python script with a unique identifier
    let temp_dir = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let random_component = thread_rng().r#gen::<u64>();
    let unique_id = format!("{}_{}", timestamp, random_component);
    let script_path = temp_dir.join(format!("benchmark_entropy_{}.py", unique_id));
    let data_str = data.iter().map(|&x| x.to_string()).collect::<Vec<_>>().join(", ");

    let script_content = format!(r#"
import infomeasure as im
import timeit
import sys

# Define the data
data = [{data_str}]

# Define the function to benchmark
def run_entropy_calculation():
    est = im.estimator(data, measure="h", approach="discrete", base="e")
    return est.result()

# Use timeit for more accurate benchmarking
# This automatically handles multiple runs and provides better timing accuracy
timer = timeit.Timer(run_entropy_calculation)
avg_time = timer.timeit(number={num_runs}) / {num_runs}

# Print the result (will be captured by Rust)
print(avg_time)
"#);

    std::fs::write(&script_path, script_content)
        .map_err(|e| format!("Failed to write temporary script: {}", e))?;

    // Run the script in the micromamba environment
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])
        .map_err(|e| format!("Failed to run Python script: {}", e))?;

    // Parse the output as a float (execution time in seconds)
    let result = output.trim().parse::<f64>()
        .map_err(|e| format!("Failed to parse output as float: {}", e))?;

    // Clean up the temporary script
    let _ = std::fs::remove_file(script_path);

    Ok(result)
}

/// Benchmarks the Python infomeasure package's entropy calculation with generic data and approach.
///
/// # Arguments
///
/// * `data` - The data to calculate entropy for
/// * `approach` - The approach to use for entropy calculation (e.g., "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
/// * `num_runs` - The number of times to run the calculation for more accurate timing
///
/// # Returns
///
/// The average execution time in seconds
pub fn benchmark_entropy_generic<T: Serialize>(data: &[T], approach: &str, kwargs: &[(String, String)], num_runs: usize) -> Result<f64, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".to_string());
    }

    // Save data to a temporary file
    let data_file_path = save_data_to_temp_file(data)?;

    // Build kwargs string
    let kwargs_str = kwargs.iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");

    // Create a temporary Python script with a unique identifier
    let temp_dir = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let random_component = thread_rng().r#gen::<u64>();
    let unique_id = format!("{}_{}", timestamp, random_component);
    let script_path = temp_dir.join(format!("benchmark_entropy_generic_{}.py", unique_id));

    let script_content = format!(r#"
import infomeasure as im
import timeit
import sys
import json

# Load data from file
with open("{}", "r") as f:
    data = json.load(f)

# Define the function to benchmark
def run_entropy_calculation():
    est = im.estimator(data, measure="h", approach="{}", base="e", {})
    return est.result()

# Use timeit for more accurate benchmarking
# This automatically handles multiple runs and provides better timing accuracy
timer = timeit.Timer(run_entropy_calculation)
avg_time = timer.timeit(number={}) / {}

# Print the result (will be captured by Rust)
print(avg_time)
"#, data_file_path.to_str().unwrap(), approach, kwargs_str, num_runs, num_runs);

    std::fs::write(&script_path, script_content)
        .map_err(|e| format!("Failed to write temporary script: {}", e))?;

    // Run the script in the micromamba environment
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])
        .map_err(|e| format!("Failed to run Python script: {}", e))?;

    // Parse the output as a float (execution time in seconds)
    let result = output.trim().parse::<f64>()
        .map_err(|e| format!("Failed to parse output as float: {}", e))?;

    // Clean up the temporary files
    let _ = std::fs::remove_file(script_path);
    let _ = std::fs::remove_file(data_file_path);

    Ok(result)
}

/// Benchmarks the Python infomeasure package's entropy calculation with multi-dimensional float data.
///
/// # Arguments
///
/// * `data` - The flat array of float data
/// * `dims` - The number of dimensions in the data
/// * `approach` - The approach to use for entropy calculation (e.g., "kernel")
/// * `kwargs` - Additional keyword arguments for the specified approach
/// * `num_runs` - The number of times to run the calculation for more accurate timing
///
/// # Returns
///
/// The average execution time in seconds
pub fn benchmark_entropy_float_nd(data: &[f64], dims: usize, approach: &str, kwargs: &[(String, String)], num_runs: usize) -> Result<f64, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".to_string());
    }

    // Check if data length is divisible by dims
    if data.len() % dims != 0 {
        return Err(format!("Data length ({}) is not divisible by the number of dimensions ({})", data.len(), dims));
    }

    // Calculate the number of samples
    let n_samples = data.len() / dims;

    // Reshape the flat array into a Vec<Vec<f64>> with shape (n_samples, dims)
    let data_vec: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| {
            (0..dims)
                .map(|j| data[i * dims + j])
                .collect()
        })
        .collect();

    // Use the generic function to benchmark entropy calculation
    benchmark_entropy_generic(&data_vec, approach, kwargs, num_runs)
}

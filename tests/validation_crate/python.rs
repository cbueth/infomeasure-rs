//! Python validation module for comparing the Rust implementation with the Python infomeasure package.
//!
//! This module provides functionality for validating the Rust implementation against
//! the Python infomeasure package.

use ndarray::Array1;
use rand::{Rng, thread_rng};
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::process::Command;

/// Checks if the micromamba environment exists.
///
/// # Returns
///
/// `true` if the environment exists, `false` otherwise.
fn environment_exists() -> bool {
    let output = Command::new("micromamba").args(["env", "list"]).output();

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            stdout.contains("infomeasure-rs-validation")
        }
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
        eprintln!("environment.yml not found at {:?}", env_file);
        return false;
    }
    match Command::new("micromamba")
        .args(["env", "create", "-f", env_file.to_str().unwrap()])
        .output()
    {
        Ok(o) if o.status.success() => true,
        Ok(o) => {
            eprintln!(
                "Failed to create env: {}",
                String::from_utf8_lossy(&o.stderr)
            );
            false
        }
        Err(e) => {
            eprintln!("Failed to execute micromamba: {}", e);
            false
        }
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
    let micromamba = which_micromamba()?;
    let out = Command::new(&micromamba)
        .args(["run", "-n", "infomeasure-rs-validation"])
        .args(args)
        .output()
        .map_err(|e| format!("failed to execute: {}", e))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).to_string())
    } else {
        Err(format!(
            "Command failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ))
    }
}

/// Gets the path to the micromamba executable.
///
/// # Returns
///
/// The path to the micromamba executable
fn which_micromamba() -> Result<String, String> {
    let out = Command::new("which")
        .arg("micromamba")
        .output()
        .map_err(|e| format!("which micromamba failed: {}", e))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err("micromamba not found".into())
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
    let uid = format!(
        "{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        thread_rng().r#gen::<u64>()
    );
    let path = temp_dir.join(format!("infomeasure_data_{}.json", uid));
    let json = serde_json::to_string(data).map_err(|e| format!("serialize failed: {}", e))?;
    let mut f = File::create(&path).map_err(|e| format!("create temp file failed: {}", e))?;
    f.write_all(json.as_bytes())
        .map_err(|e| format!("write temp file failed: {}", e))?;
    Ok(path)
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
pub fn calculate_entropy_generic<T: Serialize>(
    data: &[T],
    approach: &str,
    kwargs: &[(String, String)],
) -> Result<f64, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".into());
    }
    let data_file = save_data_to_temp_file(data)?;
    let temp_dir = std::env::temp_dir();
    let uid = format!(
        "{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        thread_rng().r#gen::<u64>()
    );
    let script_path = temp_dir.join(format!("calculate_entropy_{}.py", uid));
    let kwargs_str = kwargs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");
    let script = format!(
        r#"
import infomeasure as im, json
with open("{df}", "r") as f:
    data = json.load(f)
est = im.estimator(data, measure="h", approach="{ap}", base="e", {kw})
print(est.result())
"#,
        df = data_file.to_str().unwrap(),
        ap = approach,
        kw = kwargs_str
    );
    std::fs::write(&script_path, script).map_err(|e| format!("Failed to write script: {}", e))?;
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])?;
    let _ = std::fs::remove_file(script_path);
    let _ = std::fs::remove_file(data_file);
    output
        .trim()
        .parse::<f64>()
        .map_err(|e| format!("Failed to parse output as f64: {} (output='{}')", e, output))
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
pub fn calculate_entropy(
    data: &[i32],
    approach: &str,
    kwargs: &[(String, String)],
) -> Result<f64, String> {
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
pub fn calculate_entropy_float(
    data: &[f64],
    approach: &str,
    kwargs: &[(String, String)],
) -> Result<f64, String> {
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
pub fn calculate_local_entropy_generic<T: Serialize>(
    data: &[T],
    approach: &str,
    kwargs: &[(String, String)],
) -> Result<Vec<f64>, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".into());
    }
    let data_file = save_data_to_temp_file(data)?;
    let temp_dir = std::env::temp_dir();
    let uid = format!(
        "{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        thread_rng().r#gen::<u64>()
    );
    let script_path = temp_dir.join(format!("calculate_local_entropy_{}.py", uid));
    let kwargs_str = kwargs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");
    let script = format!(
        r#"
import infomeasure as im, json
with open("{df}", "r") as f:
    data = json.load(f)
est = im.estimator(data, measure="h", approach="{ap}", base="e", {kw})
print(json.dumps(est.local_vals().tolist()))
"#,
        df = data_file.to_str().unwrap(),
        ap = approach,
        kw = kwargs_str
    );
    std::fs::write(&script_path, script).map_err(|e| format!("Failed to write script: {}", e))?;
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])?;
    let _ = std::fs::remove_file(script_path);
    let _ = std::fs::remove_file(data_file);
    serde_json::from_str::<Vec<f64>>(output.trim())
        .map_err(|e| format!("Failed to parse output as JSON array: {}", e))
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
pub fn calculate_local_entropy(
    data: &[i32],
    approach: &str,
    kwargs: &[(String, String)],
) -> Result<Vec<f64>, String> {
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
pub fn calculate_local_entropy_float(
    data: &[f64],
    approach: &str,
    kwargs: &[(String, String)],
) -> Result<Vec<f64>, String> {
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
pub fn calculate_entropy_float_nd(
    data: &[f64],
    dims: usize,
    approach: &str,
    kwargs: &[(String, String)],
) -> Result<f64, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".into());
    }
    if data.len() % dims != 0 {
        return Err(format!(
            "Data length ({}) is not divisible by the number of dimensions ({})",
            data.len(),
            dims
        ));
    }
    let n = data.len() / dims;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..dims).map(|j| data[i * dims + j]).collect())
        .collect();
    calculate_entropy_generic(&rows, approach, kwargs)
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
pub fn calculate_local_entropy_float_nd(
    data: &[f64],
    dims: usize,
    approach: &str,
    kwargs: &[(String, String)],
) -> Result<Vec<f64>, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".into());
    }
    if data.len() % dims != 0 {
        return Err(format!(
            "Data length ({}) is not divisible by the number of dimensions ({})",
            data.len(),
            dims
        ));
    }
    let n = data.len() / dims;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..dims).map(|j| data[i * dims + j]).collect())
        .collect();
    calculate_local_entropy_generic(&rows, approach, kwargs)
}

/// Calculates cross-entropy using the Python infomeasure package with n-dimensional float data.
///
/// # Arguments
///
/// * `data_p` - The flat array of float data for the first distribution (P)
/// * `data_q` - The flat array of float data for the second distribution (Q)
/// * `dims` - The number of dimensions in the data
/// * `approach` - The approach to use for entropy calculation (e.g., "kernel", "kl")
/// * `kwargs` - Additional keyword arguments for the specified approach
///
/// # Returns
///
/// The cross-entropy value H(P||Q) calculated by the Python infomeasure package
pub fn calculate_cross_entropy_float_nd(
    data_p: &[f64],
    data_q: &[f64],
    dims: usize,
    approach: &str,
    kwargs: &[(String, String)],
) -> Result<f64, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".into());
    }
    if data_p.len() % dims != 0 || data_q.len() % dims != 0 {
        return Err(format!(
            "Data length is not divisible by the number of dimensions ({})",
            dims
        ));
    }
    let n_p = data_p.len() / dims;
    let n_q = data_q.len() / dims;
    let rows_p: Vec<Vec<f64>> = (0..n_p)
        .map(|i| (0..dims).map(|j| data_p[i * dims + j]).collect())
        .collect();
    let rows_q: Vec<Vec<f64>> = (0..n_q)
        .map(|i| (0..dims).map(|j| data_q[i * dims + j]).collect())
        .collect();

    let p_file = save_data_to_temp_file(&rows_p)?;
    let q_file = save_data_to_temp_file(&rows_q)?;

    let temp_dir = std::env::temp_dir();
    let uid = format!(
        "{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        thread_rng().r#gen::<u64>()
    );
    let script_path = temp_dir.join(format!("calculate_cross_entropy_{}.py", uid));
    let kwargs_str = kwargs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");
    let script = format!(
        r#"
import infomeasure as im, json
with open("{pf}", "r") as f:
    p = json.load(f)
with open("{qf}", "r") as f:
    q = json.load(f)
est = im.estimator(p, q, measure="h", approach="{ap}", base="e", {kw})
print(est.result())
"#,
        pf = p_file.to_str().unwrap(),
        qf = q_file.to_str().unwrap(),
        ap = approach,
        kw = kwargs_str
    );
    std::fs::write(&script_path, script).map_err(|e| format!("Failed to write script: {}", e))?;
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])?;
    let _ = std::fs::remove_file(script_path);
    let _ = std::fs::remove_file(p_file);
    let _ = std::fs::remove_file(q_file);
    output
        .trim()
        .parse::<f64>()
        .map_err(|e| format!("Failed to parse output as f64: {} (output='{}')", e, output))
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
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".into());
    }
    let temp_dir = std::env::temp_dir();
    let uid = format!(
        "{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        thread_rng().r#gen::<u64>()
    );
    let script_path = temp_dir.join(format!("benchmark_entropy_{}.py", uid));
    let data_str = data
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let script = format!(
        r#"
import infomeasure as im, timeit
data = [{data_str}]
def run():
    est = im.estimator(data, measure='h', approach='discrete', base='e')
    return est.result()
timer = timeit.Timer(run)
print(timer.timeit(number={num})/{num})
"#,
        data_str = data_str,
        num = num_runs
    );
    std::fs::write(&script_path, script).map_err(|e| format!("Failed to write script: {}", e))?;
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])?;
    let _ = std::fs::remove_file(script_path);
    output
        .trim()
        .parse::<f64>()
        .map_err(|e| format!("Failed to parse output as float: {}", e))
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
pub fn benchmark_entropy_generic<T: Serialize>(
    data: &[T],
    approach: &str,
    kwargs: &[(String, String)],
    num_runs: usize,
) -> Result<f64, String> {
    // Ensure the environment exists
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".into());
    }
    let data_file = save_data_to_temp_file(data)?;
    let temp_dir = std::env::temp_dir();
    let uid = format!(
        "{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        thread_rng().r#gen::<u64>()
    );
    let script_path = temp_dir.join(format!("benchmark_entropy_generic_{}.py", uid));
    let kwargs_str = kwargs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");
    let script = format!(
        r#"
import infomeasure as im, timeit, json
with open("{df}", "r") as f:
    data = json.load(f)
def run():
    est = im.estimator(data, measure='h', approach='{ap}', base='e', {kw})
    return est.result()
timer = timeit.Timer(run)
print(timer.timeit(number={num})/{num})
"#,
        df = data_file.to_str().unwrap(),
        ap = approach,
        kw = kwargs_str,
        num = num_runs
    );
    std::fs::write(&script_path, script).map_err(|e| format!("Failed to write script: {}", e))?;
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])?;
    let _ = std::fs::remove_file(script_path);
    let _ = std::fs::remove_file(data_file);
    output
        .trim()
        .parse::<f64>()
        .map_err(|e| format!("Failed to parse output as float: {}", e))
}

/// Benchmarks the entropy computation for multi-dimensional float data using a specified approach.
///
/// # Parameters
/// - `data`: A slice of floating-point numbers (`f64`) representing the input data.
///           The data must be structured such that it is divided evenly into `dims` dimensions.
/// - `dims`: The number of dimensions for the input data. Each row of the dataset will have `dims` elements.
/// - `approach`: A string specifying the approach or method to be used for entropy calculation.
///               The actual implementation of the entropy computation corresponding to the approach
///               should be provided via the `benchmark_entropy_generic` function.
/// - `kwargs`: A set of key-value pairs (tuples of `String`) representing additional parameters
///             or configurations required by the specified entropy computation approach.
/// - `num_runs`: The number of times to run the entropy calculation for benchmarking purposes.
///               This helps in measuring the average computation performance over multiple runs.
///
/// # Returns
/// This function returns a `Result`:
/// - `Ok(f64)`: The average runtime for the entropy computation over `num_runs` iterations if completed successfully.
/// - `Err(String)`: An error message if an issue arises, such as when the length of the input data is not divisible by `dims`.
///
/// # Errors
/// - If the input data length is not divisible by `dims`, an error `Err(String)` is returned.
///
/// # Panics
/// This function does not expect to panic if used as intended, provided the given parameters are valid.
///
/// # Example
/// ```
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let dims = 2;
/// let approach = "method";
/// let kwargs = vec![("param1".to_string(), "value1".to_string())];
/// let num_runs = 10;
///
/// match benchmark_entropy_float_nd(&data, dims, &approach, &kwargs, num_runs) {
///     Ok(result) => println!("Benchmark completed successfully: {}", result),
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
///
/// # Implementation Details
/// - The function checks if the data is evenly divisible into `dims` dimensions.
/// - Converts the 1D input data slice into a 2D vector where each inner vector represents a row with `dims` elements.
/// - Uses the `benchmark_entropy_generic` function to compute the entropy based on the provided approach and parameters.
/// - The benchmark computes the runtime across `num_runs` iterations and returns the average runtime.
pub fn benchmark_entropy_float_nd(
    data: &[f64],
    dims: usize,
    approach: &str,
    kwargs: &[(String, String)],
    num_runs: usize,
) -> Result<f64, String> {
    if data.len() % dims != 0 {
        return Err(format!(
            "Data length ({}) is not divisible by the number of dimensions ({})",
            data.len(),
            dims
        ));
    }
    let n = data.len() / dims;
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..dims).map(|j| data[i * dims + j]).collect())
        .collect();
    benchmark_entropy_generic(&rows, approach, kwargs, num_runs)
}

/// Calculates the ordinal joint entropy between two floating-point sequences.
///
/// This function accepts two slices of floating-point numbers and calculates their
/// ordinal joint entropy using the Python `infomeasure` library.
///
/// # Arguments
/// * `x` - A slice of `f64` containing the first data sequence.
/// * `y` - A slice of `f64` containing the second data sequence.
/// * `kwargs` - A slice of tuples containing keyword arguments (as `(String, String)` pairs)
///   to be passed into the Python script for configuring the entropy calculation.
///
/// # Returns
/// * `Ok(f64)` containing the calculated ordinal joint entropy if successful.
/// * `Err(String)` containing an error message if the calculation fails at any step.
///
/// # Example
/// ```rust
/// use std::collections::HashMap;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
/// let kwargs = vec![("embedding_dim".to_string(), "2".to_string())];
///
/// match calculate_ordinal_joint_entropy_two_float(&x, &y, &kwargs) {
///     Ok(result) => println!("Ordinal joint entropy: {}", result),
///     Err(e) => println!("Error: {}", e),
/// }
/// ```
pub fn calculate_ordinal_joint_entropy_two_float(
    x: &[f64],
    y: &[f64],
    kwargs: &[(String, String)],
) -> Result<f64, String> {
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".into());
    }
    let x_file = save_data_to_temp_file(x)?;
    let y_file = save_data_to_temp_file(y)?;
    let temp_dir = std::env::temp_dir();
    let uid = format!(
        "{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        thread_rng().r#gen::<u64>()
    );
    let script_path = temp_dir.join(format!("ordinal_joint_entropy_{}.py", uid));
    let kwargs_str = kwargs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");
    let script = format!(
        r#"
import json, numpy as np, infomeasure as im
from infomeasure.estimators.utils.ordinal import symbolize_series, reduce_joint_space
with open("{x_path}", "r") as f:
    x = np.array(json.load(f), dtype=float)
with open("{y_path}", "r") as f:
    y = np.array(json.load(f), dtype=float)
kwargs = dict({kwargs})
_ = im.estimator(x, measure='h', approach='ordinal', base='e', **kwargs)
_ = im.estimator(y, measure='h', approach='ordinal', base='e', **kwargs)
m = int(kwargs.get('embedding_dim'))
stable = bool(kwargs.get('stable', True))
sx = symbolize_series(x, m, 1, to_int=True, stable=stable)
sy = symbolize_series(y, m, 1, to_int=True, stable=stable)
joint = reduce_joint_space((sx, sy))
est_joint = im.estimator(joint, measure='h', approach='discrete', base='e')
print(est_joint.result())
"#,
        x_path = x_file.to_str().unwrap(),
        y_path = y_file.to_str().unwrap(),
        kwargs = kwargs_str
    );
    std::fs::write(&script_path, script)
        .map_err(|e| format!("Failed to write temporary script: {}", e))?;
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])?;
    let _ = std::fs::remove_file(script_path);
    let _ = std::fs::remove_file(x_file);
    let _ = std::fs::remove_file(y_file);
    output.trim().parse::<f64>().map_err(|e| {
        format!(
            "Failed to parse Python output as f64: {} (output='{}')",
            e, output
        )
    })
}

/// Calculate the ordinal cross-entropy between two float arrays.
///
/// # Arguments
///
/// * `x` - A slice of `f64` representing the first input dataset.
/// * `y` - A slice of `f64` representing the second input dataset.
/// * `kwargs` - A slice of key-value pairs (`Vec<(String, String)>`) that represent
///          additional parameters to be passed into the Python script for computation.
///
/// # Returns
///
/// * `Ok(f64)` - The computed ordinal cross-entropy value if the calculation succeeds.
/// * `Err(String)` - An error message if the calculation fails at any step.
///
pub fn calculate_ordinal_cross_entropy_two_float(
    x: &[f64],
    y: &[f64],
    kwargs: &[(String, String)],
) -> Result<f64, String> {
    if !ensure_environment() {
        return Err("Failed to ensure micromamba environment exists".into());
    }
    let x_file = save_data_to_temp_file(x)?;
    let y_file = save_data_to_temp_file(y)?;
    let temp_dir = std::env::temp_dir();
    let uid = format!(
        "{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        thread_rng().r#gen::<u64>()
    );
    let script_path = temp_dir.join(format!("ordinal_cross_entropy_{}.py", uid));
    let kwargs_str = kwargs
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(", ");
    let script = format!(
        r#"
import json, numpy as np, infomeasure as im
from collections import Counter
from infomeasure.estimators.utils.ordinal import symbolize_series
with open("{x_path}", "r") as f:
    x = np.array(json.load(f), dtype=float)
with open("{y_path}", "r") as f:
    y = np.array(json.load(f), dtype=float)
kwargs = dict({kwargs})
_ = im.estimator(x, measure='h', approach='ordinal', base='e', **kwargs)
_ = im.estimator(y, measure='h', approach='ordinal', base='e', **kwargs)
m = int(kwargs.get('embedding_dim'))
stable = bool(kwargs.get('stable', True))
sx = symbolize_series(x, m, 1, to_int=True, stable=stable)
sy = symbolize_series(y, m, 1, to_int=True, stable=stable)
def probs(a):
    c = Counter(a.tolist()); n = float(sum(c.values())); return {{k: v/n for k, v in c.items()}}
px = probs(sx); qy = probs(sy)
keys = set(px.keys()) & set(qy.keys())
import math
print(0.0 if len(keys)==0 else sum([-px[k]*math.log(qy[k]) for k in keys if px[k]>0.0 and qy[k]>0.0]))
"#,
        x_path = x_file.to_str().unwrap(),
        y_path = y_file.to_str().unwrap(),
        kwargs = kwargs_str
    );
    std::fs::write(&script_path, script)
        .map_err(|e| format!("Failed to write temporary script: {}", e))?;
    let output = run_in_environment(&["python", script_path.to_str().unwrap()])?;
    let _ = std::fs::remove_file(script_path);
    let _ = std::fs::remove_file(x_file);
    let _ = std::fs::remove_file(y_file);
    output.trim().parse::<f64>().map_err(|e| {
        format!(
            "Failed to parse Python output as f64: {} (output='{}')",
            e, output
        )
    })
}

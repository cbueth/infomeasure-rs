#![allow(dead_code)]

use std::process::Command;

#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_gb: Option<f64>,
    pub os: String,
}

impl Default for HardwareInfo {
    fn default() -> Self {
        detect_hardware()
    }
}

pub fn detect_hardware() -> HardwareInfo {
    let cpu_model = get_cpu_model();
    let cpu_cores = get_cpu_cores();
    let memory_gb = get_memory_gb();
    let os = get_os();

    HardwareInfo {
        cpu_model,
        cpu_cores,
        memory_gb,
        os,
    }
}

fn get_cpu_model() -> String {
    if cfg!(target_os = "macos") {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    } else if cfg!(target_os = "linux") {
        Command::new("cat")
            .args(["/proc/cpuinfo"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
            })
            .unwrap_or_else(|| "Unknown".to_string())
    } else {
        "Unknown".to_string()
    }
}

fn get_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

fn get_memory_gb() -> Option<f64> {
    if cfg!(target_os = "macos") {
        Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(|bytes| bytes as f64 / (1024.0_f64.powi(3)))
    } else if cfg!(target_os = "linux") {
        Command::new("cat")
            .args(["/proc/meminfo"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<u64>().ok())
            })
            .map(|kb| kb as f64 / (1024.0_f64.powi(2)))
    } else {
        None
    }
}

fn get_os() -> String {
    #[cfg(target_os = "macos")]
    {
        "macOS".to_string()
    }
    #[cfg(target_os = "linux")]
    {
        "Linux".to_string()
    }
    #[cfg(target_os = "windows")]
    {
        "Windows".to_string()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        "Unknown".to_string()
    }
}

impl std::fmt::Display for HardwareInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CPU: {} ({} cores)", self.cpu_model, self.cpu_cores)?;
        if let Some(mem) = self.memory_gb {
            write!(f, ", Memory: {:.1} GB", mem)?;
        }
        write!(f, ", OS: {}", self.os)
    }
}

// Box kernel compute shader for entropy calculation

// Structure for point data
struct GpuPoint {
    values: array<f32, 32>, // Support up to 32 dimensions
};

// Structure for bandwidth
struct GpuBandwidth {
    value: f32,             // Single bandwidth value for all dimensions
    dim_count: u32,         // Actual number of dimensions
    _padding: array<u32, 2>, // Padding to ensure 16-byte alignment
};

// Configuration parameters
struct GpuConfig {
    point_count: u32,
    dim_count: u32,
    normalization: f32,     // N * volume (where volume = bandwidth^dim_count)
    _padding: u32,          // Padding to ensure 16-byte alignment
};

// Bind groups
@group(0) @binding(0) var<storage, read> points: array<GpuPoint>;
@group(0) @binding(1) var<storage, read> bandwidth_info: GpuBandwidth;
@group(0) @binding(2) var<uniform> config: GpuConfig;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Main compute shader entry point
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Check if this thread is within bounds
    if (idx >= config.point_count) {
        return;
    }
    
    // Get the query point
    let query_point = points[idx];
    
    // Count neighbors within bandwidth/2 (L-infinity distance)
    var neighbor_count: f32 = 0.0;
    let r = bandwidth_info.value / 2.0;
    let r_eps = r + 1e-6; // Using slightly larger epsilon for f32
    
    // Loop through all other points
    for (var i: u32 = 0; i < config.point_count; i = i + 1) {
        // Get the neighbor point
        let neighbor_point = points[i];
        
        var in_box: bool = true;
        for (var dim: u32 = 0; dim < config.dim_count; dim = dim + 1) {
            let diff = abs(query_point.values[dim] - neighbor_point.values[dim]);
            if (diff > r_eps) {
                in_box = false;
                break;
            }
        }
        
        if (in_box) {
            neighbor_count += 1.0;
        }
    }
    
    // Normalize the count
    let normalized_count = neighbor_count / config.normalization;
    
    // Apply log transform for entropy calculation: H = -E[log(f(x))]
    // Handle the case where count is zero (should not happen in practice)
    if (normalized_count > 0.0) {
        output[idx] = -log(normalized_count);
    } else {
        output[idx] = 0.0;
    }
}
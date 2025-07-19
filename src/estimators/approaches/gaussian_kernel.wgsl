// Gaussian kernel compute shader for entropy calculation

// Structure for point data
struct GpuPoint {
    values: array<f32, 32>, // Support up to 32 dimensions
};

// Structure for scale factors
struct GpuScaleFactors {
    values: array<f32, 32>, // Support up to 32 dimensions
    dim_count: u32,         // Actual number of dimensions
    _padding: array<u32, 3>, // Padding to ensure 16-byte alignment
};

// Configuration parameters
struct GpuConfig {
    point_count: u32,
    dim_count: u32,
    normalization: f32,
    adaptive_radius: f32,
};

// Bind groups
@group(0) @binding(0) var<storage, read> points: array<GpuPoint>;
@group(0) @binding(1) var<storage, read> scale_factors: GpuScaleFactors;
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
    
    // Calculate density for this point
    var density: f32 = 0.0;
    
    // Loop through all other points
    for (var i: u32 = 0; i < config.point_count; i = i + 1) {
        // Skip self
        if (i == idx) {
            continue;
        }
        
        // Get the neighbor point
        let neighbor_point = points[i];
        
        // Calculate squared distance with improved precision for higher dimensions
        var squared_dist: f32 = 0.0;
        
        // For higher dimensions, we need to be more careful with numerical precision
        // We'll use a more stable summation algorithm (Kahan summation) for dimensions > 2
        if (config.dim_count <= 2u) {
            // Simple summation for 1D and 2D (less overhead)
            for (var dim: u32 = 0; dim < config.dim_count; dim = dim + 1) {
                let diff = query_point.values[dim] - neighbor_point.values[dim];
                let scaled_diff = diff / scale_factors.values[dim];
                squared_dist += scaled_diff * scaled_diff;
            }
        } else {
            // Kahan summation algorithm for higher dimensions to reduce floating-point errors
            var sum: f32 = 0.0;
            var c: f32 = 0.0;  // Running compensation for lost low-order bits
            
            for (var dim: u32 = 0; dim < config.dim_count; dim = dim + 1) {
                let diff = query_point.values[dim] - neighbor_point.values[dim];
                let scaled_diff = diff / scale_factors.values[dim];
                let term = scaled_diff * scaled_diff;
                
                let y = term - c;
                let t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            
            squared_dist = sum;
        }
        
        // For small bandwidths, we need to be more careful with numerical precision
        // Instead of using a hard cutoff, we'll use a soft cutoff that gradually reduces
        // the contribution of distant points
        if (squared_dist <= config.adaptive_radius) {
            // Apply Gaussian kernel: exp(-squared_dist/2)
            density += exp(-squared_dist / 2.0);
        } else if (squared_dist <= config.adaptive_radius * 1.5) {
            // For points slightly outside the adaptive radius, apply a gradual falloff
            // This helps with numerical stability, especially for small bandwidths
            let falloff = (config.adaptive_radius * 1.5 - squared_dist) / (config.adaptive_radius * 0.5);
            density += exp(-squared_dist / 2.0) * falloff;
        }
    }
    
    // Normalize the density
    let normalized_density = density / config.normalization;
    
    // Apply log transform for entropy calculation: H = -E[log(f(x))]
    // Handle the case where density is zero (should not happen in practice)
    if (normalized_density > 0.0) {
        output[idx] = -log(normalized_density);
    } else {
        output[idx] = 0.0;
    }
}
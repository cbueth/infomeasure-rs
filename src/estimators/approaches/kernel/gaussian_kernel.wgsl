// Gaussian kernel compute shader for entropy calculation

// Structure for point data
struct GpuPoint {
    values: array<f32, 32>, // Support up to 32 dimensions
};

// Structure for precision matrix
struct GpuPrecisionMatrix {
    values: array<f32, 1024>, // Support up to 32x32 dimensions
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
@group(0) @binding(1) var<storage, read> precision_matrix: GpuPrecisionMatrix;
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
    var c_density: f32 = 0.0;
    
    // Loop through all other points
    for (var i: u32 = 0; i < config.point_count; i = i + 1) {
        // Get the neighbor point
        let neighbor_point = points[i];
        
        // Calculate squared Euclidean distance first for adaptive radius check (bounding sphere)
        var squared_euclidean_dist: f32 = 0.0;
        var diffs: array<f32, 32>;
        for (var d: u32 = 0; d < config.dim_count; d = d + 1) {
            let diff = query_point.values[d] - neighbor_point.values[d];
            diffs[d] = diff;
            squared_euclidean_dist += diff * diff;
        }
        
        // Check if point is within the circumscribed sphere of the Gaussian ellipsoid
        if (squared_euclidean_dist <= config.adaptive_radius) {
            // Calculate squared Mahalanobis distance: d_M^2 = diff^T * Omega * diff
            var squared_mahalanobis_dist: f32 = 0.0;
            for (var row: u32 = 0; row < config.dim_count; row = row + 1) {
                var row_sum: f32 = 0.0;
                for (var col: u32 = 0; col < config.dim_count; col = col + 1) {
                    // Precision matrix is stored in row-major order with 32 columns
                    row_sum += precision_matrix.values[row * 32u + col] * diffs[col];
                }
                squared_mahalanobis_dist += diffs[row] * row_sum;
            }

            // Apply Gaussian kernel: exp(-squared_mahalanobis_dist/2)
            let term = exp(-max(0.0, squared_mahalanobis_dist) / 2.0);
            
            // Kahan summation for better precision
            let y = term - c_density;
            let t = density + y;
            c_density = (t - density) - y;
            density = t;
        }
    }
    
    // Normalize the density
    let normalized_density = density / config.normalization;
    
    // Apply log transform for entropy calculation: H = -E[log(f(x))]
    if (normalized_density > 0.0) {
        output[idx] = -log(normalized_density);
    } else {
        output[idx] = 0.0;
    }
}
struct Config { rows: u32, cols: u32, min_v: i32, bins: u32 };
@group(0) @binding(0) var<storage, read> data: array<i32>;
@group(0) @binding(1) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> cfg: Config;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = cfg.rows * cfg.cols;
    if (idx >= total) { return; }
    let v: i32 = data[idx];
    let bin_i32: i32 = v - cfg.min_v;
    if (bin_i32 < 0) { return; }
    let bin: u32 = u32(bin_i32);
    if (bin >= cfg.bins) { return; }
    let row: u32 = idx / cfg.cols;
    let offset: u32 = row * cfg.bins + bin;
    atomicAdd(&hist[offset], 1u);
}

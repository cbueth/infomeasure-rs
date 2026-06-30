#![allow(dead_code, unused_mut)]

use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

pub fn generate_uniform_ints(size: usize, num_states: i32, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(0..num_states)).collect()
}

pub fn generate_uniform_floats(size: usize, min: f64, max: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let uniform = Uniform::new(min, max);
    (0..size).map(|_| uniform.sample(&mut rng)).collect()
}

pub fn generate_gaussian_floats(size: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(mean, std).unwrap();
    (0..size).map(|_| normal.sample(&mut rng)).collect()
}

pub fn generate_gaussian_nd(size: usize, dims: usize, mean: f64, std: f64, seed: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(mean, std).unwrap();
    (0..size * dims).map(|_| normal.sample(&mut rng)).collect()
}

pub fn generate_correlated_pair(size: usize, correlation: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(size);
    let mut y = Vec::with_capacity(size);

    for _ in 0..size {
        let z: f64 = rng.sample(Normal::new(0.0, 1.0).unwrap());
        let w: f64 = rng.sample(Normal::new(0.0, 1.0).unwrap());
        let xi = z;
        let yi = correlation * z + (1.0 - correlation.powi(2)).sqrt() * w;
        x.push(xi);
        y.push(yi);
    }

    (x, y)
}

pub fn generate_time_series(size: usize, lag: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut source: Vec<f64> = (0..size + lag).map(|_| normal.sample(&mut rng)).collect();

    let mut target = Vec::with_capacity(size);
    for i in 0..size {
        let val = 0.7 * source[i + lag] + 0.3 * normal.sample(&mut rng);
        target.push(val);
    }

    let source = source[..size].to_vec();
    (source, target)
}

pub fn to_array1(data: &[f64]) -> Array1<f64> {
    Array1::from(data.to_vec())
}

pub fn to_array2(data: &[f64], rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap()
}

pub fn discrete_to_array1(data: &[i32]) -> Array1<i32> {
    Array1::from(data.to_vec())
}

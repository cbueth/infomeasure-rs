//! Evaluating learned representations with information-theoretic measures.
//!
//! `infomeasure` doubles as a model-agnostic evaluation layer for representation
//! learning. Given embeddings produced by an external encoder (simulated here),
//! it measures how much information about each ground-truth latent factor is
//! retained, using the k-NN (KSG) estimator, pointwise (local) MI, and a
//! permutation significance test.
//!
//! The example contrasts a *collapsed* encoder, whose dimensions mostly discard
//! their latent factor, with a *recovered* one, whose dimensions preserve it -
//! the dimensional-collapse failure mode studied in contrastive learning.

use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::stats::{StatisticalTest, StatisticalTestConfig};
use infomeasure::estimators::traits::{GlobalValue, LocalValues};
use ndarray::Array1;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

const N: usize = 400;
const D: usize = 3;
const K: usize = 5;
const NOISE: f64 = 1e-10;

/// Standard normal sample of length `n`.
fn gaussian(rng: &mut StdRng, n: usize) -> Array1<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array1::from_iter((0..n).map(|_| normal.sample(rng)))
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // Ground-truth latent factors Z, one array per factor.
    let z: Vec<Array1<f64>> = (0..D).map(|_| gaussian(&mut rng, N)).collect();

    // Simulated encoders: embedding dimension i is a noisy read-out of latent
    // factor i. The collapsed encoder mixes in mostly noise, the recovered
    // encoder keeps the signal.
    let collapsed: Vec<Array1<f64>> = z
        .iter()
        .map(|zi| {
            let noise = gaussian(&mut rng, N);
            zi.mapv(|v| 0.1 * v) + noise.mapv(|v| 0.99 * v)
        })
        .collect();
    let recovered: Vec<Array1<f64>> = z
        .iter()
        .map(|zi| {
            let noise = gaussian(&mut rng, N);
            zi.clone() + noise.mapv(|v| 0.1 * v)
        })
        .collect();

    println!("Retained information per latent factor (KSG, k = {K}):\n");
    println!("  factor | I(Z; collapsed) | I(Z; recovered)");
    println!("  -------+-----------------+----------------");
    let mut total_collapsed = 0.0;
    let mut total_recovered = 0.0;
    for i in 0..D {
        let mi_collapsed =
            MutualInformation::new_ksg(&[z[i].clone(), collapsed[i].clone()], K, NOISE)
                .global_value();
        let mi_recovered =
            MutualInformation::new_ksg(&[z[i].clone(), recovered[i].clone()], K, NOISE)
                .global_value();
        total_collapsed += mi_collapsed;
        total_recovered += mi_recovered;
        println!("  {i:>6} | {mi_collapsed:>15.4} | {mi_recovered:>15.4}");
    }
    println!("  -------+-----------------+----------------");
    println!("  total  | {total_collapsed:>15.4} | {total_recovered:>15.4}   nats\n");

    // Per-sample (pointwise) analysis of one recovered dimension: which samples
    // carry the most information, plus a permutation test for significance.
    let est = MutualInformation::new_ksg(&[z[0].clone(), recovered[0].clone()], K, NOISE);
    let pointwise = est.local_values();
    let mean = pointwise.mean().unwrap();
    let positive = pointwise.iter().filter(|&&v| v > 0.0).count();
    let min = pointwise.iter().cloned().fold(f64::INFINITY, f64::min);

    println!("Pointwise MI for factor 0 (recovered):");
    println!("  mean = {mean:.4}, #positive = {positive}/{N}, min = {min:.4}");

    let test = est
        .statistical_test(&StatisticalTestConfig::permutation(100).seed(1))
        .unwrap();
    println!(
        "  significance (permutation, {} resamples): p = {:.4}, t = {:.3}",
        test.n_tests, test.p_value, test.t_score
    );
}

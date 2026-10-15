// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parity of the statistical significance test with the Python `infomeasure`
//! package, using an *injected* permutation sequence.
//!
//! Python's resampler draws from `numpy.random.default_rng`, which cannot be
//! reproduced by the Rust RNG, so instead of matching seeds we replace the
//! Python estimator's RNG with a recorder that captures every permutation it
//! applies. Rust then replays the exact same permutations, which lets us compare
//! the null distribution and the derived statistics without any RNG agreement.

use approx::assert_relative_eq;
use infomeasure::estimators::GlobalValue;
use infomeasure::estimators::mutual_information::MutualInformation;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rstest::rstest;
use validation::python;

/// Replays the injected permutations, evaluates the measure for each, and checks
/// it against Python's `statistical_test` output elementwise, including the
/// re-derived *p*-value and *t*-score.
fn assert_statistical_test_parity(
    json: &serde_json::Value,
    observed: f64,
    null_from_permutation: impl Fn(&[usize]) -> f64,
) {
    let test_values = json["test_values"].as_array().expect("test_values");
    assert_relative_eq!(
        observed,
        json["observed_value"].as_f64().expect("observed_value"),
        epsilon = 1e-9,
        max_relative = 1e-9
    );

    let permutations: Vec<Vec<usize>> = json["permutations"]
        .as_array()
        .expect("permutations")
        .iter()
        .map(|p| {
            p.as_array()
                .expect("permutation")
                .iter()
                .map(|i| i.as_u64().expect("index") as usize)
                .collect()
        })
        .collect();
    assert_eq!(permutations.len(), test_values.len());

    let python_null: Vec<f64> = test_values
        .iter()
        .map(|v| v.as_f64().expect("value"))
        .collect();
    let rust_null: Vec<f64> = permutations
        .iter()
        .map(|p| null_from_permutation(p))
        .collect();

    for (rust, py) in rust_null.iter().zip(&python_null) {
        assert_relative_eq!(*rust, *py, epsilon = 1e-9, max_relative = 1e-9);
    }

    let n = rust_null.len() as f64;
    let rust_p = rust_null.iter().filter(|v| **v > observed).count() as f64 / n;
    assert_relative_eq!(
        rust_p,
        json["p_value"].as_f64().expect("p_value"),
        epsilon = 1e-12
    );

    let mean = rust_null.iter().sum::<f64>() / n;
    let variance = rust_null.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = variance.sqrt();
    match json["t_score"].as_f64() {
        Some(python_t) => assert_relative_eq!(
            (observed - mean) / std,
            python_t,
            epsilon = 1e-9,
            max_relative = 1e-9
        ),
        None => assert!(std <= f64::EPSILON),
    }
}

fn continuous(n: usize) -> (Array1<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(11);
    let x: Vec<f64> = (0..n).map(|_| rng.gen_range(-3.0..3.0)).collect();
    let y: Vec<f64> = x.iter().map(|&v| v + rng.gen_range(-0.5..0.5)).collect();
    (Array1::from(x), Array1::from(y))
}

fn symbols(n: usize) -> (Array1<i32>, Array1<i32>) {
    let mut rng = StdRng::seed_from_u64(11);
    let x: Vec<i32> = (0..n).map(|_| rng.gen_range(0..4)).collect();
    let y: Vec<i32> = x
        .iter()
        .map(|&v| {
            if rng.gen_bool(0.8) {
                v
            } else {
                rng.gen_range(0..4)
            }
        })
        .collect();
    (Array1::from(x), Array1::from(y))
}

fn permuted(src: &Array1<f64>, idx: &[usize]) -> Array1<f64> {
    Array1::from(idx.iter().map(|&i| src[i]).collect::<Vec<_>>())
}

type NullDraw = Box<dyn Fn(&[usize]) -> f64>;

#[rstest]
#[case("kernel")]
#[case("ksg")]
#[case("discrete")]
fn statistical_test_mi_matches_python(#[case] approach: &str) {
    const N_TESTS: usize = 16;
    const SEED: u64 = 42;

    let (json, observed, null_fn): (_, f64, NullDraw) = match approach {
        "kernel" => {
            let (x, y) = continuous(48);
            let kwargs = [
                ("kernel".to_string(), "\"gaussian\"".to_string()),
                ("bandwidth".to_string(), "1.0".to_string()),
            ];
            let observed = MutualInformation::new_kernel_with_type(
                &[x.clone(), y.clone()],
                "gaussian".to_string(),
                1.0,
            )
            .global_value();
            let json = python::statistical_test_mi(
                &[x.to_vec(), y.to_vec()],
                "kernel",
                &kwargs,
                N_TESTS,
                SEED,
                "permutation_test",
            )
            .unwrap();
            let null: NullDraw = Box::new(move |idx: &[usize]| {
                MutualInformation::new_kernel_with_type(
                    &[permuted(&x, idx), y.clone()],
                    "gaussian".to_string(),
                    1.0,
                )
                .global_value()
            });
            (json, observed, null)
        }
        "ksg" => {
            let (x, y) = continuous(48);
            let kwargs = [
                ("k".to_string(), "3".to_string()),
                ("noise_level".to_string(), "0".to_string()),
            ];
            let observed =
                MutualInformation::new_ksg(&[x.clone(), y.clone()], 3, 0.0).global_value();
            let json = python::statistical_test_mi(
                &[x.to_vec(), y.to_vec()],
                "ksg",
                &kwargs,
                N_TESTS,
                SEED,
                "permutation_test",
            )
            .unwrap();
            let null: NullDraw = Box::new(move |idx: &[usize]| {
                MutualInformation::new_ksg(&[permuted(&x, idx), y.clone()], 3, 0.0).global_value()
            });
            (json, observed, null)
        }
        "discrete" => {
            let (x, y) = symbols(48);
            let observed =
                MutualInformation::new_discrete_mle(&[x.clone(), y.clone()]).global_value();
            let json = python::statistical_test_mi(
                &[x.to_vec(), y.to_vec()],
                "discrete",
                &[],
                N_TESTS,
                SEED,
                "permutation_test",
            )
            .unwrap();
            let null: NullDraw = Box::new(move |idx: &[usize]| {
                let src = Array1::from(idx.iter().map(|&i| x[i]).collect::<Vec<_>>());
                MutualInformation::new_discrete_mle(&[src, y.clone()]).global_value()
            });
            (json, observed, null)
        }
        other => panic!("unknown approach {other}"),
    };

    assert_statistical_test_parity(&json, observed, null_fn);
}

// The ordinal family is intentionally not covered here: Python's
// `statistical_test` symbolizes once in `__init__` and `_calculate` reuses that
// cached `symbols` array, so permuting `self.data` in `_test_mi` never changes
// the result. Its null distribution is degenerate (every draw equals the
// observed value, so `p_value = 0` and `t_score = NaN`) and therefore cannot be
// compared against a real permutation test.
#[test]
fn python_ordinal_statistical_test_is_degenerate() {
    let (x, y) = continuous(48);
    let json = python::statistical_test_mi(
        &[x.to_vec(), y.to_vec()],
        "ordinal",
        &[
            ("embedding_dim".to_string(), "3".to_string()),
            ("stable".to_string(), "True".to_string()),
        ],
        8,
        42,
        "permutation_test",
    )
    .unwrap();
    let test_values = json["test_values"].as_array().unwrap();
    let observed = json["observed_value"].as_f64().unwrap();
    for value in test_values {
        assert_relative_eq!(value.as_f64().unwrap(), observed, epsilon = 1e-12);
    }
}

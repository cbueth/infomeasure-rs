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
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use infomeasure::new_kernel_te;
use ndarray::{Array1, Axis};
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

/// Asserts that Python's statistical test produced a degenerate null: every
/// draw equals the observed value.
fn assert_python_test_is_degenerate(json: &serde_json::Value) {
    let observed = json["observed_value"].as_f64().expect("observed_value");
    for value in json["test_values"].as_array().expect("test_values") {
        assert_relative_eq!(value.as_f64().unwrap(), observed, epsilon = 1e-12);
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

fn uniform(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from((0..n).map(|_| rng.gen_range(-3.0..3.0)).collect::<Vec<_>>())
}

fn permuted(src: &Array1<f64>, idx: &[usize]) -> Array1<f64> {
    Array1::from(idx.iter().map(|&i| src[i]).collect::<Vec<_>>())
}

/// Python shuffles the embedded source-history rows. For a single source-history
/// sample per window (`SRC_HIST = 1`, `STEP_SIZE = 1`) the shuffled row `i` reads
/// `source[idx[i]]`, so the same null is reproduced by reordering the raw source
/// with `perm[i] = idx[i]`.
fn te_shuffle_to_source_permutation(idx: &[usize], n: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..n).collect();
    for (i, &j) in idx.iter().enumerate() {
        perm[i] = j;
    }
    perm
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

#[rstest]
#[case("kernel")]
#[case("discrete")]
fn statistical_test_cmi_matches_python(#[case] approach: &str) {
    const N_TESTS: usize = 16;
    const SEED: u64 = 42;

    let (json, observed, null_fn): (_, f64, NullDraw) = match approach {
        "kernel" => {
            let (x, y) = continuous(48);
            let z = uniform(48, 21);
            let zv = z.to_vec();
            let kwargs = [
                ("kernel".to_string(), "\"gaussian\"".to_string()),
                ("bandwidth".to_string(), "1.0".to_string()),
            ];
            let observed = MutualInformation::new_cmi_kernel_with_type(
                &[x.clone(), y.clone()],
                &z,
                "gaussian".to_string(),
                1.0,
            )
            .global_value();
            let json = python::statistical_test(
                "cmi",
                &[x.to_vec(), y.to_vec()],
                Some(zv.as_slice()),
                "kernel",
                &kwargs,
                N_TESTS,
                SEED,
                "permutation_test",
            )
            .unwrap();
            let null: NullDraw = Box::new(move |idx: &[usize]| {
                MutualInformation::new_cmi_kernel_with_type(
                    &[permuted(&x, idx), y.clone()],
                    &z,
                    "gaussian".to_string(),
                    1.0,
                )
                .global_value()
            });
            (json, observed, null)
        }
        "discrete" => {
            let (x, y) = symbols(48);
            let z = Array1::from((0..48).map(|i| i % 4).collect::<Vec<_>>());
            let zv = z.to_vec();
            let observed =
                MutualInformation::new_cmi_discrete_mle(&[x.clone(), y.clone()], &z).global_value();
            let json = python::statistical_test(
                "cmi",
                &[x.to_vec(), y.to_vec()],
                Some(zv.as_slice()),
                "discrete",
                &[],
                N_TESTS,
                SEED,
                "permutation_test",
            )
            .unwrap();
            let null: NullDraw = Box::new(move |idx: &[usize]| {
                let src = Array1::from(idx.iter().map(|&i| x[i]).collect::<Vec<_>>());
                MutualInformation::new_cmi_discrete_mle(&[src, y.clone()], &z).global_value()
            });
            (json, observed, null)
        }
        other => panic!("unknown approach {other}"),
    };

    assert_statistical_test_parity(&json, observed, null_fn);
}

// Transfer entropy and conditional transfer entropy parity is checked for a
// single source-history sample per window (`SRC_HIST = 1`, `STEP_SIZE = 1`).
// There, Python's row shuffle of the embedded source history is equivalent to
// reordering the raw source, so `te_shuffle_to_source_permutation` reproduces
// it. For `SRC_HIST > 1` Rust resamples the raw source before embedding while
// Python permutes the embedded windows, so the two null distributions differ;
// that case is not covered here.
#[rstest]
#[case("kernel")]
#[case("discrete")]
fn statistical_test_te_matches_python(#[case] approach: &str) {
    const N_TESTS: usize = 16;
    const SEED: u64 = 42;

    let (json, observed, null_fn): (_, f64, NullDraw) = match approach {
        "kernel" => {
            let (s, d) = continuous(48);
            let s2 = s.clone().insert_axis(Axis(1));
            let d2 = d.clone().insert_axis(Axis(1));
            let kwargs = [
                ("kernel".to_string(), "\"gaussian\"".to_string()),
                ("bandwidth".to_string(), "1.0".to_string()),
            ];
            let observed =
                new_kernel_te!(&s2, &d2, 1, 1, 1, 1, 1, "gaussian".to_string(), 1.0).global_value();
            let json = python::statistical_test(
                "te",
                &[s.to_vec(), d.to_vec()],
                None,
                "kernel",
                &kwargs,
                N_TESTS,
                SEED,
                "permutation_test",
            )
            .unwrap();
            let n = s.len();
            let null: NullDraw = Box::new(move |idx: &[usize]| {
                let perm = te_shuffle_to_source_permutation(idx, n);
                let ps = Array1::from(perm.iter().map(|&i| s[i]).collect::<Vec<_>>())
                    .insert_axis(Axis(1));
                new_kernel_te!(&ps, &d2, 1, 1, 1, 1, 1, "gaussian".to_string(), 1.0).global_value()
            });
            (json, observed, null)
        }
        "discrete" => {
            let (s, d) = symbols(48);
            let observed = TransferEntropy::new_discrete_mle(&s, &d, 1, 1, 1).global_value();
            let json = python::statistical_test(
                "te",
                &[s.to_vec(), d.to_vec()],
                None,
                "discrete",
                &[],
                N_TESTS,
                SEED,
                "permutation_test",
            )
            .unwrap();
            let n = s.len();
            let null: NullDraw = Box::new(move |idx: &[usize]| {
                let perm = te_shuffle_to_source_permutation(idx, n);
                let ps = Array1::from(perm.iter().map(|&i| s[i]).collect::<Vec<_>>());
                TransferEntropy::new_discrete_mle(&ps, &d, 1, 1, 1).global_value()
            });
            (json, observed, null)
        }
        other => panic!("unknown approach {other}"),
    };

    assert_statistical_test_parity(&json, observed, null_fn);
}

// Conditional transfer entropy is not compared for the same reason as the
// ordinal family: Python's kernel and discrete CTE estimators call
// `cte_observations` without forwarding `permute_src`, so `_test_te` sets the
// RNG but the source is never permuted and the null is degenerate.
#[rstest]
#[case("kernel")]
#[case("discrete")]
fn python_cte_statistical_test_is_degenerate(#[case] approach: &str) {
    let json = match approach {
        "kernel" => {
            let (s, d) = continuous(48);
            let c = uniform(48, 31);
            python::statistical_test(
                "cte",
                &[s.to_vec(), d.to_vec()],
                Some(c.as_slice().unwrap()),
                "kernel",
                &[
                    ("kernel".to_string(), "\"gaussian\"".to_string()),
                    ("bandwidth".to_string(), "1.0".to_string()),
                ],
                8,
                42,
                "permutation_test",
            )
            .unwrap()
        }
        "discrete" => {
            let (s, d) = symbols(48);
            let c = Array1::from((0..48).map(|i| i % 4).collect::<Vec<_>>());
            python::statistical_test(
                "cte",
                &[s.to_vec(), d.to_vec()],
                Some(c.as_slice().unwrap()),
                "discrete",
                &[],
                8,
                42,
                "permutation_test",
            )
            .unwrap()
        }
        other => panic!("unknown approach {other}"),
    };

    assert!(
        json["permutations"]
            .as_array()
            .expect("permutations")
            .is_empty(),
        "expected no recorded resampling for CTE"
    );
    assert_python_test_is_degenerate(&json);
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
    assert_python_test_is_degenerate(&json);
}

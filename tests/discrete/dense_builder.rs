// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Builder API for the dense direct discrete-MLE path.
//!
//! The timing-optimised builder must return the *same* values as the generic
//! constructors, with or without a declared alphabet, and the global-only
//! terminal must refuse local values instead of returning garbage.

use approx::assert_relative_eq;
use infomeasure::estimators::mutual_information::MutualInformation;
use infomeasure::estimators::traits::{GlobalValue, LocalValues, OptionalLocalValues};
use infomeasure::estimators::transfer_entropy::TransferEntropy;
use ndarray::Array1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rstest::rstest;

fn codes(n: usize, states: i32, seed: u64) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen_range(0..states)).collect()
}

fn assert_close(a: f64, b: f64, ctx: &str) {
    let ok = (a - b).abs() <= 1e-12 + 1e-10 * a.abs().max(b.abs());
    assert!(ok, "{ctx}: {a} vs {b}");
}

/// Declaring the alphabet must not change the value, and the builder must agree
/// with the generic constructor.
#[rstest]
fn cmi_builder_global_matches_generic(#[values(5, 10, 25)] states: i32) {
    let x = codes(400, states, 1);
    let y = codes(400, states, 2);
    let z = codes(400, states, 3);

    let generic = MutualInformation::new_cmi_discrete_mle(
        &[Array1::from(x.clone()), Array1::from(y.clone())],
        &Array1::from(z.clone()),
    )
    .global_value();

    let inferred = MutualInformation::cmi_discrete_mle(&[&x, &y], &z)
        .global_only()
        .global_value();
    let known = MutualInformation::cmi_discrete_mle(&[&x, &y], &z)
        .with_alphabet(states as usize)
        .global_only()
        .global_value();

    assert_close(generic, inferred, "inferred");
    assert_close(generic, known, "known alphabet");
}

/// The local-capable terminal reproduces the generic local values too.
#[rstest]
fn cmi_builder_build_matches_generic_local(#[values(2, 3, 5)] states: i32) {
    let x = codes(300, states, 11);
    let y = codes(300, states, 12);
    let z = codes(300, states, 13);

    let generic = MutualInformation::new_cmi_discrete_mle(
        &[Array1::from(x.clone()), Array1::from(y.clone())],
        &Array1::from(z.clone()),
    )
    .local_values();

    let built = MutualInformation::cmi_discrete_mle(&[&x, &y], &z)
        .with_alphabet(states as usize)
        .build();
    let built_global = built.global_value();
    let built_local = built.local_values();

    let generic_global = MutualInformation::new_cmi_discrete_mle(
        &[Array1::from(x.clone()), Array1::from(y.clone())],
        &Array1::from(z.clone()),
    )
    .global_value();

    assert_close(generic_global, built_global, "global");
    assert_eq!(generic.len(), built_local.len());
    for (a, b) in generic.iter().zip(built_local.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-12, max_relative = 1e-10);
    }
}

/// The global-only terminal must not pretend to support local values.
#[test]
fn cmi_global_only_refuses_local_values() {
    let x = codes(50, 4, 21);
    let y = codes(50, 4, 22);
    let z = codes(50, 4, 23);

    let global = MutualInformation::cmi_discrete_mle(&[&x, &y], &z)
        .with_alphabet(4)
        .global_only();

    assert!(!global.supports_local());
    assert!(global.local_values_opt().is_err());
    assert_close(
        global.global_value(),
        MutualInformation::cmi_discrete_mle(&[&x, &y], &z)
            .with_alphabet(4)
            .build()
            .global_value(),
        "global-only vs local terminal",
    );
}

/// Transfer entropy: builder matches the generic constructor across embed
/// parameters, including strided (step > 1) embeddings.
#[rstest]
fn te_builder_matches_generic(
    #[values((1, 1, 1), (2, 1, 1), (1, 2, 2), (2, 2, 3), (3, 2, 1))] params: (usize, usize, usize),
) {
    let (l, k, tau) = params;
    let states = 4;
    let source = codes(500, states, 31);
    let dest = codes(500, states, 32);

    let generic = TransferEntropy::new_discrete_mle(
        &Array1::from(source.clone()),
        &Array1::from(dest.clone()),
        l,
        k,
        tau,
    )
    .global_value();

    let built = TransferEntropy::te_discrete_mle(&source, &dest, l, k, tau)
        .with_alphabet(states as usize)
        .global_only()
        .global_value();

    assert_close(generic, built, "te");
}

/// Conditional transfer entropy: builder matches the generic constructor.
#[rstest]
fn cte_builder_matches_generic(
    #[values((1, 1, 1, 1), (2, 1, 1, 1), (1, 2, 2, 2), (2, 2, 1, 3))] params: (
        usize,
        usize,
        usize,
        usize,
    ),
) {
    let (l, k, m, tau) = params;
    let states = 3;
    let source = codes(400, states, 41);
    let dest = codes(400, states, 42);
    let cond = codes(400, states, 43);

    let generic = TransferEntropy::new_cte_discrete_mle(
        &Array1::from(source.clone()),
        &Array1::from(dest.clone()),
        &Array1::from(cond.clone()),
        l,
        k,
        m,
        tau,
    )
    .global_value();

    let built = TransferEntropy::cte_discrete_mle(&source, &dest, &cond, l, k, m, tau)
        .with_alphabet(states as usize)
        .global_only()
        .global_value();

    assert_close(generic, built, "cte");
}

/// Beyond the dense cap the builder must fall back to the generic estimator
/// and still return the same value.
#[test]
fn cmi_builder_falls_back_beyond_cap() {
    let states = 128usize;
    let x = codes(300, states as i32, 51);
    let y = codes(300, states as i32, 52);
    let z = codes(300, states as i32, 53);

    let generic = MutualInformation::new_cmi_discrete_mle(
        &[Array1::from(x.clone()), Array1::from(y.clone())],
        &Array1::from(z.clone()),
    )
    .global_value();

    let global = MutualInformation::cmi_discrete_mle(&[&x, &y], &z)
        .with_alphabet(states)
        .global_only();
    assert!(!global.supports_local());
    assert_close(generic, global.global_value(), "fallback global");

    let built = MutualInformation::cmi_discrete_mle(&[&x, &y], &z)
        .with_alphabet(states)
        .build()
        .global_value();
    assert_close(generic, built, "fallback build");
}

/// Sanity: conditionally independent series give (near) zero CMI.
#[test]
fn cmi_conditionally_independent_is_zero() {
    let states = 5usize;
    let z = codes(500, 3, 61);
    let x = codes(500, states as i32, 62);
    let y = codes(500, states as i32, 63);

    let cmi = MutualInformation::cmi_discrete_mle(&[&x, &y], &z)
        .with_alphabet(states)
        .global_only()
        .global_value();
    assert!(cmi.abs() < 0.05, "cmi={cmi}");
}

/// MI: declared alphabet and global-only terminal agree with the generic
/// constructor, for bivariate and multivariate variants.
#[rstest]
fn mi_builder_global_matches_generic(#[values(2, 5, 10)] states: i32) {
    let x = codes(400, states, 71);
    let y = codes(400, states, 72);
    let w = codes(400, states, 73);

    let generic_xy =
        MutualInformation::new_discrete_mle(&[Array1::from(x.clone()), Array1::from(y.clone())])
            .global_value();
    let builder_xy = MutualInformation::mi_discrete_mle(&[&x, &y])
        .with_alphabet(states as usize)
        .global_only()
        .global_value();
    assert_close(generic_xy, builder_xy, "mi xy");

    let generic_xyw = MutualInformation::new_discrete_mle(&[
        Array1::from(x.clone()),
        Array1::from(y.clone()),
        Array1::from(w.clone()),
    ])
    .global_value();
    let builder_xyw = MutualInformation::mi_discrete_mle(&[&x, &y, &w])
        .with_alphabet(states as usize)
        .global_only()
        .global_value();
    assert_close(generic_xyw, builder_xyw, "mi xyw");

    let inferred = MutualInformation::mi_discrete_mle(&[&x, &y])
        .global_only()
        .global_value();
    assert_close(generic_xy, inferred, "mi inferred");
}

/// MI: the local-capable terminal reproduces the generic local values.
#[test]
fn mi_builder_build_matches_generic_local() {
    let states = 4i32;
    let x = codes(300, states, 81);
    let y = codes(300, states, 82);

    let generic =
        MutualInformation::new_discrete_mle(&[Array1::from(x.clone()), Array1::from(y.clone())]);
    let generic_local = generic.local_values();

    let built = MutualInformation::mi_discrete_mle(&[&x, &y])
        .with_alphabet(states as usize)
        .build();
    assert_close(generic.global_value(), built.global_value(), "mi global");
    let built_local = built.local_values();
    assert_eq!(generic_local.len(), built_local.len());
    for (a, b) in generic_local.iter().zip(built_local.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-12, max_relative = 1e-10);
    }
}

/// MI: the global-only terminal refuses local values and matches the local
/// terminal's average.
#[test]
fn mi_global_only_refuses_local_values() {
    let x = codes(50, 3, 91);
    let y = codes(50, 3, 92);
    let global = MutualInformation::mi_discrete_mle(&[&x, &y])
        .with_alphabet(3)
        .global_only();
    assert!(!global.supports_local());
    assert!(global.local_values_opt().is_err());
    assert_close(
        global.global_value(),
        MutualInformation::mi_discrete_mle(&[&x, &y])
            .with_alphabet(3)
            .build()
            .global_value(),
        "mi global-only vs local",
    );
}

/// MI: beyond the dense cap the builder falls back to the generic estimator.
#[test]
fn mi_builder_falls_back_beyond_cap() {
    let states = 2048usize;
    let x = codes(300, 16, 93); // only 16 used states, but declared alphabet is huge
    let y = codes(300, 16, 94);

    // With a declared alphabet far above the cap, both terminals must fall back
    // and agree with the generic value.
    let global = MutualInformation::mi_discrete_mle(&[&x, &y])
        .with_alphabet(states)
        .global_only();
    assert!(!global.supports_local());

    let inferred = MutualInformation::mi_discrete_mle(&[&x, &y])
        .global_only()
        .global_value();
    let built = MutualInformation::mi_discrete_mle(&[&x, &y])
        .with_alphabet(states)
        .build()
        .global_value();
    // Fallback computes the same reduced-code MI as the inferred dense path.
    assert_close(inferred, built, "mi fallback");
}

/// Sanity: independent MI is (near) zero, self-MI equals the entropy.
#[test]
fn mi_independent_is_zero_and_self_is_entropy() {
    let x = codes(500, 5, 95);
    let y = codes(500, 5, 96);
    let mi = MutualInformation::mi_discrete_mle(&[&x, &y])
        .with_alphabet(5)
        .global_only()
        .global_value();
    assert!(mi.abs() < 0.05, "mi={mi}");

    let self_mi = MutualInformation::mi_discrete_mle(&[&x, &x])
        .with_alphabet(5)
        .global_only()
        .global_value();
    let h = infomeasure::estimators::entropy::Entropy::new_discrete_from_slice(&x).global_value();
    assert_close(self_mi, h, "I(x;x)=H(x)");
}

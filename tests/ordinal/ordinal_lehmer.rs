// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use infomeasure::estimators::approaches::ordinal::ordinal_utils::lehmer_code;

#[test]
fn test_lehmer_code() {
    // Empty slice should result in 0
    assert_eq!(lehmer_code(&[]), 0);
    // Identity permutations should result in 0; reversed ones in (n!-1)
    for n in 1..=20 {
        let perm: Vec<usize> = (0..n).collect();
        let perm_reversed: Vec<usize> = perm.clone().into_iter().rev().collect();
        assert_eq!(
            lehmer_code(&perm),
            0,
            "identity permutation mismatch for n={}",
            n
        );

        // Use integer math for the expected value to avoid f64 precision limits
        let expected_fact: u64 = (1..=n as u64).product();

        // the reversed order is always the maximum code:
        assert_eq!(
            lehmer_code(&perm_reversed),
            expected_fact - 1,
            "reversed permutation mismatch for n={}, code={}",
            n,
            lehmer_code(&perm_reversed)
        );
    }
    assert_eq!(lehmer_code(&[1, 0]), 1);
    assert_eq!(lehmer_code(&[0, 1, 2, 3, 4, 5, 7, 6]), 1);
    assert_eq!(lehmer_code(&[0, 1, 2, 3, 4, 6, 5, 7]), 2);
    assert_eq!(lehmer_code(&[0, 1, 2, 3, 5, 4, 6, 7]), 6);
    assert_eq!(lehmer_code(&[0, 1, 2, 4, 3, 5, 6, 7]), 24);
    assert_eq!(lehmer_code(&[0, 1, 3, 2, 4, 5, 6, 7]), 120);

    // Known values for n=3
    // [0, 1, 2] -> 0
    // [0, 2, 1] -> 0*2! + 1*1! + 0*0! = 1
    // [1, 0, 2] -> 1*2! + 0*1! + 0*0! = 2
    // [1, 2, 0] -> 1*2! + 1*1! + 0*0! = 3
    // [2, 0, 1] -> 2*2! + 0*1! + 0*0! = 4
    // [2, 1, 0] -> 2*2! + 1*1! + 0*0! = 5
    assert_eq!(lehmer_code(&[0, 2, 1]), 1);
    assert_eq!(lehmer_code(&[1, 0, 2]), 2);
    assert_eq!(lehmer_code(&[1, 2, 0]), 3);
    assert_eq!(lehmer_code(&[2, 0, 1]), 4);
    assert_eq!(lehmer_code(&[2, 1, 0]), 5);
    // invalid orders are not being raised, but just straight up calculated
    assert_eq!(lehmer_code(&[0, 1, 3]), 0);
    assert_eq!(lehmer_code(&[1, 1, 1, 1]), 0);
    assert_eq!(lehmer_code(&[15, 3, 2]), 5);
}

#[test]
#[should_panic(expected = "For embedding dimensions larger than 20")]
fn test_lehmer_code_overflow_panic() {
    let perm: Vec<usize> = (0..21).collect();
    lehmer_code(&perm);
}

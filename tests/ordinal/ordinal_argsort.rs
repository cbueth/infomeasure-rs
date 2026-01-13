use infomeasure::estimators::approaches::ordinal::ordinal_utils::argsort;

#[test]
fn test_argsort_basic() {
    let window = [3.0, 1.0, 4.0, 2.0];
    let mut idx = [0usize; 4];
    argsort(&window, &mut idx, true);
    assert_eq!(idx, [1, 3, 0, 2]);

    argsort(&window, &mut idx, false);
    assert_eq!(idx, [1, 3, 0, 2]);
}

#[test]
fn test_argsort_stable_with_ties() {
    let window = [1.0, 2.0, 1.0, 0.0];
    let mut idx = [0usize; 4];
    // Indices of 1.0 are 0 and 2.
    // Stable sort should keep them as 0 then 2.
    argsort(&window, &mut idx, true);
    assert_eq!(idx, [3, 0, 2, 1]);
}

#[test]
fn test_argsort_unstable_with_ties() {
    let window = [1.0, 2.0, 1.0, 0.0];
    let mut idx = [0usize; 4];
    // Unstable sort doesn't guarantee order of ties.
    argsort(&window, &mut idx, false);

    // Verify that it is a valid sort
    for i in 0..3 {
        assert!(window[idx[i]] <= window[idx[i + 1]]);
    }

    // In this specific case, 0.0 is at index 3, 2.0 is at index 1.
    assert_eq!(idx[0], 3);
    assert_eq!(idx[3], 1);
    // idx[1] and idx[2] must be 0 and 2 in some order.
    assert!((idx[1] == 0 && idx[2] == 2) || (idx[1] == 2 && idx[2] == 0));
}

#[test]
fn test_argsort_empty() {
    let window: [f64; 0] = [];
    let mut idx: [usize; 0] = [];
    argsort(&window, &mut idx, true);
    // Should not panic
}

#[test]
fn test_argsort_single_element() {
    let window = [42.0];
    let mut idx = [0usize];
    argsort(&window, &mut idx, true);
    assert_eq!(idx, [0]);
}

#[test]
fn test_argsort_nan() {
    let window = [1.0, f64::NAN, 0.0];
    let mut idx = [0usize; 3];
    // Our implementation treats NaN as "Greater" than everything.

    // Stable:
    argsort(&window, &mut idx, true);
    // 0.0 (idx 2) < 1.0 (idx 0) < NaN (idx 1)
    assert_eq!(idx, [2, 0, 1]);

    // Unstable:
    argsort(&window, &mut idx, false);
    assert_eq!(idx, [2, 0, 1]);
}

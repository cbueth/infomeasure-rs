// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configuration for statistical significance tests.

/// Resampling scheme for a [`super::StatisticalTest`].
///
/// [`TestMethod::Permutation`] and [`TestMethod::Bootstrap`] mirror the Python
/// `infomeasure` package's `method` argument (`"permutation_test"` /
/// `"bootstrap"`) as an enum instead of a string. The remaining variants are
/// Rust-only extensions for **autocorrelated time series**, where an
/// independent shuffle destroys the serial dependence that the measure is
/// supposed to detect.
///
/// # Block-based surrogates
///
/// - [`BlockPermutation`](TestMethod::BlockPermutation) splits the source into
///   contiguous blocks of `block_len` samples (the last block may be shorter),
///   shuffles the block order, and concatenates the blocks. Within-block order
///   and the marginal multiset are preserved exactly, so short-range
///   autocorrelation survives the shuffle while long-range dependence between
///   source and target is broken.
/// - [`CircularShift`](TestMethod::CircularShift) rotates the source by a
///   uniformly random offset. It is a valid surrogate for stationary series
///   because the circular autocorrelation is preserved exactly; however, it
///   also preserves any periodicity in the series and only produces `n`
///   distinct surrogates (including the identity).
/// - [`BlockBootstrap`](TestMethod::BlockBootstrap) draws contiguous blocks of
///   the source with replacement until `n` samples are collected, then truncates
///   to `n`. It preserves short-range source structure but breaks the
///   source-target pairing. It is a source-resampling null, not a joint
///   bootstrap, and it should not be read as a confidence interval.
///
/// A `block_len` of zero is invalid and makes a statistical test return
/// [`StatsError::InvalidBlockLength`](crate::estimators::stats::StatsError::InvalidBlockLength).
/// Values larger than the series length are effectively clamped: block
/// permutation becomes the identity (one block), and block bootstrap draws one
/// full-length block, which is also the identity resample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TestMethod {
    /// Shuffle the source variable to break any dependence with the target.
    /// The null hypothesis is that the observed measure is indistinguishable
    /// from chance.
    #[default]
    Permutation,
    /// Resample the source with replacement.
    ///
    /// This breaks the source-target pairing, so it is a null-like resampling
    /// rather than a joint bootstrap. It should not be read as a confidence
    /// interval for the estimate.
    Bootstrap,
    /// Shuffle contiguous blocks of the source, preserving within-block order.
    ///
    /// This is the appropriate null test for autocorrelated series: it destroys
    /// the cross-variable dependence while retaining the short-range serial
    /// structure inside each block.
    BlockPermutation {
        /// Block length in samples. Must be `>= 1`; values `>= n` reduce to a
        /// single block and therefore to the identity permutation.
        block_len: usize,
    },
    /// Rotate the source by a uniformly random offset in `0..n`.
    ///
    /// The identity rotation (offset `0`) is a valid surrogate, so a null
    /// distribution built from circular shifts can contain the observed series
    /// itself.
    CircularShift,
    /// Resample contiguous blocks of the source with replacement.
    ///
    /// The output has the same length as the input. This preserves short-range
    /// source structure but breaks the source-target pairing. It is a
    /// source-resampling null, not a joint bootstrap, and it should not be read
    /// as a confidence interval for the estimate.
    BlockBootstrap {
        /// Block length in samples. Must be `>= 1`; values `>= n` reduce to a
        /// single full-length block, i.e. the identity resample.
        block_len: usize,
    },
}

impl TestMethod {
    /// The string form of the method.
    ///
    /// The permutation and bootstrap forms match the Python package; the
    /// block-based names are Rust-only extensions.
    pub fn as_str(self) -> &'static str {
        match self {
            TestMethod::Permutation => "permutation_test",
            TestMethod::Bootstrap => "bootstrap",
            TestMethod::BlockPermutation { .. } => "block_permutation",
            TestMethod::CircularShift => "circular_shift",
            TestMethod::BlockBootstrap { .. } => "block_bootstrap",
        }
    }

    /// The configured block length, or `None` for methods without blocks.
    pub fn block_len(self) -> Option<usize> {
        match self {
            TestMethod::BlockPermutation { block_len }
            | TestMethod::BlockBootstrap { block_len } => Some(block_len),
            TestMethod::Permutation | TestMethod::Bootstrap | TestMethod::CircularShift => None,
        }
    }

    /// Whether the surrogate preserves the source's marginal multiset.
    ///
    /// Permutation, block permutation, and circular shift reorder the source
    /// and therefore leave every marginal term (and its cached estimate)
    /// unchanged. Bootstrap and block bootstrap resample with replacement,
    /// which changes the empirical marginal distribution, so cached marginal
    /// values are no longer valid.
    pub fn preserves_marginal_multiset(self) -> bool {
        matches!(
            self,
            TestMethod::Permutation
                | TestMethod::BlockPermutation { .. }
                | TestMethod::CircularShift
        )
    }
}

/// Settings for a statistical test.
///
/// The defaults match the Python package: 100 resamples, the permutation
/// method, and a non-deterministic RNG. Use [`StatisticalTestConfig::seed`] to
/// make a run reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatisticalTestConfig {
    /// Number of resamples (permutations or bootstrap draws). Must be `>= 2`.
    pub n_tests: usize,
    /// The resampling method.
    pub method: TestMethod,
    /// RNG seed. `None` seeds from entropy; `Some` makes runs reproducible.
    pub seed: Option<u64>,
}

impl StatisticalTestConfig {
    /// The default number of resamples.
    pub const DEFAULT_N_TESTS: usize = 100;

    /// A configuration with `n_tests` resamples and the given method.
    pub fn new(n_tests: usize, method: TestMethod) -> Self {
        Self {
            n_tests,
            method,
            seed: None,
        }
    }

    /// A permutation test with `n_tests` resamples.
    pub fn permutation(n_tests: usize) -> Self {
        Self::new(n_tests, TestMethod::Permutation)
    }

    /// A bootstrap test with `n_tests` resamples.
    pub fn bootstrap(n_tests: usize) -> Self {
        Self::new(n_tests, TestMethod::Bootstrap)
    }

    /// A block-permutation test with `n_tests` resamples and blocks of
    /// `block_len` contiguous samples.
    ///
    /// The null hypothesis is still independence between source and target, but
    /// the surrogate keeps the within-block serial order, which is what makes
    /// the test valid for autocorrelated series. Choose `block_len` from the
    /// decorrelation time of the data; values `>= n` collapse to the identity
    /// permutation (one block), making the test degenerate.
    ///
    /// # Panics
    ///
    /// Does not panic. A `block_len` of zero is rejected when the test runs,
    /// returning
    /// [`StatsError::InvalidBlockLength`](crate::estimators::stats::StatsError::InvalidBlockLength).
    ///
    /// # Examples
    ///
    /// ```
    /// use infomeasure::estimators::stats::{StatisticalTestConfig, TestMethod};
    ///
    /// let cfg = StatisticalTestConfig::block_permutation(200, 20).seed(7);
    /// assert_eq!(
    ///     cfg.method,
    ///     TestMethod::BlockPermutation { block_len: 20 }
    /// );
    /// assert_eq!(cfg.method.as_str(), "block_permutation");
    /// ```
    pub fn block_permutation(n_tests: usize, block_len: usize) -> Self {
        Self::new(n_tests, TestMethod::BlockPermutation { block_len })
    }

    /// A circular-shift test with `n_tests` resamples.
    ///
    /// Each surrogate rotates the source series by a uniformly random offset.
    /// This preserves the full circular autocorrelation, so it is a good null
    /// for stationary autocorrelated series without a strong periodicity. The
    /// identity rotation is a valid surrogate and is included in the support.
    ///
    /// # Examples
    ///
    /// ```
    /// use infomeasure::estimators::stats::{StatisticalTestConfig, TestMethod};
    ///
    /// let cfg = StatisticalTestConfig::circular_shift(100).seed(1);
    /// assert_eq!(cfg.method, TestMethod::CircularShift);
    /// ```
    pub fn circular_shift(n_tests: usize) -> Self {
        Self::new(n_tests, TestMethod::CircularShift)
    }

    /// A block-bootstrap test with `n_tests` resamples and blocks of
    /// `block_len` contiguous samples.
    ///
    /// The output length always equals the input length: blocks are drawn with
    /// replacement until `n` samples are collected and the last block is
    /// truncated if necessary. This is a block-wise source-resampling null, not
    /// a joint bootstrap or a confidence interval.
    ///
    /// # Panics
    ///
    /// Does not panic. A `block_len` of zero is rejected when the test runs,
    /// returning
    /// [`StatsError::InvalidBlockLength`](crate::estimators::stats::StatsError::InvalidBlockLength).
    ///
    /// # Examples
    ///
    /// ```
    /// use infomeasure::estimators::stats::{StatisticalTestConfig, TestMethod};
    ///
    /// let cfg = StatisticalTestConfig::block_bootstrap(100, 10).seed(3);
    /// assert_eq!(
    ///     cfg.method,
    ///     TestMethod::BlockBootstrap { block_len: 10 }
    /// );
    /// ```
    pub fn block_bootstrap(n_tests: usize, block_len: usize) -> Self {
        Self::new(n_tests, TestMethod::BlockBootstrap { block_len })
    }

    /// Set the number of resamples.
    pub fn with_n_tests(mut self, n_tests: usize) -> Self {
        self.n_tests = n_tests;
        self
    }

    /// Set the resampling method.
    pub fn with_method(mut self, method: TestMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the RNG seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for StatisticalTestConfig {
    fn default() -> Self {
        Self {
            n_tests: Self::DEFAULT_N_TESTS,
            method: TestMethod::Permutation,
            seed: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn method_strings_match_python() {
        assert_eq!(TestMethod::Permutation.as_str(), "permutation_test");
        assert_eq!(TestMethod::Bootstrap.as_str(), "bootstrap");
        assert_eq!(
            TestMethod::BlockPermutation { block_len: 5 }.as_str(),
            "block_permutation"
        );
        assert_eq!(TestMethod::CircularShift.as_str(), "circular_shift");
        assert_eq!(
            TestMethod::BlockBootstrap { block_len: 5 }.as_str(),
            "block_bootstrap"
        );
    }

    #[test]
    fn block_len_accessor() {
        assert_eq!(TestMethod::Permutation.block_len(), None);
        assert_eq!(TestMethod::Bootstrap.block_len(), None);
        assert_eq!(TestMethod::CircularShift.block_len(), None);
        assert_eq!(
            TestMethod::BlockPermutation { block_len: 4 }.block_len(),
            Some(4)
        );
        assert_eq!(
            TestMethod::BlockBootstrap { block_len: 9 }.block_len(),
            Some(9)
        );
    }

    #[test]
    fn only_reordering_methods_preserve_the_marginal_multiset() {
        assert!(TestMethod::Permutation.preserves_marginal_multiset());
        assert!(TestMethod::BlockPermutation { block_len: 4 }.preserves_marginal_multiset());
        assert!(TestMethod::CircularShift.preserves_marginal_multiset());
        assert!(!TestMethod::Bootstrap.preserves_marginal_multiset());
        assert!(!TestMethod::BlockBootstrap { block_len: 4 }.preserves_marginal_multiset());
    }

    #[test]
    fn time_series_builders_set_the_method() {
        let bp = StatisticalTestConfig::block_permutation(30, 6).seed(2);
        assert_eq!(bp.method, TestMethod::BlockPermutation { block_len: 6 });
        assert_eq!(bp.n_tests, 30);
        assert_eq!(bp.seed, Some(2));

        let cs = StatisticalTestConfig::circular_shift(12);
        assert_eq!(cs.method, TestMethod::CircularShift);
        assert_eq!(cs.n_tests, 12);

        let bb = StatisticalTestConfig::block_bootstrap(8, 3);
        assert_eq!(bb.method, TestMethod::BlockBootstrap { block_len: 3 });
        assert_eq!(bb.n_tests, 8);
    }

    #[test]
    fn default_is_python_compatible() {
        let cfg = StatisticalTestConfig::default();
        assert_eq!(cfg.n_tests, 100);
        assert_eq!(cfg.method, TestMethod::Permutation);
        assert_eq!(cfg.seed, None);
    }

    #[test]
    fn builders_set_fields() {
        let cfg = StatisticalTestConfig::bootstrap(7)
            .with_method(TestMethod::Permutation)
            .seed(3);
        assert_eq!(cfg.n_tests, 7);
        assert_eq!(cfg.method, TestMethod::Permutation);
        assert_eq!(cfg.seed, Some(3));
        assert_eq!(StatisticalTestConfig::permutation(2).n_tests, 2);
    }
}

// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Configuration for statistical significance tests.

/// Resampling scheme for a [`super::StatisticalTest`].
///
/// Mirrors the Python `infomeasure` package's `method` argument
/// (`"permutation_test"` / `"bootstrap"`) as an enum instead of a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TestMethod {
    /// Shuffle the source variable to break any dependence with the target.
    /// The null hypothesis is that the observed measure is indistinguishable
    /// from chance.
    #[default]
    Permutation,
    /// Resample the data with replacement (bootstrap).
    Bootstrap,
}

impl TestMethod {
    /// The Python-compatible string form of the method.
    pub fn as_str(self) -> &'static str {
        match self {
            TestMethod::Permutation => "permutation_test",
            TestMethod::Bootstrap => "bootstrap",
        }
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

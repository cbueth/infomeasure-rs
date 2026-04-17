// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core traits for information-theoretic estimators.
//!
//! This module defines the fundamental interfaces that all estimators in the
//! `infomeasure` crate implement. These traits allow for a unified way to
//! extract results and interact with different algorithmic approaches.
//!
//! ## Core Results
//! - [`GlobalValue`]: The primary interface for obtaining a single scalar measure (e.g., total MI).
//! - [`LocalValues`]: Interface for obtaining pointwise/local values for each sample.
//! - [`OptionalLocalValues`]: A fallible variant for estimators that may not support local values.
//!
//! ## Mathematical Operations
//! - [`JointEntropy`]: Interface for estimators that can compute joint entropy of multiple variables.
//! - [`CrossEntropy`]: Interface for computing cross-entropy $H(P||Q)$ between distributions.
//! - [`MutualInformationTrait`]: Functional interface for computing MI directly.
//! - [`ConditionalMutualInformationTrait`]: Functional interface for computing CMI directly.
//!
//! ## Estimator Markers
//! These traits identify the type of measure an estimator instance represents:
//! - [`MutualInformationEstimator`]
//! - [`ConditionalMutualInformationEstimator`]
//! - [`TransferEntropyEstimator`]
//! - [`ConditionalTransferEntropyEstimator`]

use ndarray::Array1;

/// Interface for estimators that provide a single global value.
///
/// ## Theory
///
/// The global value represents the aggregate information-theoretic measure over the entire dataset.
/// For most measures based on Shannon entropy, the global value is defined as the expected value:
///
/// $$G = \mathbb{E}[L(X)] = \int P(x) L(x) \, dx$$
///
/// where $L(x)$ is the local (pointwise) information contribution.
///
/// ## Crate Architecture
///
/// `GlobalValue` is the most fundamental result trait in the `infomeasure` crate. All estimator
/// structs implement this trait to provide a unified way to extract the final scalar result,
/// regardless of the underlying estimation approach (Discrete, Kernel, KSG, etc.).
///
/// ## See Also
/// - [Estimator Usage Guide](crate::guide::estimator_usage) — How to use estimators
/// - [`LocalValues`] — For pointwise information contributions
pub trait GlobalValue {
    /// Compute and return the global value of the measure (e.g., MI in nats).
    fn global_value(&self) -> f64;
}

/// Interface for estimators that provide local (pointwise) values for each sample.
///
/// ## Theory
///
/// Local values (also known as pointwise or specific information) characterize the information
/// contribution associated with individual data points. For Shannon-based measures, the
/// global value is the expected value (average) of these local values.
///
/// For example, the local mutual information $i(x; y)$ is:
///
/// $$i(x; y) = \log \frac{p(x, y)}{p(x)p(y)}$$
///
/// and the global mutual information is $I(X; Y) = \mathbb{E}[i(x; y)]$.
///
/// ## Crate Architecture
///
/// Many estimators in this crate (especially Discrete and Kernel-based ones) can provide
/// per-sample results. This is useful for detecting outliers, analyzing temporal dynamics
/// in time series, or computing local MI for feature selection.
///
/// ## See Also
/// - [Local Mutual Information](crate::guide::mutual_information#local-mi) — Conceptual guide
/// - [`GlobalValue`] — The aggregate measure
pub trait LocalValues: GlobalValue {
    /// Compute and return the local values of the measure as an `Array1<f64>`.
    fn local_values(&self) -> Array1<f64>;

    /// Derive global_value as the mean of local values.
    fn global_from_local(&self) -> f64 {
        let local_vals = self.local_values();
        local_vals
            .mean()
            .expect("Local values should not be empty.")
    }
}

/// Optional interface for estimators that may not support local values.
///
/// ## Crate Architecture
///
/// Not all estimation algorithms can efficiently or meaningfully provide local values.
/// For example, some complex bias-corrected discrete estimators or certain high-dimensional
/// KSG variants might only produce a global aggregate.
///
/// This trait provides a fallible way to request local values, allowing generic code to
/// handle estimators that do not support them gracefully.
pub trait OptionalLocalValues {
    /// Returns true if the estimator instance supports local value extraction.
    fn supports_local(&self) -> bool;
    /// Fallible extraction of local values.
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str>;
}

/// Interface for estimators that support cross-entropy $H(P \parallel Q)$.
///
/// ## Theory
///
/// Cross-entropy measures the average number of bits (or nats) needed to identify an event
/// from a set of possibilities, if a coding scheme is used based on a distribution $Q$,
/// rather than the "true" distribution $P$:
///
/// $$H(P \parallel Q) = -\mathbb{E}_P[\log Q(X)]$$
///
/// ## See Also
/// - [Cross-Entropy Guide](crate::guide::cross_entropy) — Detailed theory and properties
pub trait CrossEntropy<Rhs = Self> {
    /// Compute the cross-entropy between this distribution (P) and another (Q).
    fn cross_entropy(&self, other: &Rhs) -> f64;
}

/// Interface for estimators that support joint entropy $H(X_1, X_2, \dots, X_n)$.
///
/// ## Theory
///
/// Joint entropy measures the uncertainty associated with a set of variables together:
///
/// $$H(X_1, \ldots, X_n) = -\sum_{x_1} \ldots \sum_{x_n} p(x_1, \ldots, x_n) \log p(x_1, \ldots, x_n)$$
///
/// ## Crate Architecture
///
/// This trait defines the functional interface for computing joint entropy. It is
/// typically implemented by factory-like structs that perform a one-shot calculation
/// without necessarily instantiating a persistent estimator object.
///
/// ## See Also
/// - [Entropy Guide](crate::guide::entropy) — Theoretical background
pub trait JointEntropy {
    /// Data type for a single series/variable.
    type Source;
    /// Additional parameters for the estimator.
    type Params;

    /// Compute the joint entropy of multiple variables.
    fn joint_entropy(series: &[Self::Source], params: Self::Params) -> f64;
}

/// Interface for estimators that support Mutual Information $I(X_1; X_2; \dots; X_n)$.
///
/// ## Theory
///
/// Mutual Information (MI) quantifies the amount of information obtained about one
/// random variable through observing the others:
///
/// $$I(X_1; \ldots; X_n) = \sum H(X_i) - H(X_1, \ldots, X_n)$$
///
/// ## See Also
/// - [Mutual Information Guide](crate::guide::mutual_information) — Conceptual background
pub trait MutualInformationTrait {
    /// Data type for a single series/variable.
    type Source;
    /// Additional parameters for the estimator.
    type Params;

    /// Compute the mutual information of multiple variables.
    fn mutual_information(series: &[Self::Source], params: Self::Params) -> f64;
}

/// Interface for estimators that support Conditional Mutual Information $I(X_1; \dots; X_n | Z)$.
///
/// ## Theory
///
/// Conditional Mutual Information (CMI) measures the expected mutual information of
/// $X$ and $Y$ given the value of $Z$:
///
/// $$I(X; Y \mid Z) = \mathbb{E}_Z [I(X; Y \mid Z=z)]$$
///
/// ## See Also
/// - [Conditional MI Guide](crate::guide::cond_mi) — Theoretical details
pub trait ConditionalMutualInformationTrait {
    /// Data type for a single series/variable.
    type Source;
    /// Data type for the conditioning variable(s).
    type Cond;
    /// Additional parameters for the estimator.
    type Params;

    /// Compute the conditional mutual information.
    fn cmi(series: &[Self::Source], cond: &Self::Cond, params: Self::Params) -> f64;
}

/// Marker trait for Mutual Information estimator instances.
///
/// This trait combines [`GlobalValue`] and [`OptionalLocalValues`] to represent a complete
/// MI estimator instance. It can be used as a trait bound for functions that operate on
/// any MI estimator.
pub trait MutualInformationEstimator: GlobalValue + OptionalLocalValues {}

/// Marker trait for Conditional Mutual Information estimator instances.
///
/// This trait combines [`GlobalValue`] and [`OptionalLocalValues`] to represent a complete
/// CMI estimator instance.
pub trait ConditionalMutualInformationEstimator: GlobalValue + OptionalLocalValues {}

/// Marker trait for Transfer Entropy estimator instances.
///
/// This trait combines [`GlobalValue`] and [`OptionalLocalValues`] to represent a complete
/// TE estimator instance.
pub trait TransferEntropyEstimator: GlobalValue + OptionalLocalValues {}

/// Marker trait for Conditional Transfer Entropy estimator instances.
///
/// This trait combines [`GlobalValue`] and [`OptionalLocalValues`] to represent a complete
/// CTE estimator instance.
pub trait ConditionalTransferEntropyEstimator: GlobalValue + OptionalLocalValues {}

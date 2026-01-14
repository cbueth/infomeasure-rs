// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use ndarray::Array1;

pub trait GlobalValue {
    /// Compute and return the global value of the measure.
    fn global_value(&self) -> f64;
}

pub trait LocalValues: GlobalValue {
    /// Compute and return the local values of the measure.
    /// To be overridden by specific measures.
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
/// Estimators that do support local values should return supports_local() = true
/// and provide local values via `Ok(Array1<f64>)`. Estimators that do not support
/// local values should return supports_local() = false and an Err with a brief reason.
pub trait OptionalLocalValues {
    fn supports_local(&self) -> bool;
    fn local_values_opt(&self) -> Result<Array1<f64>, &'static str>;
}

/// Interface for estimators that support cross-entropy $H(P||Q)$.
pub trait CrossEntropy<Rhs = Self> {
    /// Compute the cross-entropy between this distribution (P) and another (Q).
    fn cross_entropy(&self, other: &Rhs) -> f64;
}

/// Interface for estimators that support joint entropy $H(X_1, X_2, \dots, X_n)$.
pub trait JointEntropy {
    /// Data type for a single series/variable.
    type Source;
    /// Additional parameters for the estimator.
    type Params;

    /// Compute the joint entropy of multiple variables.
    fn joint_entropy(series: &[Self::Source], params: Self::Params) -> f64;
}
/// Interface for estimators that support Mutual Information $I(X_1; X_2; \dots; X_n)$.
pub trait MutualInformationTrait {
    /// Data type for a single series/variable.
    type Source;
    /// Additional parameters for the estimator.
    type Params;

    /// Compute the mutual information of multiple variables.
    fn mutual_information(series: &[Self::Source], params: Self::Params) -> f64;
}

/// Interface for estimators that support Conditional Mutual Information $I(X_1; \dots; X_n | Z)$.
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
pub trait MutualInformationEstimator: GlobalValue + OptionalLocalValues {}

/// Marker trait for Conditional Mutual Information estimator instances.
pub trait ConditionalMutualInformationEstimator: GlobalValue + OptionalLocalValues {}

/// Marker trait for Transfer Entropy estimator instances.
pub trait TransferEntropyEstimator: GlobalValue + OptionalLocalValues {}

/// Marker trait for Conditional Transfer Entropy estimator instances.
pub trait ConditionalTransferEntropyEstimator: GlobalValue + OptionalLocalValues {}

// SPDX-FileCopyrightText: 2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Composite information-theoretic measures.
//!
//! Composite measures are built on top of the entropy estimators rather than
//! introducing a new estimation approach. They are exposed as free functions and
//! small capability traits, so exactly the mathematically valid (estimator,
//! measure) combinations are expressible at compile time — no runtime whitelist
//! and no dispatch overhead.
//!
//! - [`kld`] / [`Kld`] — Kullback–Leibler divergence, for every estimator that
//!   implements [`CrossEntropy`](crate::estimators::traits::CrossEntropy).
//! - [`jsd`] — Jensen–Shannon divergence, for estimators exposing a normalized
//!   [`ProbabilityMass`](crate::estimators::traits::ProbabilityMass) (discrete,
//!   Bayes, shrinkage, ordinal).
//! - [`jsd_kernel_1d`] / [`jsd_kernel_nd`] — Jensen–Shannon divergence for
//!   continuous data via a pooled kernel density estimate.
//!
//! See the [KLD guide](crate::guide::kld) and [JSD guide](crate::guide::jsd)
//! for the theory and the full compatibility matrix.

mod jsd;
mod kld;

pub use jsd::{jsd, jsd_kernel_1d, jsd_kernel_nd};
pub use kld::{Kld, kld};

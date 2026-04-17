// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Estimator Macros
//!
//! This module documents the convenience macros provided by the `infomeasure` crate
//! for creating estimator instances with automatic dimension calculation.
//!
//! ## Why Use Macros?
//!
//! The crate uses const generics to provide compile-time type safety. However,
//! manually calculating joint dimensions (e.g., `D_JOINT = d1 + d2 + d3`) is tedious.
//! These macros handle that automatically.
//!
//! **Without macro:**
//! ```ignore
//! const D1: usize = 2;
//! const D2: usize = 3;
//! const D_JOINT: usize = D1 + D2;
//! let mi = KernelMutualInformation2::<D_JOINT, D1, D2>::new(&series, "gaussian", 0.5);
//! ```
//!
//! **With macro:**
//! ```ignore
//! let mi = new_kernel_mi!(&series, "gaussian", 0.5, 2, 3);
//! ```
//!
//! ## Available Macros
//!
//! These macros are available at the crate root (`infomeasure::`) and can be imported with:
//! ```ignore
//! use infomeasure::{new_kernel_mi, new_ksg_mi, ...};
//! ```
//!
//! Find the fitting macro for each measure and estimator type in the following table:
//!
//! | Measure | [Kernel](crate::estimators::approaches::kernel) | [KSG](crate::estimators::approaches::expfam::ksg) | [Rényi](crate::estimators::approaches::expfam::renyi) | [Tsallis](crate::estimators::approaches::expfam::tsallis) | [KL](crate::estimators::approaches::expfam::kozachenko_leonenko) |
//! |---------|--------|-----|-------|---------|-----|
//! | **[MI](crate::guide::mutual_information)** | [`new_kernel_mi`](crate::new_kernel_mi) | [`new_ksg_mi`](crate::new_ksg_mi) | [`new_renyi_mi`](crate::new_renyi_mi) | [`new_tsallis_mi`](crate::new_tsallis_mi) | [`new_kl_mi`](crate::new_kl_mi) |
//! | **[CMI](crate::guide::cond_mi)** | [`new_kernel_cmi`](crate::new_kernel_cmi) | [`new_ksg_cmi`](crate::new_ksg_cmi) | [`new_renyi_cmi`](crate::new_renyi_cmi) | [`new_tsallis_cmi`](crate::new_tsallis_cmi) | [`new_kl_cmi`](crate::new_kl_cmi) |
//! | **[TE](crate::guide::transfer_entropy)** | [`new_kernel_te`](crate::new_kernel_te) | [`new_ksg_te`](crate::new_ksg_te) | [`new_renyi_te`](crate::new_renyi_te) | [`new_tsallis_te`](crate::new_tsallis_te) | [`new_kl_te`](crate::new_kl_te) |
//! | **[CTE](crate::guide::cond_te)** | [`new_kernel_cte`](crate::new_kernel_cte) | [`new_ksg_cte`](crate::new_ksg_cte) | [`new_renyi_cte`](crate::new_renyi_cte) | [`new_tsallis_cte`](crate::new_tsallis_cte) | [`new_kl_cte`](crate::new_kl_cte) |
//!
//! ## Usage Examples
//!
//! ### 2-variable Mutual Information
//!
//! ```rust
//! use infomeasure::new_kernel_mi;
//! use ndarray::array;
//!
//! let x = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
//! let y = array![[0.2], [0.4], [0.6]];
//!
//! let mi = new_kernel_mi!(&[x, y], "gaussian".to_string(), 0.5, 2, 1);
//! ```
//!
//! ### 3-variable Conditional Mutual Information
//!
//! ```rust
//! use infomeasure::new_kernel_cmi;
//! use ndarray::array;
//!
//! let x = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
//! let y = array![[0.2], [0.4], [0.6]];
//! let z = array![[0.1], [0.2], [0.3]];
//!
//! let cmi = new_kernel_cmi!(&[x, y], &z, "gaussian".to_string(), 0.5, 2, 1, 1);
//! ```
//!
//! ### Transfer Entropy
//!
//! ```rust
//! use infomeasure::new_kernel_te;
//! use ndarray::array;
//!
//! let source = array![[0.1], [0.2], [0.3], [0.4], [0.5]];
//! let dest = array![[0.2], [0.3], [0.4], [0.5], [0.6]];
//!
//! let te = new_kernel_te!(&source, &dest, 1, 1, 1, 1, 1, "gaussian".to_string(), 0.5);
//! ```
//!
//! ## Macro Syntax Reference
//!
//! ### Mutual Information Macros
//!
//! **Kernel MI:**
//! ```ignore
//! new_kernel_mi!(&[x, y, ...], kernel, bandwidth, d1, d2, ...)
//! ```
//!
//! **KSG MI:**
//! ```ignore
//! new_ksg_mi!(&[x, y, ...], k, d1, d2, ...)
//! ```
//!
//! **Rényi/Tsallis/KL MI:**
//! ```ignore
//! new_renyi_mi!(&[x, y, ...], alpha, d1, d2, ...)
//! new_tsallis_mi!(&[x, y, ...], alpha, d1, d2, ...)
//! new_kl_mi!(&[x, y, ...], alpha, d1, d2, ...)
//! ```
//!
//! ### Conditional MI Macros
//!
//! Add condition variable as second argument:
//! ```ignore
//! new_kernel_cmi!(&[x, y], z, kernel, bandwidth, d1, d2, d_cond)
//! ```
//!
//! ### Transfer Entropy Macros
//!
//! **Kernel TE:**
//! ```ignore
//! new_kernel_te!(&source, &dest, source_history, dest_history, kernel, bandwidth, d_source, d_dest)
//! ```
//!
//! ## See Also
//!
//! - [Estimator Usage Guide](super::estimator_usage) — Full usage examples
//! - [Estimator Selection Guide](super::estimator_selection) — Choosing the right estimator

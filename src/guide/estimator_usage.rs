// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Estimator Usage Guide
//!
//! This guide explains how to use the `infomeasure` Rust crate to compute
//! information-theoretic measures.
//!
//! ## Philosophy: Type-Based API
//!
//! Unlike the Python `infomeasure` package which uses runtime string-based
//! estimator selection, this Rust crate uses **compile-time type selection**.
//! This provides:
//!
//! - **Type safety**: Invalid estimator combinations are caught at compile time
//! - **Performance**: Each estimator is a unique type the compiler can optimize
//! - **Documentation**: Types are self-documenting in your code
//!
//! ## Two Ways to Use This Crate
//!
//! ### 1. Facade Types (Recommended for Most Users)
//!
//! The simplest way to compute measures is through the facade types:
//! - [`Entropy`](crate::estimators::entropy::Entropy) — for all entropy estimators
//! - [`MutualInformation`](crate::estimators::mutual_information::MutualInformation) - for MI/CMI
//! - [`TransferEntropy`](crate::estimators::transfer_entropy::TransferEntropy) - for TE/CTE
//!
//! ### 2. Direct Estimator Types (Advanced)
//!
//! For fine-grained control, you can directly instantiate estimator types from
//! the approaches modules:
//! - [`discrete`](crate::estimators::approaches::discrete) - histogram-based estimators
//! - [`kernel`](crate::estimators::approaches::kernel) - kernel density estimation
//! - [`ordinal`](crate::estimators::approaches::ordinal) - permutation patterns
//! - [`expfam`](crate::estimators::approaches::expfam) - k-NN based (KL, KSG, Rényi, Tsallis)
//!
//! ## Computing Entropy
//!
//! ### Discrete Entropy (Categorical/Integer Data)
//!
//! Use the [`Entropy::new_discrete`](crate::estimators::entropy::Entropy::new_discrete) constructor for Maximum Likelihood Estimation:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! // Simple binary data - each symbol has probability 0.5
//! let data = array![0, 1, 0, 1, 0, 1, 0, 1];
//! let h = Entropy::new_discrete(data).global_value();
//! // For uniform binary, H = -0.5*log(0.5) - 0.5*log(0.5) = log(2) ≈ 0.693147 (nats)
//! assert_abs_diff_eq!(h, 0.693147, epsilon = 1e-4);
//! ```

//! ### Bias-Corrected Discrete Estimators
//!
//! For small samples, use bias-corrected estimators:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! let data = array![0, 1, 2, 0, 1, 0]; // Small sample: N=6, K=3
//!
//! // Miller-Madow: simple bias correction
//! let h_mm = Entropy::new_miller_madow(data.clone()).global_value();
//! assert!(h_mm > 0.0);
//!
//! // Shrinkage: regularisation toward uniform
//! let data2 = array![0, 1, 2, 0, 1, 0, 3, 4];
//! let h_shrink = Entropy::new_shrink(data2).global_value();
//! assert!(h_shrink > 0.0);
//! ```

//! ### Bias-Corrected Discrete Estimators
//!
//! For small samples, use bias-corrected estimators:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! let data = array![0, 1, 2, 0, 1, 0]; // Small sample
//!
//! // Miller-Madow: simple bias correction
//! let h_mm = Entropy::new_miller_madow(data.clone()).global_value();
//! assert!(h_mm > 0.0);
//!
//! // Shrinkage: regularisation toward uniform
//! let data2 = array![0, 1, 2, 0, 1, 0, 3, 4];
//! let h_shrink = Entropy::new_shrink(data2).global_value();
//! assert!(h_shrink > 0.0);
//!
//! // Grassberger: digamma-based correction
//! let h_grassberger = Entropy::new_grassberger(data.clone()).global_value();
//! assert!(h_grassberger > 0.0);
//! ```

//! ### Continuous Entropy (Kernel Density Estimation)
//!
//! For continuous data, use kernel-based estimators:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! // 1D continuous data - uniform-ish
//! let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let h = Entropy::new_kernel(data, 1.0).global_value();
//! assert!(h > 0.0);
//!
//! // Multi-dimensional data (2D points)
//! let data_2d = array![[1.0, 1.5], [2.0, 3.0], [4.0, 5.0]];
//! let h2 = Entropy::nd_kernel::<2>(data_2d, 1.0).global_value();
//! assert!(h2 > 0.0);
//! ```

//! ### Ordinal Entropy (Time Series)
//!
//! Ordinal (permutation) entropy is robust to amplitude variations:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! let time_series = array![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5];
//!
//! // Ordinal entropy with embedding dimension 3
//! let h = Entropy::new_ordinal(time_series.clone(), 3).global_value();
//! assert!(h.is_finite());
//!
//! // With custom step size (delay)
//! let h2 = Entropy::new_ordinal_with_step(time_series, 3, 2).global_value();
//! assert!(h2.is_finite());
//! ```

//! ### Exponential Family (k-NN Based) Entropy
//!
//! For continuous data, use Kozachenko-Leonenko (KL) entropy:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//!
//! // KL entropy (1D)
//! let h_kl = Entropy::new_kl_1d(data.clone(), 3, 1e-10).global_value();
//! assert!(h_kl > 0.0);
//!
//! // Multi-dimensional KL
//! let data_nd = array![[1.0], [2.0], [3.0], [4.0]];
//! let h_kl_nd = Entropy::kl_nd::<1>(data_nd, 3, 1e-10).global_value();
//! assert!(h_kl_nd > 0.0);
//! ```
//!
//! For Rényi and Tsallis entropy:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Rényi entropy (alpha = 2)
//! let h_renyi = Entropy::new_renyi_1d(data.clone(), 3, 2.0, 1e-10).global_value();
//! assert!(h_renyi >= 0.0);
//!
//! // Tsallis entropy (q = 1.5)
//! let h_tsallis = Entropy::new_tsallis_1d(data.clone(), 3, 1.5, 1e-10).global_value();
//! assert!(h_tsallis >= 0.0);
//! ```
//!
//! ## Computing Mutual Information
//!
//! Mutual Information (MI) quantifies the information shared between two random variables $X$ and $Y$.
//! In other words, MI measures the average reduction in uncertainty about $X$ that results from
//! learning the value of $Y$, or vice versa.
//!
//! For a comprehensive guide to mutual information including mathematical definitions,
//! properties, and interpretation, see the [Mutual Information Guide](super::mutual_information).
//!
//! ### Global Mutual Information
//!
//! $$I(X;Y) = \\sum_{x, y} p(x, y) \\log \\frac{p(x,y)}{p(x) p(y)}$$
//!
//! where $p(x,y)$ is the joint probability distribution and $p(x)$, $p(y)$ are marginals.
//! MI is zero if and only if $X$ and $Y$ are independent.
//!
//! ### Local Mutual Information
//!
//! Similar to local entropy, one can compute **local** or **point-wise mutual information**:
//!
//! $$i(x; y) = \\log_b \\left( \\frac{p(x, y)}{p(x) p(y)} \\right)$$
//!
//! The global MI is the average of local MI:
//!
//! $$I(X; Y) = \\langle i(x: y) \\rangle$$
//!
//! Local MI values can be positive or negative (negative values indicate misinforming relationships),
//! but the global MI is always non-negative.
//!
//! In this crate, local MI can be accessed via the [`LocalValues`](crate::estimators::traits::LocalValues) trait
//! on MI estimators that support it.
//!
//! ### Time-lagged Mutual Information
//!
//! For time series, variables might share information with a delay (lag):
//!
//! $$I(X_{t-u}; Y_t) = \\sum_{x_{t-u}, y_t} p(x_{t-u}, y_t) \\log \\frac{p(x_{t-u}, y_t)}{p(x_{t-u}) p(y_t)}$$
//!
//! where $u$ is the propagation time or lag between the series.
//! The transfer entropy estimators support specifying history lengths and lags.
//!
//! ### Discrete Mutual Information
//!
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! // Two discrete variables (perfectly correlated: X=Y)
//! let x = array![0, 0, 0, 0, 1, 1, 1, 1];
//! let y = array![0, 0, 0, 0, 1, 1, 1, 1];
//!
//! // Discrete MLE: perfectly correlated = log(2) ≈ 0.693147
//! let mi = MutualInformation::new_discrete_mle(&[x.clone(), y.clone()]).global_value();
//! assert_abs_diff_eq!(mi, 0.693147, epsilon = 1e-4);
//!
//! // Miller-Madow bias correction
//! let mi_mm = MutualInformation::new_discrete_miller_madow(&[x.clone(), y.clone()]).global_value();
//! assert!(mi_mm > 0.0);
//!
//! // Shrinkage estimator
//! let mi_shrink = MutualInformation::new_discrete_shrink(&[x, y]).global_value();
//! assert!(mi_shrink > 0.0);
//! ```

//! ### Continuous Mutual Information (Kernel)
//!
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//! let y = array![0.0, 0.5, 2.0, 3.0, 4.0, 5.0]; // correlated
//!
//! // Kernel MI (1D)
//! let mi = MutualInformation::new_kernel(&[x.clone(), y.clone()], 1.0).global_value();
//! assert!(mi.is_finite());
//!
//! // With specific kernel type
//! let mi_gauss = MutualInformation::new_kernel_with_type(
//!     &[x.clone(), y.clone()],
//!     "gaussian".to_string(),
//!     1.0
//! ).global_value();
//! assert!(mi_gauss.is_finite());
//! ```

//! ### Continuous Mutual Information (KSG)
//!
//! The Kraskov-Stögbauer-Grassberger estimator:
//!
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
//! let y = array![0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1];
//!
//! // KSG MI (k-nearest neighbors based)
//! let mi = MutualInformation::new_ksg(&[x, y], 3, 1e-10).global_value();
//! assert!(mi.is_finite()); // Can be 0 or positive
//! ```

//! ### Ordinal Mutual Information
//!
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
//! let y = array![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
//!
//! let mi = MutualInformation::new_ordinal(&[x.clone(), y.clone()], 3, 1, true).global_value();
//! assert!(mi.is_finite());
//! ```

//! ### Conditional Mutual Information (CMI)
//!
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! let x = array![0, 1, 0, 1, 0, 1];
//! let y = array![0, 0, 1, 1, 0, 1];
//! let z = array![0, 0, 0, 1, 1, 1];
//!
//! // Discrete CMI
//! let cmi = MutualInformation::new_cmi_discrete_mle(&[x.clone(), y.clone()], &z).global_value();
//! assert!(cmi >= 0.0);
//! ```
//!
//! ## Computing Transfer Entropy
//!
//! Transfer Entropy measures directed information flow from source to destination.
//!
//! ### Discrete Transfer Entropy
//!
//! ```rust
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // Source and destination time series
//! let source = array![0, 1, 0, 1, 0, 1, 0, 1];
//! let dest = array![0, 0, 1, 0, 1, 0, 1, 0];
//!
//! // Discrete TE (source_hist=1, dest_hist=1, step=1)
//! let te = TransferEntropy::new_discrete_mle(
//!     &source,
//!     &dest,
//!     1, // source history length
//!     1, // destination history length
//!     1  // step size
//! ).global_value();
//! assert!(te >= 0.0);
//! ```
//!
//! ### Continuous Transfer Entropy (Kernel)
//!
//! ```rust
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let source = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
//! let dest = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
//!
//! let te = TransferEntropy::new_kernel(
//!     &source,
//!     &dest,
//!     1, // src_hist
//!     1, // dest_hist
//!     1, // step_size
//!     1.0 // bandwidth
//! ).global_value();
//! assert!(te >= 0.0);
//! ```
//!
//! ### Ordinal Transfer Entropy
//!
//! ```rust
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let source = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let dest = array![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
//!
//! let te = TransferEntropy::new_ordinal(
//!     &source,
//!     &dest,
//!     3, // order
//!     1, // src_hist_len
//!     1, // dest_hist_len
//!     1, // step_size
//!     true // stable
//! ).global_value();
//! assert!(te >= 0.0);
//! ```
//!
//! ### Conditional Transfer Entropy (CTE)
//!
//! ```rust
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let source = array![0, 1, 0, 1, 0, 1, 0, 1];
//! let dest = array![0, 0, 1, 0, 1, 0, 1, 0];
//! let cond = array![0, 0, 0, 1, 0, 1, 1, 1];
//!
//! // Discrete CTE
//! let cte = TransferEntropy::new_cte_discrete_mle(
//!     &source,
//!     &dest,
//!     &cond,
//!     1, // source history length
//!     1, // destination history length
//!     1, // condition history length
//!     1  // step size
//! ).global_value();
//! assert!(cte >= 0.0);
//! ```
//!
//! ## Global vs Local Values
//!
//! Many estimators support both global (scalar) and local (per-sample) values.
//!
//! ### Getting Global Values
//!
//! All estimators implement the [`GlobalValue::global_value`](crate::estimators::traits::GlobalValue::global_value) trait:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let data = array![0, 1, 0, 1, 0, 1];
//! let estimator = Entropy::new_discrete(data);
//! let h = estimator.global_value(); // Scalar entropy
//! assert!(h >= 0.0); // Entropy is always non-negative
//! ```

//! ### Getting Local Values
//!
//! Some estimators also implement the [`LocalValues::local_values`](crate::estimators::traits::LocalValues::local_values) trait for per-sample information:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::{GlobalValue, LocalValues};
//! use ndarray::array;
//! use approx::assert_abs_diff_eq;
//!
//! let data = array![0, 1, 0, 1, 0, 1];
//! let estimator = Entropy::new_discrete(data);
//!
//! let global = estimator.global_value();
//! let local = estimator.local_values();
//!
//! // Global value should equal mean of local values
//! let local_mean = local.mean().unwrap();
//! assert_abs_diff_eq!(global, local_mean, epsilon = 1e-10);
//! ```
//!
//! ### Getting Local Values
//!
//! Some estimators also implement the [`LocalValues::local_values`](crate::estimators::traits::LocalValues::local_values) trait for per-sample information:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::{GlobalValue, LocalValues};
//! use ndarray::array;
//! use ndarray::Zip;
//! use approx::assert_abs_diff_eq;
//!
//! let data = array![0, 1, 0, 1, 0, 1];
//! let estimator = Entropy::new_discrete(data);
//!
//! let global = estimator.global_value();
//! let local = estimator.local_values();
//!
//! // Global value should equal mean of local values
//! let local_mean = local.mean().unwrap();
//! assert_abs_diff_eq!(global, local_mean, epsilon = 1e-10);
//! ```
//!
//! Note: Not all estimators support local values. Use
//! [`OptionalLocalValues::supports_local`](crate::estimators::traits::OptionalLocalValues::supports_local) to check before calling.
//!
//! ## Multi-Variable Estimators
//!
//! For estimators that support multiple random variables, you can pass a slice:
//!
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // 3 variables
//! let x = array![0, 1, 0, 1];
//! let y = array![1, 1, 0, 0];
//! let z = array![0, 0, 1, 1];
//!
//! let mi3 = MutualInformation::new_discrete_mle(&[x, y, z]).global_value();
//! assert!(mi3 >= 0.0);
//! ```
//!
//! ## Cross-Entropy
//!
//! Cross-entropy measures the information when using distribution Q to encode P:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let p = array![0, 1, 0, 1, 0, 1, 0, 1];
//! let q = array![0, 0, 1, 1, 0, 0, 1, 1];
//!
//! // Compute entropy using discrete estimators
//! let est_p = Entropy::new_discrete(p);
//! let est_q = Entropy::new_discrete(q);
//! // Cross-entropy = H(P) + D(P||Q), not directly exposed but can be derived
//! let h_p = est_p.global_value();
//! let h_q = est_q.global_value();
//! assert!(h_p >= 0.0);
//! assert!(h_q >= 0.0);
//! // Cross-entropy is always >= entropy
//! // (H(P) <= H_Q(P) = H(P) + D(P||Q), with D >= 0)
//! ```
//!
//! ## Macros for Complex Estimators
//!
//! For complex estimators like multi-dimensional kernel MI, use the provided macros:
//!
//! ```rust
//! use infomeasure::new_kernel_mi;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // 2D data for two variables
//! let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
//! let y = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];
//!
//! let mi = new_kernel_mi!(&[x, y], "box".to_string(), 1.0, 2, 2).global_value();
//! assert!(mi >= 0.0);
//! ```
//!
//! ## Typical Workflows
//!
//! This section shows common patterns for different data types and analysis goals.
//!
//! ### Discrete Data Pipeline
//!
//! For categorical or count data:
//!
//! 1. **Prepare data**: Ensure data is integer-valued (use `i32` or `u32`)
//! 2. **Choose estimator**: MLE for large samples ($N > 1000$), bias-corrected for small samples
//! 3. **Compute**: Use [`Entropy::new_discrete`](crate::estimators::entropy::Entropy::new_discrete) or [`MutualInformation::new_discrete_mle`](crate::estimators::mutual_information::MutualInformation::new_discrete_mle)
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // Raw categorical data (e.g., survey responses, sensor states)
//! let category_a = array![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
//! let category_b = array![0, 0, 1, 1, 0, 0, 1, 1, 0, 1];
//!
//! // Entropy of single variable
//! let h = Entropy::new_discrete(category_a.clone()).global_value();
//! assert!(h > 0.0);
//!
//! // Mutual information between two variables
//! let mi = MutualInformation::new_discrete_mle(&[category_a, category_b]).global_value();
//! assert!(mi >= 0.0);
//! ```
//!
//! ### Continuous Data Pipeline
//!
//! For real-valued data:
//!
//! 1. **Prepare data**: Use `ndarray::Array1<f64>` or `Array2<f64>` for multivariate
//! 2. **Choose estimator**: KSG for general use, kernel for more control over bandwidth
//! 3. **Set parameters**: $k$ (neighbors) typically 3-5 for KSG; bandwidth for kernel
//! 4. **Compute**: Use [`MutualInformation::new_ksg`](crate::estimators::mutual_information::MutualInformation::new_ksg)
//!
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // Two correlated continuous variables
//! let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
//! let y = array![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]; // y = x + 0.5
//!
//! // KSG estimator - good for moderate-dimensional continuous data
//! let mi_ksg = MutualInformation::new_ksg(&[x.clone(), y.clone()], 3, 1e-10).global_value();
//! assert!(mi_ksg > 0.0);
//!
//! // Kernel estimator - more control over bandwidth
//! let mi_kernel = MutualInformation::new_kernel(&[x, y], 1.0).global_value();
//! assert!(mi_kernel > 0.0);
//! ```
//!
//! ### Time Series Analysis Pipeline
//!
//! For temporal dependencies and causality:
//!
//! 1. **Prepare data**: Time series as `Array1<f64>`
//! 2. **Choose measure**:
//!    - Lagged MI: $I(X_{t-u}; Y_t)$ for undirected dependencies
//!    - Transfer Entropy: $T_{X \to Y}$ for directed causal influence
//!    - Conditional TE: $T_{X \to Y|Z}$ to control for confounders
//! 3. **Set history lengths**: Start with $k=l=1$, increase for longer memory
//! 4. **Compute**: Use [`TransferEntropy::new_discrete_mle`](crate::estimators::transfer_entropy::TransferEntropy::new_discrete_mle) or continuous variants
//!
//! ```rust
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! // Time series: X drives Y with delay
//! let x = array![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
//! let y = array![0, 0, 1, 0, 1, 0, 1, 0, 1, 0]; // Y(t) = X(t-1)
//!
//! // Transfer entropy X -> Y (source history, dest history, step)
//! let te_xy = TransferEntropy::new_discrete_mle(&x, &y, 1, 1, 1).global_value();
//! assert!(te_xy >= 0.0);
//!
//! // Reverse direction should be lower
//! let te_yx = TransferEntropy::new_discrete_mle(&y, &x, 1, 1, 1).global_value();
//! assert!(te_yx >= 0.0);
//! ```
//!
//! ### Using Local Values for Time-Resolved Analysis
//!
//! Some estimators support local (per-sample) values for time-resolved analysis:
//!
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::{GlobalValue, LocalValues};
//! use ndarray::array;
//! use ndarray::Zip;
//!
//! let x = array![0, 0, 1, 1, 0, 0, 1, 1];
//! let y = array![0, 0, 0, 1, 1, 1, 1, 1];
//!
//! let mi_global = MutualInformation::new_discrete_mle(&[x.clone(), y.clone()]).global_value();
//! let estimator = MutualInformation::new_discrete_mle(&[x, y]);
//! let mi_local = estimator.local_values();
//!
//! // Global MI is mean of local values
//! let local_mean = mi_local.mean().unwrap();
//! assert!((mi_global - local_mean).abs() < 1e-10);
//!
//! // Count positive local MI values (informative co-occurrences)
//! let positive_count = mi_local.mapv(|v| if v > 0.0 { 1 } else { 0 }).sum();
//! assert!(positive_count > 0);
//! ```
//!
//! ### Feature Flags
//!
//! - `gpu`: Enable GPU acceleration for kernel estimators
//! - `fast_exp`: Use fast exponential approximations (trades accuracy for speed)
//!
//! ## Related Guides
//!
//! - **[Mutual Information Guide](super::mutual_information)** - Comprehensive MI documentation
//! - **[Estimator Selection Guide](super::estimator_selection)** - Choose the right estimator
//! - **[Entropy Module Docs](../estimators/entropy/index.html)** - Detailed entropy API
//! - **[Mutual Information Module Docs](../estimators/mutual_information/index.html)** - Detailed MI API
//! - **[Transfer Entropy Module Docs](../estimators/transfer_entropy/index.html)** - Detailed TE API

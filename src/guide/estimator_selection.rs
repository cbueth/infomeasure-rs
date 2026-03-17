// SPDX-FileCopyrightText: 2025-2026 Carlson Büth <code@cbueth.de>
//
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # Estimator Selection Guide
//!
//! This guide helps you choose the appropriate estimator for your data and analysis goals.
//!
//! ## Quick Decision Tree
//!
//! ### Step 1: What type of data do you have?
//!
//! **Discrete data** (integers, categories, counts)
//! - Examples: DNA sequences, survey responses, discrete time series
//! - Use: [`discrete estimators`](crate::estimators::entropy::Entropy::new_discrete)
//!
//! **Continuous data** (real numbers)
//! - Examples: sensor readings, financial data
//! - Use: kernel, Kozachenko-Leonenko (KL), KSG, or ordinal estimators
//!
//! **Time series** (sequential measurements)
//! - Consider: ordinal estimators for temporal pattern analysis
//!
//! ### Step 2: What measure do you need?
//!
//! | Measure | Use Case | Recommended Estimators |
//! |---------|---------|----------------------|
//! | Entropy | Uncertainty quantification | See Entropy section below |
//! | Mutual Information | Variable dependence | See MI section below |
//! | Transfer Entropy | Causal flow | See TE section below |
//! | Conditional MI/TE | Controlling for confounders | Same estimators with condition |
//!
//! ## Discrete Data Selection
//!
//! Discrete estimators are used for categorical or integer-valued data.
//!
//! ### By Sample Size
//!
//! **Small samples (N < 100)**
//!
//! For small samples, bias correction is critical:
//!
//! | Scenario | Recommended | Rust Constructor | Notes |
//! |----------|-------------|------------------|-------|
//! | Correlated data | NSB | [`Entropy::new_nsb`](crate::estimators::entropy::Entropy::new_nsb) | Best for temporal/correlated |
//! | Independent data | Shrinkage | [`Entropy::new_shrink`](crate::estimators::entropy::Entropy::new_shrink) | Regularizes to uniform |
//! | Very small, balanced | Bonachela | [`Entropy::new_bonachela`](crate::estimators::entropy::Entropy::new_bonachela) | Designed for short series |
//! | Unobserved states | Chao-Shen | [`Entropy::new_chao_shen`](crate::estimators::entropy::Entropy::new_chao_shen) | Accounts for coverage |
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let small_data = array![0, 1, 0, 0, 1, 1, 0, 1, 0, 0]; // N=10
//!
//! // NSB for correlated data
//! let h_nsb = Entropy::new_nsb(small_data.clone(), None).global_value();
//!
//! // Shrinkage for independent data
//! let h_shrink = Entropy::new_shrink(small_data.clone()).global_value();
//!
//! // Chao-Shen for incomplete sampling
//! let h_chao = Entropy::new_chao_shen(small_data.clone()).global_value();
//! ```
//!
//! **Medium samples (100 ≤ N < 1000)**
//!
//! | Scenario | Recommended | Rust Constructor |
//! |----------|-------------|------------------|
//! | General use | Miller-Madow | [`Entropy::new_miller_madow`](crate::estimators::entropy::Entropy::new_miller_madow) |
//! | Bias correction | Zhang | [`Entropy::new_zhang`](crate::estimators::entropy::Entropy::new_zhang) |
//! | Advanced coverage | Chao-Wang-Jost | [`Entropy::new_chao_wang_jost`](crate::estimators::entropy::Entropy::new_chao_wang_jost) |
//!
//! **Large samples (N ≥ 1000)**
//!
//! | Scenario | Recommended | Rust Constructor |
//! |----------|-------------|------------------|
//! | Speed priority | Discrete (MLE) | [`Entropy::new_discrete`](crate::estimators::entropy::Entropy::new_discrete) |
//! | Some bias correction | Miller-Madow | [`Entropy::new_miller_madow`](crate::estimators::entropy::Entropy::new_miller_madow) |
//!
//! ### With Prior Knowledge
//!
//! If you have prior knowledge about the distribution:
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::approaches::discrete::bayes::AlphaParam;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let data = array![0, 1, 0, 0, 1, 1, 0, 1];
//!
//! // Bayesian with Jeffrey prior (alpha = 0.5)
//! let h_jeffrey = Entropy::new_bayes(data.clone(), AlphaParam::Jeffrey, None).global_value();
//!
//! // Bayesian with Laplace prior (alpha = 1.0)
//! let h_laplace = Entropy::new_bayes(data.clone(), AlphaParam::Laplace, None).global_value();
//! ```
//!
//! ## Continuous Data Selection
//!
//! ### Entropy Estimation
//!
//! | Data Characteristic | Recommended | Rust Constructor |
//! |-------------------|--------------|------------------|
//! | High-dimensional | KL (Kozachenko-Leonenko) | [`Entropy::new_kl_1d`](crate::estimators::entropy::Entropy::new_kl_1d) |
//! | Low-dimensional, large N | Kernel | [`Entropy::new_kernel`](crate::estimators::entropy::Entropy::new_kernel) |
//! | General purpose | KL | Adapts to local density |
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let continuous_data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
//!
//! // KL entropy - good for high-dimensional or small/medium samples
//! let h_kl = Entropy::new_kl_1d(continuous_data.clone(), 3, 1e-10).global_value();
//!
//! // Kernel entropy - good for low-dimensional, large samples
//! let h_kernel = Entropy::new_kernel(continuous_data.clone(), 1.0).global_value();
//! ```
//!
//! ### Mutual Information Estimation
//!
//! | Sample Size | Recommended | Rust Constructor |
//! |------------|-------------|------------------|
//! | Large | KSG | [`MutualInformation::new_ksg`](crate::estimators::mutual_information::MutualInformation::new_ksg) |
//! | Small/Medium | Kernel | [`MutualInformation::new_kernel`](crate::estimators::mutual_information::MutualInformation::new_kernel) |
//!
//! ```rust
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
//! let y = array![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]; // correlated
//!
//! // KSG - efficient for large samples
//! let mi_ksg = MutualInformation::new_ksg(&[x.clone(), y.clone()], 3, 1e-10).global_value();
//!
//! // Kernel - more control over bandwidth
//! let mi_kernel = MutualInformation::new_kernel(&[x, y], 1.0).global_value();
//! ```
//!
//! ### Transfer Entropy Estimation
//!
//! | Use Case | Recommended | Rust Constructor |
//! |----------|-------------|------------------|
//! | Most applications | Kernel TE | [`TransferEntropy::new_kernel`](crate::estimators::transfer_entropy::TransferEntropy::new_kernel) |
//! | Large samples | KSG TE | [`TransferEntropy::new_ksg`](crate::estimators::transfer_entropy::TransferEntropy::new_ksg) |
//!
//! ```rust
//! use infomeasure::estimators::transfer_entropy::TransferEntropy;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let source = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//! let dest = array![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
//!
//! // Kernel TE
//! let te_kernel = TransferEntropy::new_kernel(
//!     &source,
//!     &dest,
//!     1, 1, 1, 1.0
//! ).global_value();
//!
//! // KSG TE
//! let te_ksg = TransferEntropy::new_ksg(
//!     &source, &dest, 1, 1, 1, 3, 1e-10
//! ).global_value();
//! ```
//!
//! ## Time Series Data (Ordinal Approach)
//!
//! For time series analysis, ordinal (permutation) estimators are robust to amplitude variations:
//!
//! | Measure | Recommended | Rust Constructor |
//! |---------|--------------|------------------|
//! | Ordinal Entropy | Ordinal | [`Entropy::new_ordinal`](crate::estimators::entropy::Entropy::new_ordinal) |
//! | Ordinal MI | Ordinal MI | [`MutualInformation::new_ordinal`](crate::estimators::mutual_information::MutualInformation::new_ordinal) |
//! | Ordinal TE | Ordinal TE | [`TransferEntropy::new_ordinal`](crate::estimators::transfer_entropy::TransferEntropy::new_ordinal) |
//!
//! ### Choosing Embedding Dimension
//!
//! - **Small (2-3)**: Basic temporal patterns, computationally efficient
//! - **Medium (4-5)**: Detailed pattern analysis, balanced complexity
//! - **Large (6+)**: Fine-grained patterns, requires more data
//!
//! ```rust
//! use infomeasure::estimators::entropy::Entropy;
//! use infomeasure::estimators::mutual_information::MutualInformation;
//! use infomeasure::estimators::traits::GlobalValue;
//! use ndarray::array;
//!
//! let ts1 = array![1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0];
//! let ts2 = array![1.5, 2.5, 2.0, 3.5, 3.0, 4.5, 4.0, 5.5];
//!
//! // Ordinal entropy with embedding dimension 3
//! let h_ord = Entropy::new_ordinal(ts1.clone(), 3).global_value();
//!
//! // Ordinal MI
//! let mi_ord = MutualInformation::new_ordinal(&[ts1, ts2], 3, 1, true).global_value();
//! ```
//!
//! ## Performance Considerations
//!
//! | Estimator | Complexity | Best For |
//! |-----------|------------|----------|
//! | Discrete (MLE) | O(N) | Large samples, speed priority |
//! | Miller-Madow | O(N) | General discrete use |
//! | Shrinkage | O(N) | Small independent samples |
//! | NSB | O(N log N) | Correlated/temporal data |
//! | Kernel | O(N²) | Low-dim, large N |
//! | KSG/KL | O(N log N) | High-dim, general continuous |
//! | Ordinal | O(N) | Time series analysis |
//!
//! ## Quick Reference: Measure → Estimator Mapping
//!
//! ### Entropy
//!
//! - Discrete → [`Entropy::new_discrete`](crate::estimators::entropy::Entropy::new_discrete)
//! - Miller-Madow → [`Entropy::new_miller_madow`](crate::estimators::entropy::Entropy::new_miller_madow)
//! - Shrink → [`Entropy::new_shrink`](crate::estimators::entropy::Entropy::new_shrink)
//! - Grassberger → [`Entropy::new_grassberger`](crate::estimators::entropy::Entropy::new_grassberger)
//! - Zhang → [`Entropy::new_zhang`](crate::estimators::entropy::Entropy::new_zhang)
//! - Bayes → [`Entropy::new_bayes`](crate::estimators::entropy::Entropy::new_bayes)
//! - Bonachela → [`Entropy::new_bonachela`](crate::estimators::entropy::Entropy::new_bonachela)
//! - Chao-Shen → [`Entropy::new_chao_shen`](crate::estimators::entropy::Entropy::new_chao_shen)
//! - Chao-Wang-Jost → [`Entropy::new_chao_wang_jost`](crate::estimators::entropy::Entropy::new_chao_wang_jost)
//! - NSB → [`Entropy::new_nsb`](crate::estimators::entropy::Entropy::new_nsb)
//! - Kernel → [`Entropy::new_kernel`](crate::estimators::entropy::Entropy::new_kernel)
//! - KL → [`Entropy::new_kl_1d`](crate::estimators::entropy::Entropy::new_kl_1d)
//! - Rényi → [`Entropy::new_renyi_1d`](crate::estimators::entropy::Entropy::new_renyi_1d)
//! - Tsallis → [`Entropy::new_tsallis_1d`](crate::estimators::entropy::Entropy::new_tsallis_1d)
//! - Ordinal → [`Entropy::new_ordinal`](crate::estimators::entropy::Entropy::new_ordinal)
//!
//! ### Mutual Information
//!
//! - Discrete → [`MutualInformation::new_discrete_mle`](crate::estimators::mutual_information::MutualInformation::new_discrete_mle)
//! - Miller-Madow → [`MutualInformation::new_discrete_miller_madow`](crate::estimators::mutual_information::MutualInformation::new_discrete_miller_madow)
//! - Shrink → [`MutualInformation::new_discrete_shrink`](crate::estimators::mutual_information::MutualInformation::new_discrete_shrink)
//! - Kernel → [`MutualInformation::new_kernel`](crate::estimators::mutual_information::MutualInformation::new_kernel)
//! - KSG → [`MutualInformation::new_ksg`](crate::estimators::mutual_information::MutualInformation::new_ksg)
//! - KL → [`MutualInformation::new_kl`](crate::estimators::mutual_information::MutualInformation::new_kl)
//! - Rényi → [`MutualInformation::new_renyi`](crate::estimators::mutual_information::MutualInformation::new_renyi)
//! - Tsallis → [`MutualInformation::new_tsallis`](crate::estimators::mutual_information::MutualInformation::new_tsallis)
//! - Ordinal → [`MutualInformation::new_ordinal`](crate::estimators::mutual_information::MutualInformation::new_ordinal)
//!
//! ### Transfer Entropy
//!
//! - Discrete → [`TransferEntropy::new_discrete_mle`](crate::estimators::transfer_entropy::TransferEntropy::new_discrete_mle)
//! - Miller-Madow → [`TransferEntropy::new_discrete_miller_madow`](crate::estimators::transfer_entropy::TransferEntropy::new_discrete_miller_madow)
//! - Shrink → [`TransferEntropy::new_discrete_shrink`](crate::estimators::transfer_entropy::TransferEntropy::new_discrete_shrink)
//! - Kernel → [`TransferEntropy::new_kernel`](crate::estimators::transfer_entropy::TransferEntropy::new_kernel)
//! - KSG → [`TransferEntropy::new_ksg`](crate::estimators::transfer_entropy::TransferEntropy::new_ksg)
//! - KL → [`TransferEntropy::new_kl`](crate::estimators::transfer_entropy::TransferEntropy::new_kl)
//! - Rényi → [`TransferEntropy::new_renyi`](crate::estimators::transfer_entropy::TransferEntropy::new_renyi)
//! - Tsallis → [`TransferEntropy::new_tsallis`](crate::estimators::transfer_entropy::TransferEntropy::new_tsallis)
//! - Ordinal → [`TransferEntropy::new_ordinal`](crate::estimators::transfer_entropy::TransferEntropy::new_ordinal)
//!
//! ## Related Guides
//!
//! - **[Estimator Usage Guide](super::estimator_usage)** - How to use the API
//! - **[Entropy Guide](super::entropy)** - Entropy overview
//! - **[Mutual Information Guide](super::mutual_information)** - MI overview
//! - **[Transfer Entropy Guide](super::transfer_entropy)** - TE overview
//! - **[Conditional MI Guide](super::cond_mi)** - CMI overview
//! - **[Conditional TE Guide](super::cond_te)** - CTE overview

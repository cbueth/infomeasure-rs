# Changelog

All notable changes to this project are being documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0-alpha.1] - 2026-01-17

### 🚀 Major Features Added

#### Core Estimators
- **Discrete Entropy**: 11+ bias-corrected methods (MLE, Miller-Madow, NSB, ANSB, etc.)
- **Kernel Density Estimation**: Continuous data support with Box and Gaussian kernels
- **Ordinal Pattern Analysis**: Time series entropy with ordinal pattern methods
- **Exponential Family**: k-NN based entropy with Rényi and Tsallis generalisations
- **Kozachenko-Leonenko**: Differential entropy estimator for continuous data

#### Advanced Information Measures
- **Mutual Information**: Discrete and kernel-based MI estimators
- **Transfer Entropy**: Discrete and kernel-based TE estimators
- **Conditional Variants**: CMI and CTE estimators for conditional analysis
- **Joint Entropy**: Multi-variable entropy estimation
- **Cross Entropy**: Divergence measures between distributions

#### Performance & Architecture
- **GPU Acceleration**: Optional GPU support for large datasets
- **Factory Pattern**: Consistent API with trait-based result extraction
- **Global/Local Values**: Unified interface for entropy and local entropy estimates

---

### Migration from 0.0.x to 0.1.0
- **Breaking**: `simd` feature flag removed
- **Breaking**: `GlobalValue` and `LocalValues` refactored across all estimators
- **Added**: Comprehensive Python parity test suite
- **Improved**: GPU acceleration with automatic fallback
- **Enhanced**: Documentation with mathematical notation


---

## January 2026

### 2026-01-17 - Final Alpha Release & CI Migration
- **CI**: Migrated from `micromamba` to `uv` for Python validation
  - Lock Contention: Eliminated micromamba parallel execution bottlenecks
  - Environment Detection: Improved cross-platform compatibility for Python virtual environments
- **Testing**: Added `verbose_println!` macro for reduced test logging
- **Linting**: Added conditional GPU-specific annotations
- **Refactoring**: Centralized benchmark outputs in `internal/` directory
- **Documentation**: Expanded module-level and struct-level documentation

### 2026-01-16 - CI/CD Pipeline Refinement
- **CI**: Removed `fast_exp` from feature matrix due to accuracy degradation
- **CI**: Enforced documentation warnings in lint pipeline
- **CI**: Fixed shell hook evaluation and updated dependencies
- **Breaking**: Removed `simd` feature flag and associated code

### 2026-01-15 - Testing Infrastructure
- **CI**: Added comprehensive CI pipeline (build, test, lint, doc generation)
- **Documentation**: Extended docstrings for `NdDataset`
- **Linting**: Enabled clippy with comprehensive rules

### 2026-01-14 - License & Compliance
- **Legal**: Added SPDX license identifiers across all source files
- **Development**: Added pre-commit hooks and REUSE compliance

### 2026-01-13 - Core Estimator Implementation Day
- **🎯 Major Release**: Complete implementation of all core estimators
  - **Kernel MI/TE**: Added `KernelMutualInformation`, `KernelTransferEntropy`, `KernelConditionalMutualInformation`, `KernelConditionalTransferEntropy`
  - **Factory Macros**: Added `new_kernel_te`, `new_kernel_cte`, `new_kernel_mi`, `new_kernel_cmi` macros
  - **Facade Pattern**: Unified interface for discrete and kernel-based estimators
  - **Global/Local Values**: Refactored all estimators to use `GlobalValue` and `LocalValues`
  - **Python Parity**: Comprehensive validation against infomeasure Python package
  - **MI/TE Estimators**: Complete discrete mutual information and transfer entropy estimators
  - **Testing**: Added parity tests for all discrete MI/TE estimators

### 2026-01-12 - Joint & Cross Entropy
- **Features**: Added `JointEntropy` and `CrossEntropy` traits
- **Implementation**: Joint entropy support for all discrete estimators
- **Ordinal**: Implemented joint and cross entropy for ordinal patterns
- **Testing**: Added comprehensive tests for joint entropy and cross entropy
- **Performance**: Improved `symbolize_series_u64` and Lehmer code computation

### 2026-01-09 - Ordinal & Utility Improvements
- **Ordinal**: Added ordinal joint and cross entropy estimators
- **Utilities**: Enhanced `argsort` functionality and joint code space reduction
- **Documentation**: Integrated KaTeX support for LaTeX rendering
- **Testing**: Added unit tests for Lehmer code and related utilities

---

## September 2025

### 2025-09-21 - Exponential Family Estimators
- **Major Feature**: Added exponential-family (kNN-based) entropy estimators
- **Extension**: Implemented additional discrete entropy estimators
- **Testing**: Comprehensive Python parity validation for new estimators

---

## October 2025

### 2025-10-05 - Legacy Cleanup
- **Cleanup**: Removed legacy symbolisation function
- **Fix**: Added KL entropy and improved tests, fixed expfam and ordinal tests

---

## July 2025

### 2025-07-19 - GPU Acceleration & Testing
- **🚀 GPU Feature**: Implemented GPU-accelerated kernel entropy estimation
- **Performance**: Significant speedup for large datasets (500+ samples)
- **Testing**: Centralized test utilities into `test_helpers`
- **Usability**: Made CPU fallback less verbose and improved benchmarks

### 2025-07-13 - Project Foundation
- **🎉 Initial Release**: First implementation of infomeasure-rs library
- **Python Interface**: Initial Python parity validation framework using `micromamba`
- **Architecture**: Established core trait system and estimator patterns
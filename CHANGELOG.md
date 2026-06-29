# Changelog

## [0.3.0-rc.0](https://codeberg.org/cbueth/infomeasure-rs/releases/tag/0.3.0-rc.0) - 2026-06-29

### ❤️ Thanks to all contributors! ❤️

@cbueth

### 📚 Documentation

- fix: validate CITATION.cff against CFF 1.2.0 schema for Zenodo [[#34](https://codeberg.org/cbueth/infomeasure-rs/pulls/34)]
- feat: Add benchmarks and runtime visualisation [[#10](https://codeberg.org/cbueth/infomeasure-rs/pulls/10)]

### Misc

- fix: Restore `skipCommitsWithoutPullRequest` and enable `commentOnReleasedPullRequests` [[#32](https://codeberg.org/cbueth/infomeasure-rs/pulls/32)]

## [0.3.0-beta.1](https://codeberg.org/cbueth/infomeasure-rs/releases/tag/0.3.0-beta.1) - 2026-06-24

### Thanks to all contributors!

@cbueth

### ✨ Features

- **New `pub mod guide`**: Complete documentation guide system with 16 submodules covering all measures (estimator selection, usage scenarios, mathematical theory, references) — ~2600 lines of new documentation
- **New `doc_snippets!` macro** (`doc_macros.rs`): Shared infrastructure for consistent mathematical formula rendering across all estimators
- **Estimator API documentation overhaul**: Full mathematical theory docs with KaTeX rendering added to:
  - `entropy.rs` (+278 lines): Relationship to MI, KL-divergence, JSD; complete estimator examples
  - `mutual_information.rs` (+401 lines): Facade API docs with formulas for all MI variants
  - `transfer_entropy.rs` (+357 lines): Facade API docs with TE/CTE theory and examples
- **Module-level theory documentation** for all approach modules:
  - `expfam/renyi.rs`: Rényi α-entropy theory with kNN estimation formulas
  - `expfam/tsallis.rs`: Tsallis q-entropy theory with kNN estimation formulas
  - `kernel/mod.rs`: KDE theory with bandwidth/kernel selection guidance
  - `ordinal/mod.rs`: Permutation entropy theory with symbolization explanation
  - `traits.rs`: Full trait hierarchy documentation with mathematical notation
- **Transfer entropy slicing** (`te_slicing.rs`): Refactored and expanded TE observation slicing utilities (+192 lines)
- **Cross-reference system**: `references.rs` guide page with citations for all referenced papers

### 📚 Documentation

- docs: Proofread all doc pages [[#15](https://codeberg.org/cbueth/infomeasure-rs/pulls/15)]
- docs: Move macro dimension documentation [[#11](https://codeberg.org/cbueth/infomeasure-rs/pulls/11)]
- docs: Add guides for facade/ crate usage and estimator selection [[#9](https://codeberg.org/cbueth/infomeasure-rs/pulls/9)]
- docs: improve guide cross-references and measure relationships ([7deeaad](https://codeberg.org/cbueth/infomeasure-rs/src/commit/7deeaad4650571ad50829e5c93709dbfd6e8bfb7))
- docs: expand `Conditional TE` and `Transfer Entropy` guides with detailed examples ([61b042b](https://codeberg.org/cbueth/infomeasure-rs/src/commit/61b042bec967a4bee0240b9ef16d25d4b34108de))
- docs: add Mutual Information guide and module references ([81cc96b](https://codeberg.org/cbueth/infomeasure-rs/src/commit/81cc96bc032ab41c3d40e7a6c69a10236fe34347))
- docs: extend guides with documentation for `TransferEntropy` and `MutualInformation` ([e3903fc](https://codeberg.org/cbueth/infomeasure-rs/src/commit/e3903fc596becd4535c11f7e4bd1a04461fc359e))
- docs: guides for entropy and transfer entropy, ... ([01c301f](https://codeberg.org/cbueth/infomeasure-rs/src/commit/01c301f530b8364297994b56e5747c6989bf6986))
- docs(WIP): add guides for facade/ crate usage and estimator selection ([6f359e8](https://codeberg.org/cbueth/infomeasure-rs/src/commit/6f359e8dfdbdad62ed590187d5c6a1f887df6324))

## [0.2.0-beta.1](https://codeberg.org/cbueth/infomeasure-rs/releases/tag/0.2.0-beta.1) - 2026-03-16

### ❤️ Thanks to all contributors! ❤️

@cbueth

### 📈 Enhancement

- ci: Add container CI caching [[#5](https://codeberg.org/cbueth/infomeasure-rs/pulls/5)]

### 📦️ Dependency

- fix: Cleanup unused dependencies and CI configurations [[#7](https://codeberg.org/cbueth/infomeasure-rs/pulls/7)]
- chore: Update dependencies [[#6](https://codeberg.org/cbueth/infomeasure-rs/pulls/6)]

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

## [0.0.11] - 2026-01-17

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

## [0.0.4] - 2025-10-05

### 2025-10-05 - Legacy Cleanup
- **Cleanup**: Removed legacy symbolisation function
- **Fix**: Added KL entropy and improved tests, fixed expfam and ordinal tests

---

## [0.0.3] - 2025-09-21

### 2025-09-21 - Exponential Family Estimators
- **Major Feature**: Added exponential-family (kNN-based) entropy estimators
- **Extension**: Implemented additional discrete entropy estimators
- **Testing**: Comprehensive Python parity validation for new estimators

---

## [0.0.2] - 2025-07-19

### 2025-07-19 - GPU Acceleration & Testing
- **🚀 GPU Feature**: Implemented GPU-accelerated kernel entropy estimation
- **Performance**: Significant speedup for large datasets (500+ samples)
- **Testing**: Centralized test utilities into `test_helpers`
- **Usability**: Made CPU fallback less verbose and improved benchmarks

---

## [0.0.1] - 2025-07-13

### 2025-07-13 - Project Foundation
- **🎉 Initial Release**: First implementation of infomeasure-rs library
- **Python Interface**: Initial Python parity validation framework using `micromamba`
- **Architecture**: Established core trait system and estimator patterns

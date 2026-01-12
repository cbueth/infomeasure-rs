pub mod kernel;
pub mod discrete;
pub mod ordinal;
pub mod expfam;
pub mod common_nd;

// Unified re-exports for common estimators so tests and users can import
// infomeasure::estimators::approaches::* ergonomically.
// Discrete estimators
pub use discrete::ansb::AnsbEntropy;
pub use discrete::bayes::BayesEntropy;
pub use discrete::bonachela::BonachelaEntropy;
pub use discrete::chao_shen::ChaoShenEntropy;
pub use discrete::chao_wang_jost::ChaoWangJostEntropy;
pub use discrete::grassberger::GrassbergerEntropy;
pub use discrete::miller_madow::MillerMadowEntropy;
pub use discrete::nsb::NsbEntropy;
pub use discrete::shrink::ShrinkEntropy;
pub use discrete::zhang::ZhangEntropy;
pub use discrete::mle::DiscreteEntropy;

// Exponential family estimators
pub use expfam::renyi::RenyiEntropy;
pub use expfam::tsallis::TsallisEntropy;
pub use expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;

// Kernel and Ordinal
pub use kernel::KernelEntropy;
pub use ordinal::ordinal::OrdinalEntropy;

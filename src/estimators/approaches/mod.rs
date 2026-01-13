pub mod common_nd;
pub mod discrete;
pub mod expfam;
pub mod kernel;
pub mod ordinal;

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
pub use discrete::mle::DiscreteEntropy;
pub use discrete::nsb::NsbEntropy;
pub use discrete::shrink::ShrinkEntropy;
pub use discrete::zhang::ZhangEntropy;

// Exponential family estimators
pub use expfam::kozachenko_leonenko::KozachenkoLeonenkoEntropy;
pub use expfam::renyi::RenyiEntropy;
pub use expfam::tsallis::TsallisEntropy;

// Kernel and Ordinal
pub use kernel::KernelEntropy;
pub use ordinal::ordinal::OrdinalEntropy;

// Discrete estimators module: groups all discrete-related submodules
// and exposes them to the parent approaches module.

pub mod discrete_utils;
#[cfg(feature = "gpu_support")]
pub mod discrete_gpu;

pub mod mle;
pub mod miller_madow;
pub mod shrink;
pub mod grassberger;
pub mod zhang;
pub mod bayes;
pub mod bonachela;
pub mod chao_shen;
pub mod chao_wang_jost;
pub mod ansb;
pub mod nsb;

// Additional helpers
pub mod discrete_batch;

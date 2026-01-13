// Aggregates all submodule tests so `cargo test` runs them.
#[path = "common_nd/mod.rs"]
mod common_nd;
#[path = "discrete/mod.rs"]
mod discrete;
#[path = "discrete/discrete_mi_te.rs"]
mod discrete_mi_te;
#[path = "expfam/mod.rs"]
mod expfam;
#[path = "kernel/mod.rs"]
mod kernel;
#[path = "ordinal/mod.rs"]
mod ordinal;
#[path = "test_helpers.rs"]
pub mod test_helpers;

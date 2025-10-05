// Aggregates all submodule tests so `cargo test` runs them.
#[path = "test_helpers.rs"]
pub mod test_helpers;
#[path = "common_nd/mod.rs"]
mod common_nd;
#[path = "discrete/mod.rs"]
mod discrete;
#[path = "expfam/mod.rs"]
mod expfam;
#[path = "kernel/mod.rs"]
mod kernel;
#[path = "ordinal/mod.rs"]
mod ordinal;

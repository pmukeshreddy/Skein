//! Subcommand runners. `compile`, `verify`, `serve`, and `calibrate` run on
//! the CUDA backend by default and fall back to the CPU `NativeComputeRuntime`
//! on `--no-default-features` builds.

pub mod calibrate;
pub mod compile;
pub mod extract;
pub mod serve;
pub mod verify;

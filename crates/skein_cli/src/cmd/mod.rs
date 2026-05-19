//! Subcommand runners. `extract` is Phase A; the rest return
//! `CliError::RequiresCuda` on Phase A builds.

pub mod bench;
pub mod calibrate;
pub mod compile;
pub mod extract;
pub mod serve;
pub mod verify;

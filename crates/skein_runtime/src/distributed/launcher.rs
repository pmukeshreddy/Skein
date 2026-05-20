//! Multi-process launcher for multi-GPU serving.
//!
//! Spawns one child process per rank. Because Luminal binds every
//! `CudaRuntime` to device 0, each rank process is given a distinct physical
//! GPU through `CUDA_VISIBLE_DEVICES` — so that process's "device 0" *is* its
//! assigned GPU — together with its rank, the world size, and the shared
//! rendezvous path used to exchange the NCCL id.
//!
//! This module builds the per-rank [`std::process::Command`]s; the caller
//! spawns them (and supervises / tears them down). Keeping spawning out of the
//! builder lets the launch plan and command construction be unit-tested
//! without actually forking GPU processes.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::{CollectiveError, WorldLayout};

/// Environment variable carrying the rendezvous file path to each rank.
pub const RENDEZVOUS_ENV: &str = "SKEIN_RENDEZVOUS";
/// The standard CUDA device-isolation variable.
pub const CUDA_VISIBLE_DEVICES_ENV: &str = "CUDA_VISIBLE_DEVICES";

/// How one rank should be launched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankLaunchSpec {
    pub rank: usize,
    pub world_size: usize,
    /// Physical GPU ordinal this rank owns (becomes its visible device 0).
    pub gpu: usize,
    pub rendezvous: PathBuf,
}

/// Assign rank `i` to `gpus[i]` and build one [`RankLaunchSpec`] per rank.
/// `gpus` is the list of physical GPU ordinals to use (e.g. `[0,1,2,3]`); its
/// length is the world size.
pub fn plan_launch(gpus: &[usize], rendezvous: &Path) -> Result<Vec<RankLaunchSpec>, CollectiveError> {
    let world_size = gpus.len();
    if world_size == 0 {
        return Err(CollectiveError::BadLayout {
            rank: 0,
            world_size,
        });
    }
    Ok(gpus
        .iter()
        .enumerate()
        .map(|(rank, &gpu)| RankLaunchSpec {
            rank,
            world_size,
            gpu,
            rendezvous: rendezvous.to_path_buf(),
        })
        .collect())
}

/// Build the command for one rank: run `program args...` with
/// `CUDA_VISIBLE_DEVICES`, `SKEIN_RANK`, `SKEIN_WORLD_SIZE`, and
/// `SKEIN_RENDEZVOUS` set so the child resolves its [`WorldLayout`] and joins
/// the NCCL group on its assigned GPU.
pub fn build_command(program: &str, args: &[String], spec: &RankLaunchSpec) -> Command {
    let mut cmd = Command::new(program);
    cmd.args(args);
    cmd.env(CUDA_VISIBLE_DEVICES_ENV, spec.gpu.to_string());
    cmd.env(WorldLayout::RANK_ENV, spec.rank.to_string());
    cmd.env(WorldLayout::WORLD_SIZE_ENV, spec.world_size.to_string());
    cmd.env(RENDEZVOUS_ENV, &spec.rendezvous);
    cmd
}

/// The rendezvous path a launched rank should read, from `SKEIN_RENDEZVOUS`.
pub fn rendezvous_path_from_env() -> Option<PathBuf> {
    std::env::var_os(RENDEZVOUS_ENV).map(PathBuf::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_launch_assigns_ranks_to_gpus() {
        let specs = plan_launch(&[0, 1, 2, 3], Path::new("/tmp/rdv")).expect("plan");
        assert_eq!(specs.len(), 4);
        for (i, s) in specs.iter().enumerate() {
            assert_eq!(s.rank, i);
            assert_eq!(s.gpu, i);
            assert_eq!(s.world_size, 4);
            assert_eq!(s.rendezvous, Path::new("/tmp/rdv"));
        }
    }

    #[test]
    fn plan_launch_honors_a_gpu_subset() {
        // Use physical GPUs 2 and 5 only → ranks 0,1 map to them.
        let specs = plan_launch(&[2, 5], Path::new("/tmp/rdv")).expect("plan");
        assert_eq!(specs[0].gpu, 2);
        assert_eq!(specs[1].gpu, 5);
        assert_eq!(specs[1].world_size, 2);
    }

    #[test]
    fn plan_launch_rejects_empty() {
        assert!(matches!(
            plan_launch(&[], Path::new("/tmp/rdv")),
            Err(CollectiveError::BadLayout { .. })
        ));
    }

    #[test]
    fn build_command_sets_rank_environment() {
        let spec = RankLaunchSpec {
            rank: 1,
            world_size: 2,
            gpu: 5,
            rendezvous: PathBuf::from("/tmp/rdv"),
        };
        let cmd = build_command("skein", &["serve".to_string(), "--port".to_string()], &spec);
        assert_eq!(cmd.get_program(), "skein");
        let args: Vec<_> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(args, vec!["serve", "--port"]);
        let envs: std::collections::HashMap<String, String> = cmd
            .get_envs()
            .filter_map(|(k, v)| Some((k.to_string_lossy().into_owned(), v?.to_string_lossy().into_owned())))
            .collect();
        assert_eq!(envs.get(CUDA_VISIBLE_DEVICES_ENV).map(String::as_str), Some("5"));
        assert_eq!(envs.get(WorldLayout::RANK_ENV).map(String::as_str), Some("1"));
        assert_eq!(envs.get(WorldLayout::WORLD_SIZE_ENV).map(String::as_str), Some("2"));
        assert_eq!(envs.get(RENDEZVOUS_ENV).map(String::as_str), Some("/tmp/rdv"));
    }
}

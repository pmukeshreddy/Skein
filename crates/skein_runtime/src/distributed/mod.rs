//! Multi-process, multi-GPU execution layer.
//!
//! ## Why multi-process
//!
//! Skein's compute runtime is Luminal's `CudaRuntime`, whose constructor
//! hardcodes `CudaContext::new(0)` (GPU 0) with no device-selection API on the
//! pinned rev. A single process therefore cannot place tensor/expert shards on
//! distinct physical GPUs. The portable way around this — used by torchrun and
//! every NCCL data/tensor-parallel trainer — is **one OS process per rank**,
//! each launched with `CUDA_VISIBLE_DEVICES=<gpu>` so its "device 0" maps to a
//! different physical GPU, and the ranks joined into one NCCL communicator via
//! `ncclCommInitRank` over a shared rendezvous id.
//!
//! ## Layers
//!
//! - [`WorldLayout`] — this rank's `(rank, world_size)`, read from the
//!   environment the [`launcher`] sets.
//! - [`RankCollective`] — one rank's view of a collective group: each method
//!   runs a collective over the whole group and returns *this* rank's result.
//!   Two implementations share this trait:
//!     - [`barrier_collective::BarrierCollective`] — a CPU backend that runs
//!       the ranks as threads synchronizing through a `Barrier`. It performs
//!       the real reduction math and is the **reference used to validate the
//!       rank-parallel architecture without a GPU** (see the tests).
//!     - `crate::cuda::nccl::NcclCollective` — the GPU backend, one rank per
//!       process, transfers over NCCL (`#[cfg(feature = "cuda")]`).
//! - [`rendezvous`] — publish/fetch the NCCL unique id over a shared file so
//!   the launched ranks can `ncclCommInitRank`.
//! - [`launcher`] — spawn the per-rank processes with the right
//!   `CUDA_VISIBLE_DEVICES` / rank / world-size / rendezvous environment.
//!
//! ## What still needs the GPU host
//!
//! The remaining integration is a per-rank serving executor: each process
//! loads only its device's artifact shard, walks the artifact's
//! `SequenceStep` schedule running just its own segments, and meets its peers
//! at each `Collective` step through [`RankCollective`]. The collective
//! contract and the rank-parallel semantics are exercised on CPU here; the
//! NCCL transfers and the cross-process executor are validated on hardware.

pub mod barrier_collective;
#[cfg(feature = "cuda")]
pub mod batch_driver;
#[cfg(feature = "cuda")]
pub mod gpu_rank;
pub mod launcher;
pub mod local_topology;
pub mod rank_executor;
pub mod rendezvous;
pub mod segment_runner;

pub use barrier_collective::BarrierCollective;
#[cfg(feature = "cuda")]
pub use batch_driver::{BatchOutput, ContinuousBatchDriver, DriverMetrics};
pub use local_topology::LocalTopology;
pub use rank_executor::{LocalSegments, RankExecError, RankExecutor, ResolvedSequenceStep};
pub use segment_runner::SegmentRunner;

/// This process's place in the rank world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorldLayout {
    pub rank: usize,
    pub world_size: usize,
}

impl WorldLayout {
    /// Environment variable names the [`launcher`] sets per rank.
    pub const RANK_ENV: &'static str = "SKEIN_RANK";
    pub const WORLD_SIZE_ENV: &'static str = "SKEIN_WORLD_SIZE";

    pub fn new(rank: usize, world_size: usize) -> Result<Self, CollectiveError> {
        if world_size == 0 || rank >= world_size {
            return Err(CollectiveError::BadLayout { rank, world_size });
        }
        Ok(Self { rank, world_size })
    }

    /// Read `(SKEIN_RANK, SKEIN_WORLD_SIZE)` from the environment. Returns
    /// `None` when unset (i.e. a single-process run).
    pub fn from_env() -> Option<Result<Self, CollectiveError>> {
        let rank = std::env::var(Self::RANK_ENV).ok()?;
        let world = std::env::var(Self::WORLD_SIZE_ENV).ok()?;
        let parse = |s: String, what: &'static str| -> Result<usize, CollectiveError> {
            s.parse::<usize>()
                .map_err(|_| CollectiveError::BadEnv { what })
        };
        Some((|| {
            let rank = parse(rank, Self::RANK_ENV)?;
            let world_size = parse(world, Self::WORLD_SIZE_ENV)?;
            Self::new(rank, world_size)
        })())
    }

    pub fn is_leader(&self) -> bool {
        self.rank == 0
    }
}

/// Errors from the distributed layer. Kept separate from `RuntimeError` so the
/// CPU collective backend has no CUDA in its error surface.
#[derive(Debug, thiserror::Error)]
pub enum CollectiveError {
    #[error("invalid world layout: rank {rank} of world_size {world_size}")]
    BadLayout { rank: usize, world_size: usize },

    #[error("environment variable {what} is not a valid integer")]
    BadEnv { what: &'static str },

    #[error(
        "collective buffer length mismatch on rank {rank}: got {got}, group expects {expected}"
    )]
    LengthMismatch {
        rank: usize,
        got: usize,
        expected: usize,
    },

    #[error("collective group state was poisoned by a panicked rank")]
    Poisoned,

    #[error("rendezvous i/o on {path}: {source}")]
    Rendezvous {
        path: String,
        source: std::io::Error,
    },

    #[error("rendezvous timed out after {ms} ms waiting for {path}")]
    RendezvousTimeout { ms: u64, path: String },

    /// NCCL/CUDA failure (GPU build only).
    #[error("nccl: {0}")]
    Nccl(String),
}

/// One rank's handle on a collective group. Every method performs the
/// collective across the whole group and leaves *this* rank's result in the
/// supplied buffer (or returns it). The reduction semantics match NCCL so the
/// CPU [`BarrierCollective`] and the GPU NCCL backend are interchangeable.
///
/// Not `Send + Sync`: the NCCL backend wraps `cudarc::nccl::Comm` (a raw
/// `*mut ncclComm`), which is single-threaded. Each rank drives its own
/// collective on its own thread/process, so no cross-thread sharing is needed.
pub trait RankCollective {
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;

    /// In-place sum all-reduce: every rank ends with the elementwise sum of
    /// all ranks' input buffers. All ranks must pass equal-length buffers.
    fn all_reduce_sum(&self, buf: &mut [f32]) -> Result<(), CollectiveError>;

    /// All-gather: returns the concatenation of every rank's `buf` in rank
    /// order (length `world_size * buf.len()`), identical on every rank.
    fn all_gather(&self, buf: &[f32]) -> Result<Vec<f32>, CollectiveError>;

    /// Broadcast `root`'s buffer to every rank in place.
    fn broadcast(&self, buf: &mut [f32], root: usize) -> Result<(), CollectiveError>;

    /// In-place sum all-reduce directly on a **device** bf16 buffer — no host
    /// round-trip. `ptr` is a raw CUDA device pointer holding `elems` bf16
    /// elements on this rank's GPU; on return it holds the elementwise sum across
    /// ranks. Default: unsupported (host-only backends like the CPU barrier);
    /// the NCCL backend overrides it.
    ///
    /// # Safety
    /// `ptr` must be a valid device allocation of `elems` bf16 elements on this
    /// rank's device, alive for the duration of the call.
    unsafe fn all_reduce_sum_device_bf16(
        &self,
        _ptr: u64,
        _elems: usize,
    ) -> Result<(), CollectiveError> {
        Err(CollectiveError::Nccl(
            "device bf16 all_reduce not supported on this backend".to_string(),
        ))
    }

    /// `(shm_dev_ptr, rank, slot_bytes, max_elems)` for the custom mapped-memory
    /// all-reduce, when set up (`SKEIN_SHM_ALLREDUCE`). Under SKEIN_CAPTURE the
    /// executor launches it from luminal (shared capture stream) using this info,
    /// for all-reduces with `elems <= max_elems`. Default `None` (use NCCL).
    fn shm_all_reduce_info(&self) -> Option<(u64, i32, i32, usize)> {
        None
    }

    /// Point-to-point send of `buf` to `peer` (the pipeline-parallel stage
    /// handoff). Default: unsupported.
    fn send_f32(&self, _buf: &[f32], _peer: usize) -> Result<(), CollectiveError> {
        Err(CollectiveError::Nccl("send not supported on this backend".to_string()))
    }

    /// Point-to-point receive of `len` f32 from `peer` (the PP stage handoff).
    /// Default: unsupported.
    fn recv_f32(&self, _peer: usize, _len: usize) -> Result<Vec<f32>, CollectiveError> {
        Err(CollectiveError::Nccl("recv not supported on this backend".to_string()))
    }

    /// Deadlock-free bidirectional point-to-point exchange: in ONE NCCL group,
    /// send `send_buf` to `send_peer` AND receive `recv_len` f32 from `recv_peer`.
    /// Returns the received buffer. Used by the 1F1B pipeline-parallel decode,
    /// where each stage hands its activation forward and receives the previous
    /// job's result backward in the same step — issuing those as two separate
    /// blocking send/recv calls deadlocks (both ranks block in `send` waiting for
    /// the peer's `recv`). Grouping them (`ncclGroupStart/End`) lets NCCL resolve
    /// the pair together. Default: unsupported.
    fn send_recv_f32(
        &self,
        _send_buf: &[f32],
        _send_peer: usize,
        _recv_peer: usize,
        _recv_len: usize,
    ) -> Result<Vec<f32>, CollectiveError> {
        Err(CollectiveError::Nccl(
            "send_recv not supported on this backend".to_string(),
        ))
    }
}

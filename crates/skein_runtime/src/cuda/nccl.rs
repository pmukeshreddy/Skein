//! NCCL collectives between graph invocations.
//!
//! This is the production multi-GPU counterpart to
//! [`crate::collectives::InProcessCollective`]: the same `CollectiveKind`
//! operations, executed over NCCL across ranks. The module is
//! `#[cfg(feature = "cuda")]`-gated at the `cuda/mod.rs` level, so CPU builds
//! never see this code.
//!
//! TODO(nccl): wire each method to an NCCL communicator (init via
//! `ncclCommInitRank` with the rendezvous id, then the matching
//! `nccl{AllReduce,AllGather,ReduceScatter,AllToAll,Broadcast}` call on the
//! device buffer). Every method currently returns
//! `RuntimeError::NotImplemented`.

use crate::error::RuntimeError;
use skein_cost::collectives::CollectiveKind;

/// One rank's handle on the NCCL communicator for a process group.
pub struct NcclCommunicator {
    world_size: usize,
    rank: usize,
}

impl NcclCommunicator {
    /// Initialize this rank's communicator within a `world_size` group.
    pub fn new(world_size: usize, rank: usize) -> Result<Self, RuntimeError> {
        let _ = NcclCommunicator { world_size, rank };
        Err(RuntimeError::NotImplemented {
            what: "NcclCommunicator::new",
        })
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Run `kind` over `device_buffer` (a pointer-sized handle into device
    /// memory) across `participants`, in place.
    pub fn run(
        &self,
        kind: CollectiveKind,
        _participants: &[usize],
        _device_buffer: u64,
        _len: usize,
    ) -> Result<(), RuntimeError> {
        let _ = kind;
        Err(RuntimeError::NotImplemented {
            what: "NcclCommunicator::run",
        })
    }
}

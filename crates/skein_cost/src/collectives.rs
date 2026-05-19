//! `Collective` — a single comm operation a Plan will issue at runtime.
//!
//! The factor table comes from standard ring-algorithm analysis (Patarasuk &
//! Yuan 2009, "Bandwidth optimal all-reduce algorithms for clusters of
//! workstations"; NCCL's documentation reproduces the same formulas). The
//! factor is the *bytes-on-the-wire* multiplier — each device transmits
//! `factor × bytes` to complete the operation. Wall-clock transfer time is
//! then `factor × bytes / bottleneck_bw / efficiency`.

use serde::{Deserialize, Serialize};

/// The collective primitives Skein models. `SendRecv` is the pipeline-
/// parallel point-to-point boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollectiveKind {
    /// Ring all-reduce: every device ends up holding the sum. Used after
    /// row-parallel projections in TP.
    RingAllReduce,
    /// All-gather: every device holds every shard. Used at TP boundaries.
    AllGather,
    /// Reduce-scatter: dual of all-gather. Used when the next op consumes a
    /// sharded result.
    ReduceScatter,
    /// All-to-all: token shuffle between expert-parallel devices in MoE.
    AllToAll,
    /// Broadcast from one device to all others.
    Broadcast,
    /// Point-to-point send/recv (pipeline-parallel stage boundary).
    SendRecv,
}

impl CollectiveKind {
    /// Bytes-on-the-wire factor for `n` participants. Standard ring formulas:
    ///
    /// - `RingAllReduce`: `2(n-1)/n` (reduce-scatter then all-gather)
    /// - `AllGather` / `ReduceScatter` / `AllToAll`: `(n-1)/n`
    /// - `Broadcast` / `SendRecv`: `1` (one-shot send)
    pub fn factor(self, n: usize) -> f64 {
        let n_f = n as f64;
        match self {
            CollectiveKind::RingAllReduce => 2.0 * (n_f - 1.0) / n_f,
            CollectiveKind::AllGather
            | CollectiveKind::ReduceScatter
            | CollectiveKind::AllToAll => (n_f - 1.0) / n_f,
            CollectiveKind::Broadcast | CollectiveKind::SendRecv => 1.0,
        }
    }

    /// Key into `cost_constants.toml` `[collective_efficiency]`.
    pub fn efficiency_key(self) -> &'static str {
        match self {
            CollectiveKind::RingAllReduce => "ring_allreduce",
            CollectiveKind::AllGather => "allgather",
            CollectiveKind::ReduceScatter => "reducescatter",
            CollectiveKind::AllToAll => "alltoall",
            CollectiveKind::Broadcast => "broadcast",
            CollectiveKind::SendRecv => "send_recv",
        }
    }
}

/// One concrete collective the runtime will execute.
///
/// `bytes` is the *per-device side* of the collective — the payload size
/// each participant contributes. The factor table converts this to
/// bytes-on-the-wire.
#[derive(Debug, Clone, PartialEq)]
pub struct Collective {
    pub kind: CollectiveKind,
    /// Device indices (positions in `Cluster::all_devices()`) that
    /// participate. Must contain at least two entries.
    pub participants: Vec<u32>,
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_allreduce_factor() {
        // Patarasuk-Yuan: 2(n-1)/n.
        assert!((CollectiveKind::RingAllReduce.factor(2) - 1.0).abs() < 1e-12);
        assert!((CollectiveKind::RingAllReduce.factor(4) - 1.5).abs() < 1e-12);
        assert!((CollectiveKind::RingAllReduce.factor(8) - 1.75).abs() < 1e-12);
    }

    #[test]
    fn all_gather_factor() {
        assert!((CollectiveKind::AllGather.factor(2) - 0.5).abs() < 1e-12);
        assert!((CollectiveKind::AllGather.factor(4) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn point_to_point_factor_is_one() {
        assert_eq!(CollectiveKind::SendRecv.factor(2), 1.0);
        assert_eq!(CollectiveKind::Broadcast.factor(8), 1.0);
    }
}

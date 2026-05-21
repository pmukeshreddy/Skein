//! Per-rank forward executor for multi-process multi-GPU serving.
//!
//! Each rank process walks the artifact's global [`SequenceStep`] schedule but
//! runs **only its own device's segments**, and at every `Collective` step it
//! participates in it exchanges the named handoff tensor with its peers through
//! a [`RankCollective`]. This is the multi-process counterpart to
//! `skein_compile::TopologyExecutor` (which runs every device in one process):
//! there, all ranks' runtimes live together; here, each process owns one rank.
//!
//! The orchestration — schedule walking, rank filtering, and the
//! read → collective → write-back at each boundary — is backend-agnostic and is
//! validated on CPU with [`super::BarrierCollective`] (see the tests). On the
//! GPU host the same executor runs with the real per-device segment runtimes
//! and `crate::cuda::nccl::NcclCollective`.
//!
//! ## Scope
//!
//! This executor drives a **single collective group** — the common case being
//! tensor parallelism (`tp > 1`, one TP group), and dense expert parallelism
//! once each rank holds a `RankCollective` for its EP group. Plans that mix
//! several groups in one block (`tp × ep`, or PP `SendRecv` pairs) need a
//! per-participant-set collective registry; that extension is noted at
//! [`RankExecutor::run`] and does not change the orchestration here.

use skein_cost::collectives::CollectiveKind;
use skein_emit::segment::SequenceStep;

use super::{CollectiveError, RankCollective};

/// This rank's local segment runtimes. The real implementation wraps the
/// device's compiled `RuntimeSegment`s (feeding named inputs, executing, and
/// reading named outputs); tests use an in-memory mock. Kept as a trait so the
/// rank-parallel orchestration is exercisable without a GPU.
pub trait LocalSegments {
    /// Execute this rank's local segment `segment_idx`, consuming whatever
    /// named handoff inputs it needs and storing its named outputs.
    fn run_segment(&mut self, segment_idx: usize) -> Result<(), RankExecError>;

    /// Read a named handoff tensor's current value (handed to a collective).
    fn read(&self, name: &str) -> Result<Vec<f32>, RankExecError>;

    /// Store a collective's result back under `name` for downstream segments.
    fn write(&mut self, name: &str, data: Vec<f32>) -> Result<(), RankExecError>;
}

#[derive(Debug, thiserror::Error)]
pub enum RankExecError {
    #[error("local segment {segment_idx} failed: {detail}")]
    Segment { segment_idx: usize, detail: String },

    #[error("handoff tensor {0:?} is not present on this rank")]
    UnknownTensor(String),

    #[error(transparent)]
    Collective(#[from] CollectiveError),

    #[error(
        "collective kind {kind:?} is not supported by the single-group rank \
         executor yet (supported: RingAllReduce, AllGather, Broadcast)"
    )]
    UnsupportedCollective { kind: CollectiveKind },
}

/// Drives one rank through the global schedule. Owns this rank's local
/// segments; the collective is borrowed per [`run`](Self::run) so a long-lived
/// `RankExecutor` can be stepped repeatedly through a decode loop while the
/// NCCL communicator (which is not `Clone`) lives alongside it.
pub struct RankExecutor<S: LocalSegments> {
    rank: usize,
    runner: S,
}

impl<S: LocalSegments> RankExecutor<S> {
    pub fn new(rank: usize, runner: S) -> Self {
        Self { rank, runner }
    }

    /// Consume `self`, returning the local segment store (so callers can read
    /// the final outputs, e.g. logits, after the run).
    pub fn into_runner(self) -> S {
        self.runner
    }

    pub fn runner(&self) -> &S {
        &self.runner
    }

    pub fn runner_mut(&mut self) -> &mut S {
        &mut self.runner
    }

    /// Walk `schedule` once, executing this rank's segments and participating
    /// in the collectives it belongs to via `collective`.
    ///
    /// Single-group: `collective` is assumed to cover the participant set of
    /// every collective this rank joins (true for a tp-only plan, or a dense-EP
    /// plan within one EP group). A plan that mixes distinct participant sets
    /// per block would key a collective per participant set here instead.
    pub fn run<C: RankCollective>(
        &mut self,
        schedule: &[SequenceStep],
        collective: &C,
    ) -> Result<(), RankExecError> {
        // --- profiling: total time in local segment execution vs in NCCL
        // collectives (host read -> NCCL -> host write), summed over one forward
        // pass. Tests whether the bottleneck is compute/host staging (segments)
        // or the collective/communication path.
        let mut seg_us: u128 = 0;
        let mut comm_us: u128 = 0;
        for step in schedule {
            match step {
                SequenceStep::ExecuteSegment {
                    device_idx,
                    segment_idx,
                } => {
                    if *device_idx as usize == self.rank {
                        let t = std::time::Instant::now();
                        self.runner.run_segment(*segment_idx)?;
                        seg_us += t.elapsed().as_micros();
                    }
                }
                SequenceStep::Collective {
                    collective: kind,
                    participants,
                    tensor,
                    ..
                } => {
                    if participants.iter().any(|p| *p as usize == self.rank) {
                        let t = std::time::Instant::now();
                        let mut buf = self.runner.read(tensor)?;
                        apply_collective(collective, *kind, participants, &mut buf)?;
                        self.runner.write(tensor, buf)?;
                        comm_us += t.elapsed().as_micros();
                    }
                }
            }
        }
        tracing::info!(
            seg_us,
            comm_us,
            "SKEIN_PERF_STEP: segment-exec vs collective time for one forward pass"
        );
        Ok(())
    }
}

fn apply_collective<C: RankCollective>(
    collective: &C,
    kind: CollectiveKind,
    participants: &[u32],
    buf: &mut Vec<f32>,
) -> Result<(), RankExecError> {
    match kind {
        CollectiveKind::RingAllReduce => collective.all_reduce_sum(buf)?,
        CollectiveKind::AllGather => {
            *buf = collective.all_gather(buf)?;
        }
        CollectiveKind::Broadcast => {
            // The root is the first participant; its local index in the group
            // is 0 under the lexicographic participant ordering.
            let _ = participants;
            collective.broadcast(buf, 0)?;
        }
        kind => return Err(RankExecError::UnsupportedCollective { kind }),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::BarrierCollective;
    use skein_ir::types::Dtype;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::thread;

    /// In-memory rank runner: `run_segment` writes this rank's partial
    /// contribution for tensor "x"; `read`/`write` hit a name→buffer map.
    struct MockSegments {
        rank: usize,
        store: HashMap<String, Vec<f32>>,
    }

    impl LocalSegments for MockSegments {
        fn run_segment(&mut self, _segment_idx: usize) -> Result<(), RankExecError> {
            // Each rank contributes [rank+1, rank+1].
            self.store
                .insert("x".to_string(), vec![(self.rank as f32) + 1.0; 2]);
            Ok(())
        }
        fn read(&self, name: &str) -> Result<Vec<f32>, RankExecError> {
            self.store
                .get(name)
                .cloned()
                .ok_or_else(|| RankExecError::UnknownTensor(name.to_string()))
        }
        fn write(&mut self, name: &str, data: Vec<f32>) -> Result<(), RankExecError> {
            self.store.insert(name.to_string(), data);
            Ok(())
        }
    }

    #[test]
    fn rank_executor_runs_local_segments_and_all_reduces() {
        // Global schedule: each rank runs its own segment producing a partial,
        // then a RingAllReduce over both ranks sums the partials into "x".
        let schedule = Arc::new(vec![
            SequenceStep::ExecuteSegment {
                device_idx: 0,
                segment_idx: 0,
            },
            SequenceStep::ExecuteSegment {
                device_idx: 1,
                segment_idx: 0,
            },
            SequenceStep::Collective {
                collective: CollectiveKind::RingAllReduce,
                participants: vec![0, 1],
                tensor: "x".to_string(),
                shape: vec![2],
                dtype: Dtype::Bf16,
            },
        ]);

        let handles = BarrierCollective::group(2).expect("group");
        let mut joins = Vec::new();
        for (rank, coll) in handles.into_iter().enumerate() {
            let schedule = schedule.clone();
            joins.push(thread::spawn(move || {
                let runner = MockSegments {
                    rank,
                    store: HashMap::new(),
                };
                let mut exec = RankExecutor::new(rank, runner);
                exec.run(&schedule, &coll).expect("rank run");
                exec.into_runner().read("x").expect("x present")
            }));
        }
        for (rank, j) in joins.into_iter().enumerate() {
            let x = j.join().expect("rank thread");
            // 1 (rank 0) + 2 (rank 1) = 3 on every rank.
            assert_eq!(x, vec![3.0, 3.0], "rank {rank} got {x:?}");
        }
    }

    #[test]
    fn unsupported_collective_is_a_clear_error() {
        let schedule = vec![SequenceStep::Collective {
            collective: CollectiveKind::AllToAll,
            participants: vec![0],
            tensor: "x".to_string(),
            shape: vec![1],
            dtype: Dtype::Bf16,
        }];
        let coll = BarrierCollective::group(1).unwrap().pop().unwrap();
        let mut runner = MockSegments {
            rank: 0,
            store: HashMap::new(),
        };
        runner.store.insert("x".to_string(), vec![1.0]);
        let mut exec = RankExecutor::new(0, runner);
        let err = exec.run(&schedule, &coll).unwrap_err();
        assert!(matches!(
            err,
            RankExecError::UnsupportedCollective {
                kind: CollectiveKind::AllToAll
            }
        ));
    }
}

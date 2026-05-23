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

use skein_compile::HandoffId;
use skein_cost::collectives::CollectiveKind;

use super::{CollectiveError, RankCollective};

/// A [`SequenceStep`](skein_emit::segment::SequenceStep) with all logical tensor
/// names pre-resolved to [`HandoffId`]s (and MoE expert weights pre-resolved to
/// resident device pointers). Produced once at bootstrap by
/// [`SegmentRunner::resolve_schedule`](super::SegmentRunner::resolve_schedule)
/// from the artifact's `Vec<SequenceStep>` (which stays String-keyed on disk), so
/// the decode hot path — segments, collectives, and MoE routing — performs zero
/// string hashing per token.
#[derive(Clone, Debug)]
pub enum ResolvedSequenceStep {
    /// Run segment `segment_idx` of device `device_idx`.
    ExecuteSegment { device_idx: u32, segment_idx: usize },
    /// Issue a collective over the handoff tensor `tensor`. `elems` is the tensor's
    /// element count (from the topology shape) — needed by the receiver of a
    /// `SendRecv` (PP stage handoff) to size its recv buffer.
    Collective {
        collective: CollectiveKind,
        participants: Vec<u32>,
        tensor: HandoffId,
        elems: usize,
    },
    /// Sparse-MoE route+bind. `router_id` is the gate segment's router-logits
    /// handoff; `block` is the transformer block index (the gate buffer's layer
    /// slab); `expert_weights[e] = [(ptr, n_bytes); 3]` are the owned experts'
    /// resident weight buffers (resolved once at bootstrap); `slot_ids[s]` are the
    /// FFN segment's per-slot weight inputs; `gate_ids[s]` are the FFN gate-scalar
    /// inputs (bound once at bootstrap to fixed offsets of the resident gate
    /// buffer — see [`SegmentRunner::bind_gate_inputs`](super::SegmentRunner)).
    MoeRoute {
        device_idx: u32,
        ffn_segment_idx: usize,
        router_id: HandoffId,
        top_k: usize,
        block: usize,
        expert_weights: Vec<[(u64, usize); 3]>,
        slot_ids: Vec<[HandoffId; 3]>,
        gate_ids: Vec<HandoffId>,
    },
}

/// This rank's local segment runtimes. The real implementation wraps the
/// device's compiled `RuntimeSegment`s (feeding named inputs, executing, and
/// reading named outputs); tests use an in-memory mock. Kept as a trait so the
/// rank-parallel orchestration is exercisable without a GPU.
pub trait LocalSegments {
    /// Execute this rank's local segment `segment_idx`, consuming whatever
    /// named handoff inputs it needs and storing its named outputs.
    fn run_segment(&mut self, segment_idx: usize) -> Result<(), RankExecError>;

    /// Read a named handoff tensor's current value. Off the hot path (external
    /// accessors, the single-process `LocalTopology` walker, tests); the
    /// `RankExecutor` decode loop uses [`LocalSegments::read_by_id`].
    fn read(&self, name: &str) -> Result<Vec<f32>, RankExecError>;

    /// Store a value under a named handoff. Off-hot-path counterpart of
    /// [`LocalSegments::write_by_id`].
    fn write(&mut self, name: &str, data: Vec<f32>) -> Result<(), RankExecError>;

    /// Device buffer `(raw_ptr, bf16_elems)` of a named handoff held
    /// device-resident, if any. Off-hot-path; default `None`.
    fn output_device_ptr(&self, _name: &str) -> Option<(u64, usize)> {
        None
    }

    /// Read a handoff by pre-resolved id (the collective hot path). Default
    /// errors: only the real runner is on the id path. Mocks override.
    fn read_by_id(&self, id: HandoffId) -> Result<Vec<f32>, RankExecError> {
        Err(RankExecError::UnknownTensor(format!("id {}", id.0)))
    }

    /// Write a collective's result back by pre-resolved id (the collective hot
    /// path). Default errors. Mocks override.
    fn write_by_id(&mut self, id: HandoffId, _data: Vec<f32>) -> Result<(), RankExecError> {
        Err(RankExecError::UnknownTensor(format!("id {}", id.0)))
    }

    /// Device buffer `(raw_ptr, bf16_elems)` of a device-resident handoff by id,
    /// for in-place all-reduce. Default `None`.
    fn output_device_ptr_by_id(&self, _id: HandoffId) -> Option<(u64, usize)> {
        None
    }

    /// Launch the custom shm all-reduce on a luminal segment runtime's stream (so
    /// under SKEIN_CAPTURE it shares the capture stream with the segments).
    /// `data_ptr` is the handoff buffer; `shm_ptr` the cross-process mapped
    /// region (both raw device pointers). Default: unsupported.
    fn device_shm_all_reduce(
        &mut self,
        _data_ptr: u64,
        _shm_ptr: u64,
        _rank: i32,
        _elems: usize,
        _slot_bytes: i32,
    ) -> Result<(), RankExecError> {
        Err(RankExecError::UnknownTensor(
            "device_shm_all_reduce unsupported on this backend".to_string(),
        ))
    }

    /// Sparse-MoE route+bind with everything pre-resolved (zero string hashing):
    /// read the `router_id` logits, pick top-k experts, bind the FFN segment's
    /// `slot_ids` to the selected experts' resident `expert_weights` device
    /// buffers, and write the softmax-over-top-k gate scalars into block `block`'s
    /// slab of the resident gate buffer (whose offsets the FFN gate inputs were
    /// bound to once at bootstrap). Default: no-op (dense path). See
    /// [`ResolvedSequenceStep::MoeRoute`].
    fn route_moe_resolved(
        &mut self,
        _ffn_segment_idx: usize,
        _router_id: HandoffId,
        _top_k: usize,
        _block: usize,
        _expert_weights: &[[(u64, usize); 3]],
        _slot_ids: &[[HandoffId; 3]],
    ) -> Result<(), RankExecError> {
        Ok(())
    }
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
        schedule: &[ResolvedSequenceStep],
        collective: &C,
    ) -> Result<(), RankExecError> {
        // --- profiling (compile-time gated by `perf-trace`): total time in local
        // segment execution vs in collectives (host read -> NCCL -> host write) /
        // MoE routing, summed over one forward. With the feature off, `StepTimer`
        // is a no-op and `time_seg`/`time_comm` just run the closure.
        let mut timer = crate::perf_timing::StepTimer::new();
        // SKEIN_CAPTURE: route the device all-reduce through luminal (shared
        // capture stream). Checked once per forward.
        let capture = std::env::var_os("SKEIN_CAPTURE").is_some();
        for step in schedule {
            match step {
                ResolvedSequenceStep::ExecuteSegment {
                    device_idx,
                    segment_idx,
                } => {
                    if *device_idx as usize == self.rank {
                        timer.time_seg(|| self.runner.run_segment(*segment_idx))?;
                    }
                }
                ResolvedSequenceStep::Collective {
                    collective: kind,
                    participants,
                    tensor,
                    elems,
                } => {
                    // Pipeline-parallel stage handoff: one rank sends the boundary
                    // hidden state to the next stage's rank (the ONLY cross-GPU comm
                    // per token under PP — vs 64 all-reduces under TP). Host-staged
                    // (read -> ncclSend / ncclRecv -> write); 1x/token so it's cheap.
                    if *kind == CollectiveKind::SendRecv {
                        let sender = participants.first().copied().unwrap_or(0) as usize;
                        let receiver = participants.get(1).copied().unwrap_or(0) as usize;
                        let is_sender = self.rank == sender;
                        let is_receiver = self.rank == receiver;
                        let runner = &mut self.runner;
                        timer.time_comm(|| -> Result<(), RankExecError> {
                            if is_sender {
                                let buf = runner.read_by_id(*tensor)?;
                                collective.send_f32(&buf, receiver)?;
                            } else if is_receiver {
                                let buf = collective.recv_f32(sender, *elems)?;
                                runner.write_by_id(*tensor, buf)?;
                            }
                            Ok(())
                        })?;
                        continue;
                    }
                    if participants.iter().any(|p| *p as usize == self.rank) {
                        let runner = &mut self.runner;
                        timer.time_comm(|| -> Result<(), RankExecError> {
                            // Device-resident steady-state path: a RingAllReduce of
                            // a bf16 activation kept on-device is all-reduced IN
                            // PLACE by device pointer — no host Vec, no D2H/H2D. The
                            // producer's buffer holds the reduced result; the
                            // consumer binds it.
                            let device = match kind {
                                CollectiveKind::RingAllReduce => {
                                    runner.output_device_ptr_by_id(*tensor)
                                }
                                _ => None,
                            };
                            if let Some((ptr, elems)) = device {
                                // SKEIN_CAPTURE: launch the custom shm all-reduce
                                // from luminal so it lands on the shared capture
                                // stream (skein_runtime's cudarc can't reach
                                // luminal's stream). Else use the collective's own
                                // device all-reduce (NCCL or skein_runtime shm).
                                let routed = if capture {
                                    if let Some((shm_ptr, ar_rank, slot_bytes, max_elems)) =
                                        collective.shm_all_reduce_info()
                                        && elems <= max_elems
                                    {
                                        runner.device_shm_all_reduce(
                                            ptr, shm_ptr, ar_rank, elems, slot_bytes,
                                        )?;
                                        true
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                };
                                if !routed {
                                    unsafe { collective.all_reduce_sum_device_bf16(ptr, elems) }?;
                                }
                            } else {
                                // Host fallback: all_gather (logits), broadcast, or
                                // a tensor not held device-resident.
                                let mut buf = runner.read_by_id(*tensor)?;
                                crate::perf_counters::record_d2h(
                                    buf.len() * std::mem::size_of::<f32>(),
                                );
                                apply_collective(collective, *kind, participants, &mut buf)?;
                                crate::perf_counters::record_h2d(
                                    buf.len() * std::mem::size_of::<f32>(),
                                );
                                runner.write_by_id(*tensor, buf)?;
                            }
                            Ok(())
                        })?;
                    }
                }
                // Sparse-MoE route+bind (gated by SKEIN_SPARSE_MOE at compile);
                // dense artifacts emit no MoeRoute steps.
                ResolvedSequenceStep::MoeRoute {
                    device_idx,
                    ffn_segment_idx,
                    router_id,
                    top_k,
                    block,
                    expert_weights,
                    slot_ids,
                    // `gate_ids` are bound to the gate buffer once at bootstrap;
                    // the per-token write keys off `block`, not the ids.
                    gate_ids: _,
                } => {
                    if *device_idx as usize == self.rank {
                        timer.time_comm(|| {
                            self.runner.route_moe_resolved(
                                *ffn_segment_idx,
                                *router_id,
                                *top_k,
                                *block,
                                expert_weights,
                                slot_ids,
                            )
                        })?;
                    }
                }
            }
        }
        // Emit the per-forward record (no-op unless `perf-trace` is compiled in).
        timer.finish();
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
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::thread;

    /// Tensor "x" is interned to this id in these tests (the executor consumes a
    /// pre-resolved schedule; the runner resolves names → ids at bootstrap).
    const X: HandoffId = HandoffId(0);

    /// In-memory rank runner: `run_segment` writes this rank's partial
    /// contribution for tensor "x"; the id-based hot-path accessors map the
    /// interned id back to the name → buffer store.
    struct MockSegments {
        rank: usize,
        store: HashMap<String, Vec<f32>>,
        ids: HashMap<HandoffId, String>,
    }

    impl MockSegments {
        fn new(rank: usize) -> Self {
            let mut ids = HashMap::new();
            ids.insert(X, "x".to_string());
            Self {
                rank,
                store: HashMap::new(),
                ids,
            }
        }
        fn name_of(&self, id: HandoffId) -> Result<String, RankExecError> {
            self.ids
                .get(&id)
                .cloned()
                .ok_or_else(|| RankExecError::UnknownTensor(format!("id {}", id.0)))
        }
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
        fn read_by_id(&self, id: HandoffId) -> Result<Vec<f32>, RankExecError> {
            self.read(&self.name_of(id)?)
        }
        fn write_by_id(&mut self, id: HandoffId, data: Vec<f32>) -> Result<(), RankExecError> {
            let name = self.name_of(id)?;
            self.write(&name, data)
        }
    }

    #[test]
    fn rank_executor_runs_local_segments_and_all_reduces() {
        // Global schedule: each rank runs its own segment producing a partial,
        // then a RingAllReduce over both ranks sums the partials into "x".
        let schedule = Arc::new(vec![
            ResolvedSequenceStep::ExecuteSegment {
                device_idx: 0,
                segment_idx: 0,
            },
            ResolvedSequenceStep::ExecuteSegment {
                device_idx: 1,
                segment_idx: 0,
            },
            ResolvedSequenceStep::Collective {
                collective: CollectiveKind::RingAllReduce,
                participants: vec![0, 1],
                tensor: X,
                elems: 2,
            },
        ]);

        let handles = BarrierCollective::group(2).expect("group");
        let mut joins = Vec::new();
        for (rank, coll) in handles.into_iter().enumerate() {
            let schedule = schedule.clone();
            joins.push(thread::spawn(move || {
                let runner = MockSegments::new(rank);
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
        let schedule = vec![ResolvedSequenceStep::Collective {
            collective: CollectiveKind::AllToAll,
            participants: vec![0],
            tensor: X,
            elems: 2,
        }];
        let coll = BarrierCollective::group(1).unwrap().pop().unwrap();
        let mut runner = MockSegments::new(0);
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

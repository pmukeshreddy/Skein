//! `LocalTopology` — single-process, multi-device schedule walker over paged
//! [`SegmentRunner`]s.
//!
//! This is the in-process counterpart to [`super::RankExecutor`] (which runs one
//! rank per process and meets peers over NCCL). Here a single process owns *all*
//! devices' [`SegmentRunner`]s and walks the global [`SequenceStep`] schedule,
//! executing each device's segments and resolving every collective host-side by
//! reading the participating runners' handoff tensors, combining them, and
//! writing the result back. Each runner carries its own paged KV cache, so the
//! continuous-batching driver ([`super::batch_driver`]) gets real paged KV +
//! prefix reuse across the tp=2 model without multi-process NCCL.
//!
//! The collective set matches what the Mixtral tp=2 plan emits: `RingAllReduce`
//! (attention + MoE output sums) and `AllGather` (final logits). `Broadcast` /
//! `SendRecv` are handled for completeness.

use skein_compile::HandoffId;
use skein_cost::collectives::CollectiveKind;
use skein_emit::segment::SequenceStep;

use super::rank_executor::{LocalSegments, RankExecError, ResolvedSequenceStep};
use super::segment_runner::SegmentRunner;

/// Drives all devices' paged segment runners through one global schedule pass.
pub struct LocalTopology {
    runners: Vec<SegmentRunner>,
    sequencing: Vec<SequenceStep>,
    /// Per-runner resolved schedule (id-keyed, with sparse-MoE routing + expert
    /// weights pre-resolved per device). Empty = legacy host-only string-keyed
    /// walk (the CPU mock tests / dense host-materialized path). When present,
    /// `run_step` uses the resolved walk that handles `MoeRoute` and
    /// device-resident collectives — required for the on-device sparse Mixtral.
    resolved: Vec<Vec<ResolvedSequenceStep>>,
}

impl LocalTopology {
    pub fn new(runners: Vec<SegmentRunner>, sequencing: Vec<SequenceStep>) -> Self {
        Self {
            runners,
            sequencing,
            resolved: Vec::new(),
        }
    }

    /// Install the per-runner resolved schedules (one per device, same step order;
    /// MoeRoute expert weights resolved against each runner's resident weights).
    /// Switches `run_step` to the device-resident-aware walk.
    pub fn set_resolved(&mut self, resolved: Vec<Vec<ResolvedSequenceStep>>) {
        self.resolved = resolved;
    }

    pub fn num_devices(&self) -> usize {
        self.runners.len()
    }

    pub fn runner(&self, device: usize) -> &SegmentRunner {
        &self.runners[device]
    }

    pub fn runner_mut(&mut self, device: usize) -> &mut SegmentRunner {
        &mut self.runners[device]
    }

    pub fn runners_mut(&mut self) -> &mut [SegmentRunner] {
        &mut self.runners
    }

    /// Walk the schedule once. Uses the resolved, device-resident-aware walk when
    /// per-runner resolved schedules are installed ([`set_resolved`]), else the
    /// legacy host-only string-keyed walk (CPU tests / dense host path).
    pub fn run_step(&mut self) -> Result<(), RankExecError> {
        if self.resolved.is_empty() {
            self.run_step_host()
        } else {
            self.run_step_resolved()
        }
    }

    /// Resolved walk: executes each device's segments + MoE routing, and resolves
    /// every collective. A device-resident activation handoff (e.g. `embed_out`,
    /// per-layer RingAllReduce) is all-reduced by host-staging each device's GPU
    /// buffer (D2H bf16 -> sum -> H2D) — correct without NVLink P2P; tensors that
    /// stay host (logits AllGather) take the f32-slot path.
    fn run_step_resolved(&mut self) -> Result<(), RankExecError> {
        // Structure (step kinds/order) is identical across runners; walk by index
        // and dispatch each step using that device's own resolved entry. Extract a
        // light descriptor first so the `resolved` borrow is released before the
        // (mutable) runner operations.
        enum StepKind {
            Exec(usize, usize),
            Moe(usize),
            Coll(CollectiveKind, Vec<u32>),
        }
        let n = self.resolved[0].len();
        for i in 0..n {
            let kind = match &self.resolved[0][i] {
                ResolvedSequenceStep::ExecuteSegment {
                    device_idx,
                    segment_idx,
                } => StepKind::Exec(*device_idx as usize, *segment_idx),
                ResolvedSequenceStep::MoeRoute { device_idx, .. } => {
                    StepKind::Moe(*device_idx as usize)
                }
                ResolvedSequenceStep::Collective {
                    collective,
                    participants,
                    ..
                } => StepKind::Coll(*collective, participants.clone()),
            };
            match kind {
                StepKind::Exec(d, seg) => self.runners[d].run_segment(seg)?,
                StepKind::Moe(d) => {
                    // Use device `d`'s own resolved MoeRoute (its experts/weights).
                    if let ResolvedSequenceStep::MoeRoute {
                        ffn_segment_idx,
                        router_id,
                        top_k,
                        block,
                        expert_weights,
                        slot_ids,
                        ..
                    } = self.resolved[d][i].clone()
                    {
                        self.runners[d].route_moe_resolved(
                            ffn_segment_idx,
                            router_id,
                            top_k,
                            block,
                            &expert_weights,
                            &slot_ids,
                        )?;
                    }
                }
                StepKind::Coll(c, parts) => self.run_collective_resolved(c, &parts, i)?,
            }
        }
        Ok(())
    }

    /// Resolve one collective at resolved-step index `i`. Each participant's
    /// tensor id is that runner's own interned id (`resolved[p][i]`).
    fn run_collective_resolved(
        &mut self,
        kind: CollectiveKind,
        participants: &[u32],
        i: usize,
    ) -> Result<(), RankExecError> {
        if participants.is_empty() {
            return Ok(());
        }
        // Per-participant interned tensor id (collect first → no aliasing of
        // `resolved` while we touch `runners`).
        let mut pt: Vec<(usize, HandoffId)> = Vec::with_capacity(participants.len());
        for &p in participants {
            let pi = p as usize;
            if let ResolvedSequenceStep::Collective { tensor, .. } = &self.resolved[pi][i] {
                pt.push((pi, *tensor));
            }
        }

        // Try the device-resident all-reduce path: D2H each rank's GPU buffer.
        if matches!(kind, CollectiveKind::RingAllReduce) {
            let mut device_bufs: Vec<Vec<f32>> = Vec::with_capacity(pt.len());
            let mut all_device = true;
            for &(pi, tid) in &pt {
                match self.runners[pi].read_device_handoff(tid) {
                    Some(b) => device_bufs.push(b),
                    None => {
                        all_device = false;
                        break;
                    }
                }
            }
            if all_device {
                let sum = elementwise_sum(&device_bufs);
                for &(pi, tid) in &pt {
                    self.runners[pi].write_device_handoff(tid, &sum);
                }
                return Ok(());
            }
        }

        // Host path (logits AllGather / Broadcast / a non-device-resident tensor).
        let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(pt.len());
        for &(pi, tid) in &pt {
            bufs.push(self.runners[pi].read_by_id(tid)?);
        }
        match kind {
            CollectiveKind::RingAllReduce => {
                let sum = elementwise_sum(&bufs);
                for &(pi, tid) in &pt {
                    self.runners[pi].write_by_id(tid, sum.clone())?;
                }
            }
            CollectiveKind::AllGather => {
                let gathered: Vec<f32> = bufs.iter().flatten().copied().collect();
                for &(pi, tid) in &pt {
                    self.runners[pi].write_by_id(tid, gathered.clone())?;
                }
            }
            CollectiveKind::Broadcast => {
                let src = bufs[0].clone();
                for &(pi, tid) in &pt {
                    self.runners[pi].write_by_id(tid, src.clone())?;
                }
            }
            CollectiveKind::ReduceScatter => {
                let sum = elementwise_sum(&bufs);
                let chunk = sum.len() / pt.len().max(1);
                for (rank, &(pi, tid)) in pt.iter().enumerate() {
                    let start = rank * chunk;
                    let end = (start + chunk).min(sum.len());
                    self.runners[pi].write_by_id(tid, sum[start..end].to_vec())?;
                }
            }
            CollectiveKind::SendRecv => {
                if pt.len() >= 2 {
                    let src = self.runners[pt[0].0].read_by_id(pt[0].1)?;
                    self.runners[pt[1].0].write_by_id(pt[1].1, src)?;
                }
            }
            kind => return Err(RankExecError::UnsupportedCollective { kind }),
        }
        Ok(())
    }

    /// Legacy host-only walk: execute each device's segments in order and
    /// resolve each collective across its participants' runners (string-keyed).
    fn run_step_host(&mut self) -> Result<(), RankExecError> {
        // Clone the (small) schedule so we can mutably borrow `runners` in the
        // loop without aliasing `self.sequencing`.
        let schedule = self.sequencing.clone();
        for step in &schedule {
            match step {
                SequenceStep::ExecuteSegment {
                    device_idx,
                    segment_idx,
                } => {
                    let d = *device_idx as usize;
                    if d >= self.runners.len() {
                        return Err(RankExecError::Segment {
                            segment_idx: *segment_idx,
                            detail: format!("device {d} out of range"),
                        });
                    }
                    self.runners[d].run_segment(*segment_idx)?;
                }
                SequenceStep::Collective {
                    collective,
                    participants,
                    tensor,
                    ..
                } => {
                    self.run_collective(*collective, participants, tensor)?;
                }
                SequenceStep::MoeRoute { .. } => {}
            }
        }
        Ok(())
    }

    /// Read the tensor from every participant, combine per the collective kind,
    /// and write the result back to every participant — the host-side
    /// equivalent of the NCCL collective over the in-process runners.
    fn run_collective(
        &mut self,
        kind: CollectiveKind,
        participants: &[u32],
        tensor: &str,
    ) -> Result<(), RankExecError> {
        if participants.is_empty() {
            return Ok(());
        }
        let mut bufs: Vec<Vec<f32>> = Vec::with_capacity(participants.len());
        for &p in participants {
            bufs.push(self.runners[p as usize].read(tensor)?);
        }
        match kind {
            CollectiveKind::RingAllReduce => {
                let sum = elementwise_sum(&bufs);
                for &p in participants {
                    self.runners[p as usize].write(tensor, sum.clone())?;
                }
            }
            CollectiveKind::AllGather => {
                let gathered: Vec<f32> = bufs.iter().flatten().copied().collect();
                for &p in participants {
                    self.runners[p as usize].write(tensor, gathered.clone())?;
                }
            }
            CollectiveKind::Broadcast => {
                let src = bufs[0].clone();
                for &p in participants {
                    self.runners[p as usize].write(tensor, src.clone())?;
                }
            }
            CollectiveKind::ReduceScatter => {
                let sum = elementwise_sum(&bufs);
                let n = participants.len();
                let chunk = sum.len() / n.max(1);
                for (rank, &p) in participants.iter().enumerate() {
                    let start = rank * chunk;
                    let end = (start + chunk).min(sum.len());
                    self.runners[p as usize].write(tensor, sum[start..end].to_vec())?;
                }
            }
            CollectiveKind::SendRecv => {
                let src = bufs[0].clone();
                if let Some(&dst) = participants.get(1) {
                    self.runners[dst as usize].write(tensor, src)?;
                }
            }
            CollectiveKind::AllToAll => {
                // Not emitted by the dense tp=2 Mixtral plan; surface clearly
                // rather than silently mis-handling it.
                return Err(RankExecError::UnsupportedCollective { kind });
            }
        }
        Ok(())
    }
}

fn elementwise_sum(bufs: &[Vec<f32>]) -> Vec<f32> {
    let len = bufs.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = vec![0.0f32; len];
    for b in bufs {
        for (o, v) in out.iter_mut().zip(b) {
            *o += *v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::SegmentRunner;
    use skein_compile::{DynRuntime, DynRuntimeError, HandoffId, RuntimeSegment};
    use skein_ir::types::Dtype;
    use std::collections::HashMap;

    /// Per-device mock runtime: on execute it emits this device's partial
    /// contribution `[device+1, device+1]` for the collective tensor `x`. Drives
    /// the real `SegmentRunner`, so it implements the `_by_id` hot-path methods
    /// (delegating to `_by_name` via the interned id → name table).
    struct RankPartial {
        device: usize,
        ids: HashMap<HandoffId, String>,
    }
    impl DynRuntime for RankPartial {
        fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
            Ok(())
        }
        fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError> {
            if name == "x" {
                Ok(vec![(self.device as f32) + 1.0; 2])
            } else {
                Err(DynRuntimeError::UnknownTensor(name.to_string()))
            }
        }
        fn set_tensor_by_name(&mut self, _: &str, _: Vec<f32>) -> Result<(), DynRuntimeError> {
            Ok(())
        }
        fn set_tensor_i32_by_name(&mut self, _: &str, _: Vec<i32>) -> Result<(), DynRuntimeError> {
            Ok(())
        }
        fn register_handoff_ids(&mut self, id_for_name: &dyn Fn(&str) -> Option<HandoffId>) {
            if let Some(id) = id_for_name("x") {
                self.ids.insert(id, "x".to_string());
            }
        }
        fn get_tensor_by_id(&self, id: HandoffId) -> Result<Vec<f32>, DynRuntimeError> {
            let name = self
                .ids
                .get(&id)
                .cloned()
                .ok_or_else(|| DynRuntimeError::UnknownTensor(format!("id {}", id.0)))?;
            self.get_tensor_by_name(&name)
        }
        fn set_tensor_by_id(
            &mut self,
            id: HandoffId,
            data: Vec<f32>,
        ) -> Result<(), DynRuntimeError> {
            let name = self
                .ids
                .get(&id)
                .cloned()
                .ok_or_else(|| DynRuntimeError::UnknownTensor(format!("id {}", id.0)))?;
            self.set_tensor_by_name(&name, data)
        }
    }

    fn runner(device: usize) -> SegmentRunner {
        let seg = RuntimeSegment {
            runtime: Box::new(RankPartial {
                device,
                ids: HashMap::new(),
            }),
            input_names: vec![],
            output_names: vec!["x".to_string()],
            capture_names: vec![],
            weight_names: vec![],
            kv_cache_sizes: HashMap::new(),
        };
        SegmentRunner::new(vec![seg])
    }

    /// The single-process multi-device walker runs both devices' segments and
    /// resolves the RingAllReduce host-side: both devices end with the summed
    /// `x = [3, 3]` (1 from device 0 + 2 from device 1) — the same contract the
    /// multi-process `RankExecutor` + NCCL provides.
    #[test]
    fn local_topology_runs_segments_and_all_reduces() {
        let schedule = vec![
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
        ];
        let mut topo = LocalTopology::new(vec![runner(0), runner(1)], schedule);
        topo.run_step().expect("run_step");
        assert_eq!(topo.runner(0).read("x").unwrap(), vec![3.0, 3.0]);
        assert_eq!(topo.runner(1).read("x").unwrap(), vec![3.0, 3.0]);
    }

    /// AllGather concatenates each device's contribution onto every device.
    #[test]
    fn local_topology_all_gather_concatenates() {
        let schedule = vec![
            SequenceStep::ExecuteSegment {
                device_idx: 0,
                segment_idx: 0,
            },
            SequenceStep::ExecuteSegment {
                device_idx: 1,
                segment_idx: 0,
            },
            SequenceStep::Collective {
                collective: CollectiveKind::AllGather,
                participants: vec![0, 1],
                tensor: "x".to_string(),
                shape: vec![2],
                dtype: Dtype::Bf16,
            },
        ];
        let mut topo = LocalTopology::new(vec![runner(0), runner(1)], schedule);
        topo.run_step().expect("run_step");
        // device0 [1,1] ++ device1 [2,2] on both.
        assert_eq!(topo.runner(0).read("x").unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
        assert_eq!(topo.runner(1).read("x").unwrap(), vec![1.0, 1.0, 2.0, 2.0]);
    }
}

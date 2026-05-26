//! Shared topology executor for parity and serving.
//!
//! The executor owns no HTTP or batching policy. It receives already-built
//! runtime segments, feeds logical handoff tensors by name, walks the
//! serialized [`SequenceStep`] schedule, and returns logits / activation
//! captures to its caller.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use luminal::prelude::NodeIndex;
use safetensors::{Dtype as SafeDtype, SafeTensors};
use skein_cost::collectives::CollectiveKind;
use skein_emit::segment::{Segment, SequenceStep};

use crate::dyn_runtime::WeightDtype;
use crate::{
    CompileError, DynRuntime, DynRuntimeError, DynRuntimeWrapper, NativeComputeRuntime,
    SkeinArtifact,
};

/// Zero-buffer staging list for a segment's `Input` ops: `(node, num_bytes)`
/// for every declared weight plus every input handoff. Luminal's search
/// executes the graph and the CUDA backend needs a buffer for each `Input`;
/// these zeros are placeholders (real data is loaded after compile). The
/// `input_tokens` handoff is an i32 index tensor (4 bytes/elem) regardless of
/// its marker dtype.
/// Per-input search-staging spec: `(node, byte_len, fill)`. `fill` is the scalar
/// value to stage (0.0 for the usual zero buffers). The fp8 attention
/// `*.weight_scale` / `*.input_scale` inputs stage as **1.0** so the search's
/// `(activation / input_scale)` quantize step doesn't divide by zero and produce
/// NaN outputs (which would make every candidate genome non-viable).
pub fn segment_input_zero_bytes(segment: &Segment) -> Vec<(NodeIndex, usize, f32)> {
    let mut out = Vec::with_capacity(segment.declared.len() + segment.input_handoff.len());
    for (name, d) in segment.declared.iter() {
        let n: usize = d.shape.iter().product();
        let fill = if name.ends_with("_scale") { 1.0 } else { 0.0 };
        out.push((d.id, d.dtype.bytes_for(n as u64) as usize, fill));
    }
    for h in &segment.input_handoff {
        let n: usize = h.shape.iter().product();
        // `input_tokens` is i32; `position` and the `kvcache_*` past tensors are
        // fed as raw f32 — all 4 bytes/element, regardless of the marker dtype
        // recorded on the handoff. Everything else uses its real dtype width.
        let four_byte = h.logical_name == "input_tokens"
            || h.logical_name == "position"
            || h.logical_name.starts_with("kvcache_");
        let bytes = if four_byte {
            n * 4
        } else {
            h.dtype.bytes_for(n as u64) as usize
        };
        out.push((h.luminal_id, bytes, 0.0));
    }
    out
}

pub const DEFAULT_SEARCH_BUDGET: usize = 1;

pub trait CollectiveExecutor {
    fn execute(
        &self,
        kind: CollectiveKind,
        participants: &[usize],
        tensor_name: &str,
        runtimes: &mut [&mut dyn DynRuntime],
    ) -> Result<(), CompileError>;
}

pub struct RuntimeSegment {
    pub runtime: Box<dyn DynRuntime>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub capture_names: Vec<String>,
    pub weight_names: Vec<(String, Vec<usize>)>,
    /// For each `kvcache_*` input of this segment, the full element count of its
    /// fixed-capacity buffer (`batch * KV_CACHE_CAP * n_kv*head_dim`). The runner
    /// allocates the cache to this size and feeds it whole each step.
    pub kv_cache_sizes: HashMap<String, usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TopologyStepBatch {
    pub request_tokens: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepOutput {
    pub per_request_logits: Vec<Vec<f32>>,
    pub next_tokens: Vec<u32>,
}

pub struct TopologyExecutor<'a> {
    runtimes: &'a mut [Vec<RuntimeSegment>],
    collectives: &'a dyn CollectiveExecutor,
    sequencing: &'a [SequenceStep],
}

impl<'a> TopologyExecutor<'a> {
    pub fn new(
        runtimes: &'a mut [Vec<RuntimeSegment>],
        collectives: &'a dyn CollectiveExecutor,
        sequencing: &'a [SequenceStep],
    ) -> Self {
        Self {
            runtimes,
            collectives,
            sequencing,
        }
    }

    pub fn execute_for_parity(
        &mut self,
        tokens: &[u32],
    ) -> Result<crate::executor::StepOutput, CompileError> {
        let logits = self.execute_one(tokens)?;
        let next = argmax(&logits);
        Ok(StepOutput {
            per_request_logits: vec![logits],
            next_tokens: vec![next],
        })
    }

    pub fn execute_for_step(
        &mut self,
        batch: &TopologyStepBatch,
    ) -> Result<StepOutput, CompileError> {
        let mut per_request_logits = Vec::with_capacity(batch.request_tokens.len());
        let mut next_tokens = Vec::with_capacity(batch.request_tokens.len());
        for tokens in &batch.request_tokens {
            let logits = self.execute_one(tokens)?;
            next_tokens.push(argmax(&logits));
            per_request_logits.push(logits);
        }
        Ok(StepOutput {
            per_request_logits,
            next_tokens,
        })
    }

    pub fn execute_with_hooks(
        &mut self,
        tokens: &[u32],
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), CompileError> {
        let mut captures = BTreeMap::new();
        let logits = self.walk(tokens, Some(&mut captures))?;
        Ok((captures.into_values().collect(), logits))
    }

    fn execute_one(&mut self, tokens: &[u32]) -> Result<Vec<f32>, CompileError> {
        self.walk(tokens, None)
    }

    fn walk(
        &mut self,
        tokens: &[u32],
        mut captures: Option<&mut BTreeMap<usize, Vec<f32>>>,
    ) -> Result<Vec<f32>, CompileError> {
        // Cached-decode prefill: feed the prompt one token at a time, walking the
        // full collective schedule per token while accumulating the per-(device,
        // layer) KV cache, so attention attends over the real prompt history.
        // The logits after the *last* token are returned (compared to HF). This
        // mirrors the serving `SegmentRunner` loop, single-process here. (The KV
        // cache type lives in `skein_runtime`, which depends on this crate, so a
        // small inline f32 accumulator + name parser is used instead.)
        let prompt: Vec<u32> = if tokens.is_empty() {
            vec![0]
        } else {
            tokens.to_vec()
        };

        // Op-by-op bisection dump target (debug only; see `dump_capture`).
        let dump_dir = std::env::var_os("SKEIN_DUMP_DIR").map(std::path::PathBuf::from);

        // Per-(device, kvcache-name) accumulated past, grown by one token/step.
        let mut kv: HashMap<(usize, String), Vec<f32>> = HashMap::new();
        let mut final_logits = Vec::new();
        let last_pos = prompt.len() - 1;

        let n_devices = self.runtimes.len();

        // Batched-prefill path: SKEIN_PREFILL_SEQ>1 means the compiled graph is
        // the seq=N prefill graph — feed the WHOLE prompt in ONE forward (causal
        // attention computes every position at once; no per-token loop, no KV
        // cache input). Return the last token's logits. This is the parallel
        // prefill; the per-token loop below is the seq=1 (decode-style) path.
        let prefill_seq = std::env::var("SKEIN_PREFILL_SEQ")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(1);
        if prefill_seq > 1 {
            let n = prompt.len();
            let toks_i32: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
            let mut handoffs: HashMap<(usize, String), Vec<f32>> = HashMap::new();
            let mut handoff_i32: HashMap<(usize, String), Vec<i32>> = HashMap::new();
            for d in 0..n_devices {
                handoff_i32.insert((d, "input_tokens".to_string()), toks_i32.clone());
            }
            for step in self.sequencing {
                match step {
                    SequenceStep::ExecuteSegment {
                        device_idx,
                        segment_idx,
                    } => {
                        let device_idx = *device_idx as usize;
                        let segment_idx = *segment_idx;
                        let segment = self
                            .runtimes
                            .get_mut(device_idx)
                            .and_then(|d| d.get_mut(segment_idx))
                            .ok_or(CompileError::MissingSegment {
                                device_idx,
                                segment_idx,
                            })?;
                        for name in &segment.input_names {
                            let key = (device_idx, name.clone());
                            if let Some(data) = handoff_i32.get(&key) {
                                segment.runtime.set_tensor_i32_by_name(name, data.clone())?;
                            } else if let Some(data) = handoffs.get(&key) {
                                segment.runtime.set_tensor_by_name(name, data.clone())?;
                            }
                        }
                        segment.runtime.execute_segment()?;
                        if let Some(dump_dir) = dump_dir.as_deref() {
                            for name in &segment.capture_names {
                                let data = segment.runtime.get_tensor_by_name(name)?;
                                dump_capture(dump_dir, name, device_idx, &data);
                            }
                        }
                        for name in &segment.output_names {
                            let data = segment.runtime.get_tensor_by_name(name)?;
                            if name == "logits" {
                                final_logits = data.clone();
                            }
                            if !is_kvcache_name(name) {
                                handoffs.insert((device_idx, name.clone()), data);
                            }
                        }
                    }
                    SequenceStep::Collective {
                        collective,
                        participants,
                        tensor,
                        ..
                    } => {
                        let participants: Vec<usize> =
                            participants.iter().map(|p| *p as usize).collect();
                        let mut adapters = Vec::with_capacity(participants.len());
                        for &p in &participants {
                            let data = handoffs
                                .get(&(p, tensor.clone()))
                                .cloned()
                                .unwrap_or_default();
                            adapters.push(HandoffRuntime::new(tensor, data));
                        }
                        let mut refs: Vec<&mut dyn DynRuntime> = adapters
                            .iter_mut()
                            .map(|r| r as &mut dyn DynRuntime)
                            .collect();
                        let local_participants: Vec<usize> = (0..participants.len()).collect();
                        self.collectives.execute(
                            *collective,
                            &local_participants,
                            tensor,
                            refs.as_mut_slice(),
                        )?;
                        for (rank, &device) in participants.iter().enumerate() {
                            let data = refs[rank].get_tensor_by_name(tensor)?;
                            if tensor == "logits" {
                                final_logits = data.clone();
                            }
                            handoffs.insert((device, tensor.clone()), data);
                        }
                    }
                    // Sparse-MoE routing is handled in the multi-process serve
                    // executor; the single-process parity path runs dense.
                    SequenceStep::MoeRoute { .. } => {}
                }
            }
            // The prefill graph slices to the last token before the LM head, so
            // final_logits is already the [vocab] next-token prediction.
            let _ = n;
            return Ok(final_logits);
        }

        for (position, &tok) in prompt.iter().enumerate() {
            let mut handoffs: HashMap<(usize, String), Vec<f32>> = HashMap::new();
            let mut handoff_i32: HashMap<(usize, String), Vec<i32>> = HashMap::new();
            // Feed input_tokens to *every* device, not just device 0: with tp>1
            // a segment on another device also consumes it (e.g. the embedding
            // Cast). The serve path feeds it per-rank; this single-process
            // driver must do so for all devices or that segment's Int input
            // buffer is never bound. (Each segment only pulls it if its
            // input_names list it, so over-registering is harmless.)
            for d in 0..n_devices {
                handoff_i32.insert((d, "input_tokens".to_string()), vec![tok as i32]);
            }

            for step in self.sequencing {
                match step {
                    SequenceStep::ExecuteSegment {
                        device_idx,
                        segment_idx,
                    } => {
                        let device_idx = *device_idx as usize;
                        let segment_idx = *segment_idx;
                        let segment = self
                            .runtimes
                            .get_mut(device_idx)
                            .and_then(|d| d.get_mut(segment_idx))
                            .ok_or(CompileError::MissingSegment {
                                device_idx,
                                segment_idx,
                            })?;

                        for name in &segment.input_names {
                            let key = (device_idx, name.clone());
                            if is_kvcache_name(name) {
                                // Feed the whole fixed-capacity cache buffer
                                // (allocated zero on first use to its full size).
                                let full = segment.kv_cache_sizes.get(name).copied().unwrap_or(0);
                                let buf = kv.entry(key).or_insert_with(|| vec![0.0; full]);
                                segment.runtime.set_tensor_by_name(name, buf.clone())?;
                            } else if name == "position" {
                                segment
                                    .runtime
                                    .set_tensor_by_name(name, vec![position as f32])?;
                            } else if let Some(data) = handoff_i32.get(&key) {
                                segment.runtime.set_tensor_i32_by_name(name, data.clone())?;
                            } else if let Some(data) = handoffs.get(&key) {
                                segment.runtime.set_tensor_by_name(name, data.clone())?;
                            }
                        }

                        segment.runtime.execute_segment()?;

                        for name in &segment.output_names {
                            let data = segment.runtime.get_tensor_by_name(name)?;
                            if is_kvcache_name(name) {
                                // The new token's K/V — write into slot `position`
                                // of this layer's fixed cache.
                                let full = segment.kv_cache_sizes.get(name).copied().unwrap_or(0);
                                let buf = kv
                                    .entry((device_idx, name.clone()))
                                    .or_insert_with(|| vec![0.0; full]);
                                let off = position * data.len();
                                if !data.is_empty() && off + data.len() <= buf.len() {
                                    buf[off..off + data.len()].copy_from_slice(&data);
                                }
                            } else {
                                if name == "logits" {
                                    final_logits = data.clone();
                                }
                                handoffs.insert((device_idx, name.clone()), data);
                            }
                        }

                        // Capture per-layer activations only on the last token.
                        if position == last_pos {
                            // Op-by-op layer-0 bisection dump (gated by env): write
                            // every `dbg_*` tap this segment exposes to disk as raw
                            // little-endian f32, suffixed by device so sharded
                            // (TP-split) tensors can be reassembled in the comparison.
                            if let Some(dump_dir) = dump_dir.as_deref() {
                                for name in &segment.capture_names {
                                    let data = segment.runtime.get_tensor_by_name(name)?;
                                    dump_capture(dump_dir, name, device_idx, &data);
                                }
                            }
                            if let Some(captures) = captures.as_deref_mut() {
                                for name in &segment.capture_names {
                                    if let Some(layer_idx) = parse_hidden_after_block(name) {
                                        let data = segment.runtime.get_tensor_by_name(name)?;
                                        captures.insert(layer_idx, data);
                                    }
                                }
                            }
                        }
                    }
                    SequenceStep::Collective {
                        collective,
                        participants,
                        tensor,
                        ..
                    } => {
                        let participants: Vec<usize> =
                            participants.iter().map(|p| *p as usize).collect();
                        let mut adapters = Vec::with_capacity(participants.len());
                        for &p in &participants {
                            let data = handoffs
                                .get(&(p, tensor.clone()))
                                .cloned()
                                .unwrap_or_default();
                            adapters.push(HandoffRuntime::new(tensor, data));
                        }
                        let mut refs: Vec<&mut dyn DynRuntime> = adapters
                            .iter_mut()
                            .map(|r| r as &mut dyn DynRuntime)
                            .collect();
                        let local_participants: Vec<usize> = (0..participants.len()).collect();
                        self.collectives.execute(
                            *collective,
                            &local_participants,
                            tensor,
                            refs.as_mut_slice(),
                        )?;
                        for (rank, &device) in participants.iter().enumerate() {
                            let data = refs[rank].get_tensor_by_name(tensor)?;
                            // The logits AllGather concatenates the per-rank vocab
                            // shards into the full vocab; `final_logits` was set
                            // from the pre-gather shard in the segment-output loop,
                            // so refresh it with the gathered (full-width) result.
                            if tensor == "logits" {
                                final_logits = data.clone();
                            }
                            handoffs.insert((device, tensor.clone()), data);
                        }
                    }
                    SequenceStep::MoeRoute { .. } => {}
                }
            }
        }

        Ok(final_logits)
    }
}

/// A handoff named `kvcache_{k|v}_{layer}` carries per-layer KV cache, not a
/// cross-segment activation — the executor feeds it the accumulated past and
/// appends its step output rather than routing it as a normal handoff.
fn is_kvcache_name(name: &str) -> bool {
    name.starts_with("kvcache_")
}

pub fn load_native_runtime_segments(
    artifact: &SkeinArtifact,
) -> Result<Vec<Vec<RuntimeSegment>>, CompileError> {
    load_runtime_segments::<NativeComputeRuntime>(artifact, DEFAULT_SEARCH_BUDGET)
}

pub fn load_runtime_segments<R: crate::ComputeRuntime + 'static>(
    artifact: &SkeinArtifact,
    search_budget: usize,
) -> Result<Vec<Vec<RuntimeSegment>>, CompileError> {
    // Serve from / write to the artifact's on-disk compile cache so the second
    // and later boots of this artifact load instead of re-searching + re-NVRTC.
    let cache_dir = crate::search_cache::search_cache_dir(&artifact.root);
    crate::search_cache::enable_cubin_cache(&artifact.root);
    let timing = std::env::var_os("SKEIN_TIMING").is_some();
    let mut all_devices = Vec::with_capacity(artifact.devices.len());
    for device in &artifact.devices {
        // Opt-in single-process multi-GPU: place tp shard `d` on physical GPU
        // `d` so the full model isn't pinned to one card (frees per-GPU memory
        // for concurrent batching). The in-process collective is host-mediated,
        // so cross-GPU reduction/gather works without peer access. Default
        // (unset) keeps every shard on device 0 — the layout verify relies on.
        if std::env::var_os("SKEIN_SPREAD_DEVICES").is_some() {
            crate::set_build_device(device.device_idx);
        }
        let t_emit = std::time::Instant::now();
        let lowered = device.rebuild_graphs()?;
        let emit_ms = t_emit.elapsed().as_secs_f64() * 1e3;
        let t_replay = std::time::Instant::now();
        let mut runtime_segments = Vec::with_capacity(lowered.len());
        let n_seg = lowered.len();
        for segment in lowered {
            runtime_segments.push(compile_segment::<R>(
                segment,
                search_budget,
                Some(&cache_dir),
            )?);
        }
        let replay_ms = t_replay.elapsed().as_secs_f64() * 1e3;
        let t_w = std::time::Instant::now();
        load_weights_into_segments(&device.weights_path, runtime_segments.as_mut_slice())?;
        let weight_ms = t_w.elapsed().as_secs_f64() * 1e3;
        if timing {
            eprintln!(
                "SKEIN_TIMING device {} ({n_seg} segs): emit {emit_ms:.0}ms | graph-replay(cache) {replay_ms:.0}ms | weight-load+convert {weight_ms:.0}ms",
                device.device_idx
            );
        }
        all_devices.push(runtime_segments);
    }
    Ok(all_devices)
}

/// Load and compile the segments for a *single* device. Used by the
/// multi-process multi-GPU path, where each rank process owns exactly one
/// device and must not build the others' graphs. `device_idx` is matched
/// against `DeviceArtifactLoaded::device_idx`.
pub fn load_device_runtime_segments<R: crate::ComputeRuntime + 'static>(
    artifact: &SkeinArtifact,
    device_idx: usize,
    search_budget: usize,
) -> Result<Vec<RuntimeSegment>, CompileError> {
    let device = artifact
        .devices
        .iter()
        .find(|d| d.device_idx == device_idx)
        .ok_or(CompileError::MissingSegment {
            device_idx,
            segment_idx: 0,
        })?;
    let cache_dir = crate::search_cache::search_cache_dir(&artifact.root);
    crate::search_cache::enable_cubin_cache(&artifact.root);
    let lowered = device.rebuild_graphs()?;
    let mut runtime_segments = Vec::with_capacity(lowered.len());
    for segment in lowered {
        runtime_segments.push(compile_segment::<R>(
            segment,
            search_budget,
            Some(&cache_dir),
        )?);
    }
    load_weights_into_segments(&device.weights_path, runtime_segments.as_mut_slice())?;
    Ok(runtime_segments)
}

/// Build this device's **batched-prefill** graph (seq=N) and wire its weight
/// inputs to the already-resident weights of the decode graph by device pointer,
/// instead of loading a second ~47 GB/device copy (which would OOM a 96 GB card).
///
/// `decode_segments` must already have their weights materialized on the GPU
/// (call [`DynRuntime::materialize_weights`] on each first). Weights are bf16 in
/// the runtime weight-load path, so byte sizes are computed as `elems * 2`.
pub fn load_device_prefill_segments<R: crate::ComputeRuntime + 'static>(
    artifact: &SkeinArtifact,
    device_idx: usize,
    search_budget: usize,
    seq: usize,
    decode_segments: &[RuntimeSegment],
) -> Result<Vec<RuntimeSegment>, CompileError> {
    let device = artifact
        .devices
        .iter()
        .find(|d| d.device_idx == device_idx)
        .ok_or(CompileError::MissingSegment {
            device_idx,
            segment_idx: 0,
        })?;
    let cache_dir = crate::search_cache::search_cache_dir(&artifact.root);
    crate::search_cache::enable_cubin_cache(&artifact.root);

    // Map weight name -> (device_ptr, n_bytes) from the resident decode weights.
    let mut weight_ptrs: HashMap<String, (u64, usize)> = HashMap::new();
    for seg in decode_segments {
        for (name, shape) in &seg.weight_names {
            if let Some(ptr) = seg.runtime.weight_device_ptr_by_name(name) {
                let n_bytes = shape.iter().product::<usize>() * 2; // bf16
                weight_ptrs.insert(name.clone(), (ptr, n_bytes));
            }
        }
    }

    let lowered = device.rebuild_graphs_with_seq(seq)?;
    let mut prefill_segments = Vec::with_capacity(lowered.len());
    for segment in lowered {
        prefill_segments.push(compile_segment::<R>(
            segment,
            search_budget,
            Some(&cache_dir),
        )?);
    }
    // Share weights into the prefill graph (no second copy loaded).
    for seg in &mut prefill_segments {
        let names: Vec<String> = seg.weight_names.iter().map(|(n, _)| n.clone()).collect();
        for name in names {
            if let Some((ptr, n_bytes)) = weight_ptrs.get(&name).copied() {
                unsafe {
                    seg.runtime
                        .set_weight_device_ptr_by_name(&name, ptr, n_bytes)
                };
            }
        }
    }
    Ok(prefill_segments)
}

fn compile_segment<R: crate::ComputeRuntime + 'static>(
    mut segment: Segment,
    search_budget: usize,
    cache_dir: Option<&Path>,
) -> Result<RuntimeSegment, CompileError> {
    let input_names = segment
        .input_handoff
        .iter()
        .map(|h| h.logical_name.clone())
        .collect();
    let output_names = segment
        .output_handoff
        .iter()
        .map(|h| h.logical_name.clone())
        .collect();
    let capture_names = segment
        .op_nodes
        .keys()
        .filter(|name| name.starts_with("hidden_after_block_") || name.starts_with("dbg_"))
        .cloned()
        .collect();
    let weight_names = segment
        .declared
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.shape.clone()))
        .collect();
    let kv_cache_sizes = segment
        .input_handoff
        .iter()
        .filter(|h| h.logical_name.starts_with("kvcache_"))
        .map(|h| (h.logical_name.clone(), h.shape.iter().product::<usize>()))
        .collect();

    let mut name_to_node = HashMap::new();
    name_to_node.extend(segment.op_nodes.iter().map(|(k, v)| (k.clone(), *v)));
    name_to_node.extend(
        segment
            .declared
            .iter()
            .map(|(name, declared)| (name.clone(), declared.id)),
    );
    name_to_node.extend(
        segment
            .input_handoff
            .iter()
            .map(|h| (h.logical_name.clone(), h.luminal_id)),
    );
    let input_name_to_node = segment
        .input_handoff
        .iter()
        .map(|h| (h.logical_name.clone(), h.luminal_id))
        .collect();
    name_to_node.extend(
        segment
            .output_handoff
            .iter()
            .map(|h| (h.logical_name.clone(), h.luminal_id)),
    );

    // Bind luminal's sequence dim `s` to this segment's concrete token count.
    // Host ops authored against luminal's `s`-convention (notably the fused
    // GLUMoE MoE op) shape their output buffer as `[s, hidden]` with a symbolic
    // `s`. Skein's segments are otherwise fully static, so without this nothing
    // resolves `s` and `plan_intermediate_buffers` cannot size the buffer (the
    // search then "fails to find a viable initial genome"). luminal's own GLUMoE
    // path requires the same `set_dim('s', SEQ)` before searching. The token
    // count is `batch*seq` — the product of the leading dims of any activation
    // handoff `[batch, seq, hidden]`.
    if let Some(tokens) = segment
        .output_handoff
        .iter()
        .chain(segment.input_handoff.iter())
        .filter(|h| h.shape.len() == 3)
        .map(|h| h.shape[..h.shape.len() - 1].iter().product::<usize>())
        .find(|t| *t > 0)
    {
        segment.graph.set_dim('s', tokens);
    }

    let input_zeros = segment_input_zero_bytes(&segment);
    let runtime =
        R::build_and_search_cached(&mut segment.graph, search_budget, &input_zeros, cache_dir)?;
    let wrapped = DynRuntimeWrapper::new(runtime, segment.graph, name_to_node, input_name_to_node);
    Ok(RuntimeSegment {
        runtime: Box::new(wrapped),
        input_names,
        output_names,
        capture_names,
        weight_names,
        kv_cache_sizes,
    })
}

fn load_weights_into_segments(
    path: &Path,
    segments: &mut [RuntimeSegment],
) -> Result<(), CompileError> {
    // Memory-map the shard rather than reading it into an anonymous Vec: a
    // 46.7 GB `std::fs::read` is ~47 GB of anon heap per rank, and two PP ranks
    // loading concurrently on one node overran host RAM (OOM-killer, signal 9).
    // The mmap is file-backed (page cache, reclaimable under pressure) and read
    // by SafeTensors as a plain &[u8] slice. Read-only, so no flush needed.
    let file = std::fs::File::open(path).map_err(|source| CompileError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let bytes = unsafe {
        memmap2::Mmap::map(&file).map_err(|source| CompileError::Io {
            path: path.to_path_buf(),
            source,
        })?
    };
    let tensors = SafeTensors::deserialize(&bytes).map_err(|source| CompileError::Safetensors {
        path: path.to_path_buf(),
        source,
    })?;

    for segment in segments {
        for (name, shape) in &segment.weight_names {
            // Stacked-expert weight (on-device sparse MoE): the resident tensor
            // `[E, d1, d2]` is the concatenation of the E per-expert checkpoint
            // tensors, so `gather` can index k-of-E without N separate copies.
            if let Some(idx) = name.find(".experts.stacked_") {
                let prefix = &name[..idx]; // model.layers.{b}.block_sparse_moe
                let kind = &name[idx + ".experts.stacked_".len()..]; // gate_up | down
                // Fused gate+up is per-expert [w1 (gate); w3 (up)] concatenated;
                // down is [w2]. Matches the GLUMoE weight layout.
                let wtypes: &[&str] = match kind {
                    "gate_up" => &["w1", "w3"],
                    "down" => &["w2"],
                    _ => {
                        return Err(CompileError::MissingWeight {
                            path: path.to_path_buf(),
                            tensor: name.clone(),
                        });
                    }
                };
                let n_exp = shape[0];
                let mut buf: Vec<u8> = Vec::new();
                let mut dt: Option<WeightDtype> = None;
                for e in 0..n_exp {
                    for w in wtypes {
                        let en = format!("{prefix}.experts.{e}.{w}.weight");
                        let t = tensors.tensor(&en).map_err(|_| CompileError::MissingWeight {
                            path: path.to_path_buf(),
                            tensor: en.clone(),
                        })?;
                        dt = Some(weight_dtype(&en, t.dtype())?);
                        buf.extend_from_slice(t.data());
                    }
                }
                let dtype = dt.ok_or_else(|| CompileError::MissingWeight {
                    path: path.to_path_buf(),
                    tensor: name.clone(),
                })?;
                let expected = shape.iter().product::<usize>();
                let got = buf.len() / dtype.byte_width();
                if got != expected {
                    return Err(CompileError::TensorSizeMismatch {
                        tensor: name.clone(),
                        expected,
                        got,
                    });
                }
                segment.runtime.set_tensor_bytes_by_name(name, &buf, dtype)?;
                continue;
            }
            let tensor = tensors
                .tensor(name)
                .map_err(|_| CompileError::MissingWeight {
                    path: path.to_path_buf(),
                    tensor: name.clone(),
                })?;
            let dtype = weight_dtype(name, tensor.dtype())?;
            // Validate the byte count against the declared shape before
            // handing the raw bytes to the byte-level loader.
            let expected = shape.iter().product::<usize>();
            let got = tensor.data().len() / dtype.byte_width();
            if got != expected {
                return Err(CompileError::TensorSizeMismatch {
                    tensor: name.clone(),
                    expected,
                    got,
                });
            }
            segment
                .runtime
                .set_tensor_bytes_by_name(name, tensor.data(), dtype)?;
        }
    }
    Ok(())
}

/// Map a safetensors checkpoint dtype to the runtime's [`WeightDtype`].
fn weight_dtype(tensor: &str, dtype: SafeDtype) -> Result<WeightDtype, CompileError> {
    match dtype {
        SafeDtype::F32 => Ok(WeightDtype::F32),
        SafeDtype::BF16 => Ok(WeightDtype::Bf16),
        SafeDtype::F16 => Ok(WeightDtype::F16),
        SafeDtype::F8_E4M3 => Ok(WeightDtype::F8E4M3),
        dtype => Err(CompileError::UnsupportedWeightDtype {
            tensor: tensor.to_string(),
            dtype,
        }),
    }
}

fn parse_hidden_after_block(name: &str) -> Option<usize> {
    name.strip_prefix("hidden_after_block_")?.parse().ok()
}

/// Write a captured `dbg_*` tensor to `<dump_dir>/<name>.dev<device>.f32` as
/// raw little-endian f32. Best-effort debug instrumentation for the HF
/// parity bisection — failures are logged, never fatal.
fn dump_capture(dump_dir: &Path, name: &str, device_idx: usize, data: &[f32]) {
    if let Err(e) = std::fs::create_dir_all(dump_dir) {
        tracing::warn!(?e, "dump_capture: create_dir_all failed");
        return;
    }
    let path = dump_dir.join(format!("{name}.dev{device_idx}.f32"));
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&path, &bytes) {
        tracing::warn!(?e, path = %path.display(), "dump_capture: write failed");
    } else {
        tracing::info!(path = %path.display(), elems = data.len(), "dump_capture: wrote tap");
    }
}

fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

struct HandoffRuntime {
    tensor: String,
    data: Vec<f32>,
}

impl HandoffRuntime {
    fn new(tensor: &str, data: Vec<f32>) -> Self {
        Self {
            tensor: tensor.to_string(),
            data,
        }
    }
}

impl DynRuntime for HandoffRuntime {
    fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
        Ok(())
    }

    fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError> {
        if name == self.tensor {
            Ok(self.data.clone())
        } else {
            Err(DynRuntimeError::UnknownTensor(name.to_string()))
        }
    }

    fn set_tensor_by_name(&mut self, name: &str, data: Vec<f32>) -> Result<(), DynRuntimeError> {
        if name == self.tensor {
            self.data = data;
            Ok(())
        } else {
            Err(DynRuntimeError::UnknownTensor(name.to_string()))
        }
    }

    fn set_tensor_i32_by_name(
        &mut self,
        name: &str,
        data: Vec<i32>,
    ) -> Result<(), DynRuntimeError> {
        self.set_tensor_by_name(name, data.into_iter().map(|v| v as f32).collect())
    }
}

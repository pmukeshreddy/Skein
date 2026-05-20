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
pub fn segment_input_zero_bytes(segment: &Segment) -> Vec<(NodeIndex, usize)> {
    let mut out = Vec::with_capacity(segment.declared.len() + segment.input_handoff.len());
    for d in segment.declared.values() {
        let n: usize = d.shape.iter().product();
        out.push((d.id, d.dtype.bytes_for(n as u64) as usize));
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
        out.push((h.luminal_id, bytes));
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

        // Per-(device, kvcache-name) accumulated past, grown by one token/step.
        let mut kv: HashMap<(usize, String), Vec<f32>> = HashMap::new();
        let mut final_logits = Vec::new();
        let last_pos = prompt.len() - 1;

        for (position, &tok) in prompt.iter().enumerate() {
            let mut handoffs: HashMap<(usize, String), Vec<f32>> = HashMap::new();
            let mut handoff_i32: HashMap<(usize, String), Vec<i32>> = HashMap::new();
            handoff_i32.insert((0, "input_tokens".to_string()), vec![tok as i32]);

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
                                let past = kv.get(&key).cloned().unwrap_or_default();
                                segment.runtime.set_tensor_by_name(name, past)?;
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

                        // This step's dynamic `past` length (= position).
                        segment.runtime.set_dyn_dim('p', position);
                        segment.runtime.execute_segment()?;

                        for name in &segment.output_names {
                            let data = segment.runtime.get_tensor_by_name(name)?;
                            if is_kvcache_name(name) {
                                // The new token's K/V — append to this layer's cache.
                                kv.entry((device_idx, name.clone()))
                                    .or_default()
                                    .extend_from_slice(&data);
                            } else {
                                if name == "logits" {
                                    final_logits = data.clone();
                                }
                                handoffs.insert((device_idx, name.clone()), data);
                            }
                        }

                        // Capture per-layer activations only on the last token.
                        if position == last_pos {
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
                            handoffs.insert((device, tensor.clone()), data);
                        }
                    }
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
    let mut all_devices = Vec::with_capacity(artifact.devices.len());
    for device in &artifact.devices {
        let lowered = device.rebuild_graphs()?;
        let mut runtime_segments = Vec::with_capacity(lowered.len());
        for segment in lowered {
            runtime_segments.push(compile_segment::<R>(segment, search_budget)?);
        }
        load_weights_into_segments(&device.weights_path, runtime_segments.as_mut_slice())?;
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
    let lowered = device.rebuild_graphs()?;
    let mut runtime_segments = Vec::with_capacity(lowered.len());
    for segment in lowered {
        runtime_segments.push(compile_segment::<R>(segment, search_budget)?);
    }
    load_weights_into_segments(&device.weights_path, runtime_segments.as_mut_slice())?;
    Ok(runtime_segments)
}

fn compile_segment<R: crate::ComputeRuntime + 'static>(
    mut segment: Segment,
    search_budget: usize,
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
        .filter(|name| name.starts_with("hidden_after_block_"))
        .cloned()
        .collect();
    let weight_names = segment
        .declared
        .iter()
        .map(|(name, tensor)| (name.clone(), tensor.shape.clone()))
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
    name_to_node.extend(
        segment
            .output_handoff
            .iter()
            .map(|h| (h.logical_name.clone(), h.luminal_id)),
    );

    let input_zeros = segment_input_zero_bytes(&segment);
    let runtime =
        R::build_and_search_with_input_zeros(&mut segment.graph, search_budget, &input_zeros)?;
    let wrapped = DynRuntimeWrapper::new(runtime, segment.graph, name_to_node);
    Ok(RuntimeSegment {
        runtime: Box::new(wrapped),
        input_names,
        output_names,
        capture_names,
        weight_names,
    })
}

fn load_weights_into_segments(
    path: &Path,
    segments: &mut [RuntimeSegment],
) -> Result<(), CompileError> {
    let bytes = std::fs::read(path).map_err(|source| CompileError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|source| CompileError::Safetensors {
        path: path.to_path_buf(),
        source,
    })?;

    for segment in segments {
        for (name, shape) in &segment.weight_names {
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
        dtype => Err(CompileError::UnsupportedWeightDtype {
            tensor: tensor.to_string(),
            dtype,
        }),
    }
}

fn parse_hidden_after_block(name: &str) -> Option<usize> {
    name.strip_prefix("hidden_after_block_")?.parse().ok()
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

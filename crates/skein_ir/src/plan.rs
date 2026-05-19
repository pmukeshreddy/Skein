//! `Plan` — the output of `skein_extract`, the input of `skein_emit`.
//!
//! A `Plan` is the joint choice across the 7 axes of search: parallelism
//! placement, KV layout, batching, per-component dtype, CUDA Graphs, spec
//! decode, and prefix cache. Plus an optional P/D disaggregation pair.
//!
//! The `content_hash` is deterministic — serialized as canonical JSON (sorted
//! keys, no trailing whitespace) and BLAKE3-hashed. Artifact directories use
//! this hash; an identical Plan emitted twice resolves to the same artifact.

use serde::{Deserialize, Serialize};

use crate::error::PlanError;
use crate::ir::ModelMeta;
use crate::types::{BatchPolicy, Dtype, ExecutionConfig, KVCacheSpec};

/// The `(tp, pp, ep)` placement triple. Constrained by
/// `tp * pp * ep ≤ num_devices` and per-dim divisibility checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParallelismPlacement {
    pub tp: u32,
    pub pp: u32,
    pub ep: u32,
}

impl ParallelismPlacement {
    pub const fn devices_used(self) -> u32 {
        self.tp * self.pp * self.ep
    }
}

/// One entry per decoder block — the inner DP in `skein_extract` picks
/// `(weight, activation, kv_cache)` for each block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PerLayerDtype {
    pub weight: Dtype,
    pub activation: Dtype,
    pub kv_cache: Dtype,
}

impl PerLayerDtype {
    pub const fn uniform(d: Dtype) -> Self {
        Self {
            weight: d,
            activation: d,
            kv_cache: d,
        }
    }
}

/// The per-layer dtype assignment. `per_layer.len()` must equal the number
/// of decoder blocks in the IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DtypeMap {
    pub per_layer: Vec<PerLayerDtype>,
}

impl DtypeMap {
    /// Construct a uniform map of the requested length.
    pub fn uniform(num_blocks: usize, d: Dtype) -> Self {
        Self {
            per_layer: vec![PerLayerDtype::uniform(d); num_blocks],
        }
    }

    pub fn len(&self) -> usize {
        self.per_layer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.per_layer.is_empty()
    }
}

/// KV cache transfer mode for P/D disaggregation. `RdmaWrite` is one-sided
/// RDMA from the prefill pool's NIC to the decode pool's GPU memory;
/// `NcclByLayer` overlaps with the decode pool's first few layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvTransferMode {
    RdmaWrite,
    NcclByLayer,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransferTopology {
    pub mode: KvTransferMode,
    /// When `true`, the decode pool starts computing layer 0 while later
    /// layers' KV is still in flight from the prefill pool.
    pub layer_overlap: bool,
}

/// P/D disaggregation: separate plans for the prefill pool and the decode
/// pool, plus the KV-cache handoff topology between them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Disaggregation {
    pub prefill: Box<Plan>,
    pub decode: Box<Plan>,
    pub transfer: TransferTopology,
}

/// The complete plan. Round-trip via JSON is canonical (sorted keys).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub parallelism: ParallelismPlacement,
    pub kv: KVCacheSpec,
    pub batching: BatchPolicy,
    pub dtype_map: DtypeMap,
    pub execution: ExecutionConfig,
    /// `Some` only when the search was invoked with `--disaggregated`.
    pub disaggregation: Option<Disaggregation>,
    pub model_meta: ModelMeta,
}

impl Plan {
    pub fn num_layers(&self) -> usize {
        self.model_meta.num_layers
    }

    /// Validate that the dtype map covers every decoder block. The number of
    /// decoder blocks lives in `ModelMeta::num_layers` (which counts the
    /// transformer body, not the embedding / final norm / lm_head).
    pub fn validate(&self) -> Result<(), PlanError> {
        if self.dtype_map.len() != self.model_meta.num_layers {
            return Err(PlanError::DtypeMapLengthMismatch {
                expected: self.model_meta.num_layers,
                actual: self.dtype_map.len(),
            });
        }
        if let Some(d) = &self.disaggregation {
            d.prefill.validate()?;
            d.decode.validate()?;
        }
        Ok(())
    }

    /// Canonical content hash used for artifact directories. Two `Plan`
    /// values that compare equal hash equal; flipping any axis changes the
    /// hash.
    pub fn content_hash(&self) -> Result<blake3::Hash, PlanError> {
        // `serde_json` does not sort map keys by default. Plan has no maps —
        // all fields are flat structs or sequences — so plain serialization
        // is already canonical. If a future field adds a HashMap, this needs
        // to switch to a sorted serializer; today, it does not.
        let bytes = serde_json::to_vec(self)?;
        Ok(blake3::hash(&bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn dummy_meta(num_layers: usize) -> ModelMeta {
        ModelMeta {
            architecture: "test".into(),
            num_layers,
            hidden: 4096,
            vocab: 32000,
            max_position: 32768,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            num_experts: Some(8),
            top_k: Some(2),
            intermediate: 14336,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            tied_embeddings: false,
        }
    }

    fn dummy_plan(num_layers: usize) -> Plan {
        Plan {
            parallelism: ParallelismPlacement {
                tp: 2,
                pp: 1,
                ep: 1,
            },
            kv: KVCacheSpec {
                layout: KVLayout::Paged { page_size: 32 },
                kv_sharded: false,
            },
            batching: BatchPolicy::Continuous { max_batch: 16 },
            dtype_map: DtypeMap::uniform(num_layers, Dtype::Bf16),
            execution: ExecutionConfig {
                cuda_graphs: CudaGraphsConfig {
                    enable: false,
                    capture_classes: vec![],
                },
                spec_decode: SpecDecodeConfig {
                    enable: false,
                    draft: None,
                },
                prefix_cache: PrefixCacheConfig {
                    enable: true,
                    reuse_policy: RadixReusePolicy::LruByLastAccess,
                },
            },
            disaggregation: None,
            model_meta: dummy_meta(num_layers),
        }
    }

    #[test]
    fn placement_devices_used() {
        let p = ParallelismPlacement {
            tp: 2,
            pp: 4,
            ep: 1,
        };
        assert_eq!(p.devices_used(), 8);
    }

    #[test]
    fn validate_catches_dtype_length_mismatch() {
        let mut plan = dummy_plan(32);
        plan.dtype_map = DtypeMap::uniform(31, Dtype::Bf16); // off by one
        let err = plan.validate().unwrap_err();
        match err {
            PlanError::DtypeMapLengthMismatch { expected, actual } => {
                assert_eq!(expected, 32);
                assert_eq!(actual, 31);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn content_hash_is_stable_across_clones() {
        let plan = dummy_plan(32);
        let h1 = plan.content_hash().unwrap();
        let h2 = plan.clone().content_hash().unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn content_hash_changes_when_any_axis_changes() {
        let plan = dummy_plan(32);
        let mut other = plan.clone();
        other.parallelism.tp = 4;
        assert_ne!(plan.content_hash().unwrap(), other.content_hash().unwrap());

        let mut other = plan.clone();
        other.batching = BatchPolicy::Static { max_batch: 16 };
        assert_ne!(plan.content_hash().unwrap(), other.content_hash().unwrap());

        let mut other = plan.clone();
        other.dtype_map.per_layer[0].kv_cache = Dtype::Fp8E4m3;
        assert_ne!(plan.content_hash().unwrap(), other.content_hash().unwrap());
    }
}

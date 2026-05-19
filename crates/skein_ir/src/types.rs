//! Value types shared between the IR and the `Plan`.
//!
//! - `Dtype` / `Component` — the per-component quantization axis of search.
//! - `Shape` / `Dim` — symbolic-or-fixed tensor shape used by the IR. Runtime
//!   dimensions (batch, sequence, KV length) stay symbolic until the runtime
//!   binds them on each step.
//! - `Sharding` / `ShardRole` — produced by `skein_emit` from the chosen
//!   `Plan`. Not a free axis of search — derived from `ParallelismPlacement`.
//! - `KVCacheSpec`, `BatchPolicy`, `ExecutionConfig` — the non-precision axes
//!   of `Plan`.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Dtype + Component
// ---------------------------------------------------------------------------

/// A numeric dtype that can appear on weights, activations, or KV cache.
///
/// Bit-widths are exact: `Int4` is genuinely 4 bits per element (packed), not
/// 8-bit storage of a 4-bit value. Storage layout decisions live in
/// `skein_emit`, but the bit count drives memory accounting in `skein_cost`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    Bf16,
    Fp16,
    Fp8E4m3,
    Fp8E5m2,
    Int8,
    Int4,
}

impl Dtype {
    /// Bits per element. Used by `skein_cost` for memory pressure and by
    /// `skein_emit` for storage layout. `Int4` is *packed* — two values per
    /// byte — so this returns 4, not 8.
    pub const fn bits(self) -> u32 {
        match self {
            Dtype::Bf16 | Dtype::Fp16 => 16,
            Dtype::Fp8E4m3 | Dtype::Fp8E5m2 | Dtype::Int8 => 8,
            Dtype::Int4 => 4,
        }
    }

    /// Bytes needed to store `n_elements` values of this dtype, rounded up.
    /// The round-up matters for `Int4`: an odd count still needs the trailing
    /// half-byte.
    pub const fn bytes_for(self, n_elements: u64) -> u64 {
        let bits = self.bits() as u64 * n_elements;
        bits.div_ceil(8)
    }

    /// Whether this dtype is a float (vs. integer quantization).
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            Dtype::Bf16 | Dtype::Fp16 | Dtype::Fp8E4m3 | Dtype::Fp8E5m2
        )
    }

    /// All dtypes Skein considers, in a stable iteration order. Used by the
    /// inner DP in `skein_extract` and the drift table in `skein_calibrate`.
    pub const ALL: [Dtype; 6] = [
        Dtype::Bf16,
        Dtype::Fp16,
        Dtype::Fp8E4m3,
        Dtype::Fp8E5m2,
        Dtype::Int8,
        Dtype::Int4,
    ];
}

/// The three tensor components that can be independently quantized. Drift
/// is calibrated per `(layer_idx, component, dtype)` triple.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    Weight,
    Activation,
    KvCache,
}

impl Component {
    pub const ALL: [Component; 3] = [Component::Weight, Component::Activation, Component::KvCache];
}

// ---------------------------------------------------------------------------
// Shape + Dim
// ---------------------------------------------------------------------------

/// A symbolic-or-fixed dimension. Runtime dimensions (batch / seq / kv_len)
/// stay symbolic in the IR; `skein_emit` binds them when constructing the
/// `luminal::Graph` for each device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Dim {
    Fixed(usize),
    Batch,
    Seq,
    KvLen,
    /// Tensor-parallel-sharded fixed dim. The IR carries the un-sharded value
    /// and `skein_emit` divides by the TP group size when lowering.
    Tp(usize),
    /// Expert-parallel-sharded fixed dim (e.g. `num_experts`). `skein_emit`
    /// divides by the EP group size when lowering.
    Ep(usize),
}

/// A tensor shape. Order matters; index 0 is the leading dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape(pub Vec<Dim>);

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Number of elements when only `Fixed` dims contribute. Useful for
    /// counting parameters at IR-build time, where `Batch`/`Seq`/`KvLen` are
    /// not yet bound. Returns `None` if any dim is symbolic.
    pub fn static_numel(&self) -> Option<u64> {
        let mut n: u64 = 1;
        for d in &self.0 {
            match d {
                Dim::Fixed(v) | Dim::Tp(v) | Dim::Ep(v) => n = n.saturating_mul(*v as u64),
                Dim::Batch | Dim::Seq | Dim::KvLen => return None,
            }
        }
        Some(n)
    }
}

// ---------------------------------------------------------------------------
// Sharding + ShardRole
// ---------------------------------------------------------------------------

/// How a weight tensor is split across devices in the chosen `Plan`.
///
/// This is a *derived* property — `skein_emit` computes it from the Plan's
/// `(tp, pp, ep)` placement. It is not an independent search axis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Sharding {
    /// Tensor is fully replicated on every device in the relevant group.
    Replicated,
    /// Weight rows are split across `group_size` devices along TP axis.
    /// Used for column-parallel inputs in attention/MLP projections.
    TpRowParallel { group_size: u32 },
    /// Weight columns are split across `group_size` devices.
    /// Used for row-parallel outputs that AllReduce.
    TpColParallel { group_size: u32 },
    /// MoE experts are split across `group_size` devices; each device owns a
    /// disjoint subset of expert indices.
    ExpertParallel { group_size: u32, num_experts: u32 },
}

/// The role a particular device plays for a particular parameter, after the
/// `Plan` has been lowered. Consumed by `skein_emit` when deciding which
/// safetensors slices to load on each rank.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ShardRole {
    Replicated,
    TpOutputShard { group_size: u32, group_index: u32 },
    ExpertOwned { expert_idx: u32 },
    ExpertElsewhere { expert_idx: u32 },
    NoParams,
}

// ---------------------------------------------------------------------------
// KV cache layout
// ---------------------------------------------------------------------------

/// Layout of the KV cache. `Paged(N)` is vLLM-style paged attention with
/// page size `N` tokens; `Contiguous` is the classic per-request block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KVLayout {
    Contiguous,
    Paged { page_size: u32 },
}

/// KV cache configuration in a `Plan`. `kv_sharded = true` distributes the
/// cache across the TP group rather than replicating it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KVCacheSpec {
    pub layout: KVLayout,
    pub kv_sharded: bool,
}

// ---------------------------------------------------------------------------
// Batching policy
// ---------------------------------------------------------------------------

/// Batching strategy. `M` is the max in-flight batch size; `C` is the
/// chunked-prefill token chunk size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BatchPolicy {
    Static { max_batch: u32 },
    Continuous { max_batch: u32 },
    ContinuousChunked { max_batch: u32, chunk_tokens: u32 },
}

impl BatchPolicy {
    pub const fn max_batch(self) -> u32 {
        match self {
            BatchPolicy::Static { max_batch }
            | BatchPolicy::Continuous { max_batch }
            | BatchPolicy::ContinuousChunked { max_batch, .. } => max_batch,
        }
    }
}

// ---------------------------------------------------------------------------
// Execution config (CUDA Graphs, spec decode, prefix cache)
// ---------------------------------------------------------------------------

/// A single `(batch_size, kv_class)` pair to capture as a CUDA Graph at
/// warmup. `kv_class` is the discretized KV-length bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CaptureClass {
    pub batch_size: u32,
    pub kv_class: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CudaGraphsConfig {
    pub enable: bool,
    pub capture_classes: Vec<CaptureClass>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DraftSpec {
    /// Path to the draft model's safetensors directory.
    pub model_path: String,
    pub speculation_depth: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpecDecodeConfig {
    pub enable: bool,
    pub draft: Option<DraftSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RadixReusePolicy {
    LruByLastAccess,
    LfuByHits,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixCacheConfig {
    pub enable: bool,
    pub reuse_policy: RadixReusePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub cuda_graphs: CudaGraphsConfig,
    pub spec_decode: SpecDecodeConfig,
    pub prefix_cache: PrefixCacheConfig,
}

// ---------------------------------------------------------------------------
// Tests — value-level invariants only. Round-trip property tests live in
// `tests/round_trip.rs` so they can exercise the full crate at once.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_bytes_int4_packs() {
        // 1 elem -> 4 bits -> 1 byte (round up).
        assert_eq!(Dtype::Int4.bytes_for(1), 1);
        // 2 elems -> 8 bits -> 1 byte (exact).
        assert_eq!(Dtype::Int4.bytes_for(2), 1);
        // 3 elems -> 12 bits -> 2 bytes (round up).
        assert_eq!(Dtype::Int4.bytes_for(3), 2);
        // Large: 1024 elems -> 4096 bits -> 512 bytes.
        assert_eq!(Dtype::Int4.bytes_for(1024), 512);
    }

    #[test]
    fn dtype_bytes_bf16() {
        assert_eq!(Dtype::Bf16.bytes_for(1), 2);
        assert_eq!(Dtype::Bf16.bytes_for(1024), 2048);
    }

    #[test]
    fn shape_static_numel_includes_tp_and_ep() {
        // TP and EP are *unsharded* values in the IR.
        let s = Shape(vec![Dim::Fixed(4), Dim::Tp(4096), Dim::Ep(8)]);
        assert_eq!(s.static_numel(), Some(4 * 4096 * 8));
    }

    #[test]
    fn shape_static_numel_none_when_symbolic() {
        let s = Shape(vec![Dim::Batch, Dim::Fixed(4096)]);
        assert!(s.static_numel().is_none());
    }

    #[test]
    fn batch_policy_max_batch_extract() {
        assert_eq!(BatchPolicy::Static { max_batch: 32 }.max_batch(), 32);
        assert_eq!(
            BatchPolicy::ContinuousChunked {
                max_batch: 16,
                chunk_tokens: 1024
            }
            .max_batch(),
            16
        );
    }
}

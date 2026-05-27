//! Real [`LocalSegments`] adapter over a device's compiled segments.
//!
//! [`SegmentRunner`] wraps the `Vec<RuntimeSegment>` that
//! `skein_compile::load_runtime_segments` produces for one device and drives
//! them for the [`RankExecutor`](super::rank_executor::RankExecutor): it owns
//! the rank-local handoff store (logical-name → buffer), feeds a segment its
//! named inputs, executes it, and captures its named outputs. The collective
//! results the rank executor writes back land in the same store and are picked
//! up as inputs by later segments.
//!
//! This is backend-agnostic: the `RuntimeSegment`s carry a `dyn DynRuntime`
//! that is either the CPU `NativeComputeRuntime` or the GPU
//! `CudaComputeRuntime`. The adapter therefore compiles and is unit-tested on
//! the CPU build (with a mock `DynRuntime`); on the GPU host the identical code
//! drives the real CUDA segments.

use std::collections::{HashMap, HashSet};

use skein_compile::{HandoffId, RuntimeSegment};
use skein_emit::segment::SequenceStep;

use super::rank_executor::{LocalSegments, RankExecError, ResolvedSequenceStep};
use crate::error::RuntimeError;
use crate::kv::PagedKvCache;
use crate::kv_cache::{KvKind, parse_kvcache_name};
use crate::types::RequestId;

/// Element dtype of a device-resident handoff (segment activations are bf16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffDtype {
    Bf16,
}

/// A borrowed device-resident tensor: a producer segment's output buffer handed
/// to the next consuming segment **by device pointer** (no host round-trip).
///
/// Ownership/lifetime: the pointer is owned by the producer segment's Luminal
/// runtime arena. It is valid within one forward pass (until the producer
/// re-executes), and it is re-fetched + re-bound every decode step — so it never
/// dangles across steps. Same-rank only (one physical GPU per rank process).
#[derive(Clone, Copy, Debug)]
pub struct DeviceTensorHandle {
    /// Process-local device ordinal that owns the buffer (CUDA_VISIBLE_DEVICES
    /// maps device 0 to this rank's physical GPU).
    pub device: usize,
    /// Raw CUDA device pointer (borrowed from the producer's arena).
    pub ptr: u64,
    /// Byte size of the buffer.
    pub n_bytes: usize,
    /// Element count (`n_bytes / 2` for bf16).
    pub elems: usize,
    /// Element dtype.
    pub dtype: HandoffDtype,
}

/// Default runtime KV paging geometry for the serve path. `page_size` tokens
/// per page; `total_pages` is the per-device page budget (16 * 256 = 4096 token
/// slots, comfortably above one request's `KV_CACHE_CAP` while leaving room for
/// cached prefixes); the radix depth bounds prefix-match length.
const DEFAULT_PAGE_SIZE: u32 = 16;
const DEFAULT_TOTAL_PAGES: u32 = 256;
const DEFAULT_RADIX_DEPTH: u32 = 4096;

/// Bytes per element of the device-resident decode KV cache. The cached-decode
/// attention (`attention_fixed_cache`) runs RoPE/scores/softmax/Attn·V in **f32**
/// and writes the new token's K/V back **f32** (only the final attention output
/// is cast to the activation dtype), so the `kvcache_*` buffer the runtime sizes,
/// strides, and DtoD-appends into is f32 — 4 bytes/element, NOT the activation
/// dtype (bf16). Sizing it as bf16 (×2) under-allocates the buffer to half and,
/// in the batched write, halves the per-row slot stride `cap` — corrupting every
/// row but row 0. (Single-request decode only stayed correct because short
/// prompts never read past the first half of the under-sized buffer.)
const KV_DEVICE_ELEM_BYTES: usize = 4;

/// Parse a `u32` from the environment, falling back to `default`.
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

/// Classification of a handoff tensor, fixed once at construction from its
/// logical name. Decides how [`SegmentRunner::run_segment`] feeds it as an input
/// and captures it as an output — by index, with no per-step string matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandoffKind {
    /// Host-staged f32 scalar/small tensor (`position`, MoE gate scalars).
    F32,
    /// Integer input (`input_tokens`).
    I32,
    /// Device-resident segment→segment activation: bound/captured by device
    /// pointer with no host round-trip (host f32 fallback on the CPU backend).
    Device,
    /// Per-layer KV cache input/output (`kvcache_{k|v}_{layer}`).
    KvCache { kind: KvKind, layer: u16 },
    /// Stays on the host: all-gather/broadcast collective results and the
    /// sparse-MoE router logits (read host-side in the MoeRoute step).
    HostCollective,
}

/// One device buffer holding **all** MoE gate scalars for this rank, bf16 to
/// match the FFN gate inputs' graph dtype. Layout: block `b`, slot `s` lives at
/// element `b * top_k + s`. Each FFN segment's gate inputs are bound once
/// (persistently) to fixed offsets here at bootstrap; `route_moe_resolved` writes
/// a block's `top_k` gates with one async H2D per layer (no per-slot upload).
#[cfg(feature = "cuda")]
struct MoeGateBuffer {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    buf: cudarc::driver::CudaSlice<half::bf16>,
    top_k: usize,
}

/// Drives one device's compiled segments + the rank-local handoff store.
///
/// The handoff store is **id-indexed**: every segment input/output logical name
/// is interned to a [`HandoffId`] once at construction, and the decode hot path
/// (`run_segment`) dispatches through `Vec`-indexed slots with no `HashMap`
/// lookups or `String` clones. `name_to_id` survives only for the rare by-name
/// entry points (`read`/`write`/`route_moe_resolved`/`begin_request`).
pub struct SegmentRunner {
    segments: Vec<RuntimeSegment>,
    /// Logical name → interned id. Used at construction and by the by-name entry
    /// points (collective read/write, route_moe, begin_request); never on the
    /// per-segment input/output loop.
    name_to_id: HashMap<String, HandoffId>,
    /// id → logical name (for error messages, KV size lookup, dynamic interning).
    id_to_name: Vec<String>,
    /// id → classification, decides input-feed and output-capture dispatch.
    kind: Vec<HandoffKind>,
    /// id → f32 handoff value (collective results, KV/host fallbacks, scalars).
    f32_slots: Vec<Option<Vec<f32>>>,
    /// id → integer handoff value (e.g. `input_tokens`).
    i32_slots: Vec<Option<Vec<i32>>>,
    /// id → device-resident segment→segment activation handle (bound into the
    /// consumer by device pointer; refreshed each forward by the producer).
    device_slots: Vec<Option<DeviceTensorHandle>>,
    /// id → base device pointer of a KV-cache input's persistent on-GPU buffer.
    /// The cache lives on the GPU across decode steps; the new token's K/V is
    /// written into slot `position` via a DtoD copy, never re-uploaded from host.
    kv_device_base: Vec<Option<u64>>,
    /// id → full element count of a `kvcache_*` input's fixed-capacity buffer
    /// (from the producing segment's `kv_cache_sizes`), precomputed so the hot
    /// path needs no name lookup.
    kv_full_elems: Vec<Option<usize>>,
    /// Per segment, its input handoff ids (pre-resolved from `input_names`).
    segment_inputs: Vec<Vec<HandoffId>>,
    /// Per segment, its output handoff ids (pre-resolved from `output_names`).
    segment_outputs: Vec<Vec<HandoffId>>,
    /// Tensor names that must stay host: the all-gather (logits) / broadcast
    /// collectives, whose host `read`/`write` path is kept. RingAllReduce
    /// tensors are NOT here — they stay device-resident and are all-reduced in
    /// place on the device. (KV is handled separately and also stays host.)
    host_tensors: HashSet<String>,
    /// Paged per-layer KV cache. Handoffs named `kvcache_{k|v}_{layer}` are fed
    /// from the active request's pages (cached prefix + freshly written tokens)
    /// and each step's output is written into the page covering `position` — so
    /// the model attends over all prior tokens (cached decode) and cross-request
    /// prefixes are reused, while each segment stays fixed-shape per step.
    kv: PagedKvCache,
    /// Current decode position = number of tokens already in the cache = the
    /// dynamic `past` length the attention segments read. Set per step via
    /// [`set_position`](Self::set_position).
    position: usize,
    /// When true (batched-prefill graph), `kvcache_*` outputs are NOT written
    /// into this runner's cache at `position`; instead they are kept in
    /// `handoffs` (shape [seq, kv_dim]) so the caller can write the N tokens'
    /// K/V into the decode runner's cache. See `RankServer::forward_prefill`.
    prefill_capture: bool,
    /// When true (SKEIN_CAPTURE active), `run_segment` skips the two per-token
    /// host→device scalar updates — the `input_tokens` feed and the
    /// `set_decode_position` memcpy — so they are NOT recorded into the captured
    /// full-step graph (a recorded host→device copy would bake in a now-freed
    /// host source pointer and replay garbage). Instead they are written into
    /// their persistent device buffers every step, outside the capture window, by
    /// [`flush_step_device_inputs`](Self::flush_step_device_inputs).
    capturing: bool,
    /// MB CUDA-graph capture: the specific HostCollective OUTPUT ids (boundary
    /// carry, final logits) to keep DEVICE-RESIDENT under capture instead of
    /// D2H-ing into the captured graph. Targeted on purpose — other HostCollective
    /// outputs (e.g. MoE router logits read host-side for top-k) must NOT be
    /// diverted. The MB loop D2H-reads these outside the captured region.
    capture_passthrough_outputs: HashSet<HandoffId>,
    /// Pre-resolved id of the `input_tokens` i32 input, so `run_segment` (skip
    /// path) and `flush_step_device_inputs` (write path) agree on which input is
    /// the per-token token id.
    input_tokens_id: Option<HandoffId>,
    /// Pre-resolved id of the `position` f32 input (the absolute decode position
    /// the attention segments use for RoPE and to derive the KV length /
    /// FlashInfer indptr). Same skip/flush treatment as `input_tokens` so it is
    /// not frozen inside the captured full-step graph.
    position_id: Option<HandoffId>,
    /// Resident bf16 buffer for all MoE gate scalars (set at bootstrap by
    /// [`SegmentRunner::set_gate_buffer`]). `None` until set / on the CPU build.
    #[cfg(feature = "cuda")]
    gate_buffer: Option<MoeGateBuffer>,
    /// Continuous-batching per-request device KV. The default on-device KV path
    /// binds ONE persistent KV buffer per layer (correct for a single in-flight
    /// request), which concurrent requests would clobber. When `paged_device_kv`
    /// is on, each in-flight request gets its OWN contiguous device KV buffer per
    /// `kvcache_*` slot — bound on activation so reads/writes stay isolated. The
    /// active request is tracked so the KV input feed selects the right buffer.
    paged_device_kv: bool,
    /// The request whose pages/buffers are currently active (set by
    /// [`activate_request`](Self::activate_request)); selects which per-request
    /// KV buffer the next `run_segment` binds under `paged_device_kv`.
    active_request: Option<RequestId>,
    /// Per-request KV device buffers: `request -> (slot -> base ptr)`. Allocated
    /// lazily (zeroed) on a request's first activation and reused across its
    /// decode steps. (Benchmark scope: freed in bulk at driver teardown.)
    kv_req_bufs: HashMap<RequestId, Vec<Option<u64>>>,
    /// Decode batch width for synchronous batched decode (>1 = pack N sequences
    /// as rows of the `[N, cap, kv_dim]` KV cache, all at the shared `position`).
    /// When >1 the `kvcache_*` outputs are written with the batched (per-row,
    /// strided) KV append. `1` = single-sequence decode (default).
    decode_batch: usize,
}

/// Classify a handoff by its logical name. Mirrors the dispatch the previous
/// string-keyed `run_segment` performed: `kvcache_*` route to the paged cache,
/// `input_tokens` is the i32 input, collective/router tensors stay host, the
/// `position` scalar and MoE gate scalars are host-staged f32, everything else
/// (internal activations, weight slots) is device-resident.
fn classify_handoff(name: &str, host_tensors: &HashSet<String>) -> HandoffKind {
    if let Some((kind, layer)) = parse_kvcache_name(name) {
        HandoffKind::KvCache {
            kind,
            layer: layer as u16,
        }
    } else if name == "input_tokens" {
        HandoffKind::I32
    } else if host_tensors.contains(name) || name.starts_with("router_logits") {
        // router_logits is read host-side in the MoeRoute step; the all-gather /
        // broadcast collective tensors keep their host read/write path.
        HandoffKind::HostCollective
    } else if name == "position" || name.starts_with("moe_gate") {
        // Host-staged scalars: `position` (set per step) and the MoE gate scalars
        // (written by route_moe directly onto the FFN runtime).
        HandoffKind::F32
    } else {
        HandoffKind::Device
    }
}

/// Pick the top-`k` logits (descending) and softmax over just them. Returns the
/// selected expert indices and their gate weights (parallel arrays). This equals
/// the dense `top_k_route` weighting, which masks non-top-k logits to ~0 before
/// softmax — so the selected experts get softmax-over-top-k, identical output.
fn top_k_softmax(logits: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let n = logits.len();
    let k = k.min(n);
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let topk: Vec<usize> = order[..k].to_vec();
    let maxl = topk
        .iter()
        .map(|&i| logits[i])
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = topk.iter().map(|&i| (logits[i] - maxl).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(f32::MIN_POSITIVE);
    let gates: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    (topk, gates)
}

impl SegmentRunner {
    pub fn new(segments: Vec<RuntimeSegment>) -> Self {
        // Page geometry is overridable from the environment so a run can pick a
        // smaller page (finer-grained prefix reuse) without a recompile.
        let page_size = env_u32("SKEIN_KV_PAGE_SIZE", DEFAULT_PAGE_SIZE).max(1);
        let total_pages = env_u32("SKEIN_KV_TOTAL_PAGES", DEFAULT_TOTAL_PAGES).max(1);
        let prefix_enable = std::env::var_os("SKEIN_PREFIX_CACHE_OFF").is_none();
        Self::with_kv(
            segments,
            page_size,
            total_pages,
            prefix_enable,
            DEFAULT_RADIX_DEPTH,
        )
    }

    /// Construct with an explicit KV paging geometry (page size, page budget,
    /// prefix-cache on/off, radix depth). The serve bootstrap uses this to size
    /// pages from the device KV budget.
    pub fn with_kv(
        mut segments: Vec<RuntimeSegment>,
        page_size: u32,
        total_pages: u32,
        prefix_enable: bool,
        radix_max_depth: u32,
    ) -> Self {
        let kv = PagedKvCache::new(page_size, total_pages, prefix_enable, radix_max_depth, 0)
            .expect("valid KV paging geometry");

        // Intern every segment's input/output logical names once, in a
        // deterministic order, so the hot path dispatches by `Vec` index.
        let mut name_to_id: HashMap<String, HandoffId> = HashMap::new();
        let mut id_to_name: Vec<String> = Vec::new();
        let mut intern = |name: &str| -> HandoffId {
            if let Some(id) = name_to_id.get(name) {
                *id
            } else {
                let id = HandoffId(id_to_name.len() as u32);
                id_to_name.push(name.to_string());
                name_to_id.insert(name.to_string(), id);
                id
            }
        };
        let mut segment_inputs: Vec<Vec<HandoffId>> = Vec::with_capacity(segments.len());
        let mut segment_outputs: Vec<Vec<HandoffId>> = Vec::with_capacity(segments.len());
        for seg in &segments {
            segment_inputs.push(seg.input_names.iter().map(|n| intern(n)).collect());
            segment_outputs.push(seg.output_names.iter().map(|n| intern(n)).collect());
        }

        // Precompute each KV input's fixed buffer element count, by id (so the
        // hot path needs no `kv_cache_sizes` name lookup).
        let n = id_to_name.len();
        let mut kv_full_elems: Vec<Option<usize>> = vec![None; n];
        for seg in &segments {
            for (name, &full) in &seg.kv_cache_sizes {
                if let Some(id) = name_to_id.get(name) {
                    kv_full_elems[id.idx()] = Some(full);
                }
            }
        }

        // Build each segment's `HandoffId -> NodeIndex` table once. After this no
        // string lookups occur on the per-segment hot path.
        for seg in &mut segments {
            seg.runtime
                .register_handoff_ids(&|name| name_to_id.get(name).copied());
        }

        // Classify each handoff. `host_tensors` is empty here and supplied later
        // via `set_host_tensors`, which re-runs the classification.
        let host_tensors = HashSet::new();
        let kind: Vec<HandoffKind> = id_to_name
            .iter()
            .map(|name| classify_handoff(name, &host_tensors))
            .collect();

        Self {
            segments,
            name_to_id,
            id_to_name,
            kind,
            f32_slots: vec![None; n],
            i32_slots: vec![None; n],
            device_slots: vec![None; n],
            kv_device_base: vec![None; n],
            kv_full_elems,
            segment_inputs,
            segment_outputs,
            host_tensors,
            kv,
            position: 0,
            prefill_capture: false,
            capturing: false,
            capture_passthrough_outputs: HashSet::new(),
            input_tokens_id: None,
            position_id: None,
            #[cfg(feature = "cuda")]
            gate_buffer: None,
            paged_device_kv: false,
            active_request: None,
            kv_req_bufs: HashMap::new(),
            decode_batch: 1,
        }
    }

    /// Enable per-request device KV buffers (continuous-batch driver). Off by
    /// default so the single-request path keeps its single shared KV buffer.
    pub fn set_paged_device_kv(&mut self, on: bool) {
        self.paged_device_kv = on;
    }

    /// Set the synchronous batched decode width (rows packed per forward). >1
    /// routes `kvcache_*` outputs through the batched per-row KV append.
    pub fn set_decode_batch(&mut self, n: usize) {
        self.decode_batch = n.max(1);
    }

    /// Get (or lazily allocate, zeroed) request `req`'s own contiguous device KV
    /// buffer for KV-slot `slot` (`n_bytes` capacity). Allocated on the consuming
    /// segment's runtime (this runner's GPU) and reused across the request's
    /// decode steps. `None` if the backend has no device pointers (CPU).
    fn req_kv_buffer(
        &mut self,
        segment_idx: usize,
        req: RequestId,
        slot: usize,
        n_bytes: usize,
    ) -> Option<u64> {
        if let Some(bufs) = self.kv_req_bufs.get(&req)
            && let Some(Some(p)) = bufs.get(slot)
        {
            return Some(*p);
        }
        let ptr = self.segments[segment_idx].runtime.alloc_device_zeros(n_bytes);
        if ptr == 0 {
            return None;
        }
        let n = self.kind.len();
        let bufs = self.kv_req_bufs.entry(req).or_insert_with(|| vec![None; n]);
        if slot < bufs.len() {
            bufs[slot] = Some(ptr);
        }
        Some(ptr)
    }

    /// Resolve a logical name to its interned id, interning it on demand. Used by
    /// the by-name entry points (`write`, `set_position`, …) so a name not seen
    /// at construction (e.g. an external collective write) still gets a slot.
    /// Dynamically-interned ids belong to no segment's node table, so they are
    /// store-only — never dispatched into a runtime (segment id lists are fixed).
    fn intern_name(&mut self, name: &str) -> HandoffId {
        if let Some(id) = self.name_to_id.get(name) {
            return *id;
        }
        let id = HandoffId(self.id_to_name.len() as u32);
        self.id_to_name.push(name.to_string());
        self.name_to_id.insert(name.to_string(), id);
        self.kind.push(classify_handoff(name, &self.host_tensors));
        self.f32_slots.push(None);
        self.i32_slots.push(None);
        self.device_slots.push(None);
        self.kv_device_base.push(None);
        self.kv_full_elems.push(None);
        id
    }

    /// Tell the runner which tensor names must stay host (all-gather/broadcast
    /// collectives). Everything else — internal activations AND RingAllReduce
    /// tensors — stays device-resident; RingAllReduce is all-reduced in place on
    /// the device by the rank executor.
    pub fn set_host_tensors(&mut self, names: HashSet<String>) {
        self.host_tensors = names;
        // Re-classify now that host tensors are known (construction ran with an
        // empty set). Only `HostCollective` membership depends on this set.
        for (id, name) in self.id_to_name.iter().enumerate() {
            self.kind[id] = classify_handoff(name, &self.host_tensors);
        }
    }

    /// Device handle for a named output, if it was produced device-resident this
    /// forward (the brief's `output_device(name)`).
    pub fn output_device(&self, name: &str) -> Option<DeviceTensorHandle> {
        let id = self.name_to_id.get(name)?;
        self.device_slots[id.idx()]
    }

    /// Enable/disable batched-prefill capture: when on, `kvcache_*` outputs are
    /// kept in the handoff store (whole [seq, kv_dim] tensor) instead of being
    /// written one slot at a time into this runner's cache.
    pub fn set_prefill_capture(&mut self, on: bool) {
        self.prefill_capture = on;
    }

    /// Read a captured handoff/output tensor by name (e.g. a prefill
    /// `kvcache_*` output, or `logits`). None if not produced this run.
    pub fn read_handoff(&self, name: &str) -> Option<Vec<f32>> {
        let id = self.name_to_id.get(name)?;
        self.f32_slots[id.idx()].clone()
    }

    /// Write one token's K or V into this runner's paged cache at `slot` for
    /// `layer`. Used to land batched-prefill K/V (computed by the prefill graph)
    /// into the decode runner's cache before decoding continues.
    pub fn write_kv_slot(
        &mut self,
        kind: crate::kv_cache::KvKind,
        layer: usize,
        slot: usize,
        data: &[f32],
    ) {
        self.kv.write_slot(kind, layer, slot, data);
    }

    /// Upload all segments' weights now, as persistent GPU buffers (instead of
    /// lazily on first execute). The serve calls this on the decode runner so a
    /// sibling prefill graph can share the resident weights by device pointer.
    pub fn materialize_weights(&mut self) {
        for seg in &mut self.segments {
            seg.runtime.materialize_weights();
        }
    }

    /// The compiled segments (e.g. so a prefill loader can read resident weight
    /// device pointers to share, rather than loading a second copy).
    pub fn segments(&self) -> &[skein_compile::RuntimeSegment] {
        &self.segments
    }

    /// Free every segment's intermediate-buffer arena (re-allocated lazily on
    /// next execute). Frees the search/compile arenas so two graphs' arenas
    /// (decode + prefill) aren't resident simultaneously.
    pub fn clear_intermediates(&mut self) {
        for seg in &mut self.segments {
            seg.runtime.clear_intermediates();
        }
    }

    /// Borrow the paged KV cache (e.g. to inspect page utilisation).
    pub fn kv(&self) -> &PagedKvCache {
        &self.kv
    }

    /// Begin a request: admit it through the paged allocator (prefix match +
    /// page allocation) and make it active. Returns the number of prompt tokens
    /// reused from the prefix cache — the caller skips recomputing those.
    pub fn begin_request(&mut self, id: RequestId, tokens: &[u32]) -> Result<usize, RuntimeError> {
        let matched = self.kv.begin_request(id, tokens)?;
        self.position = matched;
        Ok(matched)
    }

    /// Grow the active request's pages by one decode token (call before the
    /// step that writes that token's KV).
    pub fn advance_kv(&mut self) -> Result<(), RuntimeError> {
        self.kv.advance(1)
    }

    /// Admit a request without making it active (continuous-batching driver,
    /// which keeps several requests in-flight). Returns the prefix-cache hit.
    pub fn admit_request(&mut self, id: RequestId, tokens: &[u32]) -> Result<usize, RuntimeError> {
        self.kv.admit_request(id, tokens)
    }

    /// Switch the active request to `id` at decode position `position` (its KV
    /// is read/written through that request's pages from now until the next
    /// switch). Keeps `self.position` in sync.
    pub fn activate_request(&mut self, id: RequestId, position: usize) -> Result<(), RuntimeError> {
        self.kv.set_active(id, position)?;
        self.position = position;
        self.active_request = Some(id);
        Ok(())
    }

    /// Release a specific request's pages (whether or not it is active).
    pub fn release_request(&mut self, id: RequestId) -> Result<(), RuntimeError> {
        self.kv.release(id)
    }

    /// Release the active request's pages back to the cache/free list.
    pub fn end_request(&mut self) {
        self.kv.end_request();
    }

    /// Prefix-cache tokens reused by the active request (for metrics).
    pub fn prefix_hit_tokens(&self) -> usize {
        self.kv.prefix_hit_tokens()
    }

    /// Pages in use across all requests on this device.
    pub fn kv_pages_in_use(&self) -> u32 {
        self.kv.in_use_pages()
    }

    /// Clear the KV cache between requests (releases the active request's
    /// pages; cached prefix bytes survive for reuse).
    pub fn reset_kv_cache(&mut self) {
        self.kv.reset();
        self.position = 0;
    }

    /// Set the current decode position for this step: the absolute index of the
    /// token being processed, which equals the number of tokens already in the
    /// cache (the dynamic `past` length). Feeds the `position` graph input and is
    /// applied to each segment's dynamic-dim map in [`run_segment`].
    pub fn set_position(&mut self, position: usize) {
        self.position = position;
        // The paged cache uses `position` as the valid-slot count for buffer
        // assembly (and grows the implicit request's pages when no explicit
        // request was begun).
        let _ = self.kv.set_position(position);
        let id = self.intern_name("position");
        self.position_id = Some(id);
        self.f32_slots[id.idx()] = Some(vec![position as f32]);
    }

    /// Seed the integer input handoff the first segment consumes (the runtime
    /// feeds the last token id each decode step, matching the executor).
    pub fn set_input_tokens(&mut self, name: &str, tokens: Vec<i32>) {
        let id = self.intern_name(name);
        self.input_tokens_id = Some(id);
        self.i32_slots[id.idx()] = Some(tokens);
    }

    /// Enable/disable captured-graph mode (SKEIN_CAPTURE). When on, `run_segment`
    /// skips the per-token `input_tokens` feed and the `set_decode_position`
    /// memcpy so they aren't recorded into the full-step graph;
    /// [`flush_step_device_inputs`](Self::flush_step_device_inputs) writes them
    /// into the persistent device buffers each step instead.
    pub fn set_capturing(&mut self, on: bool) {
        self.capturing = on;
    }

    /// MB capture: register the boundary-output ids (carry, logits) to keep
    /// device-resident under capture (see [`capture_passthrough_outputs`]). Looked
    /// up by name; unknown names are ignored.
    pub fn set_capture_passthrough_by_names(&mut self, names: &[&str]) {
        self.capture_passthrough_outputs.clear();
        for n in names {
            if let Some(id) = self.name_to_id.get(*n) {
                self.capture_passthrough_outputs.insert(*id);
            }
        }
    }

    /// MB capture: register a boundary-output id directly (the carry tensor id
    /// comes from the schedule, not a stable name).
    pub fn add_capture_passthrough_id(&mut self, id: HandoffId) {
        self.capture_passthrough_outputs.insert(id);
    }

    /// SKEIN_CAPTURE: write this step's per-token device inputs — the decode
    /// `position` (read by every captured `kv_slot_write` kernel) and the
    /// `input_tokens` id (read by the captured embedding) — directly into their
    /// persistent device buffers, OUTSIDE the captured graph. The buffers keep a
    /// stable device pointer (the captured kernels read those pointers), so the
    /// graph sees fresh values on every replay without any host→device copy being
    /// recorded into it. Must be called every decode step before replay/capture,
    /// on the shared capture stream (it is — these go through each segment's
    /// runtime, which shares that stream under SKEIN_CAPTURE).
    pub fn flush_step_device_inputs(&mut self, tokens: &[i32], position: usize) {
        self.position = position;
        let _ = self.kv.set_position(position);
        let tok_id = self.input_tokens_id;
        let pos_id = self.position_id;
        for segment_idx in 0..self.segments.len() {
            // decode position: every segment that writes a KV slot reads its own
            // runtime's device position buffer in the captured kv_slot_write.
            let writes_kv = self.segment_outputs[segment_idx]
                .iter()
                .any(|id| matches!(self.kind[id.idx()], HandoffKind::KvCache { .. }));
            if writes_kv {
                self.segments[segment_idx].runtime.set_decode_position(position);
            }
            // input token: the segment(s) consuming `input_tokens` (the embedding).
            // Use the IMMEDIATE setter (not the staging `set_tensor_i32_by_id`):
            // staged values are only uploaded at `execute_segment`, which never
            // runs on a captured-graph replay — so a staged token would freeze at
            // its capture-time value (and the capture-step upload would record a
            // freed-host-source copy into the graph). The immediate write lands in
            // the resident device buffer now, outside the captured region.
            if let Some(tok_id) = tok_id {
                if self.segment_inputs[segment_idx].iter().any(|id| *id == tok_id) {
                    let _ = self.segments[segment_idx]
                        .runtime
                        .set_input_i32_immediate_by_id(tok_id, tokens.to_vec());
                }
            }
            // position scalar: the attention segments use it for RoPE and to derive
            // the KV validity mask — must be fresh each replay. Same immediate-write
            // reasoning as the token above.
            if let Some(pos_id) = pos_id {
                if self.segment_inputs[segment_idx].iter().any(|id| *id == pos_id) {
                    let _ = self.segments[segment_idx]
                        .runtime
                        .set_input_f32_immediate_by_id(pos_id, vec![position as f32]);
                }
            }
        }
        // MB capture: also refresh each host op's per-step device state
        // (e.g. CudaGraphOp's dyn_dims_buffer) from a dyn_map seeded with the
        // current decode 'p' = position. Without this, every captured graph
        // replay sees the warmup-step `p` value baked into dyn_dims_buffer
        // and attention/MoE kernels reading dyn_dims read stale `p`.
        let mut dyn_map: HashMap<char, usize> = HashMap::new();
        dyn_map.insert('p', position);
        for segment_idx in 0..self.segments.len() {
            self.segments[segment_idx]
                .runtime
                .refresh_capture_dyn_dims(&dyn_map);
        }
    }

    /// Full-step CUDA graph (SKEIN_CAPTURE): drive capture/replay on the shared
    /// stream via any segment runtime (they all share it). `begin`/`end` wrap the
    /// pre-all-gather schedule walk; `replay` relaunches it; `has_captured` gates.
    pub fn begin_capture(&self) {
        if let Some(seg) = self.segments.first() {
            seg.runtime.begin_stream_capture();
        }
    }
    pub fn end_capture(&self) {
        if let Some(seg) = self.segments.first() {
            seg.runtime.end_stream_capture();
        }
    }
    pub fn replay_captured(&self) -> bool {
        self.segments
            .first()
            .map(|seg| seg.runtime.replay_captured())
            .unwrap_or(false)
    }
    pub fn has_captured(&self) -> bool {
        self.segments
            .first()
            .map(|seg| seg.runtime.has_captured_graph())
            .unwrap_or(false)
    }

    /// Keyed multi-graph capture (MB decode): one graph per microbatch slot
    /// `key`. `begin_capture` is shared (stream-level); end/replay/has are keyed.
    pub fn end_capture_keyed(&self, key: u64) {
        if let Some(seg) = self.segments.first() {
            seg.runtime.end_stream_capture_keyed(key);
        }
    }
    pub fn replay_captured_keyed(&self, key: u64) -> bool {
        self.segments
            .first()
            .map(|seg| seg.runtime.replay_captured_keyed(key))
            .unwrap_or(false)
    }
    pub fn has_captured_keyed(&self, key: u64) -> bool {
        self.segments
            .first()
            .map(|seg| seg.runtime.has_captured_graph_keyed(key))
            .unwrap_or(false)
    }

    /// Host-stage one side of a cross-GPU all-reduce: D2H-read the device-resident
    /// handoff `id` (bf16 -> f32) from this runner's GPU. `None` if the handoff is
    /// not device-resident (then the caller falls back to the host `f32_slots`
    /// path). Used by [`LocalTopology`](super::LocalTopology) so the single-process
    /// continuous-batch driver can all-reduce on-device activations without NVLink
    /// P2P. Any segment's runtime works — they share this runner's device context.
    pub fn read_device_handoff(&self, id: HandoffId) -> Option<Vec<f32>> {
        let (ptr, elems) = self.output_device_ptr_by_id(id)?;
        let rt = &self.segments.first()?.runtime;
        Some(unsafe { rt.read_device_bf16(ptr, elems) })
    }

    /// D2H-read a device-resident handoff by NAME (bf16 -> f32). Used by the MB
    /// capture loop to read the boundary carry / logits OUTSIDE the captured
    /// region (they are kept device-resident under capture). `None` if not bound.
    pub fn read_device_handoff_by_name(&self, name: &str) -> Option<Vec<f32>> {
        let id = *self.name_to_id.get(name)?;
        self.read_device_handoff(id)
    }

    /// MB capture: feed the boundary carry (stage 1's HostCollective input) into
    /// its persistent device buffer IMMEDIATELY, outside the captured region.
    /// Leaves `f32_slots[id]` untouched (`None`) so `run_segment`'s input feed
    /// records no host->device copy into the graph; the captured kernels read the
    /// resident buffer this writes. Returns true if the carry input was found.
    pub fn feed_capture_input_f32(&mut self, id: HandoffId, data: Vec<f32>) -> bool {
        for segment_idx in 0..self.segments.len() {
            if self.segment_inputs[segment_idx].iter().any(|i| *i == id) {
                return self.segments[segment_idx]
                    .runtime
                    .set_input_f32_immediate_by_id(id, data)
                    .is_ok();
            }
        }
        false
    }

    /// Write the reduced result back into the device-resident handoff `id`
    /// (f32 -> bf16, H2D) so the consuming segment (which binds this buffer)
    /// reads the all-reduced value. Returns false if `id` is not device-resident.
    pub fn write_device_handoff(&self, id: HandoffId, data: &[f32]) -> bool {
        let Some((ptr, _elems)) = self.output_device_ptr_by_id(id) else {
            return false;
        };
        let Some(seg) = self.segments.first() else {
            return false;
        };
        unsafe { seg.runtime.write_device_bf16(ptr, data) };
        true
    }

    /// This runner's interned [`HandoffId`] for a logical tensor name, if known.
    /// Each runner has its own `name -> id` map, so a collective tensor must be
    /// resolved per runner. `None` if the name was never interned here.
    pub fn handoff_id(&self, name: &str) -> Option<HandoffId> {
        self.name_to_id.get(name).copied()
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Does this segment produce any host-materialized output (a `HostCollective`
    /// handoff: the boundary carry, the logits, a router tensor)? Such an output
    /// triggers a device→host copy in `run_segment`, which is illegal during CUDA
    /// stream capture — the full-step-graph capture window must stop before it
    /// (the segment then runs in the un-captured host pre/post region).
    pub fn segment_has_host_output(&self, segment_idx: usize) -> bool {
        self.segment_outputs
            .get(segment_idx)
            .map(|ids| {
                ids.iter()
                    .any(|id| matches!(self.kind[id.idx()], HandoffKind::HostCollective))
            })
            .unwrap_or(false)
    }

    /// Resolve the artifact's String-keyed [`SequenceStep`] schedule into the
    /// id-keyed [`ResolvedSequenceStep`] the [`RankExecutor`](super::RankExecutor)
    /// walks on the hot path. Done once at bootstrap (after `name_to_id` is
    /// populated and — for sparse MoE — after `materialize_weights`, since the
    /// expert weight device pointers are resolved here). After this, no logical
    /// name is hashed per token: collective tensors, the router-logits handoff,
    /// the FFN slot/gate inputs, and the selected experts' resident weight
    /// buffers are all pre-resolved.
    pub fn resolve_schedule(
        &mut self,
        schedule: &[SequenceStep],
    ) -> Result<Vec<ResolvedSequenceStep>, RankExecError> {
        let mut out = Vec::with_capacity(schedule.len());
        for step in schedule {
            let resolved = match step {
                SequenceStep::ExecuteSegment {
                    device_idx,
                    segment_idx,
                } => ResolvedSequenceStep::ExecuteSegment {
                    device_idx: *device_idx,
                    segment_idx: *segment_idx,
                },
                SequenceStep::Collective {
                    collective,
                    participants,
                    tensor,
                    shape,
                    ..
                } => ResolvedSequenceStep::Collective {
                    collective: *collective,
                    participants: participants.clone(),
                    tensor: self.intern_name(tensor),
                    elems: shape.iter().product::<usize>().max(1),
                },
                SequenceStep::MoeRoute {
                    device_idx,
                    block,
                    ffn_segment_idx,
                    router_tensor,
                    top_k,
                    expert_weight_names,
                    slot_weight_names,
                    slot_gate_names,
                    ..
                } => {
                    let router_id = self.intern_name(router_tensor);
                    let slot_ids: Vec<[HandoffId; 3]> = slot_weight_names
                        .iter()
                        .map(|w| {
                            [
                                self.intern_name(&w[0]),
                                self.intern_name(&w[1]),
                                self.intern_name(&w[2]),
                            ]
                        })
                        .collect();
                    let gate_ids: Vec<HandoffId> =
                        slot_gate_names.iter().map(|g| self.intern_name(g)).collect();
                    let expert_weights =
                        self.resolve_expert_weights(*ffn_segment_idx, expert_weight_names)?;
                    ResolvedSequenceStep::MoeRoute {
                        device_idx: *device_idx,
                        ffn_segment_idx: *ffn_segment_idx,
                        router_id,
                        top_k: *top_k,
                        block: *block,
                        expert_weights,
                        slot_ids,
                        gate_ids,
                    }
                }
            };
            out.push(resolved);
        }
        Ok(out)
    }

    /// Resolve each owned expert's three weight buffers to `(device_ptr, n_bytes)`
    /// from the FFN segment's resident weights — once, at schedule resolution.
    fn resolve_expert_weights(
        &self,
        ffn_segment_idx: usize,
        expert_weight_names: &[[String; 3]],
    ) -> Result<Vec<[(u64, usize); 3]>, RankExecError> {
        let seg = self
            .segments
            .get(ffn_segment_idx)
            .ok_or_else(|| RankExecError::Segment {
                segment_idx: ffn_segment_idx,
                detail: "MoeRoute FFN segment index out of range".to_string(),
            })?;
        let mut out = Vec::with_capacity(expert_weight_names.len());
        for names in expert_weight_names {
            let mut triple = [(0u64, 0usize); 3];
            for w in 0..3 {
                let ptr = seg
                    .runtime
                    .weight_device_ptr_by_name(&names[w])
                    .ok_or_else(|| RankExecError::Segment {
                        segment_idx: ffn_segment_idx,
                        detail: format!("expert weight {} not resident", names[w]),
                    })?;
                let n_bytes = seg
                    .weight_names
                    .iter()
                    .find(|(name, _)| name == &names[w])
                    .map(|(_, shape)| shape.iter().product::<usize>() * 2)
                    .ok_or_else(|| RankExecError::Segment {
                        segment_idx: ffn_segment_idx,
                        detail: format!("expert weight {} shape unknown", names[w]),
                    })?;
                triple[w] = (ptr, n_bytes);
            }
            out.push(triple);
        }
        Ok(out)
    }

    /// Logical name for a handoff id (for error messages).
    fn id_name(&self, id: HandoffId) -> String {
        self.id_to_name
            .get(id.idx())
            .cloned()
            .unwrap_or_else(|| format!("id {}", id.0))
    }

    /// Install the resident bf16 MoE gate buffer (allocated at bootstrap on
    /// `stream`). `top_k` is the per-block slab width. CUDA only.
    #[cfg(feature = "cuda")]
    pub fn set_gate_buffer(
        &mut self,
        stream: std::sync::Arc<cudarc::driver::CudaStream>,
        buf: cudarc::driver::CudaSlice<half::bf16>,
        top_k: usize,
    ) {
        self.gate_buffer = Some(MoeGateBuffer {
            stream,
            buf,
            top_k,
        });
    }

    /// Bind each MoE FFN segment's gate-scalar inputs to fixed offsets of the
    /// resident gate buffer, **once** (persistent). Block `b`, slot `s` → element
    /// `b * top_k + s` of the buffer. After this, `route_moe_resolved` just
    /// overwrites those buffer elements per token; the FFN reads from the bound
    /// offsets. CUDA only; requires [`set_gate_buffer`](Self::set_gate_buffer)
    /// first and (for sparse MoE) resident weights.
    #[cfg(feature = "cuda")]
    pub fn bind_gate_inputs(
        &mut self,
        schedule: &[ResolvedSequenceStep],
        rank: u32,
    ) -> Result<(), RankExecError> {
        use cudarc::driver::DevicePtr as _;
        let (base, top_k) = {
            let g = self
                .gate_buffer
                .as_ref()
                .ok_or_else(|| RankExecError::Segment {
                    segment_idx: 0,
                    detail: "gate buffer not set before bind_gate_inputs".to_string(),
                })?;
            (g.buf.device_ptr(&g.stream).0, g.top_k)
        };
        let elem = std::mem::size_of::<half::bf16>();
        for step in schedule {
            let ResolvedSequenceStep::MoeRoute {
                device_idx,
                ffn_segment_idx,
                block,
                gate_ids,
                ..
            } = step
            else {
                continue;
            };
            if *device_idx != rank {
                continue;
            }
            let seg =
                self.segments
                    .get_mut(*ffn_segment_idx)
                    .ok_or_else(|| RankExecError::Segment {
                        segment_idx: *ffn_segment_idx,
                        detail: "MoeRoute FFN segment index out of range (bind_gate_inputs)"
                            .to_string(),
                    })?;
            for (s, &gid) in gate_ids.iter().enumerate() {
                let off = ((*block * top_k + s) * elem) as u64;
                let ptr = base.checked_add(off).ok_or_else(|| RankExecError::Segment {
                    segment_idx: *ffn_segment_idx,
                    detail: "gate buffer offset overflow".to_string(),
                })?;
                unsafe {
                    seg.runtime
                        .set_input_device_persistent_by_id(gid, ptr, elem);
                }
            }
        }
        Ok(())
    }

    fn segment_err(segment_idx: usize) -> impl Fn(skein_compile::DynRuntimeError) -> RankExecError {
        move |e| RankExecError::Segment {
            segment_idx,
            detail: e.to_string(),
        }
    }
}

impl LocalSegments for SegmentRunner {
    fn run_segment(&mut self, segment_idx: usize) -> Result<(), RankExecError> {
        if segment_idx >= self.segments.len() {
            return Err(RankExecError::Segment {
                segment_idx,
                detail: "segment index out of range".to_string(),
            });
        }
        let err = Self::segment_err(segment_idx);

        // SKEIN_DEVICE_KV (full-step graph piece 2): append the new token's K/V via
        // a device kernel that reads `position` from a device buffer, instead of a
        // host-issued DtoD with a baked-in destination offset.
        static DEVICE_KV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let device_kv = *DEVICE_KV.get_or_init(|| std::env::var_os("SKEIN_DEVICE_KV").is_some());

        // --- profiling (compile-time gated by `perf-trace`): host-input staging
        // vs GPU launch vs host-output capture (the dtoh in capture forces a
        // sync, so host_out_us absorbs the actual GPU compute wait). With the
        // feature off this is a zero-sized no-op — no `Instant::now()` here.
        let mut t = crate::perf_timing::SegTimer::start_in();

        // Feed inputs by pre-resolved id — no `Vec<String>` clone, no per-name
        // HashMap lookup. Each id's `kind` (fixed at construction) selects the
        // same dispatch the old string-keyed loop performed.
        for i in 0..self.segment_inputs[segment_idx].len() {
            let id = self.segment_inputs[segment_idx][i];
            let slot = id.idx();
            match self.kind[slot] {
                HandoffKind::KvCache { kind, layer } => {
                    let full = self.kv_full_elems[slot].unwrap_or(0);
                    // Continuous batching: bind THIS request's own contiguous KV
                    // buffer (per-request isolation). The shared single-buffer path
                    // below would let concurrent requests clobber each other's KV.
                    let base = if self.paged_device_kv {
                        let req = self.active_request;
                        let ptr = req.and_then(|r| {
                            self.req_kv_buffer(segment_idx, r, slot, full * KV_DEVICE_ELEM_BYTES)
                        });
                        if let Some(p) = ptr {
                            // Rebind the input to this request's buffer for this step.
                            unsafe {
                                self.segments[segment_idx]
                                    .runtime
                                    .bind_input_device_by_id(id, p, full * KV_DEVICE_ELEM_BYTES)
                            };
                        }
                        ptr
                    } else {
                        // Single-request path: bind a persistent on-GPU buffer (full
                        // capacity, f32) once and reuse it every step — NO assemble,
                        // NO H2D. The new token's K/V is written into it via DtoD.
                        self.segments[segment_idx]
                            .runtime
                            .ensure_kv_input_device_by_id(id, full * KV_DEVICE_ELEM_BYTES)
                    };
                    if let Some(b) = base {
                        self.kv_device_base[slot] = Some(b);
                    } else {
                        // Host fallback (CPU backend / no device buffer): assemble
                        // the contiguous buffer and upload it (the old path).
                        let buf = self.kv.buffer(kind, layer as usize, full);
                        crate::perf_counters::record_h2d(buf.len() * std::mem::size_of::<f32>());
                        self.segments[segment_idx]
                            .runtime
                            .set_tensor_by_id(id, buf)
                            .map_err(&err)?;
                    }
                }
                HandoffKind::Device => {
                    if let Some(h) = self.device_slots[slot] {
                        // Device-resident activation: bind the producer segment's
                        // output buffer directly by device pointer — no host Vec.
                        unsafe {
                            self.segments[segment_idx]
                                .runtime
                                .bind_input_device_by_id(id, h.ptr, h.n_bytes)
                        };
                    } else if let Some(data) = self.f32_slots[slot].clone() {
                        // CPU-backend fallback: the producer wrote host bytes.
                        crate::perf_counters::record_h2d(data.len() * std::mem::size_of::<f32>());
                        self.segments[segment_idx]
                            .runtime
                            .set_tensor_by_id(id, data)
                            .map_err(&err)?;
                    }
                }
                HandoffKind::I32 => {
                    // SKEIN_CAPTURE: the `input_tokens` feed is handled outside the
                    // captured graph by `flush_step_device_inputs` (writing into the
                    // same persistent device buffer). Doing it here would record a
                    // host→device copy into the graph from a freed host source.
                    let skip = self.capturing && Some(id) == self.input_tokens_id
                        && std::env::var_os("SKEIN_NO_SKIP").is_none();
                    if !skip {
                        if let Some(data) = self.i32_slots[slot].clone() {
                            crate::perf_counters::record_h2d(
                                data.len() * std::mem::size_of::<i32>(),
                            );
                            self.segments[segment_idx]
                                .runtime
                                .set_tensor_i32_by_id(id, data)
                                .map_err(&err)?;
                        }
                    }
                }
                HandoffKind::F32 | HandoffKind::HostCollective => {
                    // SKEIN_CAPTURE: the `position` scalar is refreshed outside the
                    // captured graph by `flush_step_device_inputs`; feeding it here
                    // would record a host→device copy from a freed source (freezing
                    // RoPE and the FlashInfer KV length). Collective results
                    // (`logits` etc.) are NOT skipped — they live in the host tail,
                    // outside the captured window, and are produced within the step.
                    let skip = self.capturing && Some(id) == self.position_id
                        && std::env::var_os("SKEIN_NO_SKIP").is_none();
                    // Peek (clone), not take: `position` is read by every attention
                    // segment within one step (it is staged once by `set_position`),
                    // so taking it would null it after the first reader. These host
                    // f32 inputs are all tiny (scalars / small collective results).
                    if !skip {
                        if let Some(data) = self.f32_slots[slot].clone() {
                            crate::perf_counters::record_h2d(
                                data.len() * std::mem::size_of::<f32>(),
                            );
                            self.segments[segment_idx]
                                .runtime
                                .set_tensor_by_id(id, data)
                                .map_err(&err)?;
                        }
                    }
                }
            }
            // A missing input is left to the segment's own defaults (e.g. a weight
            // already loaded into the runtime, or a route-bound slot); not an error.
        }

        t.mark_gpu();
        crate::perf_counters::record_segment_launch();
        self.segments[segment_idx]
            .runtime
            .execute_segment()
            .map_err(&err)?;
        t.mark_out();

        // Capture outputs by pre-resolved id. A `kvcache_*` output is the new
        // token's K/V written into slot `position`; collective/router tensors stay
        // host; every other activation is kept DEVICE-RESIDENT (record the
        // producer's output buffer ptr+size and bind it into the consumer next
        // step — no D2H).
        for i in 0..self.segment_outputs[segment_idx].len() {
            let id = self.segment_outputs[segment_idx][i];
            let slot = id.idx();
            match self.kind[slot] {
                HandoffKind::KvCache { kind, layer } => {
                    // Device-resident KV write (steady-state decode): DtoD-copy the
                    // new token's K/V into slot `position` of the resident buffer.
                    if !self.prefill_capture {
                        if let Some(base) = self.kv_device_base[slot] {
                            if let Some((_, out_bytes)) = self.segments[segment_idx]
                                .runtime
                                .output_device_ptr_by_id(id)
                            {
                                // Synchronous batched decode: write all `decode_batch`
                                // rows' new K/V into the [batch, cap, kv_dim] cache at
                                // the shared `position` (strided per row). `base` is the
                                // single batched KV buffer (full*4 bytes, f32); the
                                // per-row output is `out_bytes/batch`, and `cap` is the
                                // slot count = (full*4)/out_bytes.
                                if self.decode_batch > 1 {
                                    let rt = &self.segments[segment_idx].runtime;
                                    // Under SKEIN_CAPTURE the position memcpy is done
                                    // outside the graph by flush_step_device_inputs
                                    // (recording it would bake a freed host pointer);
                                    // the captured batched kv_slot_write reads the
                                    // device position buffer. Mirrors the single path.
                                    if !self.capturing || std::env::var_os("SKEIN_NO_SKIP").is_some() {
                                        rt.set_decode_position(self.position);
                                    }
                                    let batch = self.decode_batch;
                                    let row_bytes = out_bytes / batch;
                                    let full_bytes =
                                        self.kv_full_elems[slot].unwrap_or(0) * KV_DEVICE_ELEM_BYTES;
                                    let cap = if out_bytes > 0 { full_bytes / out_bytes } else { 0 };
                                    unsafe {
                                        rt.copy_output_to_kv_slot_batched_by_id(
                                            id, base, batch, cap, row_bytes,
                                        )
                                    };
                                    self.device_slots[slot] = None;
                                    continue;
                                }
                                if device_kv {
                                    // Device-side append: kernel computes
                                    // `base + position*out_bytes` from the device
                                    // position buffer (graph-capturable, no host
                                    // dest baked in). See piece 2 of the full-step
                                    // graph (SKEIN_DEVICE_KV).
                                    let rt = &self.segments[segment_idx].runtime;
                                    // SKEIN_CAPTURE: the position memcpy is done
                                    // outside the graph by flush_step_device_inputs
                                    // (recording it here would bake a freed host
                                    // stack pointer into the graph). The captured
                                    // kv_slot_write still reads the device buffer.
                                    if !self.capturing || std::env::var_os("SKEIN_NO_SKIP").is_some() {
                                        rt.set_decode_position(self.position);
                                    }
                                    unsafe {
                                        rt.copy_output_to_kv_slot_by_id(id, base, out_bytes)
                                    };
                                } else {
                                    let dest = base + (self.position * out_bytes) as u64;
                                    unsafe {
                                        self.segments[segment_idx]
                                            .runtime
                                            .copy_output_to_device_by_id(id, dest, out_bytes)
                                    };
                                }
                                self.device_slots[slot] = None;
                                continue;
                            }
                        }
                    }
                    // Host path: batched-prefill capture, or CPU fallback.
                    let data = self.segments[segment_idx]
                        .runtime
                        .get_tensor_by_id(id)
                        .map_err(&err)?;
                    crate::perf_counters::record_d2h(data.len() * std::mem::size_of::<f32>());
                    if self.prefill_capture {
                        self.f32_slots[slot] = Some(data);
                    } else {
                        self.kv.write_slot(kind, layer as usize, self.position, &data);
                    }
                    self.device_slots[slot] = None;
                }
                HandoffKind::HostCollective => {
                    // MB CUDA-graph capture: the boundary carry / logits are
                    // HostCollective outputs. A D2H here would be RECORDED into the
                    // captured graph with a host dest Vec freed right after, so
                    // replays would scribble freed memory. Under capture keep them
                    // device-resident (stable buffer); the MB loop D2H-reads them
                    // OUTSIDE the captured region via `read_device_handoff`.
                    if self.capturing && self.capture_passthrough_outputs.contains(&id) {
                        if let Some((ptr, n_bytes)) = self.segments[segment_idx]
                            .runtime
                            .output_device_ptr_by_id(id)
                        {
                            self.device_slots[slot] = Some(DeviceTensorHandle {
                                device: 0,
                                ptr,
                                n_bytes,
                                elems: n_bytes / 2,
                                dtype: HandoffDtype::Bf16,
                            });
                            self.f32_slots[slot] = None;
                            continue;
                        }
                    }
                    // Collective tensors and the sparse-MoE router logits stay host
                    // (read host-side: the collective read/write path, and the
                    // MoeRoute top-k pick over the router logits).
                    let data = self.segments[segment_idx]
                        .runtime
                        .get_tensor_by_id(id)
                        .map_err(&err)?;
                    crate::perf_counters::record_d2h(data.len() * std::mem::size_of::<f32>());
                    self.f32_slots[slot] = Some(data);
                    self.device_slots[slot] = None;
                }
                HandoffKind::Device | HandoffKind::F32 | HandoffKind::I32 => {
                    if let Some((ptr, n_bytes)) = self.segments[segment_idx]
                        .runtime
                        .output_device_ptr_by_id(id)
                    {
                        self.device_slots[slot] = Some(DeviceTensorHandle {
                            device: 0,
                            ptr,
                            n_bytes,
                            elems: n_bytes / 2,
                            dtype: HandoffDtype::Bf16,
                        });
                        self.f32_slots[slot] = None;
                    } else {
                        // Fallback (CPU backend / no device buffer yet): host.
                        let data = self.segments[segment_idx]
                            .runtime
                            .get_tensor_by_id(id)
                            .map_err(&err)?;
                        crate::perf_counters::record_d2h(data.len() * std::mem::size_of::<f32>());
                        self.f32_slots[slot] = Some(data);
                    }
                }
            }
        }
        // Emit the per-segment record (debug-level, and only when `perf-trace`
        // is compiled in — otherwise this is a no-op with no timing at all).
        t.finish(segment_idx);
        Ok(())
    }

    fn read(&self, name: &str) -> Result<Vec<f32>, RankExecError> {
        let id = self
            .name_to_id
            .get(name)
            .ok_or_else(|| RankExecError::UnknownTensor(name.to_string()))?;
        self.f32_slots[id.idx()]
            .clone()
            .ok_or_else(|| RankExecError::UnknownTensor(name.to_string()))
    }

    fn write(&mut self, name: &str, data: Vec<f32>) -> Result<(), RankExecError> {
        let id = self.intern_name(name);
        self.f32_slots[id.idx()] = Some(data);
        Ok(())
    }

    fn output_device_ptr(&self, name: &str) -> Option<(u64, usize)> {
        let id = self.name_to_id.get(name)?;
        self.device_slots[id.idx()].map(|h| (h.ptr, h.elems))
    }

    fn read_by_id(&self, id: HandoffId) -> Result<Vec<f32>, RankExecError> {
        self.f32_slots
            .get(id.idx())
            .and_then(|s| s.clone())
            .ok_or_else(|| RankExecError::UnknownTensor(self.id_name(id)))
    }

    fn write_by_id(&mut self, id: HandoffId, data: Vec<f32>) -> Result<(), RankExecError> {
        let slot = self
            .f32_slots
            .get_mut(id.idx())
            .ok_or_else(|| RankExecError::UnknownTensor(format!("id {}", id.0)))?;
        *slot = Some(data);
        Ok(())
    }

    fn output_device_ptr_by_id(&self, id: HandoffId) -> Option<(u64, usize)> {
        self.device_slots
            .get(id.idx())
            .copied()
            .flatten()
            .map(|h| (h.ptr, h.elems))
    }

    fn set_device_handoff_by_id(&mut self, id: HandoffId, ptr: u64, n_bytes: usize) {
        // Bind a device buffer as this handoff's device-resident value (the PP
        // receiver's just-ncclRecv'd boundary carry). The consumer segment then
        // binds it by pointer (HandoffKind::Device input path) — no H2D.
        let slot = id.idx();
        if slot < self.device_slots.len() {
            self.device_slots[slot] = Some(DeviceTensorHandle {
                device: 0,
                ptr,
                n_bytes,
                elems: n_bytes / 2,
                dtype: HandoffDtype::Bf16,
            });
            self.f32_slots[slot] = None;
        }
    }

    fn copy_output_to_device(&self, name: &str, dest_ptr: u64, n_bytes: usize) -> bool {
        let Some(&id) = self.name_to_id.get(name) else {
            return false;
        };
        // Find the segment that produces this output and copy its *computed* value
        // (D2D) into the caller's buffer — the same runtime path KV writes use, so
        // it reflects the live result rather than a stale output slot.
        for (seg, outs) in self.segment_outputs.iter().enumerate() {
            if outs.contains(&id) {
                unsafe {
                    self.segments[seg]
                        .runtime
                        .copy_output_to_device_by_id(id, dest_ptr, n_bytes);
                }
                return true;
            }
        }
        false
    }

    fn device_shm_all_reduce(
        &mut self,
        data_ptr: u64,
        shm_ptr: u64,
        rank: i32,
        elems: usize,
        slot_bytes: i32,
    ) -> Result<(), RankExecError> {
        // Any segment runtime works: all share the device's primary context +
        // (under SKEIN_CAPTURE) the one capture stream. Launch the all-reduce
        // there so it lands on that stream, capturable into the full-step graph.
        let rt = &self.segments[0].runtime;
        unsafe { rt.device_shm_all_reduce(data_ptr, shm_ptr, rank, elems, slot_bytes) };
        Ok(())
    }

    fn route_moe_resolved(
        &mut self,
        ffn_segment_idx: usize,
        router_id: HandoffId,
        top_k: usize,
        block: usize,
        expert_weights: &[[(u64, usize); 3]],
        slot_ids: &[[HandoffId; 3]],
    ) -> Result<(), RankExecError> {
        // The gate segment kept its router logits host (a `HostCollective`
        // handoff captured into `f32_slots`); read it by id — no string hashing.
        let logits = self
            .f32_slots
            .get(router_id.idx())
            .and_then(|s| s.clone())
            .ok_or_else(|| RankExecError::UnknownTensor(self.id_name(router_id)))?;
        let k = top_k.min(slot_ids.len());
        let (topk, gates) = top_k_softmax(&logits, k);
        if topk.is_empty() {
            return Ok(());
        }

        // Write this block's gate scalars (softmax over top-k, bf16 to match the
        // FFN gate input dtype) into its slab of the resident gate buffer with one
        // async H2D — the FFN gate inputs were bound to these offsets once at
        // bootstrap, so no per-slot upload and no per-token (re)bind is needed.
        #[cfg(feature = "cuda")]
        if let Some(g) = self.gate_buffer.as_mut() {
            let bf: Vec<half::bf16> = gates.iter().map(|&x| half::bf16::from_f32(x)).collect();
            let off = block * g.top_k;
            let mut view = g.buf.slice_mut(off..off + bf.len());
            g.stream
                .memcpy_htod(&bf, &mut view)
                .map_err(|e| RankExecError::Segment {
                    segment_idx: ffn_segment_idx,
                    detail: format!("gate scalar H2D failed: {e}"),
                })?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (block, &gates); // gate buffer is GPU-only (CPU computes gates only).

        // Rebind the FFN expert weight slots to the selected experts' resident
        // buffers (zero-copy, by id; re-bound every token as the selection
        // changes, so the FFN reads only the top-k experts' weights this step).
        let seg = self
            .segments
            .get_mut(ffn_segment_idx)
            .ok_or_else(|| RankExecError::Segment {
                segment_idx: ffn_segment_idx,
                detail: "MoeRoute FFN segment index out of range".to_string(),
            })?;
        let seg_err = |detail: String| RankExecError::Segment {
            segment_idx: ffn_segment_idx,
            detail,
        };
        for (slot, &expert) in topk.iter().enumerate() {
            let weights = expert_weights
                .get(expert)
                .ok_or_else(|| seg_err(format!("selected expert {expert} not in owned set")))?;
            for w in 0..3 {
                let (ptr, n_bytes) = weights[w];
                unsafe {
                    seg.runtime
                        .bind_input_device_by_id(slot_ids[slot][w], ptr, n_bytes);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skein_compile::{DynRuntime, DynRuntimeError, HandoffId};

    /// Mock helper: implement the `_by_id` hot-path methods (and the id
    /// registration that backs them) by interning a fixed name set, then
    /// delegating to the existing `_by_name` mock logic. This is the test-side
    /// mirror of how `DynRuntimeWrapper` resolves ids to graph nodes — these are
    /// pre-existing CPU mocks extended to the new method signatures, not a new
    /// component standing in for real code.
    macro_rules! mock_dyn_by_id {
        ($($name:expr),* $(,)?) => {
            fn register_handoff_ids(
                &mut self,
                id_for_name: &dyn Fn(&str) -> Option<HandoffId>,
            ) {
                $(
                    if let Some(id) = id_for_name($name) {
                        self.ids.insert(id, $name.to_string());
                    }
                )*
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
            fn set_tensor_i32_by_id(
                &mut self,
                id: HandoffId,
                data: Vec<i32>,
            ) -> Result<(), DynRuntimeError> {
                let name = self
                    .ids
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| DynRuntimeError::UnknownTensor(format!("id {}", id.0)))?;
                self.set_tensor_i32_by_name(&name, data)
            }
            fn get_tensor_by_id(&self, id: HandoffId) -> Result<Vec<f32>, DynRuntimeError> {
                let name = self
                    .ids
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| DynRuntimeError::UnknownTensor(format!("id {}", id.0)))?;
                self.get_tensor_by_name(&name)
            }
        };
    }

    /// A mock `DynRuntime`: on `execute_segment` it doubles its `in` tensor
    /// into an `out` tensor. Lets us exercise the adapter's input-feed →
    /// execute → output-capture path with no GPU and no Luminal graph.
    #[derive(Default)]
    struct DoublingRuntime {
        tensors: HashMap<String, Vec<f32>>,
        ids: HashMap<HandoffId, String>,
    }

    impl DynRuntime for DoublingRuntime {
        fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
            let input = self.tensors.get("in").cloned().unwrap_or_default();
            let doubled = input.iter().map(|v| v * 2.0).collect();
            self.tensors.insert("out".to_string(), doubled);
            Ok(())
        }
        fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError> {
            self.tensors
                .get(name)
                .cloned()
                .ok_or_else(|| DynRuntimeError::UnknownTensor(name.to_string()))
        }
        fn set_tensor_by_name(
            &mut self,
            name: &str,
            data: Vec<f32>,
        ) -> Result<(), DynRuntimeError> {
            self.tensors.insert(name.to_string(), data);
            Ok(())
        }
        fn set_tensor_i32_by_name(
            &mut self,
            name: &str,
            data: Vec<i32>,
        ) -> Result<(), DynRuntimeError> {
            self.tensors.insert(
                name.to_string(),
                data.into_iter().map(|v| v as f32).collect(),
            );
            Ok(())
        }
        mock_dyn_by_id!("in", "out");
    }

    fn mock_segment() -> RuntimeSegment {
        RuntimeSegment {
            runtime: Box::new(DoublingRuntime::default()),
            input_names: vec!["in".to_string()],
            output_names: vec!["out".to_string()],
            capture_names: vec![],
            weight_names: vec![],
            kv_cache_sizes: HashMap::new(),
        }
    }

    #[test]
    fn runs_segment_feeding_inputs_and_capturing_outputs() {
        let mut runner = SegmentRunner::new(vec![mock_segment()]);
        runner.write("in", vec![1.0, 2.0, 3.0]).unwrap();
        runner.run_segment(0).unwrap();
        assert_eq!(runner.read("out").unwrap(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn collective_result_written_back_is_visible_to_next_read() {
        let mut runner = SegmentRunner::new(vec![mock_segment()]);
        // Simulate the rank executor writing an all-reduced tensor back.
        runner.write("x", vec![9.0]).unwrap();
        assert_eq!(runner.read("x").unwrap(), vec![9.0]);
    }

    #[test]
    fn out_of_range_segment_is_an_error() {
        let mut runner = SegmentRunner::new(vec![]);
        assert!(matches!(
            runner.run_segment(0),
            Err(RankExecError::Segment { segment_idx: 0, .. })
        ));
    }

    /// Fixed-capacity KV-cache decode mechanics: a `kvcache_k_0` input is fed the
    /// whole fixed buffer each step and its per-token output is written into slot
    /// `position`. The mock always emits the token `[7, 7]`; after stepping
    /// positions 0,1,2 the cache holds it in the first three slots and zeros
    /// beyond — proving positional writes (no GPU, no Luminal).
    #[test]
    fn kv_cache_writes_into_position_slots() {
        use crate::kv_cache::KvKind;

        /// Ignores its input; emits a fixed 2-element "new token" for `kvcache_k_0`.
        #[derive(Default)]
        struct TokenEmitter {
            ids: HashMap<HandoffId, String>,
        }
        impl DynRuntime for TokenEmitter {
            fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
                Ok(())
            }
            fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError> {
                if name == "kvcache_k_0" {
                    Ok(vec![7.0, 7.0])
                } else {
                    Err(DynRuntimeError::UnknownTensor(name.to_string()))
                }
            }
            fn set_tensor_by_name(&mut self, _: &str, _: Vec<f32>) -> Result<(), DynRuntimeError> {
                Ok(())
            }
            fn set_tensor_i32_by_name(
                &mut self,
                _name: &str,
                _data: Vec<i32>,
            ) -> Result<(), DynRuntimeError> {
                Ok(())
            }
            mock_dyn_by_id!("kvcache_k_0");
        }

        // CAP = 4 slots, per-token = 2 -> full buffer length 8.
        let mut kv_cache_sizes = HashMap::new();
        kv_cache_sizes.insert("kvcache_k_0".to_string(), 8usize);
        let seg = RuntimeSegment {
            runtime: Box::new(TokenEmitter::default()),
            input_names: vec!["kvcache_k_0".to_string()],
            output_names: vec!["kvcache_k_0".to_string()],
            capture_names: vec![],
            weight_names: vec![],
            kv_cache_sizes,
        };
        let mut runner = SegmentRunner::new(vec![seg]);
        for p in 0..3 {
            runner.set_position(p);
            runner.run_segment(0).unwrap();
        }
        // The fixed `[7,7]` token was written into paged slots 0,1,2 — the
        // paged cache assembles them back into the valid-slot view (3 slots ×
        // width 2). This proves positional writes route through the page store.
        assert_eq!(
            runner.kv().peek_active(KvKind::Key, 0),
            vec![7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
        );
        // And the f32 handoff store does NOT also hold the cache tensor.
        assert!(runner.read("kvcache_k_0").is_err());
    }

    /// Capstone: the *whole* distributed stack on CPU — real [`SegmentRunner`]
    /// adapter + [`RankExecutor`] + threaded [`BarrierCollective`] — runs a
    /// two-rank schedule and produces the correct all-reduced result on both
    /// ranks. No GPU, no Luminal: this is the multi-process pipeline's
    /// orchestration validated end to end (the GPU build swaps in CUDA
    /// segments + NcclCollective behind the same interfaces).
    #[test]
    fn full_two_rank_pipeline_runs_on_cpu() {
        use crate::distributed::{BarrierCollective, RankExecutor};
        use skein_cost::collectives::CollectiveKind;
        use skein_emit::segment::SequenceStep;
        use skein_ir::types::Dtype;
        use std::sync::Arc;
        use std::thread;

        /// Per-rank segment runtime: on execute, emits this rank's partial
        /// contribution `[rank+1, rank+1]` for the collective tensor "x".
        struct RankPartial {
            rank: usize,
            tensors: HashMap<String, Vec<f32>>,
            ids: HashMap<HandoffId, String>,
        }
        impl DynRuntime for RankPartial {
            fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
                self.tensors
                    .insert("x".to_string(), vec![(self.rank as f32) + 1.0; 2]);
                Ok(())
            }
            fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError> {
                self.tensors
                    .get(name)
                    .cloned()
                    .ok_or_else(|| DynRuntimeError::UnknownTensor(name.to_string()))
            }
            fn set_tensor_by_name(
                &mut self,
                name: &str,
                data: Vec<f32>,
            ) -> Result<(), DynRuntimeError> {
                self.tensors.insert(name.to_string(), data);
                Ok(())
            }
            fn set_tensor_i32_by_name(
                &mut self,
                _name: &str,
                _data: Vec<i32>,
            ) -> Result<(), DynRuntimeError> {
                Ok(())
            }
            mock_dyn_by_id!("x");
        }

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

        let colls = BarrierCollective::group(2).expect("group");
        let mut joins = Vec::new();
        for (rank, coll) in colls.into_iter().enumerate() {
            let schedule = schedule.clone();
            joins.push(thread::spawn(move || {
                let seg = RuntimeSegment {
                    runtime: Box::new(RankPartial {
                        rank,
                        tensors: HashMap::new(),
                        ids: HashMap::new(),
                    }),
                    input_names: vec![],
                    output_names: vec!["x".to_string()],
                    capture_names: vec![],
                    weight_names: vec![],
                    kv_cache_sizes: HashMap::new(),
                };
                let mut runner = SegmentRunner::new(vec![seg]);
                // Resolve the String-keyed schedule to ids once (as bootstrap
                // does), then drive the id-based hot path.
                let resolved = runner.resolve_schedule(&schedule).expect("resolve");
                let mut exec = RankExecutor::new(rank, runner);
                exec.run(&resolved, &coll).expect("rank run");
                exec.into_runner().read("x").expect("x present")
            }));
        }
        for (rank, j) in joins.into_iter().enumerate() {
            let x = j.join().expect("rank thread");
            assert_eq!(x, vec![3.0, 3.0], "rank {rank} got {x:?}");
        }
    }

    // ── Fix #4: MoE gate routing ──────────────────────────────────────────

    /// Top-k selection + softmax-over-top-k (the gate scalars route_moe writes).
    #[test]
    fn top_k_softmax_picks_descending_and_normalizes() {
        // Experts: logits [3, 1, 5, 2]. Top-2 by value = expert 2 (5.0), 0 (3.0).
        let (idx, gates) = top_k_softmax(&[3.0, 1.0, 5.0, 2.0], 2);
        assert_eq!(idx, vec![2, 0], "top-2 picked in descending-logit order");
        // softmax over {5, 3}: 1/(1+e^-2) and e^-2/(1+e^-2).
        assert!((gates[0] - 0.880_797).abs() < 1e-5, "gate0={}", gates[0]);
        assert!((gates[1] - 0.119_203).abs() < 1e-5, "gate1={}", gates[1]);
        assert!((gates.iter().sum::<f32>() - 1.0).abs() < 1e-6, "gates sum to 1");
        // k clamps to the number of experts.
        let (idx, gates) = top_k_softmax(&[1.0, 2.0], 5);
        assert_eq!(idx, vec![1, 0]);
        assert_eq!(gates.len(), 2);
    }

    /// `route_moe_resolved` rebinds each FFN slot's three weight inputs to the
    /// selected experts' resident buffers. (The gate-scalar device write is
    /// GPU-only — `gate_buffer` is `None`/cfg'd out on this CPU build, so the
    /// gate values land via [`top_k_softmax`], asserted above; here we assert the
    /// expert→slot rebinding, which is backend-agnostic.) Uses a mock that records
    /// device-pointer bindings, the existing CPU-mock pattern.
    #[test]
    fn route_moe_resolved_rebinds_selected_experts_into_slots() {
        use std::sync::{Arc, Mutex};

        /// FFN runtime mock recording `bind_input_device_by_id(id, ptr, n_bytes)`.
        struct RecordingFfn {
            binds: Arc<Mutex<HashMap<HandoffId, (u64, usize)>>>,
        }
        impl DynRuntime for RecordingFfn {
            fn execute_segment(&mut self) -> Result<(), DynRuntimeError> {
                Ok(())
            }
            fn get_tensor_by_name(&self, name: &str) -> Result<Vec<f32>, DynRuntimeError> {
                Err(DynRuntimeError::UnknownTensor(name.to_string()))
            }
            fn set_tensor_by_name(&mut self, _: &str, _: Vec<f32>) -> Result<(), DynRuntimeError> {
                Ok(())
            }
            fn set_tensor_i32_by_name(
                &mut self,
                _: &str,
                _: Vec<i32>,
            ) -> Result<(), DynRuntimeError> {
                Ok(())
            }
            unsafe fn bind_input_device_by_id(&mut self, id: HandoffId, ptr: u64, n_bytes: usize) {
                self.binds.lock().unwrap().insert(id, (ptr, n_bytes));
            }
        }

        let binds = Arc::new(Mutex::new(HashMap::new()));
        // FFN segment with 2 slots × 3 weight inputs (block 0).
        let slot_names: Vec<String> = (0..2)
            .flat_map(|s| {
                (1..=3).map(move |w| format!("moe_slot{s}_w{w}_0"))
            })
            .collect();
        let seg = RuntimeSegment {
            runtime: Box::new(RecordingFfn {
                binds: binds.clone(),
            }),
            input_names: slot_names.clone(),
            output_names: vec![],
            capture_names: vec![],
            weight_names: vec![],
            kv_cache_sizes: HashMap::new(),
        };
        let mut runner = SegmentRunner::new(vec![seg]);

        // Router logits for 4 experts; top-2 = expert 2 (5.0) then expert 0 (3.0).
        runner
            .write("router_logits_0", vec![3.0, 1.0, 5.0, 2.0])
            .unwrap();
        let router_id = *runner.name_to_id.get("router_logits_0").unwrap();
        let id = |n: &str| *runner.name_to_id.get(n).unwrap();
        let slot_ids: Vec<[HandoffId; 3]> = vec![
            [
                id("moe_slot0_w1_0"),
                id("moe_slot0_w2_0"),
                id("moe_slot0_w3_0"),
            ],
            [
                id("moe_slot1_w1_0"),
                id("moe_slot1_w2_0"),
                id("moe_slot1_w3_0"),
            ],
        ];
        // Distinct (ptr, n_bytes) per expert so we can assert which got bound.
        let expert_weights: Vec<[(u64, usize); 3]> = (0..4)
            .map(|e| {
                let b = (e as u64 + 1) * 1000;
                [(b + 1, 11), (b + 2, 22), (b + 3, 33)]
            })
            .collect();

        runner
            .route_moe_resolved(0, router_id, 2, 0, &expert_weights, &slot_ids)
            .unwrap();

        let b = binds.lock().unwrap();
        // slot 0 ← expert 2, slot 1 ← expert 0.
        for w in 0..3 {
            assert_eq!(b[&slot_ids[0][w]], expert_weights[2][w], "slot0 w{w}");
            assert_eq!(b[&slot_ids[1][w]], expert_weights[0][w], "slot1 w{w}");
        }
    }
}

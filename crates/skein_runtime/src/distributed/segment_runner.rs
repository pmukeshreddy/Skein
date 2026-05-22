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

use skein_compile::RuntimeSegment;

use super::rank_executor::{LocalSegments, RankExecError};
use crate::error::RuntimeError;
use crate::kv::PagedKvCache;
use crate::kv_cache::parse_kvcache_name;
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

/// Parse a `u32` from the environment, falling back to `default`.
fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(default)
}

/// Drives one device's compiled segments + the rank-local handoff store.
pub struct SegmentRunner {
    segments: Vec<RuntimeSegment>,
    /// f32 handoff tensors keyed by logical name (collective results, KV-adjacent
    /// and final tensors). Internal activation handoffs use `handoffs_device`.
    handoffs: HashMap<String, Vec<f32>>,
    /// Integer handoff tensors (e.g. `input_tokens`).
    handoffs_i32: HashMap<String, Vec<i32>>,
    /// Device-resident segment->segment activation handoffs: a producer's output
    /// buffer bound directly into the consumer's input by device pointer (no
    /// host round-trip). Refreshed each forward by the producing segment.
    handoffs_device: HashMap<String, DeviceTensorHandle>,
    /// Base device pointer of each KV-cache input's persistent on-GPU buffer
    /// (logical name -> ptr). The cache lives on the GPU across decode steps; the
    /// new token's K/V is written into slot `position` via a DtoD copy, so the KV
    /// cache is never assembled or re-uploaded from host.
    kv_device_base: HashMap<String, u64>,
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
        segments: Vec<RuntimeSegment>,
        page_size: u32,
        total_pages: u32,
        prefix_enable: bool,
        radix_max_depth: u32,
    ) -> Self {
        let kv = PagedKvCache::new(page_size, total_pages, prefix_enable, radix_max_depth, 0)
            .expect("valid KV paging geometry");
        Self {
            segments,
            handoffs: HashMap::new(),
            handoffs_i32: HashMap::new(),
            handoffs_device: HashMap::new(),
            kv_device_base: HashMap::new(),
            host_tensors: HashSet::new(),
            kv,
            position: 0,
            prefill_capture: false,
        }
    }

    /// Tell the runner which tensor names must stay host (all-gather/broadcast
    /// collectives). Everything else — internal activations AND RingAllReduce
    /// tensors — stays device-resident; RingAllReduce is all-reduced in place on
    /// the device by the rank executor.
    pub fn set_host_tensors(&mut self, names: HashSet<String>) {
        self.host_tensors = names;
    }

    /// Device handle for a named output, if it was produced device-resident this
    /// forward (the brief's `output_device(name)`).
    pub fn output_device(&self, name: &str) -> Option<DeviceTensorHandle> {
        self.handoffs_device.get(name).copied()
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
        self.handoffs.get(name).cloned()
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
        self.handoffs
            .insert("position".to_string(), vec![position as f32]);
    }

    /// Seed the integer input handoff the first segment consumes (the runtime
    /// feeds the last token id each decode step, matching the executor).
    pub fn set_input_tokens(&mut self, name: &str, tokens: Vec<i32>) {
        self.handoffs_i32.insert(name.to_string(), tokens);
    }

    pub fn segment_count(&self) -> usize {
        self.segments.len()
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
        let (input_names, output_names) = {
            let seg = self
                .segments
                .get(segment_idx)
                .ok_or(RankExecError::Segment {
                    segment_idx,
                    detail: "segment index out of range".to_string(),
                })?;
            (seg.input_names.clone(), seg.output_names.clone())
        };
        let err = Self::segment_err(segment_idx);

        // --- profiling: time host-input staging vs GPU launch vs host-output
        // capture (the dtoh in capture forces a sync, so host_out_us absorbs the
        // actual GPU compute wait). Logged per segment for the profiling pass.
        let t_in = std::time::Instant::now();

        // Feed inputs. A `kvcache_*` input is fed the whole fixed-capacity cache
        // buffer for its layer (slots 0..position valid); otherwise from the
        // handoff store (i32 wins when both exist — only `input_tokens` is i32
        // and never collides with an f32 name).
        for name in &input_names {
            if let Some((kind, layer)) = parse_kvcache_name(name) {
                let full = self.segments[segment_idx]
                    .kv_cache_sizes
                    .get(name)
                    .copied()
                    .unwrap_or(0);
                // Device-resident KV: bind a persistent on-GPU buffer (full
                // capacity, bf16) once and reuse it every step — NO assemble, NO
                // H2D. The new token's K/V is written into it via DtoD below.
                let base = self.segments[segment_idx]
                    .runtime
                    .ensure_kv_input_device_by_name(name, full * 2);
                if let Some(b) = base {
                    self.kv_device_base.insert(name.clone(), b);
                } else {
                    // Host fallback (CPU backend / no device buffer): assemble the
                    // contiguous buffer and upload it (the old path).
                    let buf = self.kv.buffer(kind, layer, full);
                    crate::perf_counters::record_h2d(buf.len() * std::mem::size_of::<f32>());
                    self.segments[segment_idx]
                        .runtime
                        .set_tensor_by_name(name, buf)
                        .map_err(&err)?;
                }
            } else if let Some(h) = self.handoffs_device.get(name).copied() {
                // Device-resident activation handoff: bind the producer segment's
                // output buffer directly by device pointer — NO host Vec, no H2D.
                unsafe {
                    self.segments[segment_idx]
                        .runtime
                        .bind_input_device_by_name(name, h.ptr, h.n_bytes)
                };
            } else if let Some(data) = self.handoffs_i32.get(name).cloned() {
                crate::perf_counters::record_h2d(data.len() * std::mem::size_of::<i32>());
                self.segments[segment_idx]
                    .runtime
                    .set_tensor_i32_by_name(name, data)
                    .map_err(&err)?;
            } else if let Some(data) = self.handoffs.get(name).cloned() {
                crate::perf_counters::record_h2d(data.len() * std::mem::size_of::<f32>());
                self.segments[segment_idx]
                    .runtime
                    .set_tensor_by_name(name, data)
                    .map_err(&err)?;
            }
            // A missing input is left to the segment's own defaults (e.g. a
            // weight already loaded into the runtime); not an error here.
        }

        let host_in_us = t_in.elapsed().as_micros();

        let t_gpu = std::time::Instant::now();
        crate::perf_counters::record_segment_launch();
        self.segments[segment_idx]
            .runtime
            .execute_segment()
            .map_err(&err)?;
        let gpu_launch_us = t_gpu.elapsed().as_micros();

        let t_out = std::time::Instant::now();
        // Capture named outputs. A `kvcache_*` output is the new token's K/V —
        // written into slot `position` of the fixed cache so the next step
        // attends over it; everything else goes to the handoff store for
        // downstream segments / collectives.
        for name in &output_names {
            // KV outputs and collective tensors stay host (write_slot needs host
            // bytes; the collective read/write path is host). Every other
            // segment->segment activation output is kept DEVICE-RESIDENT: record
            // only the producer's output device buffer (ptr+size) and bind it into
            // the consumer next step — no get_tensor_by_name, no D2H.
            if let Some((kind, layer)) = parse_kvcache_name(name) {
                // Device-resident KV write (steady-state decode): DtoD-copy the
                // new token's K/V into slot `position` of the resident buffer.
                // No get_tensor (D2H), no host write_slot.
                if !self.prefill_capture {
                    if let Some(&base) = self.kv_device_base.get(name) {
                        if let Some((_, out_bytes)) = self.segments[segment_idx]
                            .runtime
                            .output_device_ptr_by_name(name)
                        {
                            let dest = base + (self.position * out_bytes) as u64;
                            unsafe {
                                self.segments[segment_idx]
                                    .runtime
                                    .copy_output_to_device_by_name(name, dest, out_bytes)
                            };
                            self.handoffs_device.remove(name);
                            continue;
                        }
                    }
                }
                // Host path: batched-prefill capture, or CPU fallback.
                let data = self.segments[segment_idx]
                    .runtime
                    .get_tensor_by_name(name)
                    .map_err(&err)?;
                crate::perf_counters::record_d2h(data.len() * std::mem::size_of::<f32>());
                if self.prefill_capture {
                    let _ = (kind, layer);
                    self.handoffs.insert(name.clone(), data);
                } else {
                    self.kv.write_slot(kind, layer, self.position, &data);
                }
                self.handoffs_device.remove(name);
            } else if self.host_tensors.contains(name) {
                let data = self.segments[segment_idx]
                    .runtime
                    .get_tensor_by_name(name)
                    .map_err(&err)?;
                crate::perf_counters::record_d2h(data.len() * std::mem::size_of::<f32>());
                self.handoffs.insert(name.clone(), data);
                self.handoffs_device.remove(name);
            } else if let Some((ptr, n_bytes)) = self.segments[segment_idx]
                .runtime
                .output_device_ptr_by_name(name)
            {
                self.handoffs_device.insert(
                    name.clone(),
                    DeviceTensorHandle {
                        device: 0,
                        ptr,
                        n_bytes,
                        elems: n_bytes / 2,
                        dtype: HandoffDtype::Bf16,
                    },
                );
                self.handoffs.remove(name);
            } else {
                // Fallback (CPU backend / no device buffer yet): host handoff.
                let data = self.segments[segment_idx]
                    .runtime
                    .get_tensor_by_name(name)
                    .map_err(&err)?;
                crate::perf_counters::record_d2h(data.len() * std::mem::size_of::<f32>());
                self.handoffs.insert(name.clone(), data);
            }
        }
        let host_out_us = t_out.elapsed().as_micros();
        tracing::info!(
            segment_idx,
            host_in_us,
            gpu_launch_us,
            host_out_us,
            "SKEIN_SEG"
        );
        Ok(())
    }

    fn read(&self, name: &str) -> Result<Vec<f32>, RankExecError> {
        self.handoffs
            .get(name)
            .cloned()
            .ok_or_else(|| RankExecError::UnknownTensor(name.to_string()))
    }

    fn write(&mut self, name: &str, data: Vec<f32>) -> Result<(), RankExecError> {
        self.handoffs.insert(name.to_string(), data);
        Ok(())
    }

    fn output_device_ptr(&self, name: &str) -> Option<(u64, usize)> {
        self.handoffs_device.get(name).map(|h| (h.ptr, h.elems))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skein_compile::{DynRuntime, DynRuntimeError};

    /// A mock `DynRuntime`: on `execute_segment` it doubles its `in` tensor
    /// into an `out` tensor. Lets us exercise the adapter's input-feed →
    /// execute → output-capture path with no GPU and no Luminal graph.
    #[derive(Default)]
    struct DoublingRuntime {
        tensors: HashMap<String, Vec<f32>>,
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
        struct TokenEmitter;
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
        }

        // CAP = 4 slots, per-token = 2 -> full buffer length 8.
        let mut kv_cache_sizes = HashMap::new();
        kv_cache_sizes.insert("kvcache_k_0".to_string(), 8usize);
        let seg = RuntimeSegment {
            runtime: Box::new(TokenEmitter),
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
                    }),
                    input_names: vec![],
                    output_names: vec!["x".to_string()],
                    capture_names: vec![],
                    weight_names: vec![],
                    kv_cache_sizes: HashMap::new(),
                };
                let runner = SegmentRunner::new(vec![seg]);
                let mut exec = RankExecutor::new(rank, runner);
                exec.run(&schedule, &coll).expect("rank run");
                exec.into_runner().read("x").expect("x present")
            }));
        }
        for (rank, j) in joins.into_iter().enumerate() {
            let x = j.join().expect("rank thread");
            assert_eq!(x, vec![3.0, 3.0], "rank {rank} got {x:?}");
        }
    }
}

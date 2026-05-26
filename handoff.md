# MB CUDA-graph capture — handoff

## Status

- **Test A** (`SKEIN_MB_CAPTURE=1 SKEIN_MB_CAPTURE_AT=9999` on prompts32): PASSES — correct capitals, ~133 tps, baseline-equivalent. Proves the immediate-input plumbing and the warmup path are clean.
- **Test B** (`SKEIN_CAPTURE=1 SKEIN_NO_SPLIT=1 SKEIN_MB_CAPTURE=1 SKEIN_MB_CAPTURE_AT=1` on prompts32): capture FIRES (8 graphs across 4 microbatches × 2 ranks), `idle_ms` drops to ~1 ms, `decode_tps` reads ~143–146, **but capitals are degenerate** ("Paris.,. being by by by…" or "Paris.梦梦梦梦…" depending on which stale-state path is the dominant one). Do not report this TPS — output is wrong.

Use **prompts32** for everything. prompts8 has a pre-existing baseline bug in the MB 1F1B loop when `n_mb=1` (reads `res[0..stage_mb]` before send_recv fills it).

## The likely-final remaining bug — dangling `kernelParams` in raw_launch

In `vendor/luminal/crates/luminal_cuda_lite/src/kernel/to_host.rs` the raw_launch path (the one that runs while `host::capture_active()` is true and the outer `cuStreamBeginCapture` is recording) used to do:

```rust
let mut params = UnifiedKernelParams::new(pv);   // local — drops at iteration end
let params_ptr = params.as_cuda_params();
cuLaunchKernel(..., params_ptr, ...);             // recorded into outer graph
```

`cuLaunchKernel` during stream capture records the `kernelParams` pointer; the pointed-to buffer must remain valid through every replay. The inner-graph path (`update_kernel_node`, ~line 800) already persists params in `state.kernel_params[idx]`. Raw_launch didn't — every recorded kernel node ended up reading freed memory on replay, which is the "model emits one repeated token forever" / "Paris.梦梦梦梦…" signature.

**Half-applied fix is currently in the working tree but DOES NOT BUILD** — I left it partway through a borrow-checker rewrite. Symptom:

```
error[E0502]: cannot borrow `state` as mutable because it is also borrowed as immutable
  --> vendor/luminal/crates/luminal_cuda_lite/src/kernel/to_host.rs:677:17
```

The line `let kernel = &state.kernels[idx];` (immutable borrow of `state`) collides with the new `state.kernel_params[idx] = …` (mutable borrow). My last edit moved the kernel borrow into an inner scope and tried to return a tuple including `kernel.kernel_name` (which is `&'static str` on `CudaGraphOp`'s kernel — confirm the type when finishing), but the file currently has the partial edit and won't compile.

## To finish the fix

1. In the raw_launch loop in `to_host.rs` (around line 633–688), extract every value that depends on `&state.kernels[idx]` inside a `{ … }` block, return them as owned values from the block, then assign `state.kernel_params[idx] = UnifiedKernelParams::new(pv)` and read `params_ptr = state.kernel_params[idx].as_cuda_params()` after the block ends (immutable borrow dropped). Also resize `state.kernel_params` to `num_kernels` once before the loop, mirroring the inner-graph path.
   - The current file already has the `if state.kernel_params.len() != num_kernels { state.kernel_params.resize_with(...) }` block in place.
   - The `let (..., kernel_name): (...) = { let kernel = &state.kernels[idx]; ...; (output_ptr, input_ptrs, grid, block, shared, kdp, pv, cu_func, kernel.kernel_name) };` block compiles in principle but check the precise types (the partial tuple may have a wrong element type — drop unused destructured names if borrowck still complains).
   - Then in the launch / kt_on / graph_kt_before/after blocks below, use `grid.0/1/2`, `block.0/1/2`, `shared`, `cu_func`, `kernel_name` from the locals instead of indexing back into `state.kernels[idx]`.

2. `cargo build --release -p skein_cli` from `/home/ubuntu/Skein` after `source /home/ubuntu/skein_env.sh`. Verify the new strings still land in the binary: `strings target/release/skein | grep -E "PROFILE_RESULT|MB_CAPTURE rank"`.

3. **Test A** (must stay correct):
   ```
   cd /home/ubuntu/Skein
   export SKEIN_MOE_GU_HOPPER=1 SKEIN_MOE_DN_HOPPER=1
   ART=/tmp/art_h100_pp2_mb8/2f1ef78dd928e50040a540954bfa88ab13739459605b6d3761b38f6fcab6ac45
   SKEIN_MB_CAPTURE=1 SKEIN_MB_CAPTURE_AT=9999 \
     bash /home/ubuntu/run_serve_param.sh "$ART" 8 /tmp/prompts32.json 2 fused_decode 1 > /tmp/A.log 2>&1
   grep -E "prompt[0-7] first_token|MB_FINAL" /tmp/A.log
   ```
   Expected: Paris / Rome / Madrid / Tokyo / Berlin / Ottawa / Canberra / Brasilia all coherent; ~133 tps.

4. **Test B** (capture fires; this is what the fix unblocks):
   ```
   SKEIN_CAPTURE=1 SKEIN_NO_SPLIT=1 SKEIN_MB_CAPTURE=1 SKEIN_MB_CAPTURE_AT=1 \
     bash /home/ubuntu/run_serve_param.sh "$ART" 8 /tmp/prompts32.json 2 fused_decode 1 > /tmp/B.log 2>&1
   grep -E "MB_CAPTURE rank|prompt[0-7] first_token|MB_FINAL|MB_BREAKDOWN|PROFILE_RESULT" /tmp/B.log
   ```
   Expected: 8 `MB_CAPTURE rank … CAPTURED key=…` lines; all 8 capitals coherent; `PROFILE_RESULT … idle_ms ≈ 1.x decode_tps > 140`.

5. If Test B is still degenerate after the kernelParams fix, the next checklist item to chase is the **inner-op `pre_execute`** at to_host.rs ~583 — it runs every call (inside the captured region), and for `MegakernelOp` resets work-queue / barriers via host→device writes; if any of those go through a sync H2D, they'd be recorded with a now-freed source. Strategy: same as kernel params — make any per-step host source it writes from a stable struct field.

## What's already in place (do NOT redo)

- Vendored luminal multi-graph capture: `captured_graph_execs: HashMap<u64, CUgraphExec>` in `CudaComputeRuntime`, keyed `end_stream_capture_keyed` / `replay_captured_keyed` / `has_captured_graph_keyed`. Threaded through `ComputeRuntime` → `DynRuntime` → `SegmentRunner`. See `runtime.rs`, `crates/skein_compile/src/{lib.rs,dyn_runtime.rs}`, `crates/skein_runtime/src/distributed/segment_runner.rs`.
- Capture-stream gate: `SKEIN_MB_CAPTURE` and/or `SKEIN_CAPTURE` selects the non-default capturable stream in `runtime.rs:273`.
- Thread-local `host::capture_active()` flag, set/cleared by `begin_stream_capture`(_keyed) and `end_stream_capture`(_keyed). Authoritative for `THREAD_LOCAL`-mode capture detection — `cuStreamGetCaptureInfo` on a non-capture-stream returns NONE even when capture is active thread-wide.
- `CudaGraphOp::execute_internal` guards: skip `needs_internal_realloc`, the `cuda_graph` reset, the dyn_dims `memcpy_htod`, and `build_graph` when `capture_active()` is true (warmup did them; the recorded raw launches handle the rest).
- `CudaGraphOp::refresh_capture_dyn_dims(stream, dyn_map)` — overrides `HostOp` default. Merges caller's `dyn_map` over `state.last_dyn_values` so static dims (`s`, batch, etc.) stay at warmup values and only the changing dim (`p`) is overridden. Called from `flush_step_device_inputs` (segment_runner.rs ~688) which builds a `HashMap{p: position}` and walks every segment's runtime.
- raw_launch gate: `std::env::var_os("SKEIN_RAW_LAUNCH").is_some() || crate::host::capture_active()` — only fires while outer capture is recording, so warmup steps still use the proven inner-graph replay.
- `is_capture()` honors `SKEIN_CAPTURE` ONLY (not `SKEIN_MB_CAPTURE`). To get the persistent-inputs + no-free behaviors you need `SKEIN_CAPTURE=1 SKEIN_NO_SPLIT=1` exported alongside `SKEIN_MB_CAPTURE=1`. NO_SPLIT suppresses the conflicting single-stream prefill capture in `forward_step_tokens`.
- MB loop in `crates/skein_runtime/src/distributed/gpu_rank.rs run_continuous_pipelined_mb`:
  - `let do_capture_step = mb_capture && k >= capture_at;` — warmup steps (k<capture_at) take the staged path EXACTLY like baseline (this is why Test A passes).
  - Capture step: `set_capturing(true)` + `flush_step_device_inputs` (writes input_tokens, position, decode_position, AND now dyn_dims for each `CudaGraphOp`) + `begin_capture` + `executor.run(&compute)` + `end_capture_keyed(key)` + `replay_captured_keyed(key)`.
  - Replay step: same `set_capturing(true)` + flush + `replay_captured_keyed(key)` (no executor.run).
  - Stage 0 carry read: `read_device_handoff(carry_id)` under capture; baseline `read_by_id` otherwise.
  - Stage 1 carry feed: `feed_capture_input_f32(carry_id, buf)` under capture (writes to the persistent device buffer; leaves `f32_slots[carry] = None` so `run_segment` records no H2D into the graph); baseline `write_by_id` otherwise.
  - Stage 1 logits: `read_device_handoff_by_name(LOGITS)` under capture; baseline `read(LOGITS)` otherwise.
  - PROFILE_RESULT line emitted at end of FINAL block when `mb_capture` is on.
- `HostCollective` output device-resident under capture is **targeted** — only for ids registered in `capture_passthrough_outputs` (registered at top of `run_continuous_pipelined_mb`: `set_capture_passthrough_by_names(&[LOGITS]); add_capture_passthrough_id(carry_id);`). MoE router logits (also `HostCollective`) are NOT diverted, so on-device top-k still works.
- Hopper MoE flags MUST be `export`ed in the parent shell before invoking `run_serve_param.sh`. The script does not set them.

## Files touched

- `vendor/luminal/crates/luminal_cuda_lite/src/runtime.rs` — capture-stream gate, keyed multi-graph fields/methods, `set_capture_active`, `refresh_capture_dyn_dims` walker.
- `vendor/luminal/crates/luminal_cuda_lite/src/host/mod.rs` — `is_capture()`, `capture_active()`/`set_capture_active()`, `HostOp::refresh_capture_dyn_dims` default.
- `vendor/luminal/crates/luminal_cuda_lite/src/kernel/to_host.rs` — capture guards in `execute_internal`, raw_launch gating on `capture_active()`, `CudaGraphOp::refresh_capture_dyn_dims` override, **half-applied persistent kernel_params fix (currently fails to compile).**
- `crates/skein_compile/src/lib.rs` — `ComputeRuntime` trait additions + CUDA impl forwards.
- `crates/skein_compile/src/dyn_runtime.rs` — `DynRuntime` trait additions + wrapper forwards.
- `crates/skein_runtime/src/distributed/segment_runner.rs` — keyed capture pass-throughs, `read_device_handoff_by_name`, `feed_capture_input_f32`, `capture_passthrough_outputs` field/setters, `flush_step_device_inputs` calls `refresh_capture_dyn_dims`.
- `crates/skein_runtime/src/distributed/gpu_rank.rs` — MB loop rewritten with `do_capture_step` split for both stages, PROFILE_RESULT emission.

## Baselines

- mb=8 / prompts32 / fp8 / hopper MoE / no capture: **133.78 tps**, per_forward 59.5 ms, ~31 ms inter-kernel idle.
- Target: idle <10 ms, decode_tps >160, strong >180.

## Reproducer commands

```bash
# baseline (no capture)
export SKEIN_MOE_GU_HOPPER=1 SKEIN_MOE_DN_HOPPER=1
ART=/tmp/art_h100_pp2_mb8/2f1ef78dd928e50040a540954bfa88ab13739459605b6d3761b38f6fcab6ac45
bash /home/ubuntu/run_serve_param.sh "$ART" 8 /tmp/prompts32.json 2 fused_decode 1 > /tmp/baseline.log 2>&1

# Test A (capture never fires)
SKEIN_MB_CAPTURE=1 SKEIN_MB_CAPTURE_AT=9999 bash /home/ubuntu/run_serve_param.sh "$ART" 8 /tmp/prompts32.json 2 fused_decode 1 > /tmp/A.log 2>&1

# Test B (capture fires)
SKEIN_CAPTURE=1 SKEIN_NO_SPLIT=1 SKEIN_MB_CAPTURE=1 SKEIN_MB_CAPTURE_AT=1 \
  bash /home/ubuntu/run_serve_param.sh "$ART" 8 /tmp/prompts32.json 2 fused_decode 1 > /tmp/B.log 2>&1
```

Build: `cd /home/ubuntu/Skein && source /home/ubuntu/skein_env.sh && cargo build --release -p skein_cli` (binary is named `skein` from package `skein_cli`).

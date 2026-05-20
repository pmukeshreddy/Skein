# Architecture diagram corrections

The architecture flowchart (referenced in the README and external presentations)
has three labels that drift from the implemented code. When the diagram is
regenerated, apply these corrections.

## 1. Mixtral 8x7B node count

**Diagram says:** `Mixtral 8x7B: 387 nodes`
**Code reality:** `Graph::layers.len() == 131` for Mixtral 8x7B
  (1 embedding + 4 x 32 decoder block layers + 1 final norm + 1 lm_head)

The 387 count corresponds to an op-level expansion where each MoE block is
expanded into router + 8 experts + combine. The IR keeps MoE as a single
Layer kind. Both views describe the same model.

**Correction:** Either
- Change `387 nodes` -> `131 layers` to match `graph.layers.len()`, or
- Keep `387 nodes` with the caption "op-level expanded (MoE blocks unrolled)"

## 2. Parity gate label

**Diagram says:** `layer-by-layer logit MSE vs external HF reference`
**Code reality:** Parity measures Skein-vs-Skein (Skein-at-bf16 reference
vs Skein-at-candidate-dtype), per the AWQ/GPTQ industry standard documented
in `docs/parity.md`.

**Correction:** `HF reference` -> `Skein-bf16 reference`

## 3. Per-device compile API

**Diagram says:** `graph.compile(CudaCompiler::<d>)` -> optimized CUDA
**Code reality:** Pinned Luminal (rev e558ce68) uses a search-based API:

```rust
cx.build_search_space::<CudaRuntime>();
let runtime = cx.search(CudaRuntime::new()?, budget);
```

**Correction:** Replace `graph.compile(CudaCompiler::<d>)` with
`cx.build_search_space::<CudaRuntime>(); cx.search(CudaRuntime::new(), budget)`

## Regeneration owner

When the diagram is regenerated, this file should be deleted or moved to
`docs/historical/` with a note about which version of the diagram these
corrections applied to.

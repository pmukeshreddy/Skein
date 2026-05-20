# Luminal integration

How Skein talks to [Luminal](https://github.com/luminal-ai/luminal) for
kernel optimization. This document is the cheat sheet to read before adding
a new op to `skein_emit::op_wiring` or a new backend to `skein_compile`. It
is **not** a Luminal tutorial — read the upstream README for that. This file
pins what Skein assumes about the Luminal API at our pinned rev.

## Pinned rev

```
luminal = git rev e558ce68498df842107ccb32e316efff4801ccc1
```

Set in:
- `Cargo.toml` (workspace) for `luminal`.
- `crates/skein_compile/Cargo.toml` for `luminal_cuda_lite` (must match).

When bumping, re-verify everything in this document against the new rev.
The Luminal HLIR surface is **not** stable; methods come and go between
revs. The "Op-availability cheat sheet" below was hand-audited at the
pinned rev.

## `ComputeRuntime` — the backend boundary

`skein_compile::ComputeRuntime` is the trait Skein code targets when it
needs Luminal to compile and execute a graph. Two impls today:

| impl                    | Available             | Backed by                          |
|-------------------------|-----------------------|------------------------------------|
| `CudaComputeRuntime`    | `--features cuda` (default) | `luminal_cuda_lite::CudaRuntime`   |
| `NativeComputeRuntime`  | Always                | `luminal::prelude::NativeRuntime`  |

Additional backends (Metal / ROCm) drop in as new impls without changing
call sites — `compile_with_luminal::<R>` selects the backend at the call
site.

Trait surface (`crates/skein_compile/src/lib.rs`):

```rust
pub trait ComputeRuntime: Sized {
    fn build_and_search(cx: &mut Graph, budget: usize) -> Result<Self, CompileError>;
    fn set_data_f32(&mut self, id: NodeIndex, data: Vec<f32>);
    fn execute(&mut self, cx: &Graph);
    fn get_data_f32(&self, id: NodeIndex) -> Vec<f32>;
}
```

The `f32` data path is what every Skein test uses.
TODO(weight-bytes): production weight loading should go through a
`set_data_bytes` entry point (not yet on the trait).

## Op-availability cheat sheet (pinned rev)

These are the `GraphTensor` methods `skein_emit::op_wiring` relies on,
each verified by reading `luminal/src/frontend/` at the pinned rev:

| Op                                       | Where                                       |
|------------------------------------------|---------------------------------------------|
| `matmul`                                 | `frontend/matmul.rs`                        |
| `permute`, `transpose`, `t`              | `frontend/movement.rs`                      |
| `expand_dim`, `expand_lhs`, `expand_rhs` | `frontend/movement.rs`                      |
| `split_dims`, `merge_dims`, `flatten`    | `frontend/movement.rs`                      |
| `unsqueeze`, `squeeze`                   | `frontend/movement.rs`                      |
| `slice`                                  | `frontend/movement.rs`                      |
| `softmax(axis)`                          | `frontend/unary.rs`                         |
| `silu`                                   | `frontend/unary.rs`                         |
| `std_norm(axis, eps)`                    | `frontend/unary.rs` — **is RmsNorm**        |
| `gather`                                 | `frontend/movement.rs` via `hlir::Gather`   |
| `cast(DType)`                            | `frontend/unary.rs`                         |
| `as_dtype(DType)`                        | `frontend/unary.rs`                         |
| `topk_indexes(k, axis)`                  | `frontend/reduction.rs`                     |
| `output()`                               | `hlir.rs`                                   |

### Gotchas verified at this rev

- **No `reshape` method.** The commented-out `luminal_nn::Embedding`
  references one but it's been removed. To re-view a tensor under a new
  shape, mutate the `GraphTensor`'s `ShapeTracker` directly:
  `t.shape = ShapeTracker::new(new_dims);`. This is a no-copy view —
  the underlying op data is unchanged.
- **Element-wise mul does NOT auto-broadcast.** `a * b` panics if shapes
  differ. Use `b.shape.expand(a.dims())` to grow a size-1 dim before
  multiplying (see `wire_moe` in `op_wiring.rs`).
- **`std_norm` is RmsNorm, not LayerNorm.** Despite the name, the
  formula is `x * 1/sqrt(mean(x²) + eps)` — no mean subtraction. This
  matches Mixtral's `RmsNorm` semantics exactly.
- **`ShapeTracker` is `pub` only via the re-export.** Use
  `luminal::prelude::ShapeTracker` (re-exported via `pub use tracker::*`
  in `shape/mod.rs`); the `luminal::shape::tracker` module itself is
  private.
- **`luminal_nn::Embedding` is fully commented out** at this rev — we
  hand-build the gather-based embedding lookup in `op_wiring.rs`.
- **`luminal_nn::MoE`** is a plain dense-matmul MoE with top-k routing;
  Mixtral SwiGLU MoE needs a hand-built block (`wire_moe`).

## Deferred op lowering (TODO)

The current op wiring is intentionally a *subset* of full Mixtral
semantics. Each gap is tracked by a `TODO(...)` tag in
`skein_emit::op_wiring` (and the others noted below); a grep for `TODO(`
across `skein_emit`/`skein_runtime` finds every deferred piece.

| Gap                  | Effect on output                                                   | Tag                  |
|----------------------|-------------------------------------------------------------------|----------------------|
| RoPE on Q/K          | No positional encoding — attention is position-blind.             | `TODO(rope)`         |
| Causal mask          | Softmax sees future positions during prefill (`seq > 1`).         | `TODO(causal-mask)`  |
| Top-k routing in MoE | Dense mixture over all experts; Mixtral picks top-2 only. Hook `topk_indexes` + `gather`. | `TODO(moe-topk)`     |
| EP token routing     | Dispatch/combine are wired structurally but tokens aren't routed. | `TODO(ep-routing)`   |
| Real KV cache        | The parity forward pass is stateless — no KV reuse.               | runtime KV work      |

## How to add a new backend

The pattern `CudaComputeRuntime` follows:

1. Add the upstream crate as an optional `cargo` dep behind a feature.
   Pin to the same Luminal rev.
2. Implement `ComputeRuntime` for the new wrapper, in its own
   `#[cfg(feature = "<flag>")] mod` inside `skein_compile`.
3. Add a `#[cfg(feature = "<flag>")] #[test]` that compile-checks
   `compile_with_luminal::<NewRuntime>` — execution tests are
   GPU-conditional.
4. Re-export the new type from `skein_compile`'s public surface so the
   CLI can pick it via `compile_with_luminal::<R>` at runtime.

Metal and ROCm follow this exact shape.

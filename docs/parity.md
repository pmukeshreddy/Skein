# Parity verification

Parity in Skein measures **Skein-at-bf16** against
**Skein-at-candidate-dtype**, layer-by-layer activation MSE plus final-logit
KL divergence. Both sides of the comparison go through Skein's own compile
path; bf16 is the reference, the search-picked dtype map is the candidate.

This matches the industry standard for post-training quantization validation.
AWQ measures AWQ-FP4 vs AWQ-FP16; GPTQ measures GPTQ-INT4 vs GPTQ-FP16;
NVIDIA Model Optimizer follows the same pattern. The reference engine and the
candidate engine are the same engine at different precisions.

## Why Skein-vs-Skein, not Skein-vs-HF

Comparing Skein's quantized output to an HF reference conflates two sources
of difference: (a) the dtype change that Skein's search introduced, and
(b) any differences between Skein's compile path and HF's eager execution.
Only (a) is what the cost model needs to predict. Skein-vs-Skein isolates
(a) cleanly.

The HF reference path remains available for a separate use case:
`skein verify --reference <hf-model>` validates that Skein's bf16 compile
output matches a known-good external implementation. This is run once per
model architecture during integration, not per-compile.

## Two-tier accuracy protection

1. **Predicted drift in the DP** (`skein_extract`). Every per-layer dtype
   combo carries a predicted KL-divergence contribution from
   `models/<model>_drift.toml`. The DP rejects any per-block dtype
   assignment whose cumulative drift exceeds `workload.slo.max_accuracy_drift`.
2. **Measured drift in the parity gate** (`skein_parity`). Predictions are
   not measurements. After compile, the gate runs Skein-bf16 and the
   candidate Skein artifact on sample prompts and checks whether the
   *measured* drift agrees with the prediction.

A Plan that passes the DP but fails the parity gate signals that the drift
table's prediction was too optimistic for the failing
`(layer, component, dtype)` triple. We update the table upward and
re-search. Over time the drift table converges to a calibrated state.

## Why real workload prompts, not synthetic

Drift behaviour is input-distribution-dependent. A model that quantizes
cleanly on short chat prompts may drift outside SLO on long-context RAG
prompts. The parity gate samples from public calibration prompts or a
user-supplied prompt corpus so verification matches the target distribution.

## External reference dtype is independent of deployment dtype

The Python external reference subprocess is verify-only and takes a string
dtype (`"bfloat16"`, `"float16"`, or `"float32"`). This is intentionally not
`skein_ir::types::Dtype`: Skein's deployment dtype enum models production
weight / activation / KV choices (`bf16`, `fp16`, `fp8`, `int8`, `int4`)
that feed cost, drift, and calibration tables.

The external reference dtype answers a different question: what precision
should the known-good external model use while validating Skein-bf16? A
CPU-only check can use `float32` for GPT-2; GPU integration usually uses
`bfloat16` for Mixtral.

## Python subprocess protocol

`PythonSubprocessReference` invokes
`crates/skein_parity/reference/verify_reference.py` with explicit piped
stdin, stdout, and stderr. Rust sends one batched JSON request:

```
{"prompts":[{"tokens":[...]}],"model_path":"...","reference_dtype":"float32"}
```

The script writes one JSON object per line:

```
{"prompt_idx":0,"per_layer_activations":[[...]],"final_logits":[...]}
```

`--tokenize-only` accepts prompt text entries and returns
`{"prompt_idx":N,"tokens":[...]}` lines. `--check-env` imports
`transformers`, `torch`, and `numpy` and exits 0 only when the reference
environment is usable. Subprocess stderr is always captured and surfaced in
`ParityError::PythonSubprocessFailed` or the timeout error.

## Activation hook convention

The external reference for `skein verify --reference` registers hooks on
decoder blocks and captures the block output hidden state: post-block
residual, before the next block's norm. The Skein side captures the matching
Luminal node named
`hidden_after_block_N`. This is equivalent to the `block_N_ffn_out` handoff
after the final residual for segmented plans, and keeps parity comparisons
on the same semantic boundary.

## Shared executor

`RealSkeinForward::load_native` reads a reloadable `SkeinArtifact`, rebuilds
segments from recipes, compiles them with `NativeComputeRuntime`, loads the
device safetensors shard, and executes through
`skein_compile::executor::TopologyExecutor`. `Server::serve` uses the same
executor inside its forward worker, so parity and live serving walk the same
`SequenceStep` schedule.

For the external verification workflow, install the reference requirements
into a venv and use `PythonSubprocessReference::new("gpt2", "float32")`. For
the production parity gate, compile Mixtral through Skein at bf16 and at the
candidate dtype map, then compare those two artifacts with
`verify_skein_pair` (CUDA build on the target GPU).

## Drift-table update protocol — monotonic up

`drift_update::update_drift_table_on_failure` enforces a never-decrease
contract:

```
new_value = max(existing_value, measured_drift)
```

Rationale:

- `skein_extract` only excludes Plans whose predicted drift exceeds SLO.
- A failing Plan was admitted because its predicted drift was below SLO.
- We must raise the predicted value above the measured value to exclude the
  Plan next time.
- Decreasing the value would let a future search pick the losing Plan
  back. Hence: never decrease.

The TOML serializer (`DriftTable::save_to_toml_file`) sorts per-layer keys
by `(layer_idx, component, dtype)` and uses `{v:?}` for floats so the
on-disk file is byte-stable across runs.

## Re-search trigger flow

```
compile artifact A
  ├── extract_plan picks Plan_A using drift_table
  ├── lower_per_device produces artifacts/<plan_hash_A>/
  └── verify_plan compares against Skein-bf16 reference
       │
       ├── passed: true → write parity_report.json, serve
       │
       └── passed: false
             ├── update_drift_table_on_failure (monotonic up)
             ├── re-invoke extract_plan
             │     └── now Plan_A is predicted as drift-violating
             │       → search picks Plan_B
             ├── compile artifact B
             └── verify_plan loop continues until pass
```

The re-search is bounded by the cost model: there are only finitely many
outer × DP combinations. In the worst case the loop terminates when all
non-bf16 combos have been measured and rejected, leaving the all-bf16
fallback (zero drift by definition).

## KL divergence — log-softmax with the max-shift trick

Both `p_logits` and `q_logits` are *unnormalized* logits, not probability
mass functions. Naively computing `log(p / q)` after a normal softmax
would underflow / overflow for any logit outside roughly `[-80, 80]`. We
instead:

```
log_softmax(x) = (x - max(x)) - log_sum_exp(x - max(x))
KL(p || q)     = Σ p_i · (log_p_i - log_q_i)
               = Σ exp(log_p_i) · (log_p_i - log_q_i)
```

This is stable for logits in `[-1e6, 1e6]` (Skein's test bound). Tiny
negative roundoff (e.g. `-1e-16`) is clamped to zero so callers never see
`KL < 0`.

## CPU vs CUDA build

| Concern                              | CPU build (`--no-default-features`)  | CUDA build (default)              |
|--------------------------------------|--------------------------------------|-----------------------------------|
| `mse` / `kl_divergence` math         | ✅ pure f32/f64                       | ✅                                |
| `ToleranceTable`                     | ✅ from `cost_constants.toml`         | ✅                                |
| `ParityReport` serde JSON            | ✅                                    | ✅                                |
| `PythonSubprocessReference::new`     | ✅ validates reference dtype + paths  | ✅                                |
| `verify_reference.py` subprocess     | ✅ CPU/GPT-2 capable                  | ✅                                |
| Real Skein forward via Luminal       | ✅ `NativeComputeRuntime`             | ✅ `CudaComputeRuntime`           |
| `drift_update` monotonic write       | ✅                                    | ✅                                |

Both builds run real implementations. The CPU build uses
`NativeComputeRuntime` + the in-process collective; the CUDA build adds the
GPU compute runtime and CUDA-specific runtime modules. Neither build fakes
hardware behavior.

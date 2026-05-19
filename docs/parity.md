# Parity — `skein_parity`

The parity gate sits *after* compile and *before* serve. It compares the
Skein artifact's per-layer activations + final logits against the HF
reference dtype selected for verification, decides pass/fail, and on
failure records the offending measurement to the drift table so the next
`skein_extract` run picks a different Plan.

## Two-tier accuracy protection

1. **Predicted drift in the DP** (`skein_extract`). Every per-layer dtype
   combo carries a predicted KL-divergence contribution from
   `models/<model>_drift.toml`. The DP rejects any per-block dtype
   assignment whose cumulative drift exceeds `workload.slo.max_accuracy_drift`.
2. **Measured drift in the parity gate** (`skein_parity`). Predictions are
   not measurements. After compile, the gate runs the real artifact against
   the HF reference on sample prompts and checks whether the *measured*
   drift agrees with the prediction.

A Plan that passes the DP but fails the parity gate signals that the drift
table's prediction was too optimistic for the failing
`(layer, component, dtype)` triple. We update the table upward and
re-search. Over time the drift table converges to a calibrated state.

## Why real workload prompts, not synthetic

Drift behaviour is input-distribution-dependent. A model that quantizes
cleanly on short chat prompts may drift outside SLO on long-context RAG
prompts. The parity gate samples from `workload.requests` (the same trace
the cost model and search used) so the verification matches the production
distribution.

## Reference dtype is independent of deployment dtype

The HuggingFace reference subprocess takes a string dtype (`"bfloat16"`,
`"float16"`, or `"float32"`). This is intentionally not
`skein_ir::types::Dtype`: Skein's deployment dtype enum models production
weight / activation / KV choices (`bf16`, `fp16`, `fp8`, `int8`, `int4`)
that feed cost, drift, and calibration tables.

The reference dtype answers a different question: what precision should the
ground-truth model use while parity compares final logits and hook
activations? Mac development can use `float32` for CPU-friendly GPT-2 checks;
H100 verification usually uses `bfloat16` for Mixtral.

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

The HF reference registers hooks on decoder blocks and captures the block
output hidden state: post-block residual, before the next block's norm. The
Skein side captures the matching Luminal node named
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

Mac workflow: install the reference requirements into a venv, use
`PythonSubprocessReference::new("gpt2", "float32")` for the HF side, and use
the tiny-artifact fixture for native Skein execution. H100 workflow: use the
Mixtral checkpoint path and `bfloat16`, then swap the native runtime and
mock collective for CUDA/NCCL composition under the CUDA feature.

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
  └── verify_plan compares against HF reference
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

## Phase A vs Phase B feature matrix

| Concern                              | Phase A (Mac)                                 | Phase B (H100, `--features cuda`) |
|--------------------------------------|-----------------------------------------------|-----------------------------------|
| `mse` / `kl_divergence` math         | ✅ pure f32/f64                                | ✅                                |
| `ToleranceTable`                     | ✅ from `cost_constants.toml`                  | ✅                                |
| `ParityReport` serde JSON            | ✅                                             | ✅                                |
| `MockReference` (fixture loader)     | ✅                                             | ✅ (kept for unit tests)          |
| `PhaseAStub` (drift simulator)       | ✅                                             | ✅ (kept for unit tests)          |
| `PythonSubprocessReference::new`     | ✅ validates reference dtype + paths           | ✅                                |
| `verify_reference.py` subprocess     | ✅ CPU/GPT-2 capable                           | ✅                                |
| Real Skein forward via Luminal       | ✅ native tiny artifacts                       | ✅                                |
| `drift_update` monotonic write       | ✅ (exercised against synthetic drift table)   | ✅                                |

New Prompt-2 code paths run real implementations on Mac where a native
composition exists. CUDA-specific modules remain feature-gated; the Mac path
does not fake CUDA-only hardware behavior.

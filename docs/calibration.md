# Calibration — `skein_calibrate`

Offline tool that produces two files from a real measurement run:

- `cluster/cost_constants.toml` — calibrated per-hardware efficiency
  constants plus all the other tunables `skein_cost` and `skein_runtime`
  consume.
- `models/<model>_drift.toml` — calibrated per-`(layer, component, dtype)`
  drift table the DP in `skein_extract` uses.

The pipeline has two halves: the data plumbing (corpus loader, statistical
aggregation, deterministic TOML writers, the `CalibrationReport` audit
struct) and the samplers that produce `KernelMeasurement` and
`DriftMeasurement` rows. The samplers are generic over the compute runtime:
the CPU `NativeComputeRuntime` yields valid pipeline checks (not production
constants), while `CudaComputeRuntime` on the target GPU yields the
production numbers.

## Calibration philosophy

Calibrate once per `(hardware, model)` pair, reuse across compiles. A
run on an H100 SXM5 with Mixtral 8x7B writes constants that the cost
model and the DP consult for every subsequent `skein compile` on that
hardware/model combo. Re-calibration is incremental — running with a
partial corpus updates only what was measured and preserves everything
else.

## Corpus design

`crates/skein_calibrate/corpus/<model>.toml` ships per supported model.

### Kernel samples — what to sample

The corpus walks the model's actual op mix:

- **GEMM at attention shapes.** Q/K/V/O at `[num_heads × head_dim, hidden]`.
  GQA models add `[num_kv_heads × head_dim, hidden]` for K/V.
- **GEMM at MLP shapes.** w1/w3 at `[intermediate, hidden]`, w2 at
  `[hidden, intermediate]`. (For MoE the same shapes, multiplied by
  the per-expert footprint.)
- **Attention at workload shapes.** `[batch, num_heads, seq_len,
  head_dim]` for representative `(batch, seq_len)` pairs from the
  workload trace.
- **Elementwise.** RmsNorm at hidden-state shape.

`repeats` per sample: 50-100 is enough to make the median stable; the
shipped Mixtral corpus uses 100 for GEMMs and 50 for the attention /
elementwise rows.

### Drift prompts — public corpus by default

Calibration prompts come from a public calibration corpus by default, with
operator override at runtime. This follows the AWQ/GPTQ/ModelOpt pattern:
use a fixed, documented text calibration set for quantization drift, not the
token-count-only serving workload trace.

Skein ships `crates/skein_calibrate/corpus/drift_prompts.jsonl`, a small
development corpus derived from Lewis Carroll's *Alice's Adventures in
Wonderland*, Project Gutenberg EBook #11. The work is public domain in the
United States. The fixture is for development and CI plumbing; production
calibration should pass `--drift-prompts-path` with the operator's chosen
C4/WikiText/Pile-style corpus.

Instead the corpus carries a `DriftPromptSource`:

```toml
workload_trace_path = "traces/sharegpt_2000.jsonl"
n_drift_prompts     = 50
sampling_strategy   = "stratified_by_length"   # uniform_random | first_n | stratified_by_length
sampling_seed       = 0
```

`prompt_sampling::sample_drift_prompts` reads JSONL with one
`{"prompt": "..."}` object per line, applies the configured strategy, and
returns the sampled prompt list. `sampler::sample_drift` takes that list and
measures Skein-vs-Skein drift: bf16 artifact as reference, candidate artifact
as the measured path.

The three strategies:

| Strategy             | Behaviour                                                                 |
|----------------------|---------------------------------------------------------------------------|
| `first_n`            | Trace order, first `n` prompts. Useful for fixed-position smoke tests.    |
| `uniform_random`     | Deterministic shuffle via BLAKE3 of `(seed, index)`, take first `n`.      |
| `stratified_by_length` | Sort by byte length, slice into `n` equally-ranked buckets, pick the midpoint. Spans short → long. |

Determinism is part of the contract: same `(trace, n, strategy, seed)` →
byte-identical sampled list. Drift calibration depends on this so
re-runs don't introduce noise the parity gate then chases.

**Prompt file location.** The Mixtral corpus carries a `workload_trace_path`
field. The CLI can override it with `--drift-prompts-path`; if the corpus
points at the (non-bundled) ShareGPT trace, the CLI substitutes the bundled
public-domain prompt corpus.

**When to re-sample.** Drift behaviour is input-distribution-sensitive
— a model that quantizes cleanly on short chat prompts can drift
outside SLO on long-context RAG prompts. If the workload trace shifts
(Loop 3 in the feedback diagram fires a re-calibration), the
drift-prompt sample changes automatically the next time you re-run
`skein calibrate` against the updated trace. The corpus stays stable.

### Kernel samples stay enumerated

Architecture facts — Mixtral's 4096-wide GEMMs, GQA-flavoured K/V
projections, MoE expert FFN shapes — are model structure, not
workload-derived data. They stay enumerated in the corpus's
`[[kernel_samples]]` entries.

## Why median for efficiency, p95 for drift

**Median for efficiency.** A kernel's measured runtime has heavy tails
from warmup, page faults, kernel launches, and so on. The median is the
"typical sustained" perf the cost model wants to predict — the wall
time the DP scores Plans against. Outliers shouldn't make us pessimistic.

**P95 (linearly interpolated) for drift.** The drift table predicts
worst-case behavior. The DP rejects per-block dtype combos whose
predicted drift exceeds SLO; the parity gate enforces measured drift
under SLO. If we picked the *median* drift, the DP would admit Plans
that fail parity 50% of the time. P95 picks something the search can
trust as an upper bound for almost every prompt — matching the
"monotonic up" protocol from `skein_parity::drift_update`. Documented
formula:

```
sort ascending
pos     = 0.95 × (n - 1)
lo, hi  = floor(pos), min(lo + 1, n - 1)
result  = sorted[lo] × (1 - frac) + sorted[hi] × frac   where frac = pos - lo
```

This is the NumPy / "Type 7" definition. Linear interpolation rather
than nearest-rank so a small corpus doesn't snap to the nearest sample.

## Why preserving unmeasured fields matters

A practical calibration run may cover only the (op, dtype) pairs the
operator cares about — say, just `(Gemm, Bf16)` and `(Gemm, Fp8E4m3)`.
The writer must leave `[efficiency.attention]` and
`[efficiency.elementwise]` untouched in that case. Otherwise:

- Re-running calibration with a partial corpus would *delete* prior
  measurements, regressing the DP's quality.
- The on-disk file would contain zeros or default-derived values for cells
  that already have known, calibrated values from a previous run.

The cost-constants writer reads `base: &CostConstants` and overlays
fitted values from a `HashMap<(OpKind, Dtype), f64>`. Sections /
dtypes absent from `fitted` come straight from `base`. The drift
writer follows the same pattern: per-layer overrides not present in
`fitted` are preserved from the base table.

## Recalibration cadence

- **When Luminal updates.** New kernel-search heuristics or fused ops
  change the achieved efficiency. Re-run the kernel sampler; the
  drift prompts can be skipped (model accuracy is independent of
  Luminal kernel selection).
- **When the calibration corpus shifts.** Re-run the drift sampler with the
  refreshed public/operator-supplied prompt set. Kernel efficiency can be
  skipped (hardware behaviour didn't change).
- **When hardware changes.** Re-run the kernel sampler under the new
  hardware kind. Drift is independent of hardware (it's compared
  against the bf16 reference, which is identical across H100 / B100
  / etc).

Three independent axes: hardware × Luminal version × prompt corpus. The
writer's preservation property lets these be calibrated independently.

## Deterministic output

Both writers produce byte-stable TOML on the same inputs. Two callers
running with identical `(base, fitted, hardware, timestamp)` triples
get byte-identical files. This is verified by
`cost_writer_byte_stable` and `drift_writer_byte_stable`.

- Sections are emitted in a fixed canonical order.
- Map iteration sorted by key (hardware kind in `[peak_tflops.*]`,
  `(layer, component, dtype)` in the drift table).
- Floats formatted via `{v:?}` (Rust's shortest round-trip form,
  bit-stable across runs and platforms).
- Timestamp passed in as a parameter; callers in tests pin it, the
  CLI passes the wall-clock time at calibration start.

Byte stability matters because both files feed the content-addressed
artifact hashing path. A calibration that drifts in formatting would
inflate the cache invalidation rate without any actual change.

## CPU vs CUDA build

| Concern                                    | CPU build (`--no-default-features`) | CUDA build (default)        |
|--------------------------------------------|-------------------------------------|-----------------------------|
| `CalibrationCorpus` loader + validation    | ✅                                  | ✅                          |
| `KernelMeasurement` / `DriftMeasurement`   | ✅                                  | ✅                          |
| `aggregate_kernel_measurements` (median)   | ✅                                  | ✅                          |
| `aggregate_drift_measurements` (p95)       | ✅                                  | ✅                          |
| `write_cost_constants` (preserve-unmeasured) | ✅                                | ✅                          |
| `write_drift_table` (preserve-unmeasured)  | ✅                                  | ✅                          |
| `sampler::sample_kernel_runtimes`          | ✅ NativeRuntime timings            | ✅ CUDA timings             |
| `sampler::sample_drift`                    | ✅ Skein bf16 vs candidate          | ✅ Skein bf16 vs candidate  |
| Top-level `calibrate()`                    | ✅ non-production warning            | ✅ production path           |

`skein calibrate` on a CPU build prints a warning because CPU timings must
not be committed as production constants. Its value is exercising the same
compile/execute/write flow before renting GPU time.

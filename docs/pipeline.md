# Compile Pipeline

`skein compile` is now the full artifact pipeline:

1. Load model config, cluster topology, workload trace, drift table, and cost constants.
2. Run `skein_extract::extract_plan` to choose the lowest-cost feasible `Plan`.
3. Lower each device with `skein_emit::lower_per_device`.
4. Compile every lowered segment through `compile_with_luminal::<R>` using the selected runtime.
5. Materialize per-device safetensors shards and write a reloadable `SkeinArtifact`.
6. Build a bf16 reference artifact for the same model and run advisory Skein-vs-Skein parity.
7. Write `parity_report.json`, update the drift table on parity failure, and update `LATEST`.

The artifact directory is content-addressed by `Plan::content_hash()`:

```text
artifacts/<plan_hash>/
  plan.json
  topology.json
  metadata.json
  parity_report.json
  reference_bf16/
  device_0/
    graph_segments.bin
    weights.safetensors
    io.json
```

## Parity Policy

Compile follows the TensorRT-LLM / NVIDIA Model Optimizer pattern: artifact
production and accuracy gating are separate by default. Parity runs inside
compile because Skein uses the result to refine the drift table, but a failed
parity report does not fail the compile unless `--enforce-parity` is set.

Default behavior:

- artifact is written;
- `parity_report.json` is written;
- drift table is updated monotonically on failure;
- `LATEST` points at the compiled artifact;
- command exits zero.

CI behavior:

- pass `--enforce-parity`;
- the same report and drift update happen;
- parity failure returns `CliError::ParityFailed`.

`skein verify --enforce` is the standalone validation gate for an existing
artifact. Without `--reference`, verify uses `<artifact>/reference_bf16`,
which compile writes during advisory parity.

## Failure Modes

- Input loading errors stop before search.
- No feasible plan stops before lowering.
- Lowering or Luminal compile errors stop before artifact activation.
- Existing content-addressed artifact directories are reloaded rather than
  silently overwritten.
- Parity failures update the drift table and only become hard failures when
  `--enforce-parity` or `skein verify --enforce` is used.

## CPU and GPU builds

The default build selects `CudaComputeRuntime` for compile / verify /
calibrate on an NVIDIA host. Building with `--no-default-features` selects
`NativeComputeRuntime`, which exercises the same pipeline with CPU timings
and native Luminal execution.

The CPU build is for pipeline correctness and CI; the GPU build is the
production target for calibrated cost constants and final numeric
validation.

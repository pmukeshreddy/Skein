# Runtime — `skein_runtime`

Five subsystems compose into the `Server`:

```
                       ┌─────────────────────────┐
                       │         Server          │
                       ├─────────────────────────┤
                       │  submit / serve / swap  │
                       └────┬──────┬──────┬──────┘
                            │      │      │
            ┌───────────────┴┐    │    ┌──┴──────────────────┐
            │ ContinuousBatcher│    │    │     HotSwap         │
            │  + admission     │    │    │ drain → mount      │
            │  + queue/inflight│    │    │     → atomic swap  │
            │  + chunked split │    │    └─────────────────────┘
            └───────────────┬─┘    │
                            │      │
                       ┌────┴──────┴──┐
                       │ PagedKVAllocator
                       │  + RadixPrefixTree
                       └────────────────┘
                                  │
                          ┌───────┴────────┐
                          │  ProfileHooks   │
                          │  Prometheus/OTel│
                          └────────────────┘
```

`Server::serve` runs an axum completion endpoint backed by the shared
topology executor and the in-process runtime pieces; the CUDA build swaps in
the GPU compute runtime and CUDA-specific modules behind the same surfaces.

## Paged KV with refcount + LRU eviction

Each page holds `page_size` tokens of K + V. A page has:

```
Page { id, capacity_tokens, refcount, generation }
```

State transitions:

```
free ──admit──► active (refcount > 0)
   ▲                   │ release (drops to 0)
   │                   ▼
   │           cached_lru (radix still maps it)
   │                   │ evicted (bump generation, drop radix entry)
   └───────────────────┘
```

- **free** holds pages no longer referenced and not retained by the radix
  cache; first to be claimed on the next allocation.
- **cached_lru** holds released pages the radix tree still maps. They are
  evictable; eviction bumps `generation` so any stale reuse attempt fails
  a generation check.
- **active** holds pages with at least one `PageTable` referencing them.

LRU + refcount: cached pages share lifetime. Eviction picks the oldest
`refcount == 0` cached page. Active pages (`refcount > 0`) are never
evicted.

## Radix prefix tree — cross-request reuse

A per-token trie, capped at `radix_max_depth`. Each node at depth
`d * page_size` carries an `Option<(PageId, generation_at_insert)>`. On
admit:

1. Walk the new prompt through the trie, harvesting `(PageId, generation)`
   pairs at every page boundary that's present.
2. Filter by current generation — stale entries (a generation bump on the
   page) are rejected and the walk stops there.
3. Bump the per-page `refcount` on matched pages; the new request shares
   them with whoever else still owns them.
4. Allocate fresh pages for the remaining tokens.
5. Insert the new request's full `(tokens, pages, generations)` triple so
   future requests with overlapping prefixes can reuse.

`refcount_inc / refcount_dec` walk the per-node refcount along the
inserter's full token path so the radix knows when an entire subtree is
unreferenced.

## SLO-aware admission

`LatencyEstimator` (constants from `[runtime_estimator]` in
`cluster/cost_constants.toml`) predicts:

```
prefill_ms(N)  = prefill_per_token_us  × N / 1000
tpot_ms(b)     = per_token_decode_us_at_b1 × b^batch_scaling_exponent / 1000
```

`AdmissionDecision`:

| Predicate                                    | Decision                              |
|----------------------------------------------|---------------------------------------|
| `prefill_ms(prompt_len) > slo.ttft_p95_ms`   | `Reject { PromptTooLong }`            |
| `current_inflight >= max_batch`              | `Delay { until_ms = now + one step }` |
| `tpot_ms(current_inflight + 1) > slo.tpot_p95_ms` | `Delay`                          |
| otherwise                                    | `Admit`                               |

Rejections are *terminal* — no queue helps. Delays are queued FIFO; the
runtime re-considers them when capacity frees up.

## Chunked-prefill state machine

For `BatchPolicy::ContinuousChunked { chunk_tokens, .. }`, a prompt of
`N` tokens decomposes into `ceil(N / chunk_tokens)` `(start, end)` ranges
processed one per step. `ChunkPlan::current_range` returns the next
range; `advance` moves the cursor. The request transitions to
`Phase::Decode` exactly when `is_done()` returns `true`.

## Hot-swap protocol

`HotSwap::swap_artifact` runs three stages in order:

1. **Verify shape compatibility.** Read `<old>/device_0/io.json` and
   `<new>/device_0/io.json`. New artifact must accept the same *input*
   tensors (name + shape + dtype). Output tensors may differ — the new
   Plan might quantize differently. Mismatch → `IncompatibleArtifact`
   *without* touching disk or starting drain.
2. **Drain in-flight.** Poll the in-flight set every 1 ms until empty or
   the deadline (`drain_timeout_seconds`) hits. Timeout → `DrainTimeout`
   with the residual count.
3. **Atomic symlink swap.** Write `LATEST.new.<nanotime> -> new_target`
   then `rename(2)` to `LATEST`. POSIX `rename` is atomic on the same
   filesystem; concurrent readers see `LATEST` pointing to either the
   old or new target, never broken.

Symlink states across a swap:

```
T = 0     LATEST -> old_artifact
T = 1     LATEST -> old_artifact   LATEST.new.123 -> new_artifact
T = 2     LATEST -> new_artifact   (rename absorbed LATEST.new.123)
```

`LATEST` is always a valid existing symlink between states.

## Observability metrics

`ProfileHooks` exposes a `prometheus::Registry` with:

| Metric                                | Type      | What it measures                          |
|---------------------------------------|-----------|-------------------------------------------|
| `skein_steps_total`                   | counter   | Forward steps executed                    |
| `skein_step_compute_us_total`         | counter   | Cumulative compute time, microseconds     |
| `skein_step_comm_us_total`            | counter   | Cumulative comm time (0 on the CPU build) |
| `skein_kv_pages_in_use`               | gauge     | Live KV pages                             |
| `skein_batch_size_histogram`          | histogram | Per-step batch size distribution          |
| `skein_requests_total{status}`        | counter   | Requests completed by status              |
| `skein_ttft_ms_histogram`             | histogram | Time-to-first-token, ms                   |
| `skein_tpot_ms_histogram`             | histogram | Per-token decode latency, ms              |
| `skein_prefix_cache_hit_tokens_total` | counter   | Prompt tokens served from the radix cache |

The Prometheus exporter is a tiny tokio TCP listener. `serve_metrics(port)`
binds the configured port; `serve_metrics_ephemeral` binds `:0` and
returns the kernel-assigned port (used by tests).

Trace emission uses `tracing::info_span!` so spans appear under any
installed `tracing-subscriber`.
TODO(otlp): wire the full OpenTelemetry-OTLP exporter.

## CPU vs CUDA build

| Concern                                | CPU build (`--no-default-features`) | CUDA build (default)         |
|----------------------------------------|-------------------------------------|------------------------------|
| `PagedKVAllocator` + `RadixPrefixTree` | ✅                                  | ✅                          |
| `ContinuousBatcher` (admission, queue, retire, chunking) | ✅                | ✅                          |
| `HotSwap` (drain + atomic symlink swap) | ✅                                 | ✅                          |
| `ProfileHooks` + Prometheus exporter   | ✅                                  | ✅                          |
| `Server::new` + `submit`               | ✅                                  | ✅                          |
| `Server::serve` forward-pass driver    | ✅ axum + in-process worker         | ✅ CUDA/NCCL composition    |
| NCCL between graph runs                 | n/a                                 | ✅                          |
| CUDA Graphs capture + dispatch         | eager dispatch                      | ✅                          |
| RDMA for prefill/decode KV handoff     | local copy                          | ✅                          |
| OpenTelemetry-OTLP exporter            | tracing spans only                  | tracing spans only (TODO)   |

Everything under `src/cuda/` lives behind `#[cfg(feature = "cuda")]`; the
CPU build never compiles it. The CPU serving composition uses the same
executor contract as the CUDA path, with native segment execution and
in-process collective math.

## Runtime abstractions

Three runtime-facing abstractions decouple the serving loop from the backend:

| Abstraction | CPU build | CUDA build |
|---|---|---|
| `CollectiveBackend` | In-process collective math over runtime tensors | NCCL-backed collectives |
| `KernelDispatcher` | Eager segment execution | CUDA Graph capture/dispatch |
| `KvTransport` | Local in-process page copy | RDMA transfer for prefill/decode disaggregation |

`InProcessCollective` is not a skip or a fake success path. It reads the
named tensor from every participant, performs the collective math in Rust,
and writes the correct per-rank result back. `RingAllReduce` sums
element-wise, `AllGather` concatenates shards, `ReduceScatter` sums then
splits, and `AllToAll` redistributes equal rank chunks.

`EagerDispatcher` is the production-shape CPU dispatcher: warmup is a no-op,
and dispatch calls `DynRuntime::execute_segment()` exactly once for the step.
The CUDA Graph dispatcher replaces it without changing the runtime segment
API.

`LocalKvTransport` gives the same transfer contract as RDMA for a
single-process deployment. It validates the request shape and leaves actual
page ownership with the KV allocator.

The compile crate also exposes a `DynRuntime` name-to-`NodeIndex` layer so
collectives and dispatchers can address logical tensors without knowing the
concrete `ComputeRuntime` type.

At the pinned Luminal rev, `luminal::Graph` is not `Send` or `Sync` because
it owns non-thread-safe op trait objects. `DynRuntimeWrapper` therefore keeps
graph execution single-threaded. If a later Luminal revision makes `Graph`
thread-safe, the trait can regain `Send + Sync` without changing the logical
tensor API.

## Axum Server Path

`Server::serve` is now an axum HTTP server with `/v1/completions` and
`/metrics`. The existing Prometheus TCP exporter is unchanged; axum's
`/metrics` endpoint simply encodes the same registry for callers that are
already talking to the completion port.

Because Luminal graphs are single-threaded at the pinned rev, axum state
does not hold a `TopologyExecutor` directly. Instead, serve starts a forward
worker thread, and that thread constructs the `SkeinArtifact`, native
runtime segments, `InProcessCollective`, and `TopologyExecutor` locally. HTTP
handlers tokenize/admit requests, send work to the worker, and stream SSE
token events back to the client.

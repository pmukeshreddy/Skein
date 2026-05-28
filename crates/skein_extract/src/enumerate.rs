//! Outer enumeration: list every `GlobalConfig` the search will consider
//! before hard-constraint filtering.
//!
//! Each component enumerator returns a small, deterministic list. The full
//! product is built by nested loops (no `itertools` dep). Total raw count
//! ≈ ~84,000 on the canonical Mixtral 2× H100 setup; ~50–200 survive the
//! constraints. See `docs/search_algorithms.md` for the derivation.

use skein_ir::cluster::ClusterSpec;
use skein_ir::ir::Graph;
use skein_ir::plan::ParallelismPlacement;
use skein_ir::types::{
    BatchPolicy, CudaGraphsConfig, KVLayout, PrefixCacheConfig, RadixReusePolicy, SpecDecodeConfig,
};

use crate::candidate::GlobalConfig;

// --- public entry point ---

pub fn enumerate_global_configs(cluster: &skein_cost::Cluster, ir: &Graph) -> Vec<GlobalConfig> {
    let parallelism_options = enumerate_parallelism(cluster.spec(), ir);
    let kv_options = enumerate_kv_layouts();
    let batch_options = enumerate_batch_policies();
    let cg_options = enumerate_cuda_graphs_configs();
    let spec_options = enumerate_spec_decode_configs();
    let prefix_options = enumerate_prefix_cache_configs();

    let mut out: Vec<GlobalConfig> = Vec::new();
    for parallelism in &parallelism_options {
        for &(kv_layout, kv_shard) in &kv_options {
            for &batch in &batch_options {
                for cuda_graphs in &cg_options {
                    for spec_decode in &spec_options {
                        for prefix_cache in &prefix_options {
                            out.push(GlobalConfig {
                                parallelism: *parallelism,
                                kv_layout,
                                kv_shard,
                                batch,
                                cuda_graphs: cuda_graphs.clone(),
                                spec_decode: spec_decode.clone(),
                                prefix_cache: *prefix_cache,
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

// --- component enumerators ---

pub fn enumerate_parallelism(spec: &ClusterSpec, ir: &Graph) -> Vec<ParallelismPlacement> {
    let mut out = Vec::new();
    let n = spec.num_devices;
    let ep_max: u32 = match ir.meta.num_experts {
        Some(num_experts) => (num_experts as u32).min(8),
        None => 1,
    };
    for tp in [1u32, 2, 4, 8] {
        for pp in [1u32, 2, 4] {
            let mut ep = 1u32;
            loop {
                if tp * pp * ep > n {
                    break;
                }
                out.push(ParallelismPlacement { tp, pp, ep });
                if ep == ep_max {
                    break;
                }
                ep *= 2;
                if ep > ep_max {
                    break;
                }
            }
        }
    }
    out
}

pub fn enumerate_kv_layouts() -> Vec<(KVLayout, bool)> {
    let mut out: Vec<(KVLayout, bool)> = Vec::new();
    out.push((KVLayout::Contiguous, false));
    out.push((KVLayout::Contiguous, true));
    for n in [16u32, 32, 64, 128] {
        out.push((KVLayout::Paged { page_size: n }, false));
        out.push((KVLayout::Paged { page_size: n }, true));
    }
    out
}

pub fn enumerate_batch_policies() -> Vec<BatchPolicy> {
    // Prune: keep `Continuous(M)` for the standard `M` ladder, plus one
    // representative `ContinuousChunked(M, 1024)` per `M`. `Static` is
    // dropped: decoder serving uses continuous batching as the default.
    let mut ladder: Vec<u32> = vec![1, 2, 4, 8, 16, 32, 64];
    // SKEIN_FORCE_BATCH pins the compiled decode batch width (used to build
    // batched-throughput / continuous-PP microbatch artifacts). The forced
    // value may sit off the power-of-two ladder — e.g. a stage_microbatch of 5
    // — so inject it here. Without this the outer `SKEIN_FORCE_BATCH` filter in
    // `extract_plan` has no matching candidate and rejects every plan ("no
    // feasible plan found"). This only widens the search space when explicitly
    // requested; the default ladder is unchanged.
    if let Some(fb) = std::env::var("SKEIN_FORCE_BATCH")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        && fb >= 1
        && !ladder.contains(&fb)
    {
        ladder.push(fb);
    }
    let mut out = Vec::new();
    for m in ladder {
        out.push(BatchPolicy::Continuous { max_batch: m });
        out.push(BatchPolicy::ContinuousChunked {
            max_batch: m,
            chunk_tokens: 1024,
        });
    }
    out
}

pub fn enumerate_cuda_graphs_configs() -> Vec<CudaGraphsConfig> {
    // Plan search only considers cuda_graphs=false today — but note the core
    // benefit is already delivered on the GPU build, just one layer down: the
    // Luminal `CudaRuntime` compiles each kernel subgraph into a real CUDA
    // graph by *manual node construction* (`cuGraphAddKernelNode` +
    // `cuGraphInstantiateWithFlags` + `cuGraphLaunch`; not stream capture), so
    // per-subgraph kernel-launch overhead is amortized regardless of this flag.
    // This flag would instead control Skein's *step-level* graph (one graph
    // across all segments + collectives for a whole decode step), whose
    // `skein_runtime::cuda::CudaGraphCache` is still a stub. That, too, is
    // buildable via Luminal's `pub CudaGraphHandle::add_kernel_node` (no stream
    // capture needed — the earlier "legacy stream not capturable" concern does
    // not apply, since nothing here captures). It stays gated only because the
    // step-level cache isn't implemented, not because it's blocked. Widen this
    // enumeration when that cache lands.
    vec![CudaGraphsConfig {
        enable: false,
        capture_classes: Vec::new(),
    }]
}

pub fn enumerate_spec_decode_configs() -> Vec<SpecDecodeConfig> {
    // Plan search must only consider spec_decode=false until skein_runtime's
    // SpeculativeDecoder lands. The blocking architectural prerequisite is
    // GPU-side KV cache ownership: PagedKVAllocator currently tracks page
    // metadata only, while spec-decode rollback requires owning the KV tensor
    // contents. Widen this enumeration when the prerequisite is met.
    vec![SpecDecodeConfig {
        enable: false,
        draft: None,
    }]
}

pub fn enumerate_prefix_cache_configs() -> Vec<PrefixCacheConfig> {
    // Prefix caching is cost-neutral in the analytic model, so only the
    // enabled variant ships (the runtime always benefits from it).
    vec![PrefixCacheConfig {
        enable: true,
        reuse_policy: RadixReusePolicy::LruByLastAccess,
    }]
}

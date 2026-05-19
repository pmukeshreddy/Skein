//! Test 2 + Test 3 — hard-constraint behaviour.

mod common;
use common::*;

use skein_extract::candidate::GlobalConfig;
use skein_extract::constraints;
use skein_ir::cluster::ClusterSpec;
use skein_ir::plan::ParallelismPlacement;
use skein_ir::types::{
    BatchPolicy, CudaGraphsConfig, KVLayout, PrefixCacheConfig, RadixReusePolicy, SpecDecodeConfig,
};

fn mk_global(tp: u32, pp: u32, ep: u32) -> GlobalConfig {
    GlobalConfig {
        parallelism: ParallelismPlacement { tp, pp, ep },
        kv_layout: KVLayout::Contiguous,
        kv_shard: false,
        batch: BatchPolicy::Continuous { max_batch: 8 },
        cuda_graphs: CudaGraphsConfig {
            enable: false,
            capture_classes: vec![],
        },
        spec_decode: SpecDecodeConfig {
            enable: false,
            draft: None,
        },
        prefix_cache: PrefixCacheConfig {
            enable: false,
            reuse_policy: RadixReusePolicy::LruByLastAccess,
        },
    }
}

// Test 2 — `tp=3` violates `hidden % tp == 0` for Mixtral's hidden=4096.
#[test]
fn constraints_reject_undivisible_tp() {
    let ir = load_mixtral_ir();
    assert_eq!(ir.meta.hidden, 4096);

    // tp=3 doesn't divide 4096.
    let bad = mk_global(3, 1, 1);
    assert!(!constraints::divisibility_tp(&bad, &ir));

    // tp=2, 4 do.
    for tp in [1u32, 2, 4, 8, 16] {
        let g = mk_global(tp, 1, 1);
        // Only powers of two that divide 4096 should pass.
        let expected = 4096 % tp as usize == 0;
        assert_eq!(constraints::divisibility_tp(&g, &ir), expected);
    }
}

// Test 3 — TP=1 OOMs on a small-memory cluster even with int4 weights.
//
// The spec's original framing was "TP=1 on 80 GB H100 OOMs". After we worked
// the numbers (see PR notes for Phase A Step 3), Mixtral 8x7B at int4
// weights + fp8 KV at batch=8 / kv_len=2048 actually fits in 80 GB.
// `global_memory_fits` therefore *does* allow TP=1 in that configuration —
// which is correct: the inner DP gets a chance to pick an int4 dtype map.
//
// To exercise the rejection path we use a contrived 8 GB device. Mixtral's
// non-decoder weights alone (≈ 263 MB at bf16) plus int4 decoder weights
// (≈ 23 GB) blow through any 8 GB cap. The constraint must reject.
#[test]
fn constraints_reject_oom_global() {
    let ir = load_mixtral_ir();
    let cost_model = load_cost_model();

    let tiny_cluster_toml = r#"
num_devices = 1
[[node]]
id               = "node0"
devices          = ["d0"]
device_kind      = "h100_sxm5"
device_memory_gb = 8
"#;
    let spec = ClusterSpec::from_toml_str(tiny_cluster_toml).unwrap();
    let cluster = skein_cost::Cluster::from_spec(spec);

    let g = mk_global(1, 1, 1);
    assert!(!constraints::global_memory_fits(
        &g,
        &cluster,
        &ir,
        &cost_model
    ));
    // And the umbrella `reject` returns GlobalMemory.
    assert_eq!(
        constraints::reject(&g, &cluster, &ir, &cost_model),
        Some(constraints::RejectReason::GlobalMemory)
    );
}

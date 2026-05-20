//! Cached-decode KV wiring — structural test for the STEP 3 fix.
//!
//! `wire_block_attention` must lower each attention block as a cached decode
//! step: it reads the accumulated past from runtime-fed `kvcache_{k,v}_{block}`
//! inputs plus a `position` scalar, and hands the new token's K/V back as
//! `kvcache_{k,v}_{block}` outputs (which `skein_runtime::SegmentRunner` appends
//! to the per-layer cache). This test asserts those handoffs are present for
//! every block across the device's segments — i.e. the runtime KV cache is
//! actually wired into the graph (it was not, before STEP 3: attention was a
//! stateless seq=1 pass at position 0). The numerical correctness of the decode
//! is validated end-to-end on the GPU; the dynamic-`past` attention math cannot
//! run on the CPU `NativeRuntime` (see `kv_cache.rs`).

mod common;
use common::*;

use std::collections::HashSet;

use skein_emit::build_device_graph;

#[test]
fn cached_decode_handoffs_wired_for_every_block() {
    let ir = load_mixtral_ir();
    let cluster = load_2x_h100_cluster();
    let plan = mk_plan(ir.meta.clone(), 2, 1, 1);

    // Aggregate the input/output handoff names across all of device 0's
    // segments (under tp=2 each block's attention lives in its own segment).
    let lowered = build_device_graph(&plan, &cluster, &ir, 0).expect("build device graph");
    let mut inputs: HashSet<String> = HashSet::new();
    let mut outputs: HashSet<String> = HashSet::new();
    for seg in &lowered.segments {
        for h in &seg.input_handoff {
            inputs.insert(h.logical_name.clone());
        }
        for h in &seg.output_handoff {
            outputs.insert(h.logical_name.clone());
        }
    }

    assert!(
        inputs.contains("position"),
        "cached decode needs a `position` input; got inputs {inputs:?}"
    );

    for block in 0..ir.meta.num_layers {
        let k = format!("kvcache_k_{block}");
        let v = format!("kvcache_v_{block}");
        assert!(inputs.contains(&k), "missing past-cache input {k}");
        assert!(inputs.contains(&v), "missing past-cache input {v}");
        assert!(outputs.contains(&k), "missing new-K/V output {k}");
        assert!(outputs.contains(&v), "missing new-K/V output {v}");
    }
}

//! Test 6 — retire releases KV pages and closes the token streamer.

mod common;
use common::*;

use std::sync::{Arc, Mutex};

use skein_ir::types::BatchPolicy;
use skein_runtime::ContinuousBatcher;
use skein_runtime::batcher::AdmissionDecision;
use skein_runtime::kv::PagedKVAllocator;
use skein_runtime::token_stream::TokenStreamer;
use skein_runtime::types::{IncomingRequest, RequestId};

#[tokio::test]
async fn batcher_retire_releases_kv() {
    let cost_constants = load_cost_constants();
    let workload = mk_workload(500, 50);
    let plan = mk_plan(32, false, BatchPolicy::Continuous { max_batch: 8 });
    let kv = Arc::new(Mutex::new(
        PagedKVAllocator::new(&plan, 1024 * 32, 1, 4096).unwrap(),
    ));
    let mut batcher = ContinuousBatcher::new(&plan, &workload, &cost_constants, kv.clone());

    // Admit one request and advance KV state so pages exist.
    let req = IncomingRequest {
        id: RequestId(42),
        prompt_tokens: (0..96).collect(),
        max_output_tokens: 8,
        arrival_ms: 0,
    };
    let (mut streamer, sender) = TokenStreamer::paired();
    assert!(matches!(
        batcher.admit(req, sender, 0),
        AdmissionDecision::Admit
    ));
    {
        // Mirror the runtime: hand the KV allocator the same prompt.
        let mut k = kv.lock().unwrap();
        k.admit(RequestId(42), &(0..96).collect::<Vec<u32>>())
            .unwrap();
    }
    assert!(kv.lock().unwrap().in_use_pages() >= 3);

    batcher.retire(RequestId(42)).unwrap();

    // KV pages reclaimed.
    assert_eq!(kv.lock().unwrap().in_use_pages(), 0);

    // Token streamer is closed: `.recv()` returns `None` on the next read.
    assert!(streamer.recv().await.is_none());
}

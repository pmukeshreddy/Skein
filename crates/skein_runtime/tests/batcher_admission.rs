//! Test 5 — SLO-aware admission delays new requests once in-flight saturates.

mod common;
use common::*;

use std::sync::{Arc, Mutex};

use skein_ir::types::BatchPolicy;
use skein_runtime::ContinuousBatcher;
use skein_runtime::batcher::AdmissionDecision;
use skein_runtime::kv::PagedKVAllocator;
use skein_runtime::token_stream::TokenStreamer;
use skein_runtime::types::{IncomingRequest, RequestId};

#[test]
fn batcher_admission_slo_aware() {
    let cost_constants = load_cost_constants();
    // tpot_p95_ms = 200 ms — generous, so the latency-prediction gate
    // stays out of the way; the *capacity* gate (max_batch = 8) is what
    // we're exercising. At batch=8 the estimator predicts ~37 ms tpot
    // (8 ms × 8^0.7), well under 200 ms.
    let workload = mk_workload(/*ttft*/ 500, /*tpot*/ 200);
    let plan = mk_plan(32, false, BatchPolicy::Continuous { max_batch: 8 });
    let kv = PagedKVAllocator::new(&plan, 1024 * 32, 1, 4096).unwrap();
    let kv = Arc::new(Mutex::new(kv));
    let mut batcher = ContinuousBatcher::new(&plan, &workload, &cost_constants, kv);

    let mut admitted = 0;
    let mut delayed = 0;
    for i in 0..100u64 {
        let req = IncomingRequest {
            id: RequestId(i + 1),
            prompt_tokens: (0..16).collect(),
            max_output_tokens: 64,
            arrival_ms: i, // arrivals 0..100 ms apart by 1ms each
        };
        let (_streamer, sender) = TokenStreamer::paired();
        match batcher.admit(req, sender, i) {
            AdmissionDecision::Admit => admitted += 1,
            AdmissionDecision::Delay { until_ms } => {
                assert!(until_ms > i, "delay must be in the future");
                delayed += 1;
            }
            AdmissionDecision::Reject { .. } => {
                panic!("rejecting reasonable prompts is not expected here");
            }
        }
    }
    // The first `max_batch` requests admit; further requests delay until
    // some inflight retires (which doesn't happen in this test).
    assert_eq!(admitted, 8);
    assert!(
        delayed >= 90,
        "expected ≥ 90 delays after saturation, got {delayed}"
    );
}

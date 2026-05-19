//! Test 7 — chunked prefill emits exactly ceil(N / chunk_size) prefill slices.

// `common` is intentionally not imported here — chunking tests need no Plan
// fixtures, only `BatchPolicy` from skein_ir. Keeping the module unimported
// avoids a `dead_code` warning while leaving the file in the standard
// integration-test shape.

use skein_ir::types::BatchPolicy;
use skein_runtime::batcher::chunking::plan_chunks;
use skein_runtime::types::{IncomingRequest, RequestId};

#[test]
fn chunked_prefill_split_correctness() {
    let prompt_tokens: Vec<u32> = (0..32_000u32).collect();
    let request = IncomingRequest {
        id: RequestId(1),
        prompt_tokens,
        max_output_tokens: 1,
        arrival_ms: 0,
    };
    let policy = BatchPolicy::ContinuousChunked {
        max_batch: 8,
        chunk_tokens: 2048,
    };
    let mut plan = plan_chunks(&request, policy);
    assert_eq!(plan.total_chunks, 16);

    let mut total_processed: u32 = 0;
    let mut chunks_emitted: u32 = 0;
    let prompt_len = request.prompt_tokens.len() as u32;
    while let Some((start, end)) = plan.current_range(prompt_len) {
        total_processed += end - start;
        chunks_emitted += 1;
        plan.advance();
    }
    assert_eq!(chunks_emitted, 16);
    assert_eq!(total_processed, 32_000);
    assert!(plan.is_done());
}

// A short prompt that's shorter than the chunk size still yields exactly
// one chunk (covering all tokens).
#[test]
fn short_prompt_yields_one_chunk() {
    let request = IncomingRequest {
        id: RequestId(2),
        prompt_tokens: (0..100u32).collect(),
        max_output_tokens: 1,
        arrival_ms: 0,
    };
    let policy = BatchPolicy::ContinuousChunked {
        max_batch: 4,
        chunk_tokens: 2048,
    };
    let mut plan = plan_chunks(&request, policy);
    assert_eq!(plan.total_chunks, 1);
    let (start, end) = plan.current_range(100).unwrap();
    assert_eq!((start, end), (0, 100));
    plan.advance();
    assert!(plan.is_done());
}

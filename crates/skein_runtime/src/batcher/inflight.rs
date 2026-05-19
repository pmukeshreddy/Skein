//! The in-flight set — requests currently being processed.

use std::collections::HashMap;

use skein_ir::types::BatchPolicy;

use crate::batcher::chunking::{ChunkPlan, plan_chunks};
use crate::token_stream::TokenSender;
use crate::types::{IncomingRequest, RequestId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    Decode,
}

pub struct InflightRequest {
    pub request: IncomingRequest,
    pub sender: TokenSender,
    pub phase: Phase,
    pub admitted_at_ms: u64,
    /// `Some` when the policy is `ContinuousChunked`. Tracks which prefill
    /// chunk the request is on. Once all chunks complete, the request
    /// transitions to `Phase::Decode`.
    pub chunks: Option<ChunkPlan>,
    /// Output tokens emitted so far. Used by `retire` to seal the stream.
    pub output_tokens_emitted: u32,
}

#[derive(Default)]
pub struct InflightSet {
    by_id: HashMap<RequestId, InflightRequest>,
}

impl InflightSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(
        &mut self,
        request: IncomingRequest,
        sender: TokenSender,
        policy: BatchPolicy,
        admitted_at_ms: u64,
    ) {
        let chunks = matches!(policy, BatchPolicy::ContinuousChunked { .. })
            .then(|| plan_chunks(&request, policy));
        let entry = InflightRequest {
            phase: Phase::Prefill,
            admitted_at_ms,
            chunks,
            output_tokens_emitted: 0,
            request,
            sender,
        };
        self.by_id.insert(entry.request.id, entry);
    }

    pub fn remove(&mut self, id: RequestId) -> Option<InflightRequest> {
        self.by_id.remove(&id)
    }

    pub fn get(&self, id: RequestId) -> Option<&InflightRequest> {
        self.by_id.get(&id)
    }

    pub fn get_mut(&mut self, id: RequestId) -> Option<&mut InflightRequest> {
        self.by_id.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&RequestId, &InflightRequest)> {
        self.by_id.iter()
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

//! Chunked-prefill state machine.

use skein_ir::types::BatchPolicy;

use crate::types::IncomingRequest;

#[derive(Debug, Clone)]
pub struct ChunkPlan {
    pub chunk_size: u32,
    pub total_chunks: u32,
    /// Index of the next chunk to process. Steps advance this monotonically
    /// until `next_chunk == total_chunks`, at which point the request
    /// transitions to decode.
    pub next_chunk: u32,
}

impl ChunkPlan {
    pub fn is_done(&self) -> bool {
        self.next_chunk >= self.total_chunks
    }

    /// Token range `[start, end)` covered by `next_chunk`. Returns `None`
    /// if no chunks remain.
    pub fn current_range(&self, prompt_len: u32) -> Option<(u32, u32)> {
        if self.is_done() {
            return None;
        }
        let start = self.next_chunk * self.chunk_size;
        let end = (start + self.chunk_size).min(prompt_len);
        Some((start, end))
    }

    pub fn advance(&mut self) {
        if self.next_chunk < self.total_chunks {
            self.next_chunk += 1;
        }
    }
}

/// Build the `ChunkPlan` for a request under `BatchPolicy::ContinuousChunked`.
pub fn plan_chunks(request: &IncomingRequest, policy: BatchPolicy) -> ChunkPlan {
    let chunk_size = match policy {
        BatchPolicy::ContinuousChunked { chunk_tokens, .. } => chunk_tokens,
        _ => request.prompt_tokens.len() as u32,
    };
    let chunk_size = chunk_size.max(1);
    let prompt_len = request.prompt_tokens.len() as u32;
    let total_chunks = prompt_len.div_ceil(chunk_size).max(1);
    ChunkPlan {
        chunk_size,
        total_chunks,
        next_chunk: 0,
    }
}

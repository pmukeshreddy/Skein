//! FIFO admission queue. Requests are enqueued when admission decides
//! `Delay` and pulled when capacity frees up.

use std::collections::VecDeque;

use crate::token_stream::TokenSender;
use crate::types::IncomingRequest;

#[derive(Debug)]
pub struct QueueEntry {
    pub request: IncomingRequest,
    pub sender: TokenSender,
    pub ready_at_ms: u64,
}

#[derive(Debug, Default)]
pub struct AdmissionQueue {
    entries: VecDeque<QueueEntry>,
}

impl AdmissionQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, request: IncomingRequest, sender: TokenSender, ready_at_ms: u64) {
        self.entries.push_back(QueueEntry {
            request,
            sender,
            ready_at_ms,
        });
    }

    pub fn pop_ready(&mut self, now_ms: u64) -> Option<QueueEntry> {
        // FIFO over readiness: only the head matters; if its time hasn't
        // come, the rest haven't either (the queue is push_back-only and
        // delays grow with inflight load — non-decreasing).
        if matches!(self.entries.front(), Some(e) if e.ready_at_ms <= now_ms) {
            self.entries.pop_front()
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

//! Per-request output streamer. A thin wrapper around `tokio::sync::mpsc`
//! so callers can `.recv().await` tokens as they're produced. The sender
//! lives in the runtime; the receiver hands back to the front door.

use tokio::sync::mpsc;

use crate::types::TokenOutput;

/// One-direction stream of `TokenOutput`s for a single request.
pub struct TokenStreamer {
    rx: mpsc::Receiver<TokenOutput>,
}

/// Producer side of the stream. The runtime keeps this; the front door gets
/// only the [`TokenStreamer`].
#[derive(Clone, Debug)]
pub struct TokenSender {
    tx: mpsc::Sender<TokenOutput>,
}

impl TokenStreamer {
    /// Create a paired streamer + sender. Buffer of 64 outputs is more than
    /// enough for any reasonable per-step decode batch — the receiver
    /// usually drains as fast as the runtime produces.
    pub fn paired() -> (Self, TokenSender) {
        let (tx, rx) = mpsc::channel(64);
        (Self { rx }, TokenSender { tx })
    }

    pub async fn recv(&mut self) -> Option<TokenOutput> {
        self.rx.recv().await
    }
}

impl TokenSender {
    pub async fn send(
        &self,
        t: TokenOutput,
    ) -> Result<(), tokio::sync::mpsc::error::SendError<TokenOutput>> {
        self.tx.send(t).await
    }

    /// Close the sender. The receiver's next `.recv()` returns `None`.
    pub fn close(self) {
        drop(self);
    }
}

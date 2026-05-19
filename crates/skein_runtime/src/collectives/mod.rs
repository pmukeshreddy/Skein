//! Runtime collective backends.
//!
//! `MockCollective` performs the same tensor transformations as the
//! distributed collective, but inside one process over `DynRuntime` tensor
//! buffers. It is the Mac composition for Prompt 2.

use skein_compile::{CollectiveExecutor, DynRuntime};
use skein_cost::collectives::CollectiveKind;

pub trait CollectiveBackend: Send + Sync {
    fn execute(
        &self,
        kind: CollectiveKind,
        participants: &[usize],
        tensor_name: &str,
        runtimes: &mut [&mut dyn DynRuntime],
    ) -> Result<(), CollectiveError>;
}

#[derive(Debug, thiserror::Error)]
pub enum CollectiveError {
    #[error("collective participant {participant} is outside runtime set of {num_runtimes}")]
    InvalidParticipant {
        participant: usize,
        num_runtimes: usize,
    },

    #[error("collective needs at least one participant")]
    EmptyParticipants,

    #[error("collective tensor shapes differ: expected {expected}, got {got}")]
    ShapeMismatch { expected: usize, got: usize },

    #[error("collective tensor length {len} is not divisible by {parts} participants")]
    UnevenSplit { len: usize, parts: usize },

    #[error("runtime tensor access failed: {0}")]
    Runtime(#[from] skein_compile::DynRuntimeError),
}

pub struct MockCollective {
    num_devices: usize,
}

impl MockCollective {
    pub fn new(num_devices: usize) -> Self {
        Self { num_devices }
    }
}

impl CollectiveBackend for MockCollective {
    fn execute(
        &self,
        kind: CollectiveKind,
        participants: &[usize],
        tensor_name: &str,
        runtimes: &mut [&mut dyn DynRuntime],
    ) -> Result<(), CollectiveError> {
        if participants.is_empty() {
            return Err(CollectiveError::EmptyParticipants);
        }
        for &p in participants {
            if p >= runtimes.len() || p >= self.num_devices {
                return Err(CollectiveError::InvalidParticipant {
                    participant: p,
                    num_runtimes: runtimes.len().min(self.num_devices),
                });
            }
        }

        let tensors = participants
            .iter()
            .map(|&p| runtimes[p].get_tensor_by_name(tensor_name))
            .collect::<Result<Vec<_>, _>>()?;
        ensure_same_len(&tensors)?;

        match kind {
            CollectiveKind::RingAllReduce => {
                let sum = elementwise_sum(&tensors);
                for &p in participants {
                    runtimes[p].set_tensor_by_name(tensor_name, sum.clone())?;
                }
            }
            CollectiveKind::AllGather => {
                let gathered = tensors.iter().flatten().copied().collect::<Vec<_>>();
                for &p in participants {
                    runtimes[p].set_tensor_by_name(tensor_name, gathered.clone())?;
                }
            }
            CollectiveKind::ReduceScatter => {
                let sum = elementwise_sum(&tensors);
                let chunks = split_even(&sum, participants.len())?;
                for (rank, &p) in participants.iter().enumerate() {
                    runtimes[p].set_tensor_by_name(tensor_name, chunks[rank].clone())?;
                }
            }
            CollectiveKind::AllToAll => {
                let per_rank_chunks = tensors
                    .iter()
                    .map(|t| split_even(t, participants.len()))
                    .collect::<Result<Vec<_>, _>>()?;
                for (rank, &p) in participants.iter().enumerate() {
                    let mut out = Vec::new();
                    for chunks in &per_rank_chunks {
                        out.extend_from_slice(&chunks[rank]);
                    }
                    runtimes[p].set_tensor_by_name(tensor_name, out)?;
                }
            }
            CollectiveKind::Broadcast => {
                let src = tensors[0].clone();
                for &p in participants {
                    runtimes[p].set_tensor_by_name(tensor_name, src.clone())?;
                }
            }
            CollectiveKind::SendRecv => {
                let src = tensors[0].clone();
                if let Some(&dst) = participants.get(1) {
                    runtimes[dst].set_tensor_by_name(tensor_name, src)?;
                }
            }
        }
        Ok(())
    }
}

impl CollectiveExecutor for MockCollective {
    fn execute(
        &self,
        kind: CollectiveKind,
        participants: &[usize],
        tensor_name: &str,
        runtimes: &mut [&mut dyn DynRuntime],
    ) -> Result<(), skein_compile::CompileError> {
        <Self as CollectiveBackend>::execute(self, kind, participants, tensor_name, runtimes)
            .map_err(|e| skein_compile::CompileError::Collective(e.to_string()))
    }
}

fn ensure_same_len(tensors: &[Vec<f32>]) -> Result<(), CollectiveError> {
    let expected = tensors.first().map_or(0, Vec::len);
    for tensor in tensors {
        if tensor.len() != expected {
            return Err(CollectiveError::ShapeMismatch {
                expected,
                got: tensor.len(),
            });
        }
    }
    Ok(())
}

fn elementwise_sum(tensors: &[Vec<f32>]) -> Vec<f32> {
    let mut out = vec![0.0; tensors.first().map_or(0, Vec::len)];
    for tensor in tensors {
        for (dst, src) in out.iter_mut().zip(tensor) {
            *dst += *src;
        }
    }
    out
}

fn split_even(tensor: &[f32], parts: usize) -> Result<Vec<Vec<f32>>, CollectiveError> {
    if parts == 0 {
        return Err(CollectiveError::EmptyParticipants);
    }
    if tensor.len() % parts != 0 {
        return Err(CollectiveError::UnevenSplit {
            len: tensor.len(),
            parts,
        });
    }
    let chunk = tensor.len() / parts;
    Ok(tensor.chunks(chunk).map(|c| c.to_vec()).collect())
}

#[cfg(feature = "cuda")]
pub struct NcclCollective {
    _world_size: usize,
}

#[cfg(feature = "cuda")]
impl NcclCollective {
    pub fn new(world_size: usize) -> Self {
        Self {
            _world_size: world_size,
        }
    }
}

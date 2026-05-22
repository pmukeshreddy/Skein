//! CUDA-gated runtime paths. Everything in this module is behind
//! `#[cfg(feature = "cuda")]`; the non-feature build sees an empty module.

#[cfg(feature = "cuda")]
pub mod nccl;
#[cfg(feature = "cuda")]
pub mod rdma;

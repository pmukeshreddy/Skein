//! Kernel dispatch abstraction.

use skein_compile::DynRuntime;
use skein_ir::plan::Plan;

use crate::batcher::StepBatch;

pub trait KernelDispatcher: Send + Sync {
    fn dispatch(
        &mut self,
        batch: &StepBatch,
        runtime: &mut dyn DynRuntime,
    ) -> Result<DispatchOutcome, DispatchError>;

    fn warmup(&mut self, plan: &Plan, runtime: &mut dyn DynRuntime) -> Result<(), DispatchError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchOutcome {
    Eager,
}

#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("runtime dispatch failed: {0}")]
    Runtime(#[from] skein_compile::DynRuntimeError),
}

pub struct EagerDispatcher;

impl KernelDispatcher for EagerDispatcher {
    fn dispatch(
        &mut self,
        _batch: &StepBatch,
        runtime: &mut dyn DynRuntime,
    ) -> Result<DispatchOutcome, DispatchError> {
        runtime.execute_segment()?;
        Ok(DispatchOutcome::Eager)
    }

    fn warmup(&mut self, _plan: &Plan, _runtime: &mut dyn DynRuntime) -> Result<(), DispatchError> {
        Ok(())
    }
}

#[cfg(feature = "cuda")]
pub struct CudaGraphDispatcher {
    captured: std::collections::HashMap<(u32, u32), ()>,
}

#[cfg(feature = "cuda")]
impl CudaGraphDispatcher {
    pub fn new() -> Self {
        Self {
            captured: std::collections::HashMap::new(),
        }
    }

    pub fn captured_len(&self) -> usize {
        self.captured.len()
    }
}

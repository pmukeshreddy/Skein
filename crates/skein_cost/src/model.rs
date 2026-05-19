//! `CostModel::total_cost` — the entry point. Iterates devices, sums the
//! five per-device terms, returns the max.

use std::path::Path;

use skein_ir::ir::Graph;
use skein_ir::plan::Plan;

use crate::bubble::bubble_time;
use crate::cluster::{Cluster, DeviceIdx, Placement};
use crate::comm::comm_time_on_device;
use crate::compute::{compute_time, kernels_per_step_on_device};
use crate::constants::CostConstants;
use crate::error::CostError;
use crate::launch::launch_overhead;
use crate::memory::memory_penalty;
use crate::workload_ctx::WorkloadCtx;

/// Wall-clock cost in microseconds. `f64` so the value compares directly via
/// `PartialOrd`; the search loop uses `partial_cmp` to pick the minimum.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Cost(pub f64);

impl Cost {
    pub fn microseconds(us: f64) -> Self {
        Self(us)
    }

    pub fn as_us(self) -> f64 {
        self.0
    }

    pub fn as_ms(self) -> f64 {
        self.0 / 1000.0
    }
}

#[derive(Debug, Clone)]
pub struct CostModel {
    constants: CostConstants,
}

impl CostModel {
    pub fn load(path: &Path) -> Result<Self, CostError> {
        Ok(CostModel {
            constants: CostConstants::load(path)?,
        })
    }

    pub fn from_toml_str(s: &str) -> Result<Self, CostError> {
        Ok(CostModel {
            constants: CostConstants::from_toml_str(s)?,
        })
    }

    pub fn constants(&self) -> &CostConstants {
        &self.constants
    }

    /// Top-level scoring API. Returns `max_d per_device_cost(d)` in
    /// microseconds.
    pub fn total_cost(
        &self,
        plan: &Plan,
        ir: &Graph,
        cluster: &Cluster,
    ) -> Result<Cost, CostError> {
        // Validate Plan/IR shape up front. The DP feeds Plans into this hot
        // path many times — surface a shape mismatch once, here, not deep
        // inside compute_time.
        if plan.dtype_map.per_layer.len() != ir.meta.num_layers {
            return Err(CostError::DtypeMapShape {
                expected: ir.meta.num_layers,
                actual: plan.dtype_map.per_layer.len(),
            });
        }
        let wl = WorkloadCtx::decode_step(plan, &self.constants);
        let mut max_us: f64 = 0.0;
        for d in 0..cluster.num_devices() {
            let c = self.per_device_cost(plan, ir, cluster, &wl, d)?;
            if c > max_us {
                max_us = c;
            }
        }
        Ok(Cost::microseconds(max_us))
    }

    /// Compute the five-term sum for one device.
    pub fn per_device_cost(
        &self,
        plan: &Plan,
        ir: &Graph,
        cluster: &Cluster,
        wl: &WorkloadCtx,
        device: DeviceIdx,
    ) -> Result<f64, CostError> {
        let placement = Placement::from_plan(plan);
        // Idle devices contribute zero to the per-device max, but they can
        // still surface a topology error if the Plan's parallelism doesn't
        // fit — that's caught by extract's constraints, not here.
        if placement.stage_of(device).is_none() {
            return Ok(0.0);
        }

        let compute = self.compute_on_device(plan, ir, cluster, wl, device)?;
        let comm = comm_time_on_device(plan, ir, cluster, &self.constants, wl, device)?;
        let mem = memory_penalty(plan, ir, cluster, &self.constants, device)?;
        let bubble = bubble_time(plan, compute);
        let kernels = kernels_per_step_on_device(ir, plan, device);
        let launch = launch_overhead(plan, &self.constants, kernels);

        Ok(compute + comm + mem + bubble + launch)
    }

    /// Per-device compute time: sum over layers assigned to this device's
    /// stage. Convenience for tests and for the `bubble` term.
    pub fn compute_on_device(
        &self,
        plan: &Plan,
        ir: &Graph,
        cluster: &Cluster,
        wl: &WorkloadCtx,
        device: DeviceIdx,
    ) -> Result<f64, CostError> {
        let placement = Placement::from_plan(plan);
        let Some(stage) = placement.stage_of(device) else {
            return Ok(0.0);
        };
        let num_blocks = ir.meta.num_layers;
        let mut total = 0.0_f64;
        for layer in &ir.layers {
            let on_stage = match layer.block_idx {
                Some(b) => crate::cluster::block_to_stage(b, num_blocks, placement.pp) == stage,
                None => match &layer.kind {
                    skein_ir::ir::LayerKind::Embedding(_) => stage == 0,
                    skein_ir::ir::LayerKind::Lmhead(_) => stage == placement.pp - 1,
                    _ => stage == 0,
                },
            };
            if on_stage {
                total += compute_time(layer, &ir.meta, plan, cluster, &self.constants, wl, device)?;
            }
        }
        Ok(total)
    }

    /// Expose the single-collective time for direct use (tests, the future
    /// `skein_extract` cost-of-comm pre-filter, etc.).
    pub fn comm_time_one(
        &self,
        c: &crate::collectives::Collective,
        cluster: &Cluster,
    ) -> Result<f64, CostError> {
        crate::comm::comm_time(c, cluster, &self.constants)
    }

    /// Compute time for one decoder block at a candidate weight dtype —
    /// the primitive `skein_extract`'s inner DP uses to score each combo.
    /// Activation and KV dtypes don't change compute; only weight dtype
    /// drives the GEMM/attention path in the cost model.
    #[allow(clippy::too_many_arguments)]
    pub fn block_compute_time(
        &self,
        block_idx: usize,
        ir: &skein_ir::ir::Graph,
        weight_dtype: skein_ir::types::Dtype,
        placement: crate::cluster::Placement,
        cluster: &Cluster,
        wl: &WorkloadCtx,
        device: DeviceIdx,
    ) -> Result<f64, CostError> {
        crate::compute::block_compute_time(
            block_idx,
            ir,
            weight_dtype,
            placement,
            cluster,
            &self.constants,
            wl,
            device,
        )
    }
}

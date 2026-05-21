//! Parsed `cost_constants.toml` — the only source of magic numbers in the
//! cost model. Constants are loaded once at `CostModel::load` and reused for
//! every `total_cost` call.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;
use skein_ir::types::Dtype;

use crate::collectives::CollectiveKind;
use crate::compute::OpKind;
use crate::error::CostError;

/// Map keyed by `Dtype`. Stored as a struct with six fields so the TOML
/// parser surfaces a clear "missing field `bf16`" error instead of returning
/// `None` at lookup time.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct DtypeTable {
    pub bf16: f64,
    pub fp16: f64,
    pub fp8_e4m3: f64,
    pub fp8_e5m2: f64,
    pub int8: f64,
    pub int4: f64,
}

impl DtypeTable {
    pub fn get(self, d: Dtype) -> f64 {
        match d {
            // F32 is never a planner-selected layer dtype (absent from
            // Dtype::ALL); it only appears on emitter handoffs. Fall back to the
            // bf16 cost coefficient so any incidental lookup is well-defined.
            Dtype::F32 | Dtype::Bf16 => self.bf16,
            Dtype::Fp16 => self.fp16,
            Dtype::Fp8E4m3 => self.fp8_e4m3,
            Dtype::Fp8E5m2 => self.fp8_e5m2,
            Dtype::Int8 => self.int8,
            Dtype::Int4 => self.int4,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct CollectiveEffTable {
    pub ring_allreduce: f64,
    pub allgather: f64,
    pub reducescatter: f64,
    pub alltoall: f64,
    pub broadcast: f64,
    pub send_recv: f64,
}

impl CollectiveEffTable {
    pub fn get(self, c: CollectiveKind) -> f64 {
        match c {
            CollectiveKind::RingAllReduce => self.ring_allreduce,
            CollectiveKind::AllGather => self.allgather,
            CollectiveKind::ReduceScatter => self.reducescatter,
            CollectiveKind::AllToAll => self.alltoall,
            CollectiveKind::Broadcast => self.broadcast,
            CollectiveKind::SendRecv => self.send_recv,
        }
    }
}

/// Representative `(batch, kv_len)` pair the cost model uses to score a
/// decode step. Per-step cost determines throughput at a fixed SLO, which is
/// what the cost model ranks Plans on.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct RepresentativeWorkload {
    pub prefill_tokens: u32,
    pub decode_kv_tokens: u32,
}

/// DP discretization knobs consumed by `skein_extract`. Carried here so all
/// tunables live in one TOML file.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct DpConfig {
    pub memory_buckets: u32,
    pub drift_buckets: u32,
}

/// Runtime knobs consumed by `skein_runtime` — buffer sizes, port defaults,
/// timeouts.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct RuntimeConfig {
    pub metrics_buffer_capacity: u32,
    pub drain_timeout_seconds: u32,
    pub prometheus_port: u16,
    pub radix_max_depth: u32,
}

/// SLO-aware admission estimator constants. The runtime uses these to
/// decide `Admit` / `Delay` / `Reject` without launching a forward pass.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct RuntimeEstimator {
    pub per_token_decode_us_at_b1: f64,
    pub batch_scaling_exponent: f64,
    pub prefill_per_token_us: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CostConstants {
    /// Peak compute, indexed by `device_kind` then `Dtype`.
    pub peak_tflops: HashMap<String, DtypeTable>,
    /// Sustained efficiency relative to peak, indexed by op kind then
    /// `Dtype`. Op kinds: `gemm`, `attention`, `elementwise`.
    pub efficiency: HashMap<String, DtypeTable>,
    pub collective_efficiency: CollectiveEffTable,
    pub launch_us_per_kernel: f64,
    pub overshoot_us_per_gb: f64,
    /// Per-dtype activation-MSE tolerance for `skein_parity`. Sourced from
    /// `[parity_tolerance_mse]` in `cost_constants.toml`.
    pub parity_tolerance_mse: DtypeTable,
    pub representative_workload: RepresentativeWorkload,
    pub dp: DpConfig,
    pub runtime: RuntimeConfig,
    pub runtime_estimator: RuntimeEstimator,
}

impl CostConstants {
    pub fn load(path: &Path) -> Result<Self, CostError> {
        let s = std::fs::read_to_string(path).map_err(|source| CostError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml_str(&s)
    }

    pub fn from_toml_str(s: &str) -> Result<Self, CostError> {
        Ok(toml::from_str(s)?)
    }

    pub fn peak_tflops_for(&self, kind: &str, dtype: Dtype) -> Result<f64, CostError> {
        let table = self
            .peak_tflops
            .get(kind)
            .ok_or_else(|| CostError::UnknownDeviceKind {
                kind: kind.to_string(),
            })?;
        let v = table.get(dtype);
        if v.is_finite() && v > 0.0 {
            Ok(v)
        } else {
            // Zero/negative/NaN in the TOML is a configuration bug, not a
            // runtime condition. Surface it loudly.
            Err(CostError::MissingPeakTflopsDtype {
                kind: kind.to_string(),
                dtype,
            })
        }
    }

    pub fn efficiency_for(&self, op: OpKind, dtype: Dtype) -> Result<f64, CostError> {
        let table = self
            .efficiency
            .get(op.efficiency_key())
            .ok_or(CostError::UnknownEfficiencyOp { op })?;
        let v = table.get(dtype);
        if v.is_finite() && v > 0.0 {
            Ok(v)
        } else {
            Err(CostError::MissingEfficiencyDtype { op, dtype })
        }
    }

    pub fn collective_eff(&self, c: CollectiveKind) -> f64 {
        self.collective_efficiency.get(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
launch_us_per_kernel = 7.0
overshoot_us_per_gb = 1000000.0

[peak_tflops.h100_sxm5]
bf16     = 989.0
fp16     = 989.0
fp8_e4m3 = 1979.0
fp8_e5m2 = 1979.0
int8     = 1979.0
int4     = 3958.0

[efficiency.gemm]
bf16     = 0.78
fp16     = 0.78
fp8_e4m3 = 0.72
fp8_e5m2 = 0.72
int8     = 0.68
int4     = 0.55

[efficiency.attention]
bf16     = 0.64
fp16     = 0.64
fp8_e4m3 = 0.58
fp8_e5m2 = 0.58
int8     = 0.52
int4     = 0.40

[efficiency.elementwise]
bf16     = 0.95
fp16     = 0.95
fp8_e4m3 = 0.92
fp8_e5m2 = 0.92
int8     = 0.90
int4     = 0.85

[collective_efficiency]
ring_allreduce = 0.85
allgather      = 0.90
reducescatter  = 0.88
alltoall       = 0.75
broadcast      = 0.95
send_recv      = 0.92

[parity_tolerance_mse]
bf16     = 1.0e-3
fp16     = 1.0e-3
fp8_e4m3 = 5.0e-3
fp8_e5m2 = 5.0e-3
int8     = 1.0e-2
int4     = 2.0e-2

[representative_workload]
prefill_tokens   = 1024
decode_kv_tokens = 2048

[dp]
memory_buckets = 100
drift_buckets  = 50

[runtime]
metrics_buffer_capacity = 100000
drain_timeout_seconds   = 60
prometheus_port         = 9090
radix_max_depth         = 4096

[runtime_estimator]
per_token_decode_us_at_b1 = 8000.0
batch_scaling_exponent    = 0.7
prefill_per_token_us      = 25.0
"#;

    #[test]
    fn parses_sample() {
        let c = CostConstants::from_toml_str(SAMPLE).unwrap();
        assert_eq!(c.peak_tflops_for("h100_sxm5", Dtype::Bf16).unwrap(), 989.0);
        assert_eq!(c.peak_tflops_for("h100_sxm5", Dtype::Int4).unwrap(), 3958.0);
        assert_eq!(c.efficiency_for(OpKind::Gemm, Dtype::Bf16).unwrap(), 0.78);
        assert_eq!(
            c.efficiency_for(OpKind::Attention, Dtype::Fp8E4m3).unwrap(),
            0.58
        );
        assert_eq!(
            c.efficiency_for(OpKind::Elementwise, Dtype::Int4).unwrap(),
            0.85
        );
        assert_eq!(c.collective_eff(CollectiveKind::RingAllReduce), 0.85);
        assert_eq!(c.collective_eff(CollectiveKind::SendRecv), 0.92);
        assert_eq!(c.representative_workload.decode_kv_tokens, 2048);
        assert_eq!(c.launch_us_per_kernel, 7.0);
    }

    #[test]
    fn unknown_device_kind_errors() {
        let c = CostConstants::from_toml_str(SAMPLE).unwrap();
        let err = c.peak_tflops_for("a100_sxm4", Dtype::Bf16).unwrap_err();
        match err {
            CostError::UnknownDeviceKind { kind } => assert_eq!(kind, "a100_sxm4"),
            other => panic!("unexpected: {other:?}"),
        }
    }
}

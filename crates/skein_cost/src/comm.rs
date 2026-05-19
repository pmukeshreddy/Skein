//! Communication cost. Two pieces:
//!
//! 1. `comm_time(&collective)` — single-collective microseconds: factor ×
//!    bytes / bottleneck_bandwidth / efficiency + path latency.
//! 2. `collectives_on_device(&plan, &ir, device)` — enumerates which
//!    collectives this device participates in for one decode step.
//!
//! The "collectives a Plan issues" rules:
//!
//! - **TP > 1**, per decoder block: two `RingAllReduce`s (post-attention
//!   o_proj output reduce + post-MLP/MoE output reduce). Payload is one
//!   hidden-state tensor each.
//! - **EP > 1**, per MoE block: two `AllToAll`s (token dispatch + combine).
//! - **PP > 1**: one `SendRecv` per (pp - 1) stage boundary per forward
//!   step. Modelled on devices that lie on the boundary.

use skein_ir::ir::{Graph, LayerKind};
use skein_ir::plan::Plan;

use crate::cluster::{Cluster, DeviceIdx, Placement, block_to_stage};
use crate::collectives::{Collective, CollectiveKind};
use crate::compute::activation_dtype;
use crate::constants::CostConstants;
use crate::error::CostError;
use crate::workload_ctx::WorkloadCtx;

/// Wall-clock cost in microseconds for executing a single collective on the
/// chosen cluster topology.
pub fn comm_time(
    collective: &Collective,
    cluster: &Cluster,
    constants: &CostConstants,
) -> Result<f64, CostError> {
    if collective.participants.len() < 2 {
        return Err(CostError::DegenerateCollective {
            kind: collective.kind,
            n: collective.participants.len(),
        });
    }
    let path = cluster
        .topology()
        .collective_path(&collective.participants)?;
    let n = collective.participants.len();
    let factor = collective.kind.factor(n);
    let eff = constants.collective_eff(collective.kind);
    let bw_gbps = path.min_bandwidth_gbps();
    let latency = path.total_latency_us();

    // Transfer term: bytes-on-the-wire / sustained bandwidth.
    //   factor × bytes (B) / (bw_gbps × 1e9 (B/s) × eff)   →   seconds
    //   × 1e6                                              →   microseconds
    let transfer_us = if bw_gbps.is_finite() {
        (factor * collective.bytes as f64) / (bw_gbps * eff * 1e9) * 1e6
    } else {
        // Empty path (single-host self-collective): no transfer, only the
        // launch cost which is captured by the launch-overhead term.
        0.0
    };
    Ok(latency + transfer_us)
}

/// Enumerate every collective `device` participates in during one forward
/// step. Empty list when this device is idle (`device >= devices_used`) or
/// when the Plan does not require any communication.
pub fn collectives_on_device(plan: &Plan, ir: &Graph, device: DeviceIdx) -> Vec<Collective> {
    let placement = Placement::from_plan(plan);
    let Some(stage) = placement.stage_of(device) else {
        return Vec::new();
    };
    let num_blocks = ir.meta.num_layers;
    let hidden = ir.meta.hidden as u64;
    let max_batch = plan.batching.max_batch() as u64;
    let seq = 1_u64; // decode-step

    let mut out: Vec<Collective> = Vec::new();

    // TP collectives — one per Attention output and one per MoE/MLP output,
    // for every block assigned to this device's stage.
    if placement.tp > 1 {
        let tp_group = placement.tp_group(device);
        for layer in &ir.layers {
            let Some(block) = layer.block_idx else {
                continue;
            };
            if block_to_stage(block, num_blocks, placement.pp) != stage {
                continue;
            }
            match &layer.kind {
                LayerKind::Attention(_) | LayerKind::Mlp(_) | LayerKind::Moe(_) => {
                    let dt = activation_dtype(plan, Some(block));
                    let bytes = dt.bytes_for(max_batch * seq * hidden);
                    out.push(Collective {
                        kind: CollectiveKind::RingAllReduce,
                        participants: tp_group.clone(),
                        bytes,
                    });
                }
                _ => {}
            }
        }
    }

    // EP collectives — two AllToAll per MoE block (dispatch + combine).
    if placement.ep > 1 {
        let ep_group = placement.ep_group(device);
        for layer in &ir.layers {
            let Some(block) = layer.block_idx else {
                continue;
            };
            if block_to_stage(block, num_blocks, placement.pp) != stage {
                continue;
            }
            if let LayerKind::Moe(m) = &layer.kind {
                let dt = activation_dtype(plan, Some(block));
                // Each token routes to `top_k` experts; per-token AllToAll
                // payload is roughly `top_k × hidden` activations.
                let bytes = dt.bytes_for(max_batch * seq * (m.top_k as u64) * hidden);
                out.push(Collective {
                    kind: CollectiveKind::AllToAll,
                    participants: ep_group.clone(),
                    bytes,
                });
                out.push(Collective {
                    kind: CollectiveKind::AllToAll,
                    participants: ep_group.clone(),
                    bytes,
                });
            }
        }
    }

    // PP collectives — one SendRecv per stage boundary this device
    // participates in. Modelled per direction (forward only at decode time).
    if placement.pp > 1 {
        let pp_group = placement.pp_group(device);
        // Device on stage s communicates with stage s+1 if s < pp - 1, and
        // with stage s - 1 if s > 0. Each boundary is a 2-party SendRecv.
        if stage < placement.pp - 1 {
            let next = pp_group[(stage + 1) as usize];
            let dt = activation_dtype(plan, Some(0)); // boundary-token dtype
            let bytes = dt.bytes_for(max_batch * seq * hidden);
            out.push(Collective {
                kind: CollectiveKind::SendRecv,
                participants: vec![device, next],
                bytes,
            });
        }
    }

    out
}

/// Total comm time (µs) for `device` over one forward step.
pub fn comm_time_on_device(
    plan: &Plan,
    ir: &Graph,
    cluster: &Cluster,
    constants: &CostConstants,
    _wl: &WorkloadCtx,
    device: DeviceIdx,
) -> Result<f64, CostError> {
    let mut total = 0.0_f64;
    for c in collectives_on_device(plan, ir, device) {
        total += comm_time(&c, cluster, constants)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::CostConstants;
    use skein_ir::cluster::ClusterSpec;

    const TWO_H100: &str = r#"
num_devices = 2
[[node]]
id               = "node0"
devices          = ["d0", "d1"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
[[link]]
endpoints      = ["d0", "d1"]
kind           = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us     = 1.0
"#;

    const CONSTANTS: &str = r#"
launch_us_per_kernel = 7.0
overshoot_us_per_gb  = 1000000.0
[peak_tflops.h100_sxm5]
bf16=989 fp16=989 fp8_e4m3=1979 fp8_e5m2=1979 int8=1979 int4=3958
[efficiency.gemm]
bf16=0.78 fp16=0.78 fp8_e4m3=0.72 fp8_e5m2=0.72 int8=0.68 int4=0.55
[efficiency.attention]
bf16=0.64 fp16=0.64 fp8_e4m3=0.58 fp8_e5m2=0.58 int8=0.52 int4=0.40
[efficiency.elementwise]
bf16=0.95 fp16=0.95 fp8_e4m3=0.92 fp8_e5m2=0.92 int8=0.90 int4=0.85
[collective_efficiency]
ring_allreduce=0.85 allgather=0.90 reducescatter=0.88 alltoall=0.75 broadcast=0.95 send_recv=0.92
[representative_workload]
prefill_tokens=1024
decode_kv_tokens=2048
[dp]
memory_buckets=100
drift_buckets=50
"#;

    #[test]
    fn allreduce_two_devices_realistic_magnitude() {
        // Build a 2x H100 cluster and time a 32 MB allreduce.
        let spec = ClusterSpec::from_toml_str(TWO_H100).unwrap();
        let cluster = Cluster::from_spec(spec);
        let constants = CostConstants::from_toml_str(
            // The compact inline-table syntax in CONSTANTS isn't quite TOML;
            // use a normal block here.
            r#"
launch_us_per_kernel = 7.0
overshoot_us_per_gb  = 1000000.0
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
"#,
        )
        .unwrap();
        // 32 MB payload.
        let c = Collective {
            kind: CollectiveKind::RingAllReduce,
            participants: vec![0, 1],
            bytes: 32 * 1024 * 1024,
        };
        let t = comm_time(&c, &cluster, &constants).unwrap();
        // factor(2)=1, payload=33_554_432, bw=900 GB/s × 0.85 eff, latency 1 µs.
        //   transfer = 33_554_432 / (900e9 × 0.85) × 1e6 ≈ 43.86 µs.
        //   total ≈ 44.86 µs.
        assert!(t > 40.0 && t < 50.0, "expected ~44 µs, got {t}");
    }

    // Suppress unused warning on the unused CONSTANTS string — kept for
    // documentation but the test above uses an inline block instead.
    #[test]
    fn constants_str_compiles() {
        let _ = CONSTANTS;
    }
}

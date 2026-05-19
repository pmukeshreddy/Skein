//! `Cluster` — owned wrapper around `skein_ir::cluster::ClusterSpec` plus
//! the derived `Topology` graph, per-device kind/memory lookups, and the
//! TP/PP/EP device-placement helpers used throughout the cost model.

use skein_ir::cluster::ClusterSpec;

use crate::error::CostError;
use crate::topology::Topology;

/// Device index in `[0, num_devices)`. Matches the position of the device id
/// in `ClusterSpec::all_devices()`.
pub type DeviceIdx = u32;

#[derive(Debug, Clone)]
pub struct Cluster {
    spec: ClusterSpec,
    topology: Topology,
    device_kinds: Vec<String>,
    device_memory_bytes: Vec<u64>,
}

impl Cluster {
    pub fn from_spec(spec: ClusterSpec) -> Self {
        let topology = Topology::from_spec(&spec);
        // Lay out per-device kind/memory in the same order as
        // `ClusterSpec::all_devices()` so device indices line up everywhere.
        let mut device_kinds: Vec<String> = Vec::with_capacity(spec.num_devices as usize);
        let mut device_memory_bytes: Vec<u64> = Vec::with_capacity(spec.num_devices as usize);
        for node in &spec.nodes {
            // 1 GB = 10^9 bytes (decimal). Datasheet "80 GB" is decimal.
            let mem_bytes = (node.device_memory_gb as u64) * 1_000_000_000;
            for _ in &node.devices {
                device_kinds.push(node.device_kind.clone());
                device_memory_bytes.push(mem_bytes);
            }
        }
        Cluster {
            spec,
            topology,
            device_kinds,
            device_memory_bytes,
        }
    }

    pub fn spec(&self) -> &ClusterSpec {
        &self.spec
    }

    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    pub fn num_devices(&self) -> u32 {
        self.spec.num_devices
    }

    pub fn device_kind(&self, idx: DeviceIdx) -> Result<&str, CostError> {
        self.device_kinds
            .get(idx as usize)
            .map(|s| s.as_str())
            .ok_or(CostError::DeviceOutOfRange {
                idx,
                total: self.spec.num_devices,
            })
    }

    pub fn device_memory_bytes(&self, idx: DeviceIdx) -> Result<u64, CostError> {
        self.device_memory_bytes
            .get(idx as usize)
            .copied()
            .ok_or(CostError::DeviceOutOfRange {
                idx,
                total: self.spec.num_devices,
            })
    }

    /// Smallest device cap across the cluster. The skein_extract memory-fit
    /// constraint conservatively uses this — a Plan must fit the tightest
    /// device, not just the largest.
    pub fn min_device_memory_bytes(&self) -> u64 {
        self.device_memory_bytes
            .iter()
            .copied()
            .min()
            .expect("Cluster has zero devices — ClusterSpec validation should reject this")
    }
}

/// Device placement helper: how `(tp, pp, ep)` map onto device indices.
///
/// Phase A assumes a simple lexicographic layout: device `i` belongs to
///   `stage = i / (tp × ep)`,
///   `tp_idx = (i % (tp × ep)) / ep`,
///   `ep_idx = i % ep`.
///
/// `tp × pp × ep ≤ num_devices`; devices past `tp × pp × ep` are idle. The
/// extract-stage constraint check (`skein_extract::constraints`) rejects
/// placements that don't actually use the cluster.
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub tp: u32,
    pub pp: u32,
    pub ep: u32,
}

impl Placement {
    pub fn from_plan(plan: &skein_ir::plan::Plan) -> Self {
        // `parallelism.tp/pp/ep` are at least 1 in any valid Plan.
        Placement {
            tp: plan.parallelism.tp.max(1),
            pp: plan.parallelism.pp.max(1),
            ep: plan.parallelism.ep.max(1),
        }
    }

    pub fn devices_used(self) -> u32 {
        self.tp * self.pp * self.ep
    }

    /// `(stage_idx, tp_idx, ep_idx)` for a given device index. `None` for
    /// devices outside the placement.
    pub fn device_coords(self, device: DeviceIdx) -> Option<(u32, u32, u32)> {
        if device >= self.devices_used() {
            return None;
        }
        let stage = device / (self.tp * self.ep);
        let within = device % (self.tp * self.ep);
        let tp_idx = within / self.ep;
        let ep_idx = within % self.ep;
        Some((stage, tp_idx, ep_idx))
    }

    /// Devices that share a TP group with `device`.
    pub fn tp_group(self, device: DeviceIdx) -> Vec<DeviceIdx> {
        let Some((stage, _, ep_idx)) = self.device_coords(device) else {
            return Vec::new();
        };
        (0..self.tp)
            .map(|t| stage * self.tp * self.ep + t * self.ep + ep_idx)
            .collect()
    }

    /// Devices that share an EP group with `device` (same stage, same TP
    /// index, varying EP index).
    pub fn ep_group(self, device: DeviceIdx) -> Vec<DeviceIdx> {
        let Some((stage, tp_idx, _)) = self.device_coords(device) else {
            return Vec::new();
        };
        (0..self.ep)
            .map(|e| stage * self.tp * self.ep + tp_idx * self.ep + e)
            .collect()
    }

    /// Devices that share a PP group with `device` (same TP and EP index
    /// across all stages). Used for pipeline send/recv.
    pub fn pp_group(self, device: DeviceIdx) -> Vec<DeviceIdx> {
        let Some((_, tp_idx, ep_idx)) = self.device_coords(device) else {
            return Vec::new();
        };
        (0..self.pp)
            .map(|s| s * self.tp * self.ep + tp_idx * self.ep + ep_idx)
            .collect()
    }

    /// Stage index for `device`. Returns `None` if the device is idle.
    pub fn stage_of(self, device: DeviceIdx) -> Option<u32> {
        self.device_coords(device).map(|(s, _, _)| s)
    }
}

/// Even block-to-stage mapping. With `num_blocks % pp != 0` the leading
/// stages absorb the remainder one block each.
pub fn block_to_stage(block_idx: usize, num_blocks: usize, pp: u32) -> u32 {
    let pp = pp.max(1) as usize;
    let base = num_blocks / pp;
    let rem = num_blocks % pp;
    let mut acc = 0;
    for s in 0..pp {
        let stage_size = base + if s < rem { 1 } else { 0 };
        if block_idx < acc + stage_size {
            return s as u32;
        }
        acc += stage_size;
    }
    (pp - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_2tp_1pp_1ep() {
        let p = Placement {
            tp: 2,
            pp: 1,
            ep: 1,
        };
        assert_eq!(p.devices_used(), 2);
        assert_eq!(p.device_coords(0), Some((0, 0, 0)));
        assert_eq!(p.device_coords(1), Some((0, 1, 0)));
        assert_eq!(p.tp_group(0), vec![0, 1]);
        assert_eq!(p.tp_group(1), vec![0, 1]);
        assert_eq!(p.pp_group(0), vec![0]);
    }

    #[test]
    fn placement_2tp_2pp() {
        let p = Placement {
            tp: 2,
            pp: 2,
            ep: 1,
        };
        assert_eq!(p.devices_used(), 4);
        assert_eq!(p.device_coords(0), Some((0, 0, 0)));
        assert_eq!(p.device_coords(2), Some((1, 0, 0)));
        assert_eq!(p.device_coords(3), Some((1, 1, 0)));
        // Stage 0 holds {0,1}, stage 1 holds {2,3}.
        assert_eq!(p.tp_group(2), vec![2, 3]);
        // Pipeline group of device 0 is {0, 2}, since both are tp_idx=0 in
        // their respective stages.
        assert_eq!(p.pp_group(0), vec![0, 2]);
    }

    #[test]
    fn block_distribution_even() {
        // 32 blocks, pp=2 → 16 each.
        assert_eq!(block_to_stage(0, 32, 2), 0);
        assert_eq!(block_to_stage(15, 32, 2), 0);
        assert_eq!(block_to_stage(16, 32, 2), 1);
        assert_eq!(block_to_stage(31, 32, 2), 1);
    }

    #[test]
    fn block_distribution_with_remainder() {
        // 32 blocks, pp=3 → 11, 11, 10.
        assert_eq!(block_to_stage(10, 32, 3), 0);
        assert_eq!(block_to_stage(11, 32, 3), 1);
        assert_eq!(block_to_stage(21, 32, 3), 1);
        assert_eq!(block_to_stage(22, 32, 3), 2);
        assert_eq!(block_to_stage(31, 32, 3), 2);
    }
}

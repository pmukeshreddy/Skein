//! Topology graph built from a `ClusterSpec`. Devices are vertices, links
//! are typed undirected edges. The cost model walks this graph to compute
//! per-collective bandwidth bottleneck and latency.
//!
//! `Topology::collective_path` picks the *critical path* among a set of
//! participants: the pair of participants with the largest sum-of-latency
//! shortest path. Bandwidth is the min along that path. This matches how
//! ring collectives are limited by their slowest link.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use skein_ir::cluster::{ClusterSpec, LinkKind};

use crate::error::CostError;

#[derive(Debug, Clone, Copy)]
pub struct Hop {
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
    pub kind: LinkKind,
}

#[derive(Debug, Clone, Default)]
pub struct Path {
    pub hops: Vec<Hop>,
}

impl Path {
    /// Bandwidth bottleneck (min over hops). Infinity for an empty (loopback)
    /// path so the comm transfer term contributes nothing — there is no link
    /// to traverse.
    pub fn min_bandwidth_gbps(&self) -> f64 {
        if self.hops.is_empty() {
            f64::INFINITY
        } else {
            self.hops
                .iter()
                .map(|h| h.bandwidth_gbps)
                .fold(f64::INFINITY, f64::min)
        }
    }

    /// Total latency: sum over hops. Zero for an empty path.
    pub fn total_latency_us(&self) -> f64 {
        self.hops.iter().map(|h| h.latency_us).sum()
    }
}

/// Adjacency-list topology. Devices are indexed `0..n_devices` matching the
/// position of each device id in `ClusterSpec::all_devices()`.
#[derive(Debug, Clone)]
pub struct Topology {
    n_devices: u32,
    /// `edges[i]` is the list of `(neighbor_idx, hop)` pairs.
    edges: Vec<Vec<(u32, Hop)>>,
}

impl Topology {
    pub fn from_spec(spec: &ClusterSpec) -> Self {
        // Build a name → idx map matching `ClusterSpec::all_devices()`.
        let names: Vec<&str> = spec.all_devices();
        let n = names.len() as u32;
        let mut name_to_idx = std::collections::HashMap::with_capacity(names.len());
        for (i, &name) in names.iter().enumerate() {
            name_to_idx.insert(name, i as u32);
        }
        let mut edges: Vec<Vec<(u32, Hop)>> = vec![Vec::new(); names.len()];
        for link in &spec.links {
            // The `ClusterSpec` parser has already validated that both
            // endpoints exist; the lookup here is infallible by construction.
            let a = name_to_idx[link.endpoints[0].as_str()];
            let b = name_to_idx[link.endpoints[1].as_str()];
            let hop = Hop {
                bandwidth_gbps: link.bandwidth_gbps,
                latency_us: link.latency_us,
                kind: link.kind,
            };
            edges[a as usize].push((b, hop));
            edges[b as usize].push((a, hop));
        }
        Topology {
            n_devices: n,
            edges,
        }
    }

    pub fn num_devices(&self) -> u32 {
        self.n_devices
    }

    /// Latency-weighted shortest path from `from` to `to`. Returns the empty
    /// path for `from == to`. Returns `Err(NoPath)` if the graph is
    /// disconnected.
    pub fn pair_path(&self, from: u32, to: u32) -> Result<Path, CostError> {
        if from == to {
            return Ok(Path::default());
        }
        if from >= self.n_devices {
            return Err(CostError::DeviceOutOfRange {
                idx: from,
                total: self.n_devices,
            });
        }
        if to >= self.n_devices {
            return Err(CostError::DeviceOutOfRange {
                idx: to,
                total: self.n_devices,
            });
        }
        // Dijkstra on a small graph. Total node count is the number of
        // devices in the cluster — at the largest expected sizes (single
        // node H100 SuperPOD: 256) this is trivial.
        let n = self.n_devices as usize;
        let mut dist = vec![f64::INFINITY; n];
        let mut prev: Vec<Option<(u32, Hop)>> = vec![None; n];
        let mut heap: BinaryHeap<Entry> = BinaryHeap::new();
        dist[from as usize] = 0.0;
        heap.push(Entry {
            node: from,
            dist: 0.0,
        });

        while let Some(Entry { node, dist: d }) = heap.pop() {
            if d > dist[node as usize] {
                continue;
            }
            if node == to {
                break;
            }
            for &(next, hop) in &self.edges[node as usize] {
                let alt = d + hop.latency_us;
                if alt < dist[next as usize] {
                    dist[next as usize] = alt;
                    prev[next as usize] = Some((node, hop));
                    heap.push(Entry {
                        node: next,
                        dist: alt,
                    });
                }
            }
        }

        if !dist[to as usize].is_finite() {
            return Err(CostError::NoPath { from, to });
        }

        // Walk `prev` backwards to recover hops in order.
        let mut hops: Vec<Hop> = Vec::new();
        let mut cursor = to;
        while cursor != from {
            let (p, hop) = prev[cursor as usize].expect(
                "Dijkstra reached `to` with finite distance but no predecessor; \
                 this can only happen if the graph was mutated during the walk",
            );
            hops.push(hop);
            cursor = p;
        }
        hops.reverse();
        Ok(Path { hops })
    }

    /// Critical path for a collective involving `participants`: the pair of
    /// participants whose shortest path has the highest total latency. Ties
    /// broken by lower min bandwidth (the worse bottleneck wins).
    ///
    /// For a ring collective on `n` participants this returns the worst link
    /// in the ring, which is the bottleneck that determines the collective's
    /// completion time.
    pub fn collective_path(&self, participants: &[u32]) -> Result<Path, CostError> {
        if participants.len() < 2 {
            // Caller's job to filter out degenerate collectives; surface as
            // a clear error if it slips through.
            return Err(CostError::DegenerateCollective {
                kind: crate::collectives::CollectiveKind::SendRecv,
                n: participants.len(),
            });
        }
        let mut worst: Option<Path> = None;
        for i in 0..participants.len() {
            for j in (i + 1)..participants.len() {
                let candidate = self.pair_path(participants[i], participants[j])?;
                worst = Some(match worst {
                    None => candidate,
                    Some(cur) => {
                        if is_worse(&candidate, &cur) {
                            candidate
                        } else {
                            cur
                        }
                    }
                });
            }
        }
        // `worst` is `Some` because `participants.len() >= 2` ⇒ at least one
        // pair was considered.
        Ok(worst.expect("loop iterates at least once when participants.len() >= 2"))
    }
}

/// `a` is worse than `b` when its diameter (total latency) is larger, or
/// equal-and-its-bottleneck-bandwidth is smaller.
fn is_worse(a: &Path, b: &Path) -> bool {
    let (la, ba) = (a.total_latency_us(), a.min_bandwidth_gbps());
    let (lb, bb) = (b.total_latency_us(), b.min_bandwidth_gbps());
    la > lb || (la == lb && ba < bb)
}

// --- BinaryHeap support: f64 isn't `Ord`, so wrap in a min-heap entry.

#[derive(Debug)]
struct Entry {
    node: u32,
    dist: f64,
}

impl PartialEq for Entry {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist && self.node == o.node
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Entry {
    fn cmp(&self, o: &Self) -> Ordering {
        // Min-heap: smaller dist is "greater" in heap order.
        // f64 is finite here (we never push INFINITY), so total_cmp is safe.
        o.dist.total_cmp(&self.dist).then(self.node.cmp(&o.node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    const FOUR_GPU_TWO_NODE: &str = r#"
num_devices = 4
[[node]]
id               = "node0"
devices          = ["d0", "d1"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
[[node]]
id               = "node1"
devices          = ["d2", "d3"]
device_kind      = "h100_sxm5"
device_memory_gb = 80
[[link]]
endpoints      = ["d0", "d1"]
kind           = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us     = 1.0
[[link]]
endpoints      = ["d2", "d3"]
kind           = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us     = 1.0
[[link]]
endpoints      = ["d1", "d2"]
kind           = "infiniband_400g"
bandwidth_gbps = 400.0
latency_us     = 5.0
"#;

    #[test]
    fn two_device_pair_path() {
        let spec = ClusterSpec::from_toml_str(TWO_H100).unwrap();
        let t = Topology::from_spec(&spec);
        let path = t.pair_path(0, 1).unwrap();
        assert_eq!(path.hops.len(), 1);
        assert_eq!(path.min_bandwidth_gbps(), 900.0);
        assert_eq!(path.total_latency_us(), 1.0);
    }

    #[test]
    fn self_path_is_empty() {
        let spec = ClusterSpec::from_toml_str(TWO_H100).unwrap();
        let t = Topology::from_spec(&spec);
        let path = t.pair_path(0, 0).unwrap();
        assert!(path.hops.is_empty());
        assert_eq!(path.total_latency_us(), 0.0);
        assert!(path.min_bandwidth_gbps().is_infinite());
    }

    #[test]
    fn multi_hop_finds_bottleneck() {
        // Note: this serde-derived parser treats unknown LinkKind variants as
        // an error, so this fixture uses `infiniband_400g` which is in the
        // enum (see `skein_ir::cluster::LinkKind`).
        // Hmm — actually we declared `Infiniband400g` which serializes as
        // `infiniband400g`. Use that name to keep the test green.
        let toml = FOUR_GPU_TWO_NODE.replace("infiniband_400g", "infiniband400g");
        let spec = ClusterSpec::from_toml_str(&toml).unwrap();
        let t = Topology::from_spec(&spec);
        // d0 → d3 must traverse d0—NVLink—d1—IB—d2—NVLink—d3
        let path = t.pair_path(0, 3).unwrap();
        assert_eq!(path.hops.len(), 3);
        assert_eq!(path.min_bandwidth_gbps(), 400.0);
        assert_eq!(path.total_latency_us(), 1.0 + 5.0 + 1.0);
    }

    #[test]
    fn collective_path_picks_critical_pair() {
        let toml = FOUR_GPU_TWO_NODE.replace("infiniband_400g", "infiniband400g");
        let spec = ClusterSpec::from_toml_str(&toml).unwrap();
        let t = Topology::from_spec(&spec);
        let path = t.collective_path(&[0, 1, 2, 3]).unwrap();
        // The worst pair is (d0, d3) — three hops, bottleneck 400 Gbps.
        assert_eq!(path.min_bandwidth_gbps(), 400.0);
        assert_eq!(path.total_latency_us(), 7.0);
    }

    #[test]
    fn disconnected_graph_errors() {
        let toml = r#"
num_devices = 2
[[node]]
id = "node0"
devices = ["d0"]
device_kind = "h100_sxm5"
device_memory_gb = 80
[[node]]
id = "node1"
devices = ["d1"]
device_kind = "h100_sxm5"
device_memory_gb = 80
"#;
        let spec = ClusterSpec::from_toml_str(toml).unwrap();
        let t = Topology::from_spec(&spec);
        assert!(matches!(t.pair_path(0, 1), Err(CostError::NoPath { .. })));
    }
}

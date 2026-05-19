//! `ClusterSpec` — the topology graph parsed from `cluster/<name>.toml`.
//!
//! The cluster is modeled as an undirected graph: vertices are devices,
//! edges are typed links (NVLink Gen4/Gen5, PCIe 5, InfiniBand 400 G, RoCE
//! 100 G, TCP 10 G). `skein_cost` walks this graph to compute multi-hop
//! collective costs.
//!
//! Validation is strict: device counts must agree, every link endpoint must
//! resolve to a known device, no self-links, no duplicates.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

use crate::error::ClusterError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    NvlinkGen4,
    NvlinkGen5,
    Pcie5,
    Infiniband400g,
    Roce100g,
    Tcp10g,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub devices: Vec<String>,
    pub device_kind: String,
    pub device_memory_gb: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Link {
    /// Two device ids. The link is undirected; ordering is not significant
    /// but the parser preserves the file's order for deterministic hashing.
    pub endpoints: [String; 2],
    pub kind: LinkKind,
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusterSpec {
    pub num_devices: u32,
    #[serde(default, rename = "node")]
    pub nodes: Vec<Node>,
    #[serde(default, rename = "link")]
    pub links: Vec<Link>,
}

impl ClusterSpec {
    pub fn from_toml_str(s: &str) -> Result<Self, ClusterError> {
        let spec: ClusterSpec = toml::from_str(s)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn from_toml_file(path: &Path) -> Result<Self, ClusterError> {
        let s = std::fs::read_to_string(path).map_err(|source| ClusterError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_toml_str(&s)
    }

    fn validate(&self) -> Result<(), ClusterError> {
        // Every device id must be unique across nodes; every node id unique.
        let mut node_ids: HashSet<&str> = HashSet::new();
        let mut device_ids: HashSet<&str> = HashSet::new();
        let mut total_devices: u32 = 0;
        for node in &self.nodes {
            if !node_ids.insert(node.id.as_str()) {
                return Err(ClusterError::DuplicateNode(node.id.clone()));
            }
            for d in &node.devices {
                if !device_ids.insert(d.as_str()) {
                    return Err(ClusterError::DuplicateDevice(d.clone()));
                }
                total_devices += 1;
            }
        }
        if total_devices != self.num_devices {
            return Err(ClusterError::DeviceCountMismatch {
                declared: self.num_devices,
                actual: total_devices,
            });
        }
        // Every link endpoint must resolve; no self-links.
        for link in &self.links {
            let [a, b] = &link.endpoints;
            if a == b {
                return Err(ClusterError::SelfLink {
                    a: a.clone(),
                    b: b.clone(),
                });
            }
            if !device_ids.contains(a.as_str()) {
                return Err(ClusterError::UnknownDeviceInLink { device: a.clone() });
            }
            if !device_ids.contains(b.as_str()) {
                return Err(ClusterError::UnknownDeviceInLink { device: b.clone() });
            }
        }
        Ok(())
    }

    /// All device ids in stable iteration order (the order they appear in
    /// the TOML, node-by-node).
    pub fn all_devices(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .flat_map(|n| n.devices.iter().map(|d| d.as_str()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO_H100: &str = r#"
num_devices = 2

[[node]]
id = "node0"
devices = ["d0", "d1"]
device_kind = "h100_sxm5"
device_memory_gb = 80

[[link]]
endpoints = ["d0", "d1"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
"#;

    #[test]
    fn parse_two_h100_sxm5() {
        let spec = ClusterSpec::from_toml_str(TWO_H100).unwrap();
        assert_eq!(spec.num_devices, 2);
        assert_eq!(spec.nodes.len(), 1);
        assert_eq!(spec.links.len(), 1);
        assert_eq!(spec.all_devices(), vec!["d0", "d1"]);
        assert_eq!(spec.links[0].kind, LinkKind::NvlinkGen4);
    }

    #[test]
    fn rejects_device_count_mismatch() {
        let bad = r#"
num_devices = 4

[[node]]
id = "node0"
devices = ["d0", "d1"]
device_kind = "h100_sxm5"
device_memory_gb = 80
"#;
        let err = ClusterSpec::from_toml_str(bad).unwrap_err();
        match err {
            ClusterError::DeviceCountMismatch {
                declared: 4,
                actual: 2,
            } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_device_in_link() {
        let bad = r#"
num_devices = 1

[[node]]
id = "node0"
devices = ["d0"]
device_kind = "h100_sxm5"
device_memory_gb = 80

[[link]]
endpoints = ["d0", "d99"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
"#;
        let err = ClusterSpec::from_toml_str(bad).unwrap_err();
        match err {
            ClusterError::UnknownDeviceInLink { device } => assert_eq!(device, "d99"),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_self_link() {
        let bad = r#"
num_devices = 1

[[node]]
id = "node0"
devices = ["d0"]
device_kind = "h100_sxm5"
device_memory_gb = 80

[[link]]
endpoints = ["d0", "d0"]
kind = "nvlink_gen4"
bandwidth_gbps = 900.0
latency_us = 1.0
"#;
        let err = ClusterSpec::from_toml_str(bad).unwrap_err();
        assert!(matches!(err, ClusterError::SelfLink { .. }));
    }

    #[test]
    fn rejects_duplicate_device() {
        let bad = r#"
num_devices = 2

[[node]]
id = "node0"
devices = ["d0"]
device_kind = "h100_sxm5"
device_memory_gb = 80

[[node]]
id = "node1"
devices = ["d0"]
device_kind = "h100_sxm5"
device_memory_gb = 80
"#;
        let err = ClusterSpec::from_toml_str(bad).unwrap_err();
        assert!(matches!(err, ClusterError::DuplicateDevice(d) if d == "d0"));
    }
}

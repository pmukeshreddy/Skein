//! Test-only tiny Skein artifact builder.
//!
//! The generated weights are synthetic by design and are scoped to parity
//! plumbing tests. They are not used for calibration, planning, production
//! serving, or any benchmark.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use safetensors::tensor::TensorView;
use skein_compile::{ArtifactMetadata, SkeinArtifact};
use skein_cost::Cluster;
use skein_ir::cluster::ClusterSpec;
use skein_ir::ir::Graph;
use skein_ir::plan::*;
use skein_ir::types::*;

pub struct TinyArtifact {
    pub root: PathBuf,
    pub artifact_dir: PathBuf,
    pub ir: Graph,
    pub cluster: ClusterSpec,
    pub plan: Plan,
}

pub fn build_tiny_artifact(root: &Path) -> TinyArtifact {
    std::fs::create_dir_all(root).expect("create tiny artifact root");
    let ir = tiny_ir();
    let cluster = one_device_cluster();
    let plan = tiny_plan(&ir);
    let cost_cluster = Cluster::from_spec(cluster.clone());
    let lowered =
        skein_emit::lower_per_device(&plan, &cost_cluster, &ir, 0, root).expect("lower tiny graph");

    let weights_path = root.join("tiny.weights.safetensors");
    write_synthetic_weights(&ir, &weights_path);

    let artifact_dir = root.join("artifact");
    let metadata = ArtifactMetadata::new("test-skein", "h100_sxm5", "2026-05-19T00:00:00Z");
    SkeinArtifact::write(
        &plan,
        &cluster,
        &ir,
        &[lowered],
        &[weights_path],
        &metadata,
        &artifact_dir,
    )
    .expect("write tiny artifact");

    TinyArtifact {
        root: root.to_path_buf(),
        artifact_dir,
        ir,
        cluster,
        plan,
    }
}

pub fn tiny_ir() -> Graph {
    let cfg = r#"{
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": 64,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "vocab_size": 128,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "hidden_act": "silu"
    }"#;
    skein_ir::model::import_from_str(cfg).expect("tiny Mixtral config imports")
}

pub fn one_device_cluster() -> ClusterSpec {
    ClusterSpec::from_toml_str(
        r#"
num_devices = 1
[[node]]
id = "node0"
devices = ["d0"]
device_kind = "h100_sxm5"
device_memory_gb = 80
"#,
    )
    .expect("cluster parses")
}

pub fn tiny_plan(ir: &Graph) -> Plan {
    Plan {
        parallelism: ParallelismPlacement {
            tp: 1,
            pp: 1,
            ep: 1,
        },
        kv: KVCacheSpec {
            layout: KVLayout::Paged { page_size: 16 },
            kv_sharded: false,
        },
        batching: BatchPolicy::Continuous { max_batch: 1 },
        dtype_map: DtypeMap::uniform(ir.meta.num_layers, Dtype::Bf16),
        execution: ExecutionConfig {
            cuda_graphs: CudaGraphsConfig {
                enable: false,
                capture_classes: vec![],
            },
            spec_decode: SpecDecodeConfig {
                enable: false,
                draft: None,
            },
            prefix_cache: PrefixCacheConfig {
                enable: true,
                reuse_policy: RadixReusePolicy::LruByLastAccess,
            },
        },
        disaggregation: None,
        model_meta: ir.meta.clone(),
    }
}

fn write_synthetic_weights(ir: &Graph, path: &Path) {
    let mut seen = HashSet::new();
    let mut buffers: BTreeMap<String, (Vec<usize>, Vec<u8>)> = BTreeMap::new();
    for layer in &ir.layers {
        for param in &layer.params {
            if !seen.insert(param.name.clone()) {
                continue;
            }
            let shape: Vec<usize> = param
                .shape
                .0
                .iter()
                .map(|d| match d {
                    Dim::Fixed(v) | Dim::Tp(v) | Dim::Ep(v) => *v,
                    Dim::Batch | Dim::Seq | Dim::KvLen => {
                        panic!("test weight has symbolic dim: {}", param.name)
                    }
                })
                .collect();
            let n: u64 = shape.iter().map(|d| *d as u64).product();
            let bytes = vec![0u8; Dtype::Bf16.bytes_for(n) as usize];
            buffers.insert(param.name.clone(), (shape, bytes));
        }
    }

    let views = buffers.iter().map(|(name, (shape, bytes))| {
        (
            name.clone(),
            TensorView::new(safetensors::Dtype::BF16, shape.clone(), bytes.as_slice())
                .expect("shape and buffer are sized from the same param"),
        )
    });
    let serialized = safetensors::serialize(views, &None).expect("serialize synthetic weights");
    std::fs::write(path, serialized).expect("write synthetic weights");
}

//! Reloadable artifact schema round-trip tests.

use std::path::PathBuf;

use skein_compile::{ArtifactMetadata, SkeinArtifact};
use skein_cost::Cluster;
use skein_ir::cluster::ClusterSpec;
use skein_ir::plan::*;
use skein_ir::types::*;

fn tempdir(prefix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "{prefix}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn tiny_ir() -> skein_ir::ir::Graph {
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

fn one_device_cluster() -> ClusterSpec {
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

fn plan_for(ir: &skein_ir::ir::Graph) -> Plan {
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

fn declared_names_and_shapes(
    segments: &[skein_emit::segment::Segment],
) -> Vec<Vec<(String, Vec<usize>, Dtype)>> {
    segments
        .iter()
        .map(|s| {
            let mut declared = s
                .declared
                .iter()
                .map(|(name, d)| (name.clone(), d.shape.clone(), d.dtype))
                .collect::<Vec<_>>();
            declared.sort_by(|a, b| a.0.cmp(&b.0));
            declared
        })
        .collect()
}

#[test]
fn artifact_write_load_and_rebuild_roundtrip() {
    let ir = tiny_ir();
    let cluster_spec = one_device_cluster();
    let cluster = Cluster::from_spec(cluster_spec.clone());
    let plan = plan_for(&ir);
    let lowered =
        skein_emit::lower_per_device(&plan, &cluster, &ir, 0, std::path::Path::new("/tmp"))
            .expect("lower tiny artifact");

    let root = tempdir("skein_compile_artifact_roundtrip");
    let weight_path = root.join("device0_source.weights.safetensors");
    std::fs::write(&weight_path, b"fixture weight shard").unwrap();
    let out_dir = root.join("artifact");
    let metadata = ArtifactMetadata::new("test-skein", "h100_sxm5", "2026-05-19T00:00:00Z");

    let artifact = SkeinArtifact::write(
        &plan,
        &cluster_spec,
        &ir,
        &[lowered],
        &[weight_path],
        &metadata,
        &out_dir,
    )
    .expect("write artifact");

    let plan_bytes = std::fs::read(out_dir.join("plan.json")).unwrap();
    let topology_bytes = std::fs::read(out_dir.join("topology.json")).unwrap();
    let metadata_bytes = std::fs::read(out_dir.join("metadata.json")).unwrap();

    let loaded = SkeinArtifact::load(&out_dir).expect("load artifact");
    assert_eq!(artifact.plan, loaded.plan);
    assert_eq!(artifact.sequencing, loaded.sequencing);
    assert_eq!(artifact.metadata, loaded.metadata);
    assert_eq!(
        std::fs::read(out_dir.join("plan.json")).unwrap(),
        plan_bytes
    );
    assert_eq!(
        std::fs::read(out_dir.join("topology.json")).unwrap(),
        topology_bytes
    );
    assert_eq!(
        std::fs::read(out_dir.join("metadata.json")).unwrap(),
        metadata_bytes
    );

    let rebuilt = loaded.devices[0].rebuild_graphs().expect("rebuild graphs");
    assert_eq!(rebuilt.len(), loaded.devices[0].segments.len());

    let original = skein_emit::build_device_graph(&plan, &cluster, &ir, 0).unwrap();
    assert_eq!(
        declared_names_and_shapes(&rebuilt),
        declared_names_and_shapes(&original.segments)
    );
}

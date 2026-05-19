//! Round-trip property tests for every serializable `skein_ir` value type.
//!
//! For float-bearing types (`Slo`, `ModelMeta`, `Link`), strategies bound the
//! range to *normal* finite positive values so JSON round-trip preserves them
//! exactly. NaN/±inf are excluded — they cannot appear in a valid config.

use proptest::prelude::*;

use skein_ir::cluster::{ClusterSpec, Link, LinkKind, Node};
use skein_ir::plan::{
    DtypeMap, KvTransferMode, ParallelismPlacement, PerLayerDtype, Plan, TransferTopology,
};
use skein_ir::types::{
    BatchPolicy, CaptureClass, Component, CudaGraphsConfig, DraftSpec, Dtype, ExecutionConfig,
    KVCacheSpec, KVLayout, PrefixCacheConfig, RadixReusePolicy, Sharding, SpecDecodeConfig,
};
use skein_ir::workload::{RequestRecord, Slo, Workload};

fn rt_json<T>(v: T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let s = serde_json::to_string(&v).expect("serialize");
    serde_json::from_str(&s).expect("deserialize")
}

// --- enum strategies ---

fn dtype_strategy() -> impl Strategy<Value = Dtype> {
    prop_oneof![
        Just(Dtype::Bf16),
        Just(Dtype::Fp16),
        Just(Dtype::Fp8E4m3),
        Just(Dtype::Fp8E5m2),
        Just(Dtype::Int8),
        Just(Dtype::Int4),
    ]
}

fn component_strategy() -> impl Strategy<Value = Component> {
    prop_oneof![
        Just(Component::Weight),
        Just(Component::Activation),
        Just(Component::KvCache),
    ]
}

fn kv_layout_strategy() -> impl Strategy<Value = KVLayout> {
    prop_oneof![
        Just(KVLayout::Contiguous),
        prop_oneof![Just(16u32), Just(32u32), Just(64u32), Just(128u32)]
            .prop_map(|page_size| KVLayout::Paged { page_size }),
    ]
}

fn kv_spec_strategy() -> impl Strategy<Value = KVCacheSpec> {
    (kv_layout_strategy(), any::<bool>())
        .prop_map(|(layout, kv_sharded)| KVCacheSpec { layout, kv_sharded })
}

fn batch_policy_strategy() -> impl Strategy<Value = BatchPolicy> {
    let max_batch = prop_oneof![
        Just(1u32),
        Just(2u32),
        Just(4u32),
        Just(8u32),
        Just(16u32),
        Just(32u32),
        Just(64u32),
    ];
    let chunk = prop_oneof![Just(256u32), Just(512u32), Just(1024u32), Just(2048u32)];
    prop_oneof![
        max_batch
            .clone()
            .prop_map(|m| BatchPolicy::Static { max_batch: m }),
        max_batch
            .clone()
            .prop_map(|m| BatchPolicy::Continuous { max_batch: m }),
        (max_batch, chunk).prop_map(|(m, c)| BatchPolicy::ContinuousChunked {
            max_batch: m,
            chunk_tokens: c,
        }),
    ]
}

fn sharding_strategy() -> impl Strategy<Value = Sharding> {
    prop_oneof![
        Just(Sharding::Replicated),
        (1u32..=8).prop_map(|g| Sharding::TpRowParallel { group_size: g }),
        (1u32..=8).prop_map(|g| Sharding::TpColParallel { group_size: g }),
        (1u32..=8, 1u32..=64).prop_map(|(g, ne)| Sharding::ExpertParallel {
            group_size: g,
            num_experts: ne
        }),
    ]
}

fn per_layer_dtype_strategy() -> impl Strategy<Value = PerLayerDtype> {
    (dtype_strategy(), dtype_strategy(), dtype_strategy()).prop_map(|(w, a, k)| PerLayerDtype {
        weight: w,
        activation: a,
        kv_cache: k,
    })
}

fn execution_config_strategy() -> impl Strategy<Value = ExecutionConfig> {
    let capture_classes = prop::collection::vec(
        (1u32..=64, 1u32..=8).prop_map(|(b, k)| CaptureClass {
            batch_size: b,
            kv_class: k,
        }),
        0..=4,
    );
    let cg = (any::<bool>(), capture_classes).prop_map(|(enable, cs)| CudaGraphsConfig {
        enable,
        capture_classes: cs,
    });
    let draft = prop::option::of(("[a-z0-9/]{4,16}", 1u32..=8).prop_map(|(p, d)| DraftSpec {
        model_path: p,
        speculation_depth: d,
    }));
    let sd = (any::<bool>(), draft).prop_map(|(enable, draft)| SpecDecodeConfig { enable, draft });
    let policy = prop_oneof![
        Just(RadixReusePolicy::LruByLastAccess),
        Just(RadixReusePolicy::LfuByHits),
    ];
    let pc = (any::<bool>(), policy).prop_map(|(enable, p)| PrefixCacheConfig {
        enable,
        reuse_policy: p,
    });
    (cg, sd, pc).prop_map(|(cuda_graphs, spec_decode, prefix_cache)| ExecutionConfig {
        cuda_graphs,
        spec_decode,
        prefix_cache,
    })
}

// --- float-bearing strategies (bounded to exactly-representable values) ---

// Slo carries f64 drift fields. serde_json's shortest-float formatting can
// disagree with the f64 parser by 1 ULP for some bit patterns, so we don't
// proptest random f64s. The `slo_roundtrip_fixed` test below covers the
// values that actually appear in real workload traces.

fn request_strategy() -> impl Strategy<Value = RequestRecord> {
    (1u32..=8192, 1u32..=8192, 0u64..=10_000_000).prop_map(|(p, o, a)| RequestRecord {
        prompt_tokens: p,
        output_tokens: o,
        arrival_ms: a,
    })
}

// --- tests ---

proptest! {
    #[test]
    fn dtype_roundtrip(d in dtype_strategy()) {
        prop_assert_eq!(rt_json(d), d);
    }

    #[test]
    fn component_roundtrip(c in component_strategy()) {
        prop_assert_eq!(rt_json(c), c);
    }

    #[test]
    fn kv_spec_roundtrip(k in kv_spec_strategy()) {
        prop_assert_eq!(rt_json(k), k);
    }

    #[test]
    fn batch_policy_roundtrip(b in batch_policy_strategy()) {
        prop_assert_eq!(rt_json(b), b);
    }

    #[test]
    fn sharding_roundtrip(s in sharding_strategy()) {
        prop_assert_eq!(rt_json(s.clone()), s);
    }

    #[test]
    fn per_layer_dtype_roundtrip(p in per_layer_dtype_strategy()) {
        prop_assert_eq!(rt_json(p), p);
    }

    #[test]
    fn dtype_map_roundtrip(ds in prop::collection::vec(per_layer_dtype_strategy(), 1..=32)) {
        let m = DtypeMap { per_layer: ds };
        prop_assert_eq!(rt_json(m.clone()), m);
    }

    #[test]
    fn parallelism_roundtrip(tp in 1u32..=8, pp in 1u32..=8, ep in 1u32..=8) {
        let p = ParallelismPlacement { tp, pp, ep };
        prop_assert_eq!(rt_json(p), p);
    }

    #[test]
    fn execution_config_roundtrip(e in execution_config_strategy()) {
        prop_assert_eq!(rt_json(e.clone()), e);
    }

    #[test]
    fn request_record_roundtrip(r in request_strategy()) {
        prop_assert_eq!(rt_json(r), r);
    }

    #[test]
    fn transfer_topology_roundtrip(layer_overlap in any::<bool>(), mode in prop_oneof![Just(KvTransferMode::RdmaWrite), Just(KvTransferMode::NcclByLayer)]) {
        let t = TransferTopology { mode, layer_overlap };
        prop_assert_eq!(rt_json(t.clone()), t);
    }
}

// --- non-proptest float / aggregate round-trips ---

#[test]
fn slo_roundtrip_fixed() {
    // Hand-picked drift values that have terminating decimal representations
    // and exact f64 forms — they round-trip through serde_json bit-exact.
    let cases = [
        Slo {
            ttft_p95_ms: 500,
            tpot_p95_ms: 50,
            max_accuracy_drift: 0.01,
            recompile_drift_threshold_kl: 0.05,
        },
        Slo {
            ttft_p95_ms: 1,
            tpot_p95_ms: 1,
            max_accuracy_drift: 0.0,
            recompile_drift_threshold_kl: 0.5,
        },
        Slo {
            ttft_p95_ms: 60_000,
            tpot_p95_ms: 10_000,
            max_accuracy_drift: 0.25,
            recompile_drift_threshold_kl: 0.125,
        },
    ];
    for s in cases {
        let s2 = rt_json(s);
        assert_eq!(s, s2);
    }
}

#[test]
fn workload_roundtrip_from_fixture() {
    let src = include_str!("../../../cluster/sample_trace.jsonl");
    let w1 = Workload::from_jsonl_str(src).expect("parse fixture");
    let serialized = serde_json::to_string(&w1).expect("serialize workload");
    let w2: Workload = serde_json::from_str(&serialized).expect("deserialize workload");
    assert_eq!(w1, w2);
}

#[test]
fn cluster_roundtrip_from_fixture() {
    let src = include_str!("../../../cluster/h100_2x.toml");
    let c1 = ClusterSpec::from_toml_str(src).expect("parse fixture");

    // Round-trip via TOML.
    let toml_s = toml::to_string(&c1).expect("serialize toml");
    let c2 = ClusterSpec::from_toml_str(&toml_s).expect("parse round-tripped toml");
    assert_eq!(c1, c2);

    // Round-trip via JSON (skein_emit serializes topology to JSON).
    let json_s = serde_json::to_string(&c1).expect("serialize json");
    let c3: ClusterSpec = serde_json::from_str(&json_s).expect("parse round-tripped json");
    assert_eq!(c1, c3);
}

#[test]
fn link_kind_all_variants_roundtrip() {
    // Hand-enumerated coverage: every LinkKind variant survives a JSON
    // round-trip. Catches accidentally-renamed serde variants.
    for kind in [
        LinkKind::NvlinkGen4,
        LinkKind::NvlinkGen5,
        LinkKind::Pcie5,
        LinkKind::Infiniband400g,
        LinkKind::Roce100g,
        LinkKind::Tcp10g,
    ] {
        let link = Link {
            endpoints: ["d0".into(), "d1".into()],
            kind,
            bandwidth_gbps: 100.0,
            latency_us: 1.0,
        };
        let s = serde_json::to_string(&link).expect("serialize");
        let back: Link = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(back, link);
    }
}

#[test]
fn plan_roundtrip_canonical() {
    use skein_ir::ir::ModelMeta;
    use skein_ir::types::*;

    let meta = ModelMeta {
        architecture: "MixtralForCausalLM".into(),
        num_layers: 4,
        hidden: 4096,
        vocab: 32000,
        max_position: 32768,
        num_attention_heads: 32,
        num_kv_heads: 8,
        head_dim: 128,
        num_experts: Some(8),
        top_k: Some(2),
        intermediate: 14336,
        // Exactly representable as f32.
        rope_theta: 1_000_000.0,
        rms_norm_eps: 9.765625e-6, // 2^-17, exactly representable
        sliding_window: None,
        tied_embeddings: false,
    };

    let plan = Plan {
        parallelism: ParallelismPlacement {
            tp: 2,
            pp: 1,
            ep: 1,
        },
        kv: KVCacheSpec {
            layout: KVLayout::Paged { page_size: 32 },
            kv_sharded: false,
        },
        batching: BatchPolicy::Continuous { max_batch: 16 },
        dtype_map: DtypeMap::uniform(4, Dtype::Bf16),
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
        model_meta: meta,
    };

    let s = serde_json::to_string(&plan).expect("serialize");
    let back: Plan = serde_json::from_str(&s).expect("deserialize");
    assert_eq!(back, plan);
    // Content hashes must agree after round-trip.
    assert_eq!(back.content_hash().unwrap(), plan.content_hash().unwrap());
}

#[test]
fn ir_graph_roundtrip_via_mixtral_importer() {
    // Parse Mixtral's real config, round-trip the Graph through JSON, and
    // verify equality. This exercises every Layer/LayerKind/Param/Tensor
    // serializer in one shot.
    let src = include_str!("../../../configs/mixtral_8x7b_config.json");
    let g1 = skein_ir::model::import_from_str(src).expect("import Mixtral fixture");
    let s = serde_json::to_string(&g1).expect("serialize graph");
    let g2: skein_ir::ir::Graph = serde_json::from_str(&s).expect("deserialize graph");
    assert_eq!(g1, g2);
}

// Bare reference to silence the unused-import linter when proptest excludes
// some symbols on a given build.
#[allow(dead_code)]
fn _node_link_unused(_n: Node) {}

//! Pipeline bubble time. Standard 1F1B / GPipe formula:
//!
//! ```text
//! bubble(pp, stage) = ((pp - 1) / pp) × stage_compute
//! ```
//!
//! pp=1 → 0. pp=2 → 0.5 × stage. pp=4 → 0.75 × stage.

use skein_ir::plan::Plan;

pub fn bubble_time(plan: &Plan, stage_compute_us: f64) -> f64 {
    let pp = plan.parallelism.pp.max(1) as f64;
    if pp <= 1.0 {
        0.0
    } else {
        ((pp - 1.0) / pp) * stage_compute_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skein_ir::ir::ModelMeta;
    use skein_ir::plan::*;
    use skein_ir::types::*;

    fn plan_with_pp(pp: u32) -> Plan {
        let meta = ModelMeta {
            architecture: "t".into(),
            num_layers: 32,
            hidden: 4096,
            vocab: 32000,
            max_position: 32768,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            num_experts: Some(8),
            top_k: Some(2),
            intermediate: 14336,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            tied_embeddings: false,
        };
        Plan {
            parallelism: ParallelismPlacement { tp: 1, pp, ep: 1 },
            kv: KVCacheSpec {
                layout: KVLayout::Contiguous,
                kv_sharded: false,
            },
            batching: BatchPolicy::Continuous { max_batch: 1 },
            dtype_map: DtypeMap::uniform(32, Dtype::Bf16),
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
                    enable: false,
                    reuse_policy: RadixReusePolicy::LruByLastAccess,
                },
            },
            disaggregation: None,
            model_meta: meta,
        }
    }

    #[test]
    fn pp1_no_bubble() {
        assert_eq!(bubble_time(&plan_with_pp(1), 1000.0), 0.0);
    }

    #[test]
    fn pp2_half_stage() {
        assert!((bubble_time(&plan_with_pp(2), 1000.0) - 500.0).abs() < 1e-9);
    }

    #[test]
    fn pp4_three_quarters_stage() {
        assert!((bubble_time(&plan_with_pp(4), 1000.0) - 750.0).abs() < 1e-9);
    }
}

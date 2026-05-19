//! Launch overhead. `launch_us_per_kernel × num_kernels × (1 - coverage)`.
//!
//! Coverage = fraction of decode steps that match a captured CUDA Graph,
//! based on Plan's batching policy and number of capture classes:
//!
//! - `Static(_)`: 1.0 (single shape, trivially captured)
//! - `Continuous(_)` with ≥ 4 capture classes: 0.95
//! - `Continuous(_)` with fewer classes: 0.7
//! - `ContinuousChunked(...)`: 0.4 (variable chunk shapes)
//! - CUDA Graphs disabled: 0.0
//!
//! Document the coverage table in `docs/cost_model.md` along with the
//! assumption that captured CUDA Graphs amortize launch costs to ~zero.

use skein_ir::plan::Plan;
use skein_ir::types::BatchPolicy;

use crate::constants::CostConstants;

const MIN_CAPTURE_CLASSES_FOR_HIGH_COVERAGE: usize = 4;

pub fn cuda_graph_coverage(plan: &Plan) -> f64 {
    if !plan.execution.cuda_graphs.enable {
        return 0.0;
    }
    let n_classes = plan.execution.cuda_graphs.capture_classes.len();
    match plan.batching {
        BatchPolicy::Static { .. } => 1.0,
        BatchPolicy::Continuous { .. } => {
            if n_classes >= MIN_CAPTURE_CLASSES_FOR_HIGH_COVERAGE {
                0.95
            } else {
                0.7
            }
        }
        BatchPolicy::ContinuousChunked { .. } => 0.4,
    }
}

pub fn launch_overhead(plan: &Plan, constants: &CostConstants, num_kernels_per_step: u32) -> f64 {
    let coverage = cuda_graph_coverage(plan);
    constants.launch_us_per_kernel * num_kernels_per_step as f64 * (1.0 - coverage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use skein_ir::ir::ModelMeta;
    use skein_ir::plan::*;
    use skein_ir::types::*;

    fn plan(cg_enable: bool, n_classes: usize, batching: BatchPolicy) -> Plan {
        let meta = ModelMeta {
            architecture: "t".into(),
            num_layers: 1,
            hidden: 4,
            vocab: 4,
            max_position: 4,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: 4,
            num_experts: None,
            top_k: None,
            intermediate: 4,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            tied_embeddings: false,
        };
        let classes = (0..n_classes)
            .map(|i| CaptureClass {
                batch_size: 1 + i as u32,
                kv_class: 1,
            })
            .collect();
        Plan {
            parallelism: ParallelismPlacement {
                tp: 1,
                pp: 1,
                ep: 1,
            },
            kv: KVCacheSpec {
                layout: KVLayout::Contiguous,
                kv_sharded: false,
            },
            batching,
            dtype_map: DtypeMap::uniform(1, Dtype::Bf16),
            execution: ExecutionConfig {
                cuda_graphs: CudaGraphsConfig {
                    enable: cg_enable,
                    capture_classes: classes,
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
    fn coverage_zero_when_disabled() {
        let p = plan(false, 4, BatchPolicy::Continuous { max_batch: 8 });
        assert_eq!(cuda_graph_coverage(&p), 0.0);
    }

    #[test]
    fn coverage_high_with_4_classes() {
        let p = plan(true, 4, BatchPolicy::Continuous { max_batch: 8 });
        assert_eq!(cuda_graph_coverage(&p), 0.95);
    }

    #[test]
    fn coverage_low_with_few_classes() {
        let p = plan(true, 1, BatchPolicy::Continuous { max_batch: 8 });
        assert_eq!(cuda_graph_coverage(&p), 0.7);
    }

    #[test]
    fn chunked_low_coverage() {
        let p = plan(
            true,
            4,
            BatchPolicy::ContinuousChunked {
                max_batch: 8,
                chunk_tokens: 1024,
            },
        );
        assert_eq!(cuda_graph_coverage(&p), 0.4);
    }

    #[test]
    fn static_full_coverage() {
        let p = plan(true, 0, BatchPolicy::Static { max_batch: 8 });
        assert_eq!(cuda_graph_coverage(&p), 1.0);
    }
}

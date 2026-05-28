//! `GlobalConfig` — the outer-loop's one-row candidate. Holds every Plan
//! field that isn't decided by the inner DP. The DP returns a `DtypeMap`,
//! and `compose_plan` assembles the full `Plan`.

use skein_ir::ir::ModelMeta;
use skein_ir::plan::{DtypeMap, ParallelismPlacement, Plan};
use skein_ir::types::{
    BatchPolicy, CudaGraphsConfig, ExecutionConfig, KVCacheSpec, KVLayout, PrefixCacheConfig,
    SpecDecodeConfig,
};

#[derive(Debug, Clone)]
pub struct GlobalConfig {
    pub parallelism: ParallelismPlacement,
    pub kv_layout: KVLayout,
    pub kv_shard: bool,
    pub batch: BatchPolicy,
    pub cuda_graphs: CudaGraphsConfig,
    pub spec_decode: SpecDecodeConfig,
    pub prefix_cache: PrefixCacheConfig,
}

/// Assemble a `Plan` from a `(GlobalConfig, DtypeMap, ModelMeta)` triple.
pub fn compose_plan(global: GlobalConfig, dtype_map: DtypeMap, model_meta: ModelMeta) -> Plan {
    Plan {
        parallelism: global.parallelism,
        kv: KVCacheSpec {
            layout: global.kv_layout,
            kv_sharded: global.kv_shard,
        },
        batching: global.batch,
        dtype_map,
        execution: ExecutionConfig {
            cuda_graphs: global.cuda_graphs,
            spec_decode: global.spec_decode,
            prefix_cache: global.prefix_cache,
        },
        disaggregation: None,
        model_meta,
    }
}

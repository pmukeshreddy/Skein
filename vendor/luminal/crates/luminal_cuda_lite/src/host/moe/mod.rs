use std::sync::{Arc, OnceLock};

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND},
        extract_expr,
    },
    op::{EgglogOp, LLIROp},
    prelude::*,
    shape::Expression,
};

use crate::{
    compile_module_image_for_current_device,
    cudarc::{
        cublas::sys::cublasOperation_t,
        cublaslt::{
            CudaBlasLT, MatmulShared,
            sys::{
                cublasComputeType_t, cublasLtMatmul, cublasLtMatmulAlgoGetHeuristic,
                cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, cublasLtMatmulDescCreate,
                cublasLtMatmulDescDestroy, cublasLtMatmulDescSetAttribute,
                cublasLtMatmulHeuristicResult_t, cublasLtMatmulPreference_t,
                cublasLtMatmulPreferenceAttributes_t, cublasLtMatmulPreferenceCreate,
                cublasLtMatmulPreferenceDestroy, cublasLtMatmulPreferenceSetAttribute,
                cublasLtMatrixLayout_t, cublasLtMatrixLayoutCreate, cublasLtMatrixLayoutDestroy,
                cudaDataType,
            },
        },
        driver::{
            CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
        },
    },
    host::{DeviceBuffer, HostOp},
    try_create_cublaslt,
};

const WORKSPACE_SIZE: usize = 32 * 1024 * 1024; // 32 MiB

/// Fused GLU-MoE HostOp matched via egglog pattern.
///
/// Replaces the expert computation subgraph (expert gathers + matmuls + gated
/// activation + weighted sum) with an efficient cuBLASLt implementation.
///
/// Inputs (graph edges, in order):
///   0: x              [seq, hidden]                        F32
///   1: topk_indices   [seq, k]                             Int
///   2: topk_values    [seq, k]                             F32
///   3: gate_up_w      [E, gate_up_dim, hidden]             BF16
///   4: down_w         [E, hidden, intermediate]             BF16
///   5: mode_aux
///      - SwiGLU/SwiGLUNormalized: ignored (rewriter wires `topk_values` again)
///      - GemmaGELU: per_expert_scale [E]                   F32
///
/// Output: [seq, hidden] F32
pub struct GLUMoE {
    pub(crate) mode: GLUMoEMode,
    /// Product of gate_up weight dimensions per expert (gate_up_dim * hidden) used for gather stride
    gu_io: Expression,
    /// Product of down weight dimensions per expert (hidden * intermediate) used for gather stride
    dn_io: Expression,
    /// K dimension of gate_up matmul (= hidden)
    gu_matmul_k: Expression,
    /// K dimension of down matmul (= intermediate)
    dn_matmul_k: Expression,
    /// K experts to sum over (= top_k)
    output_k: Expression,
    /// Total elements in a single gate_up expert weight matrix
    gu_within_range: Expression,
    /// Total elements in a single down expert weight matrix
    dn_within_range: Expression,
    cublaslt: OnceLock<Arc<CudaBlasLT>>,
    #[allow(clippy::type_complexity)]
    module: OnceLock<(
        Arc<CudaModule>,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
    )>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GLUMoEMode {
    SwiGLU,
    GemmaGELU,
    SwiGLUNormalized,
}

impl GLUMoEMode {
    fn from_mode_id(mode_id: usize) -> Self {
        match mode_id {
            0 => Self::SwiGLU,
            1 => Self::GemmaGELU,
            2 => Self::SwiGLUNormalized,
            other => {
                panic!("Unknown GLUMoE mode id: {other}");
            }
        }
    }

    fn activation_kernel_mode(self) -> i32 {
        match self {
            Self::SwiGLU | Self::SwiGLUNormalized => 0,
            Self::GemmaGELU => 1,
        }
    }
}

impl Default for GLUMoE {
    fn default() -> Self {
        Self {
            mode: GLUMoEMode::SwiGLU,
            gu_io: Expression::default(),
            dn_io: Expression::default(),
            gu_matmul_k: Expression::default(),
            dn_matmul_k: Expression::default(),
            output_k: Expression::default(),
            gu_within_range: Expression::default(),
            dn_within_range: Expression::default(),
            cublaslt: OnceLock::new(),
            module: OnceLock::new(),
        }
    }
}

impl std::fmt::Debug for GLUMoE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GLUMoE")
            .field("mode", &self.mode)
            .field("gu_io", &self.gu_io)
            .field("dn_io", &self.dn_io)
            .field("gu_matmul_k", &self.gu_matmul_k)
            .field("dn_matmul_k", &self.dn_matmul_k)
            .field("output_k", &self.output_k)
            .finish()
    }
}

impl Clone for GLUMoE {
    fn clone(&self) -> Self {
        Self {
            mode: self.mode,
            gu_io: self.gu_io,
            dn_io: self.dn_io,
            gu_matmul_k: self.gu_matmul_k,
            dn_matmul_k: self.dn_matmul_k,
            output_k: self.output_k,
            gu_within_range: self.gu_within_range,
            dn_within_range: self.dn_within_range,
            cublaslt: OnceLock::new(),
            module: OnceLock::new(),
        }
    }
}

impl GLUMoE {
    fn get_cublaslt(&self, stream: &Arc<CudaStream>) -> anyhow::Result<Arc<CudaBlasLT>> {
        if let Some(cublaslt) = self.cublaslt.get() {
            return Ok(cublaslt.clone());
        }
        let created = try_create_cublaslt(stream.clone()).map_err(|message| {
            anyhow::anyhow!("cuBLASLt unavailable on this machine: {message}")
        })?;
        let _ = self.cublaslt.set(created.clone());
        Ok(created)
    }

    #[allow(clippy::type_complexity)]
    fn get_kernels(
        &self,
        stream: &Arc<CudaStream>,
    ) -> &(
        Arc<CudaModule>,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
    ) {
        self.module.get_or_init(|| {
            let src = r#"
#include <cuda_bf16.h>

extern "C" __global__ void f32_to_bf16(unsigned long long in_ptr, unsigned long long out_ptr, int n) {
    const float* in_ = (const float*)in_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in_[i]);
}

extern "C" __global__ void glu_activation_bf16(
    unsigned long long gate_up_ptr,
    unsigned long long out_ptr,
    int intermediate,
    int mode
) {
    const __nv_bfloat16* gate_up = (const __nv_bfloat16*)gate_up_ptr;
    __nv_bfloat16* out = (__nv_bfloat16*)out_ptr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate) {
        float gate = __bfloat162float(gate_up[i]);
        float up   = __bfloat162float(gate_up[i + intermediate]);
        float activated;
        if (mode == 0) {
            activated = gate / (1.0f + expf(-gate));
        } else {
            float scaled = 1.5957691216f * gate * (1.0f + 0.044715f * gate * gate);
            activated = gate / (1.0f + expf(-scaled));
        }
        out[i] = __float2bfloat16(activated * up);
    }
}

// Fully on-device fused MoE: expert selection (topk_idx) and weighting
// (topk_vals) are read straight from device memory, so the op never copies
// indices to the host or issues per-expert GEMMs from a host loop. For decode
// (seq small) these are GEMVs — memory-bandwidth bound — so a coalesced custom
// kernel matches cuBLAS while removing the per-layer device->host sync.

// gate_up GEMV + gated activation, fused. One block per (intermediate index o,
// (token,slot)); the block reduces the gate row and up row dot-products over
// `hidden`, then writes silu(gate)*up (act_mode 0) / gelu(gate)*up (act_mode 1).
// gate_up GEMV + gated activation, fused. float4 (8 bf16) vectorized loads +
// warp-shuffle reduction. One block per (intermediate o, (token,slot)).
extern "C" __global__ void moe_gate_up_act(
    unsigned long long x_bf16_ptr,    // [seq, hidden] bf16
    unsigned long long topk_idx_ptr,  // [seq, idx_stride] i32
    unsigned long long gate_up_ptr,   // [E, gate_up_dim, hidden] bf16
    unsigned long long hid_ptr,       // [seq*top_k, intermediate] bf16
    int hidden, int intermediate, int gate_up_dim, int top_k, int idx_stride, int seq, int act_mode
) {
    int o = blockIdx.x;
    int tj = blockIdx.y;
    int t = tj / top_k;
    int slot = tj % top_k;
    if (t >= seq || o >= intermediate) return;
    int expert = ((const int*)topk_idx_ptr)[(long long)t * idx_stride + slot];
    const __nv_bfloat16* x = (const __nv_bfloat16*)x_bf16_ptr + (long long)t * hidden;
    const __nv_bfloat16* W = (const __nv_bfloat16*)gate_up_ptr + (long long)expert * gate_up_dim * hidden;
    const __nv_bfloat16* gate_row = W + (long long)o * hidden;
    const __nv_bfloat16* up_row   = W + (long long)(o + intermediate) * hidden;

    float gacc = 0.f, uacc = 0.f;
    int n4 = hidden >> 3;
    const float4* xp = (const float4*)x;
    const float4* gp = (const float4*)gate_row;
    const float4* upp = (const float4*)up_row;
    for (int j = threadIdx.x; j < n4; j += blockDim.x) {
        float4 xb = xp[j], gb = gp[j], ub = upp[j];
        const __nv_bfloat16* xh = (const __nv_bfloat16*)&xb;
        const __nv_bfloat16* gh = (const __nv_bfloat16*)&gb;
        const __nv_bfloat16* uh = (const __nv_bfloat16*)&ub;
        #pragma unroll
        for (int e = 0; e < 8; e++) { float xf = __bfloat162float(xh[e]); gacc += __bfloat162float(gh[e]) * xf; uacc += __bfloat162float(uh[e]) * xf; }
    }
    for (int j = n4 * 8 + (int)threadIdx.x; j < hidden; j += blockDim.x) {
        float xf = __bfloat162float(x[j]); gacc += __bfloat162float(gate_row[j]) * xf; uacc += __bfloat162float(up_row[j]) * xf;
    }
    for (int s = 16; s > 0; s >>= 1) { gacc += __shfl_down_sync(0xffffffffu, gacc, s); uacc += __shfl_down_sync(0xffffffffu, uacc, s); }
    __shared__ float sg[32];
    __shared__ float su[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = (blockDim.x + 31) >> 5;
    if (lane == 0) { sg[warp] = gacc; su[warp] = uacc; }
    __syncthreads();
    if (warp == 0) {
        gacc = (lane < nwarp) ? sg[lane] : 0.f;
        uacc = (lane < nwarp) ? su[lane] : 0.f;
        for (int s = 16; s > 0; s >>= 1) { gacc += __shfl_down_sync(0xffffffffu, gacc, s); uacc += __shfl_down_sync(0xffffffffu, uacc, s); }
        if (lane == 0) {
            float gate = gacc, up = uacc, act;
            if (act_mode == 0) { act = gate / (1.0f + expf(-gate)); }
            else { float sc = 1.5957691216f * gate * (1.0f + 0.044715f * gate * gate); act = gate / (1.0f + expf(-sc)); }
            ((__nv_bfloat16*)hid_ptr)[(long long)tj * intermediate + o] = __float2bfloat16(act * up);
        }
    }
}

// down GEMV + weighted combine, fused. float4 vectorized loads + warp-shuffle.
// One block per (hidden h, token t); loops the selected experts.
extern "C" __global__ void moe_down_combine(
    unsigned long long hid_ptr,       // [seq*top_k, intermediate] bf16
    unsigned long long topk_idx_ptr,  // [seq, idx_stride] i32
    unsigned long long topk_vals_ptr, // [seq, vals_stride] f32
    unsigned long long scale_ptr,     // [E] f32 (gemma) or unused
    unsigned long long down_ptr,      // [E, hidden, intermediate] bf16
    unsigned long long out_ptr,       // [seq, hidden] f32
    int hidden, int intermediate, int top_k, int idx_stride, int vals_stride, int seq,
    int normalize, int use_scale
) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    if (t >= seq || h >= hidden) return;
    const int* idx = (const int*)topk_idx_ptr;
    const float* vals = (const float*)topk_vals_ptr;
    float inv_norm = 1.0f;
    if (normalize) {
        float ssum = 0.f;
        for (int j = 0; j < top_k; j++) ssum += vals[(long long)t * vals_stride + j];
        inv_norm = (ssum != 0.f) ? (1.0f / ssum) : 0.f;
    }
    int n4 = intermediate >> 3;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = (blockDim.x + 31) >> 5;
    __shared__ float sd[32];
    float out_acc = 0.f;
    for (int jx = 0; jx < top_k; jx++) {
        int expert = idx[(long long)t * idx_stride + jx];
        float w = vals[(long long)t * vals_stride + jx] * inv_norm;
        if (use_scale) w *= ((const float*)scale_ptr)[expert];
        const __nv_bfloat16* D = (const __nv_bfloat16*)down_ptr + (long long)expert * hidden * intermediate + (long long)h * intermediate;
        const __nv_bfloat16* hd = (const __nv_bfloat16*)hid_ptr + (long long)(t * top_k + jx) * intermediate;
        const float4* dp = (const float4*)D;
        const float4* hp = (const float4*)hd;
        float dot = 0.f;
        for (int j = threadIdx.x; j < n4; j += blockDim.x) {
            float4 db = dp[j], hb = hp[j];
            const __nv_bfloat16* dh = (const __nv_bfloat16*)&db;
            const __nv_bfloat16* hh = (const __nv_bfloat16*)&hb;
            #pragma unroll
            for (int e = 0; e < 8; e++) dot += __bfloat162float(dh[e]) * __bfloat162float(hh[e]);
        }
        for (int j = n4 * 8 + (int)threadIdx.x; j < intermediate; j += blockDim.x) dot += __bfloat162float(D[j]) * __bfloat162float(hd[j]);
        for (int s = 16; s > 0; s >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, s);
        if (lane == 0) sd[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float r = (lane < nwarp) ? sd[lane] : 0.f;
            for (int s = 16; s > 0; s >>= 1) r += __shfl_down_sync(0xffffffffu, r, s);
            if (lane == 0) out_acc += w * r;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ((float*)out_ptr)[(long long)t * hidden + h] = out_acc;
}
"#;
            let ptx = compile_module_image_for_current_device(stream.context(), src).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let f32_to_bf16 = module.load_function("f32_to_bf16").unwrap();
            let activation = module.load_function("glu_activation_bf16").unwrap();
            let gate_up_act = module.load_function("moe_gate_up_act").unwrap();
            let down_combine = module.load_function("moe_down_combine").unwrap();
            (module, f32_to_bf16, activation, gate_up_act, down_combine)
        })
    }
}

impl EgglogOp for GLUMoE {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "GLUMoE",
            &[
                ("gu_io", EXPRESSION),
                ("dn_io", EXPRESSION),
                ("gu_matmul_k", EXPRESSION),
                ("dn_matmul_k", EXPRESSION),
                ("output_k", EXPRESSION),
                ("gu_within_range", EXPRESSION),
                ("dn_within_range", EXPRESSION),
                ("mode", EXPRESSION),
            ],
        )
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![
            Rule::raw(
                "(rule
                (
                    (= ?e (Op (GLUMoE ?gu_io ?dn_io ?gu_matmul_k ?dn_matmul_k ?output_k ?gu_within_range ?dn_within_range ?mode) ?inputs))
                )
                (
                    (set (dtype ?e) (F32))
                )
                :ruleset dtype_prop
            )",
            ),
            Rule::raw(include_str!["glumoe_rewrite.egg"]),
        ]
    }

    fn n_inputs(&self) -> usize {
        6
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let gu_io = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let dn_io = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let gu_matmul_k = extract_expr(egraph, kind_children[2], expr_cache).unwrap();
        let dn_matmul_k = extract_expr(egraph, kind_children[3], expr_cache).unwrap();
        let output_k = extract_expr(egraph, kind_children[4], expr_cache).unwrap();
        let gu_within_range = extract_expr(egraph, kind_children[5], expr_cache).unwrap();
        let dn_within_range = extract_expr(egraph, kind_children[6], expr_cache).unwrap();
        let mode_expr = extract_expr(egraph, kind_children[7], expr_cache).unwrap();
        let mode_id = mode_expr
            .to_usize()
            .unwrap_or_else(|| panic!("GLUMoE mode must be static, got expression: {mode_expr}"));
        let mode = GLUMoEMode::from_mode_id(mode_id);

        let extracted = GLUMoE {
            mode,
            gu_io,
            dn_io,
            gu_matmul_k,
            dn_matmul_k,
            output_k,
            gu_within_range,
            dn_within_range,
            cublaslt: OnceLock::new(),
            module: OnceLock::new(),
        };

        let op = LLIROp::new::<dyn HostOp>(Box::new(extracted) as Box<dyn HostOp>);
        // Return the 6 IR inputs: x, topk_idx, topk_values, gate_up_w, down_w, mode_aux
        (op, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for GLUMoE {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if inputs.len() < 6 {
            anyhow::bail!("GLUMoE expected at least 6 inputs, got {}", inputs.len());
        }

        // Resolve dimensions
        let hidden = self
            .gu_matmul_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE hidden dimension is unresolved"))?;
        let intermediate = self
            .dn_matmul_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE intermediate dimension is unresolved"))?;
        let top_k = self
            .output_k
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE top-k dimension is unresolved"))?;
        let gu_io = self
            .gu_io
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE gate/up stride is unresolved"))?;
        let dn_io = self
            .dn_io
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE down stride is unresolved"))?;

        if hidden == 0 || intermediate == 0 {
            anyhow::bail!(
                "GLUMoE got zero-sized matmul dimensions: hidden={hidden}, intermediate={intermediate}"
            );
        }
        if top_k == 0 {
            return Ok(());
        }
        if gu_io % hidden != 0 {
            anyhow::bail!("GLUMoE gate/up stride {gu_io} is not divisible by hidden {hidden}");
        }
        if dn_io % intermediate != 0 {
            anyhow::bail!(
                "GLUMoE down stride {dn_io} is not divisible by intermediate {intermediate}"
            );
        }

        let gate_up_dim = gu_io / hidden; // gate_up_dim = 2 * intermediate for GLU
        let down_hidden = dn_io / intermediate;
        if gate_up_dim != intermediate * 2 {
            anyhow::bail!(
                "GLUMoE expected gate/up dim {} to equal 2 * intermediate {}",
                gate_up_dim,
                intermediate * 2
            );
        }
        if down_hidden != hidden {
            anyhow::bail!("GLUMoE down hidden {down_hidden} does not match hidden {hidden}");
        }

        let output_bytes = self
            .output_bytes()
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("GLUMoE output byte size is unresolved"))?;
        if output_bytes % (hidden * 4) != 0 {
            anyhow::bail!(
                "GLUMoE output bytes {output_bytes} are not divisible by hidden bytes {}",
                hidden * 4
            );
        }
        let seq = output_bytes / (hidden * 4);
        if seq == 0 {
            return Ok(());
        }

        let get_buffer = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers.get(&node).copied().ok_or_else(|| {
                anyhow::anyhow!("GLUMoE missing {name} buffer for LLIR node {node:?}")
            })
        };

        // Get input/output buffers
        let x_buf = get_buffer("x", inputs[0])?; // [seq, hidden] F32
        let topk_idx_buf = get_buffer("topk indices", inputs[1])?; // [seq, k] Int
        let topk_vals_buf = get_buffer("topk values", inputs[2])?; // [seq, k] F32
        let gate_up_buf = get_buffer("gate/up weights", inputs[3])?; // [E, gate_up_dim, hidden] BF16
        let down_buf = get_buffer("down weights", inputs[4])?; // [E, hidden, intermediate] BF16
        let mode_aux_buf = get_buffer("mode aux", inputs[5])?;
        let output_buf = get_buffer("output", self_node)?; // [seq, hidden] F32

        let min_topk_bytes = seq * top_k * 4;
        if x_buf.len() < output_bytes {
            anyhow::bail!(
                "GLUMoE x buffer too small: have {} bytes, need {output_bytes}",
                x_buf.len()
            );
        }
        if topk_idx_buf.len() < min_topk_bytes {
            anyhow::bail!(
                "GLUMoE topk index buffer too small: have {} bytes, need {min_topk_bytes}",
                topk_idx_buf.len()
            );
        }
        if topk_vals_buf.len() < min_topk_bytes {
            anyhow::bail!(
                "GLUMoE topk value buffer too small: have {} bytes, need {min_topk_bytes}",
                topk_vals_buf.len()
            );
        }
        if output_buf.len() < output_bytes {
            anyhow::bail!(
                "GLUMoE output buffer too small: have {} bytes, need {output_bytes}",
                output_buf.len()
            );
        }

        let gu_stride_bytes = gate_up_dim * hidden * 2;
        let down_stride_bytes = hidden * intermediate * 2;
        if gu_stride_bytes == 0 || gate_up_buf.len() % gu_stride_bytes != 0 {
            anyhow::bail!(
                "GLUMoE gate/up weight buffer has {} bytes, not a multiple of per-expert stride {gu_stride_bytes}",
                gate_up_buf.len()
            );
        }
        let num_experts = gate_up_buf.len() / gu_stride_bytes;
        if num_experts == 0 {
            anyhow::bail!("GLUMoE has no expert weights");
        }
        if down_buf.len() < num_experts * down_stride_bytes {
            anyhow::bail!(
                "GLUMoE down weight buffer too small: have {} bytes, need {}",
                down_buf.len(),
                num_experts * down_stride_bytes
            );
        }

        // Get raw device pointer addresses
        let x_ptr = buf_ptr(x_buf, stream);
        let gate_up_ptr = buf_ptr(gate_up_buf, stream);
        let down_ptr = buf_ptr(down_buf, stream);
        let output_ptr = buf_ptr(output_buf, stream);

        // Fully on-device dispatch: topk_idx / topk_vals stay on the GPU; the
        // two fused kernels read them directly, so there is NO per-layer
        // device->host copy and NO host-side per-expert GEMM loop.
        let kernels = self.get_kernels(stream);
        let f32_to_bf16_fn = &kernels.1;
        let gate_up_act_fn = &kernels.3;
        let down_combine_fn = &kernels.4;

        let topk_idx_ptr = buf_ptr(topk_idx_buf, stream);
        let topk_vals_ptr = buf_ptr(topk_vals_buf, stream);

        // Row strides come from buffer sizes (no copy of the data itself).
        let topk_idx_elems = topk_idx_buf.len() / 4;
        let topk_vals_elems = topk_vals_buf.len() / 4;
        if seq == 0 || !topk_idx_elems.is_multiple_of(seq) || !topk_vals_elems.is_multiple_of(seq)
        {
            anyhow::bail!(
                "GLUMoE topk buffers (idx {topk_idx_elems}, vals {topk_vals_elems}) not divisible by seq {seq}"
            );
        }
        let idx_stride = topk_idx_elems / seq;
        let vals_stride = topk_vals_elems / seq;
        if idx_stride < top_k || vals_stride < top_k {
            anyhow::bail!(
                "GLUMoE topk row stride (idx {idx_stride}, vals {vals_stride}) smaller than top_k {top_k}"
            );
        }
        let _ = num_experts;

        // Scratch: x as bf16 [seq, hidden]; gated hidden [seq*top_k, intermediate] bf16.
        let x_bf16_buf = unsafe { stream.alloc::<u8>(seq * hidden * 2)? };
        let hid_buf = unsafe { stream.alloc::<u8>(seq * top_k * intermediate * 2)? };
        let xbf16_ptr = slice_ptr(&x_bf16_buf, stream);
        let hid_ptr = slice_ptr(&hid_buf, stream);

        // x F32 -> BF16.
        let n_cast = (seq * hidden) as i32;
        let cast_blocks = (n_cast as u32).div_ceil(256);
        unsafe {
            stream
                .launch_builder(f32_to_bf16_fn)
                .arg(&x_ptr)
                .arg(&xbf16_ptr)
                .arg(&n_cast)
                .launch(LaunchConfig {
                    grid_dim: (cast_blocks, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        let act_mode: i32 = self.mode.activation_kernel_mode();
        let normalize: i32 = match self.mode {
            GLUMoEMode::SwiGLU => 0,
            GLUMoEMode::SwiGLUNormalized | GLUMoEMode::GemmaGELU => 1,
        };
        let use_scale: i32 = matches!(self.mode, GLUMoEMode::GemmaGELU) as i32;
        let scale_ptr: u64 = if use_scale == 1 {
            buf_ptr(mode_aux_buf, stream)
        } else {
            0
        };

        let hidden_i = hidden as i32;
        let intermediate_i = intermediate as i32;
        let gate_up_dim_i = gate_up_dim as i32;
        let top_k_i = top_k as i32;
        let idx_stride_i = idx_stride as i32;
        let vals_stride_i = vals_stride as i32;
        let seq_i = seq as i32;

        // gate_up GEMV + gated activation. Grid: (intermediate, seq*top_k).
        unsafe {
            stream
                .launch_builder(gate_up_act_fn)
                .arg(&xbf16_ptr)
                .arg(&topk_idx_ptr)
                .arg(&gate_up_ptr)
                .arg(&hid_ptr)
                .arg(&hidden_i)
                .arg(&intermediate_i)
                .arg(&gate_up_dim_i)
                .arg(&top_k_i)
                .arg(&idx_stride_i)
                .arg(&seq_i)
                .arg(&act_mode)
                .launch(LaunchConfig {
                    grid_dim: (intermediate as u32, (seq * top_k) as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        // down GEMV + weighted combine. Grid: (hidden, seq).
        unsafe {
            stream
                .launch_builder(down_combine_fn)
                .arg(&hid_ptr)
                .arg(&topk_idx_ptr)
                .arg(&topk_vals_ptr)
                .arg(&scale_ptr)
                .arg(&down_ptr)
                .arg(&output_ptr)
                .arg(&hidden_i)
                .arg(&intermediate_i)
                .arg(&top_k_i)
                .arg(&idx_stride_i)
                .arg(&vals_stride_i)
                .arg(&seq_i)
                .arg(&normalize)
                .arg(&use_scale)
                .launch(LaunchConfig {
                    grid_dim: (hidden as u32, seq as u32, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }

        Ok(())
    }

    fn output_size(&self) -> Expression {
        // Output is [seq, hidden] F32 → seq * hidden elements
        // But seq is dynamic. We derive from first input size / hidden.
        // Actually, output_bytes is what matters for allocation:
        Expression::from('s') * self.gu_matmul_k
    }

    fn output_bytes(&self) -> Expression {
        Expression::from('s') * self.gu_matmul_k * 4 // F32
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("GLUMoE")
    }
}

// ============================================================
// Helpers
// ============================================================

fn buf_ptr(buf: DeviceBuffer, _stream: &Arc<CudaStream>) -> u64 {
    buf.ptr()
}

fn slice_ptr(buf: &CudaSlice<u8>, stream: &Arc<CudaStream>) -> u64 {
    let (ptr, _guard) = buf.device_ptr(stream);
    ptr
}

#[allow(clippy::too_many_arguments)]
fn cublas_matmul(
    stream: &Arc<CudaStream>,
    cublaslt: &Arc<CudaBlasLT>,
    workspace_ptr: u64,
    m: u64,
    n: u64,
    k: u64,
    a_ptr: u64,
    a_op: cublasOperation_t,
    lda: i64,
    b_ptr: u64,
    b_op: cublasOperation_t,
    ldb: i64,
    c_ptr: u64,
    ldc: i64,
    dtype: cudaDataType,
    compute: cublasComputeType_t,
    alpha: f32,
    beta: f32,
) -> anyhow::Result<()> {
    let scale_type = cudaDataType::CUDA_R_32F;

    let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
    let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
    let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
    let mut algo_count: i32 = 0;

    unsafe {
        cublasLtMatmulDescCreate(&mut matmul_desc, compute, scale_type).result()?;
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &a_op as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &b_op as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;

        let (a_rows, a_cols) = if a_op == cublasOperation_t::CUBLAS_OP_N {
            (m, k)
        } else {
            (k, m)
        };
        let (b_rows, b_cols) = if b_op == cublasOperation_t::CUBLAS_OP_N {
            (k, n)
        } else {
            (n, k)
        };

        cublasLtMatrixLayoutCreate(&mut a_desc, dtype, a_rows, a_cols, lda).result()?;
        cublasLtMatrixLayoutCreate(&mut b_desc, dtype, b_rows, b_cols, ldb).result()?;
        cublasLtMatrixLayoutCreate(&mut c_desc, dtype, m, n, ldc).result()?;

        cublasLtMatmulPreferenceCreate(&mut preference).result()?;
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
        .result()?;

        cublasLtMatmulAlgoGetHeuristic(
            *cublaslt.handle(),
            matmul_desc,
            a_desc,
            b_desc,
            c_desc,
            c_desc,
            preference,
            1,
            &mut heuristic,
            &mut algo_count,
        )
        .result()?;

        if algo_count == 0 {
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(c_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
            return Err(anyhow::anyhow!("No suitable cuBLASLT algorithm found"));
        }

        cublasLtMatmul(
            *cublaslt.handle(),
            matmul_desc,
            &alpha as *const _ as *const std::ffi::c_void,
            a_ptr as *const std::ffi::c_void,
            a_desc,
            b_ptr as *const std::ffi::c_void,
            b_desc,
            &beta as *const _ as *const std::ffi::c_void,
            c_ptr as *const std::ffi::c_void,
            c_desc,
            c_ptr as *mut std::ffi::c_void,
            c_desc,
            &heuristic.algo,
            workspace_ptr as *mut std::ffi::c_void,
            WORKSPACE_SIZE,
            stream.cu_stream() as *mut _,
        )
        .result()?;

        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cublas_matmul_mixed(
    stream: &Arc<CudaStream>,
    cublaslt: &Arc<CudaBlasLT>,
    workspace_ptr: u64,
    m: u64,
    n: u64,
    k: u64,
    a_ptr: u64,
    a_op: cublasOperation_t,
    lda: i64,
    b_ptr: u64,
    b_op: cublasOperation_t,
    ldb: i64,
    c_ptr: u64,
    ldc: i64,
    alpha: f32,
    beta: f32,
) -> anyhow::Result<()> {
    let ab_dtype = cudaDataType::CUDA_R_16BF;
    let cd_dtype = cudaDataType::CUDA_R_32F;
    let compute = cublasComputeType_t::CUBLAS_COMPUTE_32F;
    let scale_type = cudaDataType::CUDA_R_32F;

    let mut matmul_desc: cublasLtMatmulDesc_t = std::ptr::null_mut();
    let mut a_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut b_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut c_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut d_desc: cublasLtMatrixLayout_t = std::ptr::null_mut();
    let mut preference: cublasLtMatmulPreference_t = std::ptr::null_mut();
    let mut heuristic: cublasLtMatmulHeuristicResult_t = unsafe { std::mem::zeroed() };
    let mut algo_count: i32 = 0;

    unsafe {
        cublasLtMatmulDescCreate(&mut matmul_desc, compute, scale_type).result()?;
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &a_op as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;
        cublasLtMatmulDescSetAttribute(
            matmul_desc,
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            &b_op as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<cublasOperation_t>(),
        )
        .result()?;

        let (a_rows, a_cols) = if a_op == cublasOperation_t::CUBLAS_OP_N {
            (m, k)
        } else {
            (k, m)
        };
        let (b_rows, b_cols) = if b_op == cublasOperation_t::CUBLAS_OP_N {
            (k, n)
        } else {
            (n, k)
        };

        cublasLtMatrixLayoutCreate(&mut a_desc, ab_dtype, a_rows, a_cols, lda).result()?;
        cublasLtMatrixLayoutCreate(&mut b_desc, ab_dtype, b_rows, b_cols, ldb).result()?;
        cublasLtMatrixLayoutCreate(&mut c_desc, cd_dtype, m, n, ldc).result()?;
        cublasLtMatrixLayoutCreate(&mut d_desc, cd_dtype, m, n, ldc).result()?;

        cublasLtMatmulPreferenceCreate(&mut preference).result()?;
        cublasLtMatmulPreferenceSetAttribute(
            preference,
            cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &WORKSPACE_SIZE as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<usize>(),
        )
        .result()?;

        cublasLtMatmulAlgoGetHeuristic(
            *cublaslt.handle(),
            matmul_desc,
            a_desc,
            b_desc,
            c_desc,
            d_desc,
            preference,
            1,
            &mut heuristic,
            &mut algo_count,
        )
        .result()?;

        if algo_count == 0 {
            cublasLtMatmulPreferenceDestroy(preference);
            cublasLtMatrixLayoutDestroy(d_desc);
            cublasLtMatrixLayoutDestroy(c_desc);
            cublasLtMatrixLayoutDestroy(b_desc);
            cublasLtMatrixLayoutDestroy(a_desc);
            cublasLtMatmulDescDestroy(matmul_desc);
            return Err(anyhow::anyhow!(
                "No suitable cuBLASLT algorithm found for mixed matmul"
            ));
        }

        cublasLtMatmul(
            *cublaslt.handle(),
            matmul_desc,
            &alpha as *const _ as *const std::ffi::c_void,
            a_ptr as *const std::ffi::c_void,
            a_desc,
            b_ptr as *const std::ffi::c_void,
            b_desc,
            &beta as *const _ as *const std::ffi::c_void,
            c_ptr as *const std::ffi::c_void,
            c_desc,
            c_ptr as *mut std::ffi::c_void,
            d_desc,
            &heuristic.algo,
            workspace_ptr as *mut std::ffi::c_void,
            WORKSPACE_SIZE,
            stream.cu_stream() as *mut _,
        )
        .result()?;

        cublasLtMatmulPreferenceDestroy(preference);
        cublasLtMatrixLayoutDestroy(d_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
    }
    Ok(())
}

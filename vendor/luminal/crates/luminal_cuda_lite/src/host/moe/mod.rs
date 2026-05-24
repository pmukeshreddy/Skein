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
        CudaFunction, // quantize_fp8_rows
        CudaFunction, // moe_gate_up_act_fp8
        CudaFunction, // moe_down_combine_fp8
        CudaFunction, // moe_gate_up_act_fp8_v2
        CudaFunction, // moe_down_combine_fp8_v2
        CudaFunction, // moe_gate_up_act_fp8_grouped (batch>1: read each expert once)
        CudaFunction, // zero_f32
        CudaFunction, // moe_down_combine_fp8_grouped
        CudaFunction, // moe_gate_up_act_fp8_binned (token-permuted, active-only)
        CudaFunction, // moe_down_combine_fp8_binned
        CudaFunction, // moe_down_gemv_fp8 (v1 two-kernel: per-expert GEMV)
        CudaFunction, // moe_sum_fp8 (v1 two-kernel: weighted reduction)
    )>,
    /// fp8 (E4M3) weight cache (SKEIN_MOE_FP8): raw device pointers to the
    /// quantized resident expert weights + per-row scales. Computed once on the
    /// first execute and leaked (resident for the process, like the bf16 weights).
    fp8: OnceLock<Fp8Weights>,
}

/// Raw device pointers to the fp8-quantized expert weights and per-row scales.
#[derive(Clone, Copy)]
struct Fp8Weights {
    gate_up_ptr: u64,
    gate_up_scale_ptr: u64,
    down_ptr: u64,
    down_scale_ptr: u64,
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
            fp8: OnceLock::new(),
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
            fp8: OnceLock::new(),
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
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction,
        CudaFunction, // moe_down_gemv_fp8
        CudaFunction, // moe_sum_fp8
    ) {
        self.module.get_or_init(|| {
            let src = r#"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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

// ---- fp8 (E4M3) weight-only path (SKEIN_MOE_FP8) ----
// Quantize a [n_rows, row_len] bf16 matrix to fp8 E4M3 with one scale per row
// (scale = rowmax/448). One block per row.
extern "C" __global__ void quantize_fp8_rows(
    unsigned long long bf16_ptr, unsigned long long fp8_ptr, unsigned long long scale_ptr,
    int n_rows, int row_len
) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    const __nv_bfloat16* w = (const __nv_bfloat16*)bf16_ptr + (long long)row * row_len;
    float m = 0.f;
    for (int j = threadIdx.x; j < row_len; j += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(w[j])));
    for (int s = 16; s > 0; s >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, s));
    __shared__ float sm[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = (blockDim.x + 31) >> 5;
    if (lane == 0) sm[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (lane < nwarp) ? sm[lane] : 0.f;
        for (int s = 16; s > 0; s >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, s));
        if (lane == 0) { float sc = (m > 0.f) ? (m / 448.0f) : 1.0f; sm[0] = sc; ((float*)scale_ptr)[row] = sc; }
    }
    __syncthreads();
    float inv = 1.0f / sm[0];
    unsigned char* out = (unsigned char*)fp8_ptr + (long long)row * row_len;
    for (int j = threadIdx.x; j < row_len; j += blockDim.x) {
        out[j] = __nv_fp8_e4m3(__bfloat162float(w[j]) * inv).__x;
    }
}

extern "C" __global__ void moe_gate_up_act_fp8(
    unsigned long long x_bf16_ptr, unsigned long long topk_idx_ptr,
    unsigned long long gate_up_ptr, unsigned long long gate_up_scale_ptr, unsigned long long hid_ptr,
    int hidden, int intermediate, int gate_up_dim, int top_k, int idx_stride, int seq, int act_mode
) {
    int o = blockIdx.x;
    int tj = blockIdx.y;
    int t = tj / top_k, slot = tj % top_k;
    if (t >= seq || o >= intermediate) return;
    int expert = ((const int*)topk_idx_ptr)[(long long)t * idx_stride + slot];
    const __nv_bfloat16* x = (const __nv_bfloat16*)x_bf16_ptr + (long long)t * hidden;
    const unsigned char* W = (const unsigned char*)gate_up_ptr + (long long)expert * gate_up_dim * hidden;
    const unsigned char* gate_row = W + (long long)o * hidden;
    const unsigned char* up_row   = W + (long long)(o + intermediate) * hidden;
    const float* sc = (const float*)gate_up_scale_ptr + (long long)expert * gate_up_dim;
    float gscale = sc[o], uscale = sc[o + intermediate];
    float gacc = 0.f, uacc = 0.f;
    int n16 = hidden >> 4;
    const uint4* gp4 = (const uint4*)gate_row;
    const uint4* up4 = (const uint4*)up_row;
    for (int j = threadIdx.x; j < n16; j += blockDim.x) {
        uint4 gw = gp4[j], uw = up4[j];
        const __nv_fp8_e4m3* gh = (const __nv_fp8_e4m3*)&gw;
        const __nv_fp8_e4m3* uh = (const __nv_fp8_e4m3*)&uw;
        int b = j << 4;
        #pragma unroll
        for (int e = 0; e < 16; e++) { float xf = __bfloat162float(x[b + e]); gacc += (float)gh[e] * xf; uacc += (float)uh[e] * xf; }
    }
    for (int j = (n16 << 4) + (int)threadIdx.x; j < hidden; j += blockDim.x) {
        float xf = __bfloat162float(x[j]);
        __nv_fp8_e4m3 gq; gq.__x = gate_row[j];
        __nv_fp8_e4m3 uq; uq.__x = up_row[j];
        gacc += (float)gq * xf;
        uacc += (float)uq * xf;
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
            float gate = gacc * gscale, up = uacc * uscale, act;
            if (act_mode == 0) { act = gate / (1.0f + expf(-gate)); }
            else { float scx = 1.5957691216f * gate * (1.0f + 0.044715f * gate * gate); act = gate / (1.0f + expf(-scx)); }
            ((__nv_bfloat16*)hid_ptr)[(long long)tj * intermediate + o] = __float2bfloat16(act * up);
        }
    }
}

extern "C" __global__ void moe_down_combine_fp8(
    unsigned long long hid_ptr, unsigned long long topk_idx_ptr, unsigned long long topk_vals_ptr,
    unsigned long long scale_ptr, unsigned long long down_ptr, unsigned long long down_scale_ptr,
    unsigned long long out_ptr,
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
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = (blockDim.x + 31) >> 5;
    __shared__ float sd[32];
    float out_acc = 0.f;
    for (int jx = 0; jx < top_k; jx++) {
        int expert = idx[(long long)t * idx_stride + jx];
        float w = vals[(long long)t * vals_stride + jx] * inv_norm;
        if (use_scale) w *= ((const float*)scale_ptr)[expert];
        const unsigned char* D = (const unsigned char*)down_ptr + (long long)expert * hidden * intermediate + (long long)h * intermediate;
        const __nv_bfloat16* hd = (const __nv_bfloat16*)hid_ptr + (long long)(t * top_k + jx) * intermediate;
        float dscale = ((const float*)down_scale_ptr)[(long long)expert * hidden + h];
        float dot = 0.f;
        int n16 = intermediate >> 4;
        const uint4* dp4 = (const uint4*)D;
        for (int j = threadIdx.x; j < n16; j += blockDim.x) {
            uint4 dw = dp4[j];
            const __nv_fp8_e4m3* dh = (const __nv_fp8_e4m3*)&dw;
            int b = j << 4;
            #pragma unroll
            for (int e = 0; e < 16; e++) dot += (float)dh[e] * __bfloat162float(hd[b + e]);
        }
        for (int j = (n16 << 4) + (int)threadIdx.x; j < intermediate; j += blockDim.x) {
            __nv_fp8_e4m3 dq; dq.__x = D[j];
            dot += (float)dq * __bfloat162float(hd[j]);
        }
        for (int s = 16; s > 0; s >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, s);
        if (lane == 0) sd[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float r = (lane < nwarp) ? sd[lane] : 0.f;
            for (int s = 16; s > 0; s >>= 1) r += __shfl_down_sync(0xffffffffu, r, s);
            if (lane == 0) out_acc += w * dscale * r;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) ((float*)out_ptr)[(long long)t * hidden + h] = out_acc;
}

// ---- v1 two-kernel split (vLLM pattern, default decode path) ----------------
// KERNEL 1 (heavy): one block computes ONE expert's down-GEMV for ONE output
// element. Grid (hidden, seq*top_k) — matches gate_up, so the down phase gets
// the same block count (28672 vs the old 4096) and hides HBM latency far better
// than the old single grid that looped top_k inside the block with a per-expert
// __syncthreads. Applies the per-output down weight_scale and writes the partial
// (no accumulation; the cheap KERNEL 2 does the weighted reduction).
extern "C" __global__ void moe_down_gemv_fp8(
    unsigned long long hid_ptr, unsigned long long topk_idx_ptr,
    unsigned long long down_ptr, unsigned long long down_scale_ptr,
    unsigned long long partials_ptr,
    int hidden, int intermediate, int top_k, int idx_stride, int seq
) {
    int h = blockIdx.x;
    int tj = blockIdx.y;
    int t = tj / top_k, jx = tj % top_k;
    if (t >= seq || h >= hidden) return;
    int expert = ((const int*)topk_idx_ptr)[(long long)t * idx_stride + jx];
    const unsigned char* D = (const unsigned char*)down_ptr + (long long)expert * hidden * intermediate + (long long)h * intermediate;
    const __nv_bfloat16* hd = (const __nv_bfloat16*)hid_ptr + (long long)(t * top_k + jx) * intermediate;
    float dscale = ((const float*)down_scale_ptr)[(long long)expert * hidden + h];
    float dot = 0.f;
    int n16 = intermediate >> 4;
    const uint4* dp4 = (const uint4*)D;
    for (int j = threadIdx.x; j < n16; j += blockDim.x) {
        uint4 dw = dp4[j];
        const __nv_fp8_e4m3* dh = (const __nv_fp8_e4m3*)&dw;
        int b = j << 4;
        #pragma unroll
        for (int e = 0; e < 16; e++) dot += (float)dh[e] * __bfloat162float(hd[b + e]);
    }
    for (int j = (n16 << 4) + (int)threadIdx.x; j < intermediate; j += blockDim.x) {
        __nv_fp8_e4m3 dq; dq.__x = D[j];
        dot += (float)dq * __bfloat162float(hd[j]);
    }
    for (int s = 16; s > 0; s >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, s);
    __shared__ float sd[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = (blockDim.x + 31) >> 5;
    if (lane == 0) sd[warp] = dot;
    __syncthreads();
    if (warp == 0) {
        float r = (lane < nwarp) ? sd[lane] : 0.f;
        for (int s = 16; s > 0; s >>= 1) r += __shfl_down_sync(0xffffffffu, r, s);
        if (lane == 0) ((float*)partials_ptr)[(long long)(t * top_k + jx) * hidden + h] = dscale * r;
    }
}

// KERNEL 2 (cheap, ~15 lines): mirror of vLLM's moe_sum_kernel
// (csrc/moe/moe_align_sum_kernels.cu). One thread per output element sums the
// top_k partials, applying routing_weight x normalize (x optional expert scale).
extern "C" __global__ void moe_sum_fp8(
    unsigned long long partials_ptr, unsigned long long topk_idx_ptr, unsigned long long topk_vals_ptr,
    unsigned long long scale_ptr, unsigned long long out_ptr,
    int hidden, int top_k, int idx_stride, int vals_stride, int seq,
    int normalize, int use_scale
) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
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
    float out_acc = 0.f;
    for (int jx = 0; jx < top_k; jx++) {
        int expert = idx[(long long)t * idx_stride + jx];
        float w = vals[(long long)t * vals_stride + jx] * inv_norm;
        if (use_scale) w *= ((const float*)scale_ptr)[expert];
        out_acc += w * ((const float*)partials_ptr)[(long long)(t * top_k + jx) * hidden + h];
    }
    ((float*)out_ptr)[(long long)t * hidden + h] = out_acc;
}

// ---- v2 (SKEIN_MOE_V2): warp-per-output-row + shared-memory staging of the
// reused activation, single warp-shuffle reduction (no cross-warp shared reduce).
// MEASURED REGRESSION (kept gated-off for the record): on RTX PRO 6000 Blackwell
// at the decode shape (seq=1, top-2, hidden=4096, intermediate=7168) v2 is SLOWER
// than v1 — gate_up 73%->70% peak, down 58%->33% peak. v1's much higher block
// count (14336 / 4096 blocks) hides latency better than v2's shared-staging
// barrier + lower occupancy (1024 blocks of 128 threads). Default stays v1.
// gate_up: 8 rows/block (256 thr), x staged in shared (reused by all 8 warps).
extern "C" __global__ void moe_gate_up_act_fp8_v2(
    unsigned long long x_bf16_ptr, unsigned long long topk_idx_ptr,
    unsigned long long gate_up_ptr, unsigned long long gate_up_scale_ptr, unsigned long long hid_ptr,
    int hidden, int intermediate, int gate_up_dim, int top_k, int idx_stride, int seq, int act_mode
) {
    extern __shared__ __nv_bfloat16 sx[];   // [hidden]
    int tj = blockIdx.y;
    int t = tj / top_k, slot = tj % top_k;
    if (t >= seq) return;
    const __nv_bfloat16* xg = (const __nv_bfloat16*)x_bf16_ptr + (long long)t * hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) sx[j] = xg[j];
    __syncthreads();
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
    int o = blockIdx.x * wpb + warp;
    if (o >= intermediate) return;
    int expert = ((const int*)topk_idx_ptr)[(long long)t * idx_stride + slot];
    const unsigned char* W = (const unsigned char*)gate_up_ptr + (long long)expert * gate_up_dim * hidden;
    const unsigned char* gate_row = W + (long long)o * hidden;
    const unsigned char* up_row   = W + (long long)(o + intermediate) * hidden;
    const float* sc = (const float*)gate_up_scale_ptr + (long long)expert * gate_up_dim;
    float gscale = sc[o], uscale = sc[o + intermediate];
    float gacc = 0.f, uacc = 0.f;
    int n16 = hidden >> 4;
    const uint4* gp4 = (const uint4*)gate_row;
    const uint4* up4 = (const uint4*)up_row;
    for (int j = lane; j < n16; j += 32) {
        uint4 gw = gp4[j], uw = up4[j];
        const __nv_fp8_e4m3* gh = (const __nv_fp8_e4m3*)&gw;
        const __nv_fp8_e4m3* uh = (const __nv_fp8_e4m3*)&uw;
        int b = j << 4;
        #pragma unroll
        for (int e = 0; e < 16; e++) { float xf = __bfloat162float(sx[b + e]); gacc += (float)gh[e] * xf; uacc += (float)uh[e] * xf; }
    }
    for (int j = (n16 << 4) + lane; j < hidden; j += 32) {
        float xf = __bfloat162float(sx[j]);
        __nv_fp8_e4m3 gq; gq.__x = gate_row[j];
        __nv_fp8_e4m3 uq; uq.__x = up_row[j];
        gacc += (float)gq * xf; uacc += (float)uq * xf;
    }
    for (int s = 16; s > 0; s >>= 1) { gacc += __shfl_down_sync(0xffffffffu, gacc, s); uacc += __shfl_down_sync(0xffffffffu, uacc, s); }
    if (lane == 0) {
        float gate = gacc * gscale, up = uacc * uscale, act;
        if (act_mode == 0) { act = gate / (1.0f + expf(-gate)); }
        else { float scx = 1.5957691216f * gate * (1.0f + 0.044715f * gate * gate); act = gate / (1.0f + expf(-scx)); }
        ((__nv_bfloat16*)hid_ptr)[(long long)tj * intermediate + o] = __float2bfloat16(act * up);
    }
}

// down: 4 rows/block (128 thr); both experts' hd staged in shared (reused by all
// 4 warps), top_k looped inside the warp so out[h] is written exactly once.
extern "C" __global__ void moe_down_combine_fp8_v2(
    unsigned long long hid_ptr, unsigned long long topk_idx_ptr, unsigned long long topk_vals_ptr,
    unsigned long long scale_ptr, unsigned long long down_ptr, unsigned long long down_scale_ptr,
    unsigned long long out_ptr,
    int hidden, int intermediate, int top_k, int idx_stride, int vals_stride, int seq,
    int normalize, int use_scale
) {
    extern __shared__ __nv_bfloat16 shd[];  // [top_k, intermediate]
    int t = blockIdx.y;
    if (t >= seq) return;
    const int* idx = (const int*)topk_idx_ptr;
    const float* vals = (const float*)topk_vals_ptr;
    // Stage this token's top_k expert activations (contiguous) into shared.
    const __nv_bfloat16* hbase = (const __nv_bfloat16*)hid_ptr + (long long)(t * top_k) * intermediate;
    for (int j = threadIdx.x; j < top_k * intermediate; j += blockDim.x) shd[j] = hbase[j];
    __syncthreads();
    float inv_norm = 1.0f;
    if (normalize) {
        float ssum = 0.f;
        for (int j = 0; j < top_k; j++) ssum += vals[(long long)t * vals_stride + j];
        inv_norm = (ssum != 0.f) ? (1.0f / ssum) : 0.f;
    }
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wpb = blockDim.x >> 5;
    int h = blockIdx.x * wpb + warp;
    if (h >= hidden) return;
    int n16 = intermediate >> 4;
    float out_acc = 0.f;
    for (int jx = 0; jx < top_k; jx++) {
        int expert = idx[(long long)t * idx_stride + jx];
        float w = vals[(long long)t * vals_stride + jx] * inv_norm;
        if (use_scale) w *= ((const float*)scale_ptr)[expert];
        const unsigned char* D = (const unsigned char*)down_ptr + (long long)expert * hidden * intermediate + (long long)h * intermediate;
        float dscale = ((const float*)down_scale_ptr)[(long long)expert * hidden + h];
        const __nv_bfloat16* hd = shd + (long long)jx * intermediate;
        const uint4* dp4 = (const uint4*)D;
        float dot = 0.f;
        for (int j = lane; j < n16; j += 32) {
            uint4 dw = dp4[j];
            const __nv_fp8_e4m3* dh = (const __nv_fp8_e4m3*)&dw;
            int b = j << 4;
            #pragma unroll
            for (int e = 0; e < 16; e++) dot += (float)dh[e] * __bfloat162float(hd[b + e]);
        }
        for (int j = (n16 << 4) + lane; j < intermediate; j += 32) {
            __nv_fp8_e4m3 dq; dq.__x = D[j];
            dot += (float)dq * __bfloat162float(hd[j]);
        }
        for (int s = 16; s > 0; s >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, s);
        if (lane == 0) out_acc += w * dscale * dot;
    }
    if (lane == 0) ((float*)out_ptr)[(long long)t * hidden + h] = out_acc;
}

// ---- GROUPED fp8 MoE (batch>1): one block per (output row, EXPERT) instead of
// per (output row, token). Each block stages its expert's weight row in shared
// memory ONCE (read from HBM once) and loops the tokens routing to that expert,
// reusing the staged weight. This amortizes the dominant expert-weight HBM
// traffic across all tokens that share an expert, so MoE work no longer scales
// ~linearly with the batch — the fix that lets batched decode actually speed up.
extern "C" __global__ void moe_gate_up_act_fp8_grouped(
    unsigned long long x_bf16_ptr, unsigned long long topk_idx_ptr,
    unsigned long long gate_up_ptr, unsigned long long gate_up_scale_ptr, unsigned long long hid_ptr,
    int hidden, int intermediate, int gate_up_dim, int top_k, int idx_stride, int seq, int act_mode,
    int num_experts
) {
    int o = blockIdx.x, e = blockIdx.y;
    if (o >= intermediate || e >= num_experts) return;
    const unsigned char* W = (const unsigned char*)gate_up_ptr + (long long)e * gate_up_dim * hidden;
    const unsigned char* gate_row = W + (long long)o * hidden;
    const unsigned char* up_row   = W + (long long)(o + intermediate) * hidden;
    const float* sc = (const float*)gate_up_scale_ptr + (long long)e * gate_up_dim;
    float gscale = sc[o], uscale = sc[o + intermediate];
    extern __shared__ unsigned char sh[];
    unsigned char* sg = sh;            // hidden fp8 bytes
    unsigned char* su = sh + hidden;   // hidden fp8 bytes
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) { sg[j] = gate_row[j]; su[j] = up_row[j]; }
    __syncthreads();
    __shared__ float rg[32];
    __shared__ float ru[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = (blockDim.x + 31) >> 5;
    int njobs = seq * top_k;
    for (int tj = 0; tj < njobs; tj++) {
        int t = tj / top_k, slot = tj % top_k;
        if (((const int*)topk_idx_ptr)[(long long)t * idx_stride + slot] != e) continue;
        const __nv_bfloat16* x = (const __nv_bfloat16*)x_bf16_ptr + (long long)t * hidden;
        float gacc = 0.f, uacc = 0.f;
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            float xf = __bfloat162float(x[j]);
            __nv_fp8_e4m3 gq; gq.__x = sg[j];
            __nv_fp8_e4m3 uq; uq.__x = su[j];
            gacc += (float)gq * xf; uacc += (float)uq * xf;
        }
        for (int s = 16; s > 0; s >>= 1) { gacc += __shfl_down_sync(0xffffffffu, gacc, s); uacc += __shfl_down_sync(0xffffffffu, uacc, s); }
        if (lane == 0) { rg[warp] = gacc; ru[warp] = uacc; }
        __syncthreads();
        if (warp == 0) {
            gacc = (lane < nwarp) ? rg[lane] : 0.f;
            uacc = (lane < nwarp) ? ru[lane] : 0.f;
            for (int s = 16; s > 0; s >>= 1) { gacc += __shfl_down_sync(0xffffffffu, gacc, s); uacc += __shfl_down_sync(0xffffffffu, uacc, s); }
            if (lane == 0) {
                float gate = gacc * gscale, up = uacc * uscale, act;
                if (act_mode == 0) act = gate / (1.0f + expf(-gate));
                else { float scx = 1.5957691216f * gate * (1.0f + 0.044715f * gate * gate); act = gate / (1.0f + expf(-scx)); }
                ((__nv_bfloat16*)hid_ptr)[(long long)tj * intermediate + o] = __float2bfloat16(act * up);
            }
        }
        __syncthreads();
    }
}

// Zero an f32 buffer (the grouped down accumulates into out via atomicAdd, so the
// output must start at 0). Capturable (plain kernel launch on the stream).
extern "C" __global__ void zero_f32(unsigned long long ptr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ((float*)ptr)[i] = 0.f;
}

extern "C" __global__ void moe_down_combine_fp8_grouped(
    unsigned long long hid_ptr, unsigned long long topk_idx_ptr, unsigned long long topk_vals_ptr,
    unsigned long long scale_ptr, unsigned long long down_ptr, unsigned long long down_scale_ptr,
    unsigned long long out_ptr,
    int hidden, int intermediate, int top_k, int idx_stride, int vals_stride, int seq,
    int normalize, int use_scale, int num_experts
) {
    int h = blockIdx.x, e = blockIdx.y;
    if (h >= hidden || e >= num_experts) return;
    const unsigned char* D = (const unsigned char*)down_ptr + (long long)e * hidden * intermediate + (long long)h * intermediate;
    float dscale = ((const float*)down_scale_ptr)[(long long)e * hidden + h];
    extern __shared__ unsigned char sD[]; // intermediate fp8 bytes
    for (int j = threadIdx.x; j < intermediate; j += blockDim.x) sD[j] = D[j];
    __syncthreads();
    __shared__ float rd[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = (blockDim.x + 31) >> 5;
    const int* idx = (const int*)topk_idx_ptr;
    const float* vals = (const float*)topk_vals_ptr;
    int njobs = seq * top_k;
    for (int tj = 0; tj < njobs; tj++) {
        int t = tj / top_k, slot = tj % top_k;
        if (idx[(long long)t * idx_stride + slot] != e) continue;
        float w = vals[(long long)t * vals_stride + slot];
        if (normalize) {
            float ssum = 0.f;
            for (int k = 0; k < top_k; k++) ssum += vals[(long long)t * vals_stride + k];
            w = (ssum != 0.f) ? (w / ssum) : 0.f;
        }
        if (use_scale) w *= ((const float*)scale_ptr)[e];
        const __nv_bfloat16* hd = (const __nv_bfloat16*)hid_ptr + (long long)tj * intermediate;
        float dot = 0.f;
        for (int j = threadIdx.x; j < intermediate; j += blockDim.x) {
            __nv_fp8_e4m3 dq; dq.__x = sD[j];
            dot += (float)dq * __bfloat162float(hd[j]);
        }
        for (int s = 16; s > 0; s >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, s);
        if (lane == 0) rd[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float r = (lane < nwarp) ? rd[lane] : 0.f;
            for (int s = 16; s > 0; s >>= 1) r += __shfl_down_sync(0xffffffffu, r, s);
            if (lane == 0) atomicAdd(&((float*)out_ptr)[(long long)t * hidden + h], w * dscale * r);
        }
        __syncthreads();
    }
}

// ---- BINNED (token-permuted) fp8 MoE (SKEIN_MOE_PERMUTED, lockstep/no-capture):
// the host sorts the (token,slot) pairs by expert and launches one block-COLUMN
// per ACTIVE expert (grid.y = num_active, empty experts skipped). Each block
// stages its expert's fp8 weight row in shared ONCE and loops only that expert's
// CONTIGUOUS permuted rows (perm[s..s+n]) — coherent weight reuse, no per-token
// scatter, no blocks for unused experts. `hid` stays in ORIGINAL [seq*top_k,
// intermediate] layout (indexed by the original tj), so `down` reads it normally.
extern "C" __global__ void moe_gate_up_act_fp8_binned(
    unsigned long long x_bf16_ptr, unsigned long long perm_ptr,
    unsigned long long blk_e_ptr, unsigned long long blk_s_ptr, unsigned long long blk_n_ptr,
    unsigned long long gate_up_ptr, unsigned long long gate_up_scale_ptr, unsigned long long hid_ptr,
    int hidden, int intermediate, int gate_up_dim, int top_k, int act_mode
) {
    int o = blockIdx.x, ai = blockIdx.y;
    if (o >= intermediate) return;
    int e = ((const int*)blk_e_ptr)[ai];
    int s = ((const int*)blk_s_ptr)[ai];
    int n = ((const int*)blk_n_ptr)[ai];
    if (n <= 0) return;
    const unsigned char* W = (const unsigned char*)gate_up_ptr + (long long)e * gate_up_dim * hidden;
    const unsigned char* gate_row = W + (long long)o * hidden;
    const unsigned char* up_row   = W + (long long)(o + intermediate) * hidden;
    const float* sc = (const float*)gate_up_scale_ptr + (long long)e * gate_up_dim;
    float gscale = sc[o], uscale = sc[o + intermediate];
    extern __shared__ unsigned char sh[];
    unsigned char* sg = sh; unsigned char* su = sh + hidden;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x) { sg[j] = gate_row[j]; su[j] = up_row[j]; }
    __syncthreads();
    __shared__ float rg[32]; __shared__ float ru[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = (blockDim.x + 31) >> 5;
    const int* perm = (const int*)perm_ptr;
    for (int r = 0; r < n; r++) {
        int tj = perm[s + r];
        int t = tj / top_k;
        const __nv_bfloat16* x = (const __nv_bfloat16*)x_bf16_ptr + (long long)t * hidden;
        float gacc = 0.f, uacc = 0.f;
        for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
            float xf = __bfloat162float(x[j]);
            __nv_fp8_e4m3 gq; gq.__x = sg[j];
            __nv_fp8_e4m3 uq; uq.__x = su[j];
            gacc += (float)gq * xf; uacc += (float)uq * xf;
        }
        for (int z = 16; z > 0; z >>= 1) { gacc += __shfl_down_sync(0xffffffffu, gacc, z); uacc += __shfl_down_sync(0xffffffffu, uacc, z); }
        if (lane == 0) { rg[warp] = gacc; ru[warp] = uacc; }
        __syncthreads();
        if (warp == 0) {
            gacc = (lane < nwarp) ? rg[lane] : 0.f;
            uacc = (lane < nwarp) ? ru[lane] : 0.f;
            for (int z = 16; z > 0; z >>= 1) { gacc += __shfl_down_sync(0xffffffffu, gacc, z); uacc += __shfl_down_sync(0xffffffffu, uacc, z); }
            if (lane == 0) {
                float gate = gacc * gscale, up = uacc * uscale, act;
                if (act_mode == 0) act = gate / (1.0f + expf(-gate));
                else { float scx = 1.5957691216f * gate * (1.0f + 0.044715f * gate * gate); act = gate / (1.0f + expf(-scx)); }
                ((__nv_bfloat16*)hid_ptr)[(long long)tj * intermediate + o] = __float2bfloat16(act * up);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void moe_down_combine_fp8_binned(
    unsigned long long hid_ptr, unsigned long long perm_ptr,
    unsigned long long blk_e_ptr, unsigned long long blk_s_ptr, unsigned long long blk_n_ptr,
    unsigned long long topk_vals_ptr, unsigned long long scale_ptr,
    unsigned long long down_ptr, unsigned long long down_scale_ptr, unsigned long long out_ptr,
    int hidden, int intermediate, int top_k, int vals_stride, int normalize, int use_scale
) {
    int h = blockIdx.x, ai = blockIdx.y;
    if (h >= hidden) return;
    int e = ((const int*)blk_e_ptr)[ai];
    int s = ((const int*)blk_s_ptr)[ai];
    int n = ((const int*)blk_n_ptr)[ai];
    if (n <= 0) return;
    const unsigned char* D = (const unsigned char*)down_ptr + (long long)e * hidden * intermediate + (long long)h * intermediate;
    float dscale = ((const float*)down_scale_ptr)[(long long)e * hidden + h];
    extern __shared__ unsigned char sD[];
    for (int j = threadIdx.x; j < intermediate; j += blockDim.x) sD[j] = D[j];
    __syncthreads();
    __shared__ float rd[32];
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nwarp = (blockDim.x + 31) >> 5;
    const int* perm = (const int*)perm_ptr;
    const float* vals = (const float*)topk_vals_ptr;
    for (int r = 0; r < n; r++) {
        int tj = perm[s + r];
        int t = tj / top_k, slot = tj % top_k;
        float w = vals[(long long)t * vals_stride + slot];
        if (normalize) {
            float ssum = 0.f;
            for (int k = 0; k < top_k; k++) ssum += vals[(long long)t * vals_stride + k];
            w = (ssum != 0.f) ? (w / ssum) : 0.f;
        }
        if (use_scale) w *= ((const float*)scale_ptr)[e];
        const __nv_bfloat16* hd = (const __nv_bfloat16*)hid_ptr + (long long)tj * intermediate;
        float dot = 0.f;
        for (int j = threadIdx.x; j < intermediate; j += blockDim.x) {
            __nv_fp8_e4m3 dq; dq.__x = sD[j];
            dot += (float)dq * __bfloat162float(hd[j]);
        }
        for (int z = 16; z > 0; z >>= 1) dot += __shfl_down_sync(0xffffffffu, dot, z);
        if (lane == 0) rd[warp] = dot;
        __syncthreads();
        if (warp == 0) {
            float rr = (lane < nwarp) ? rd[lane] : 0.f;
            for (int z = 16; z > 0; z >>= 1) rr += __shfl_down_sync(0xffffffffu, rr, z);
            if (lane == 0) atomicAdd(&((float*)out_ptr)[(long long)t * hidden + h], w * dscale * rr);
        }
        __syncthreads();
    }
}
"#;
            let ptx = compile_module_image_for_current_device(stream.context(), src).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let f32_to_bf16 = module.load_function("f32_to_bf16").unwrap();
            let activation = module.load_function("glu_activation_bf16").unwrap();
            let gate_up_act = module.load_function("moe_gate_up_act").unwrap();
            let down_combine = module.load_function("moe_down_combine").unwrap();
            let quant_fp8 = module.load_function("quantize_fp8_rows").unwrap();
            let gate_up_act_fp8 = module.load_function("moe_gate_up_act_fp8").unwrap();
            let down_combine_fp8 = module.load_function("moe_down_combine_fp8").unwrap();
            let gate_up_act_fp8_v2 = module.load_function("moe_gate_up_act_fp8_v2").unwrap();
            let down_combine_fp8_v2 = module.load_function("moe_down_combine_fp8_v2").unwrap();
            let gate_up_act_fp8_grouped = module.load_function("moe_gate_up_act_fp8_grouped").unwrap();
            let zero_f32 = module.load_function("zero_f32").unwrap();
            let down_combine_fp8_grouped = module.load_function("moe_down_combine_fp8_grouped").unwrap();
            let gate_up_act_fp8_binned = module.load_function("moe_gate_up_act_fp8_binned").unwrap();
            let down_combine_fp8_binned = module.load_function("moe_down_combine_fp8_binned").unwrap();
            let down_gemv_fp8 = module.load_function("moe_down_gemv_fp8").unwrap();
            let sum_fp8 = module.load_function("moe_sum_fp8").unwrap();
            (
                module,
                f32_to_bf16,
                activation,
                gate_up_act,
                down_combine,
                quant_fp8,
                gate_up_act_fp8,
                down_combine_fp8,
                gate_up_act_fp8_v2,
                down_combine_fp8_v2,
                gate_up_act_fp8_grouped,
                zero_f32,
                down_combine_fp8_grouped,
                gate_up_act_fp8_binned,
                down_combine_fp8_binned,
                down_gemv_fp8,
                sum_fp8,
            )
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
            fp8: OnceLock::new(),
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
        // fp8 (SKEIN_MOE_FP8): quantize the resident bf16 expert weights to E4M3
        // with per-output-row scales ONCE (cached + leaked), halving the weight
        // bytes the decode GEMVs read.
        let use_fp8 = std::env::var_os("SKEIN_MOE_FP8").is_some();
        let fp8w: Option<Fp8Weights> = if use_fp8 {
            let quant_fn = kernels.5.clone();
            Some(*self.fp8.get_or_init(|| {
                let gu_rows = num_experts * gate_up_dim;
                let dn_rows = num_experts * hidden;
                let gu_fp8 = unsafe { stream.alloc::<u8>(num_experts * gate_up_dim * hidden).unwrap() };
                let gu_scale = unsafe { stream.alloc::<u8>(gu_rows * 4).unwrap() };
                let dn_fp8 = unsafe { stream.alloc::<u8>(num_experts * hidden * intermediate).unwrap() };
                let dn_scale = unsafe { stream.alloc::<u8>(dn_rows * 4).unwrap() };
                let gu_fp8_ptr = slice_ptr(&gu_fp8, stream);
                let gu_scale_ptr = slice_ptr(&gu_scale, stream);
                let dn_fp8_ptr = slice_ptr(&dn_fp8, stream);
                let dn_scale_ptr = slice_ptr(&dn_scale, stream);
                let gu_rows_i = gu_rows as i32;
                let hidden_i = hidden as i32;
                let dn_rows_i = dn_rows as i32;
                let inter_i = intermediate as i32;
                unsafe {
                    stream
                        .launch_builder(&quant_fn)
                        .arg(&gate_up_ptr).arg(&gu_fp8_ptr).arg(&gu_scale_ptr)
                        .arg(&gu_rows_i).arg(&hidden_i)
                        .launch(LaunchConfig { grid_dim: (gu_rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                        .unwrap();
                    stream
                        .launch_builder(&quant_fn)
                        .arg(&down_ptr).arg(&dn_fp8_ptr).arg(&dn_scale_ptr)
                        .arg(&dn_rows_i).arg(&inter_i)
                        .launch(LaunchConfig { grid_dim: (dn_rows as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                        .unwrap();
                }
                stream.synchronize().unwrap();
                std::mem::forget(gu_fp8);
                std::mem::forget(gu_scale);
                std::mem::forget(dn_fp8);
                std::mem::forget(dn_scale);
                Fp8Weights {
                    gate_up_ptr: gu_fp8_ptr,
                    gate_up_scale_ptr: gu_scale_ptr,
                    down_ptr: dn_fp8_ptr,
                    down_scale_ptr: dn_scale_ptr,
                }
            }))
        } else {
            None
        };

        // Scratch: x as bf16 [seq, hidden]; gated hidden [seq*top_k, intermediate] bf16.
        // SKEIN_CAPTURE: these are read/written by the captured gate_up / down
        // kernels, so a per-call alloc (freed at scope end) would make the replayed
        // graph touch freed/reused addresses — garbage. Use persistent scratch
        // (distinct keys: x and hid coexist within the op; one buffer each is safe
        // because the MoE layers run serialized on the shared capture stream).
        let _x_owned;
        let _hid_owned;
        let _partials_owned;
        let (xbf16_ptr, hid_ptr) = if super::is_capture() {
            (
                super::capture_scratch(stream, 2, seq * hidden * 2),
                super::capture_scratch(stream, 3, seq * top_k * intermediate * 2),
            )
        } else {
            let x_bf16_buf = unsafe { stream.alloc::<u8>(seq * hidden * 2)? };
            let hid_buf = unsafe { stream.alloc::<u8>(seq * top_k * intermediate * 2)? };
            let xp = slice_ptr(&x_bf16_buf, stream);
            let hp = slice_ptr(&hid_buf, stream);
            _x_owned = x_bf16_buf;
            _hid_owned = hid_buf;
            (xp, hp)
        };

        if std::env::var_os("SKEIN_FI_LOG").is_some() {
            use std::sync::atomic::{AtomicU64, Ordering};
            static N: AtomicU64 = AtomicU64::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            if n < 100 && n % 16 == 0 {
                eprintln!("SKEIN_MOE call#{n} seq={seq} hidden={hidden} top_k={top_k} inter={intermediate} fp8={}", fp8w.is_some());
            }
        }
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

        // PERMUTED/BINNED fp8 MoE (SKEIN_MOE_PERMUTED; lockstep / no capture — the
        // host D2H of topk_idx can't be recorded into a captured graph). The host
        // sorts the (token,slot) pairs by expert into CONTIGUOUS bins and launches
        // one block-column per ACTIVE expert; the binned kernels read each expert's
        // weight ONCE and loop only that expert's rows — the real batched-MoE fix
        // (no per-token scatter, no blocks for unused experts).
        let permuted = fp8w.is_some()
            && seq > 1
            && std::env::var_os("SKEIN_MOE_PERMUTED").is_some();
        let mut _perm_keep: Option<CudaSlice<u8>> = None;
        let mut _be_keep: Option<CudaSlice<u8>> = None;
        let mut _bs_keep: Option<CudaSlice<u8>> = None;
        let mut _bn_keep: Option<CudaSlice<u8>> = None;
        let (perm_ptr, be_ptr, bs_ptr, bn_ptr, num_active) = if permuted {
            let n_idx = seq * idx_stride;
            let mut idx_bytes = vec![0u8; n_idx * 4];
            unsafe {
                crate::cudarc::driver::result::memcpy_dtoh_async(
                    &mut idx_bytes,
                    topk_idx_buf.ptr(),
                    stream.cu_stream(),
                )?;
            }
            stream.synchronize()?;
            let idx_host: &[i32] =
                unsafe { std::slice::from_raw_parts(idx_bytes.as_ptr() as *const i32, n_idx) };
            let mut bins: Vec<Vec<i32>> = vec![Vec::new(); num_experts];
            for t in 0..seq {
                for slot in 0..top_k {
                    let e = idx_host[t * idx_stride + slot];
                    if e >= 0 && (e as usize) < num_experts {
                        bins[e as usize].push((t * top_k + slot) as i32);
                    }
                }
            }
            let mut perm: Vec<i32> = Vec::with_capacity(seq * top_k);
            let (mut be, mut bs, mut bn): (Vec<i32>, Vec<i32>, Vec<i32>) =
                (Vec::new(), Vec::new(), Vec::new());
            for (e, bin) in bins.iter().enumerate() {
                if bin.is_empty() {
                    continue;
                }
                be.push(e as i32);
                bs.push(perm.len() as i32);
                bn.push(bin.len() as i32);
                perm.extend_from_slice(bin);
            }
            let na = be.len();
            let up = |v: &[i32]| -> CudaSlice<u8> {
                stream
                    .clone_htod(unsafe {
                        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                    })
                    .unwrap()
            };
            let pd = up(&perm);
            let bed = up(&be);
            let bsd = up(&bs);
            let bnd = up(&bn);
            let pp = slice_ptr(&pd, stream);
            let bep = slice_ptr(&bed, stream);
            let bsp = slice_ptr(&bsd, stream);
            let bnp = slice_ptr(&bnd, stream);
            _perm_keep = Some(pd);
            _be_keep = Some(bed);
            _bs_keep = Some(bsd);
            _bn_keep = Some(bnd);
            (pp, bep, bsp, bnp, na)
        } else {
            (0u64, 0u64, 0u64, 0u64, 0usize)
        };

        // --- Optional kernel timing (SKEIN_MOE_KTIME): CUDA events around the two
        // MoE GEMVs, accumulated into module statics, logged with achieved GB/s vs
        // peak every 64 calls. Adds a per-call stream sync so tok/s is INVALID
        // during a ktime run; only the per-kernel GPU µs / GB/s are meaningful.
        static KTIME: OnceLock<bool> = OnceLock::new();
        static KT_GU_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static KT_DN_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static KT_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let ktime = *KTIME.get_or_init(|| std::env::var_os("SKEIN_MOE_KTIME").is_some());
        static V2: OnceLock<bool> = OnceLock::new();
        let use_v2 = *V2.get_or_init(|| std::env::var_os("SKEIN_MOE_V2").is_some());
        // A/B: SKEIN_MOE_DOWN_MONO forces the old single-kernel down (top_k loop in
        // one block) instead of the two-kernel GEMV+sum split — same binary, for an
        // apples-to-apples end-to-end delta.
        static DOWN_MONO: OnceLock<bool> = OnceLock::new();
        let down_mono = *DOWN_MONO.get_or_init(|| std::env::var_os("SKEIN_MOE_DOWN_MONO").is_some());
        let kt_events = if ktime {
            // CU_EVENT_DEFAULT (=0) keeps timing enabled; None would default to
            // CU_EVENT_DISABLE_TIMING and elapsed_ms would fail.
            let f = Some(crate::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
            let ctx = stream.context();
            Some((
                ctx.new_event(f).unwrap(),
                ctx.new_event(f).unwrap(),
                ctx.new_event(f).unwrap(),
            ))
        } else {
            None
        };

        // GROUPED fp8 (batch>1): one block per (output row, EXPERT), expert weight
        // read once into shared and reused across the tokens routing to it — so MoE
        // HBM traffic stops scaling ~linearly with the batch. Default-on for seq>1;
        // SKEIN_MOE_GROUPED_OFF forces the per-token path.
        let num_experts_i = num_experts as i32;
        // Opt-in: at batch=8 the per-token kernel is faster (its re-reads are
        // L2-cached, while grouped also pays for unused experts). Grouped wins at
        // larger batch where per-token's linear HBM growth dominates L2.
        let grouped = seq > 1 && std::env::var_os("SKEIN_MOE_GROUPED").is_some();
        let grid_gu_grouped = LaunchConfig {
            grid_dim: (intermediate as u32, num_experts as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (hidden * 2) as u32, // gate_row + up_row fp8
        };
        // gate_up GEMV + gated activation. Grid: (intermediate, seq*top_k).
        let grid_gu = LaunchConfig {
            grid_dim: (intermediate as u32, (seq * top_k) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        if let Some((a, _, _)) = &kt_events {
            a.record(stream).unwrap();
        }
        // v2 (SKEIN_MOE_V2, fp8 only): warp-per-row, 8 rows/block, x staged in shared.
        let grid_gu_v2 = LaunchConfig {
            grid_dim: ((intermediate as u32).div_ceil(8), (seq * top_k) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (hidden * 2) as u32,
        };
        unsafe {
            if let Some(f8) = fp8w {
                let gu_w = f8.gate_up_ptr;
                let gu_s = f8.gate_up_scale_ptr;
                if permuted {
                    let cfg = LaunchConfig {
                        grid_dim: (intermediate as u32, num_active as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: (hidden * 2) as u32,
                    };
                    stream
                        .launch_builder(&kernels.13)
                        .arg(&xbf16_ptr).arg(&perm_ptr).arg(&be_ptr).arg(&bs_ptr).arg(&bn_ptr)
                        .arg(&gu_w).arg(&gu_s).arg(&hid_ptr)
                        .arg(&hidden_i).arg(&intermediate_i).arg(&gate_up_dim_i).arg(&top_k_i).arg(&act_mode)
                        .launch(cfg)?;
                } else if grouped {
                    stream
                        .launch_builder(&kernels.10)
                        .arg(&xbf16_ptr).arg(&topk_idx_ptr).arg(&gu_w).arg(&gu_s).arg(&hid_ptr)
                        .arg(&hidden_i).arg(&intermediate_i).arg(&gate_up_dim_i).arg(&top_k_i)
                        .arg(&idx_stride_i).arg(&seq_i).arg(&act_mode).arg(&num_experts_i)
                        .launch(grid_gu_grouped)?;
                } else {
                let (gu_fn, gu_cfg) = if use_v2 {
                    (&kernels.8, grid_gu_v2)
                } else {
                    (&kernels.6, grid_gu)
                };
                stream
                    .launch_builder(gu_fn)
                    .arg(&xbf16_ptr).arg(&topk_idx_ptr).arg(&gu_w).arg(&gu_s).arg(&hid_ptr)
                    .arg(&hidden_i).arg(&intermediate_i).arg(&gate_up_dim_i).arg(&top_k_i)
                    .arg(&idx_stride_i).arg(&seq_i).arg(&act_mode)
                    .launch(gu_cfg)?;
                }
            } else {
                stream
                    .launch_builder(gate_up_act_fn)
                    .arg(&xbf16_ptr).arg(&topk_idx_ptr).arg(&gate_up_ptr).arg(&hid_ptr)
                    .arg(&hidden_i).arg(&intermediate_i).arg(&gate_up_dim_i).arg(&top_k_i)
                    .arg(&idx_stride_i).arg(&seq_i).arg(&act_mode)
                    .launch(grid_gu)?;
            }
        }

        if let Some((_, b, _)) = &kt_events {
            b.record(stream).unwrap();
        }

        // down GEMV + weighted combine. Grid: (hidden, seq).
        let grid_dn = LaunchConfig {
            grid_dim: (hidden as u32, seq as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // v2 (SKEIN_MOE_V2, fp8 only): warp-per-row, 4 rows/block, both experts'
        // hd staged in shared (28 KB), top_k looped inside the warp.
        let grid_dn_v2 = LaunchConfig {
            grid_dim: ((hidden as u32).div_ceil(4), seq as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: (top_k * intermediate * 2) as u32,
        };
        let grid_dn_grouped = LaunchConfig {
            grid_dim: (hidden as u32, num_experts as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: intermediate as u32, // D row fp8
        };
        unsafe {
            if let Some(f8) = fp8w {
                let dn_w = f8.down_ptr;
                let dn_s = f8.down_scale_ptr;
                let n_out = (seq * hidden) as i32;
                let zcfg = LaunchConfig {
                    grid_dim: ((n_out as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                if permuted {
                    // binned down accumulates into out via atomicAdd → zero first.
                    stream.launch_builder(&kernels.11).arg(&output_ptr).arg(&n_out).launch(zcfg)?;
                    let cfg = LaunchConfig {
                        grid_dim: (hidden as u32, num_active as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: intermediate as u32,
                    };
                    stream
                        .launch_builder(&kernels.14)
                        .arg(&hid_ptr).arg(&perm_ptr).arg(&be_ptr).arg(&bs_ptr).arg(&bn_ptr)
                        .arg(&topk_vals_ptr).arg(&scale_ptr).arg(&dn_w).arg(&dn_s).arg(&output_ptr)
                        .arg(&hidden_i).arg(&intermediate_i).arg(&top_k_i)
                        .arg(&vals_stride_i).arg(&normalize).arg(&use_scale)
                        .launch(cfg)?;
                } else if grouped {
                    // The grouped down accumulates into out via atomicAdd, so zero it
                    // first (capturable plain-kernel memset on the stream).
                    stream
                        .launch_builder(&kernels.11)
                        .arg(&output_ptr).arg(&n_out)
                        .launch(zcfg)?;
                    stream
                        .launch_builder(&kernels.12)
                        .arg(&hid_ptr).arg(&topk_idx_ptr).arg(&topk_vals_ptr).arg(&scale_ptr)
                        .arg(&dn_w).arg(&dn_s).arg(&output_ptr)
                        .arg(&hidden_i).arg(&intermediate_i).arg(&top_k_i).arg(&idx_stride_i)
                        .arg(&vals_stride_i).arg(&seq_i).arg(&normalize).arg(&use_scale).arg(&num_experts_i)
                        .launch(grid_dn_grouped)?;
                } else if use_v2 {
                    stream
                        .launch_builder(&kernels.9)
                        .arg(&hid_ptr).arg(&topk_idx_ptr).arg(&topk_vals_ptr).arg(&scale_ptr)
                        .arg(&dn_w).arg(&dn_s).arg(&output_ptr)
                        .arg(&hidden_i).arg(&intermediate_i).arg(&top_k_i).arg(&idx_stride_i)
                        .arg(&vals_stride_i).arg(&seq_i).arg(&normalize).arg(&use_scale)
                        .launch(grid_dn_v2)?;
                } else if down_mono {
                    // Old single-kernel down (A/B baseline via SKEIN_MOE_DOWN_MONO).
                    stream
                        .launch_builder(&kernels.7)
                        .arg(&hid_ptr).arg(&topk_idx_ptr).arg(&topk_vals_ptr).arg(&scale_ptr)
                        .arg(&dn_w).arg(&dn_s).arg(&output_ptr)
                        .arg(&hidden_i).arg(&intermediate_i).arg(&top_k_i).arg(&idx_stride_i)
                        .arg(&vals_stride_i).arg(&seq_i).arg(&normalize).arg(&use_scale)
                        .launch(grid_dn)?;
                } else {
                    // v1 two-kernel split (vLLM pattern). KERNEL 1: per-expert GEMV
                    // on grid (hidden, seq*top_k) — same block count as gate_up, no
                    // top_k loop, no per-expert __syncthreads — writes scaled partials.
                    // KERNEL 2: cheap weighted reduction over top_k → final out.
                    // partials [seq, top_k, hidden] f32. Under capture it must be a
                    // persistent scratch (distinct key 4) so the replayed graph hits a
                    // stable address; off-capture it's a per-call alloc kept alive to
                    // function scope (matches x/hid scratch lifetimes).
                    let partials_ptr = if super::is_capture() {
                        super::capture_scratch(stream, 4, seq * top_k * hidden * 4)
                    } else {
                        let buf = stream.alloc::<u8>(seq * top_k * hidden * 4)?;
                        let p = slice_ptr(&buf, stream);
                        _partials_owned = buf;
                        p
                    };
                    let grid_gemv = LaunchConfig {
                        grid_dim: (hidden as u32, (seq * top_k) as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    stream
                        .launch_builder(&kernels.15)
                        .arg(&hid_ptr).arg(&topk_idx_ptr).arg(&dn_w).arg(&dn_s).arg(&partials_ptr)
                        .arg(&hidden_i).arg(&intermediate_i).arg(&top_k_i).arg(&idx_stride_i).arg(&seq_i)
                        .launch(grid_gemv)?;
                    let grid_sum = LaunchConfig {
                        grid_dim: ((hidden as u32).div_ceil(128), seq as u32, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    stream
                        .launch_builder(&kernels.16)
                        .arg(&partials_ptr).arg(&topk_idx_ptr).arg(&topk_vals_ptr).arg(&scale_ptr).arg(&output_ptr)
                        .arg(&hidden_i).arg(&top_k_i).arg(&idx_stride_i).arg(&vals_stride_i).arg(&seq_i)
                        .arg(&normalize).arg(&use_scale)
                        .launch(grid_sum)?;
                }
            } else {
                stream
                    .launch_builder(down_combine_fn)
                    .arg(&hid_ptr).arg(&topk_idx_ptr).arg(&topk_vals_ptr).arg(&scale_ptr)
                    .arg(&down_ptr).arg(&output_ptr)
                    .arg(&hidden_i).arg(&intermediate_i).arg(&top_k_i).arg(&idx_stride_i)
                    .arg(&vals_stride_i).arg(&seq_i).arg(&normalize).arg(&use_scale)
                    .launch(grid_dn)?;
            }
        }

        if let Some((a, b, c)) = &kt_events {
            use std::sync::atomic::Ordering::Relaxed;
            c.record(stream).unwrap();
            c.synchronize().unwrap();
            // elapsed_ms is `end - self`: gate_up = a->b, down = b->c.
            let gu_us = a.elapsed_ms(b).unwrap() as f64 * 1000.0;
            let dn_us = b.elapsed_ms(c).unwrap() as f64 * 1000.0;
            KT_GU_NS.fetch_add((gu_us * 1000.0) as u64, Relaxed);
            KT_DN_NS.fetch_add((dn_us * 1000.0) as u64, Relaxed);
            let n = KT_CALLS.fetch_add(1, Relaxed) + 1;
            if n % 64 == 0 {
                // fp8 weight bytes read per call: top_k experts' full matrices.
                let gu_bytes = (gate_up_dim * hidden * top_k) as f64;
                let dn_bytes = (hidden * intermediate * top_k) as f64;
                let gu_us_avg = KT_GU_NS.load(Relaxed) as f64 / 1000.0 / n as f64;
                let dn_us_avg = KT_DN_NS.load(Relaxed) as f64 / 1000.0 / n as f64;
                let gu_gbs = gu_bytes / (gu_us_avg * 1e-6) / 1e9;
                let dn_gbs = dn_bytes / (dn_us_avg * 1e-6) / 1e9;
                let peak = 1790.0;
                eprintln!(
                    "SKEIN_MOE_KTIME n={n} gate_up: {gu_us_avg:.1}us {gu_gbs:.0}GB/s ({:.0}% peak) | down: {dn_us_avg:.1}us {dn_gbs:.0}GB/s ({:.0}% peak) | total {:.1}us/call",
                    gu_gbs / peak * 100.0,
                    dn_gbs / peak * 100.0,
                    gu_us_avg + dn_us_avg,
                );
            }
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

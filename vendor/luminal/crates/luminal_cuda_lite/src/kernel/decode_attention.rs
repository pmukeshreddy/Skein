//! Fused single-query (decode) GQA attention as a direct NVRTC custom op.
//!
//! Replaces the primitive QK^T → scale → mask → softmax → AV lowering for the
//! seq=1 cached-decode attention (`attention_fixed_cache`). That lowering is a
//! broadcasted `Mul` + `SumReduce` over the FULL static cache
//! (`KV_CACHE_CAP = 2048`), so it (a) materializes a `[batch, n_heads, CAP,
//! head_dim]` intermediate and (b) does work for every one of the 2048 cache
//! slots regardless of the real context length — which is why attention
//! dominates the decode forward and scales with the microbatch.
//!
//! This op does the whole thing in one kernel, reading the fixed-cap KV cache
//! in place and EARLY-EXITING at the runtime `position` (only slots `0..=pos`
//! are touched). One block per `(batch_row, query_head)`; the GQA mapping
//! `kv = head / groups` is computed in-kernel. All tensors are F32.
//!
//! Opaque to egglog by design (same approach as `kernel::matmul2d`): we inject
//! it directly via `cx.custom_op`, we are not trying to fuse with neighbours.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    dtype::DType, op::CustomOp, op::LLIROp, prelude::FxHashMap, prelude::GraphTensor,
    shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

/// Fused decode attention. All dims static (baked into the CUDA source); the
/// only runtime value is `position`, read from the `position` input tensor.
#[derive(Debug, Clone)]
pub struct FusedDecodeAttnKernel {
    pub batch: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub cap: usize,
}

impl KernelOp for FusedDecodeAttnKernel {
    #[allow(clippy::type_complexity)]
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let groups = self.n_heads / self.n_kv_heads;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let nwarp = self.head_dim.div_ceil(32);

        // out: [B,H,D]  q: [B,H,D] (rotated)  k/v: [B,C,KV,D] (cache, new token
        // already selected into slot pos)  position: [1] f32.
        // One block per (b, head); blockDim.x == head_dim. Dynamic shared holds
        // qs[D] + sc[C]; static shared holds the warp-reduction scratch.
        let kernel = format!(
            r#"
extern "C" __global__ void fused_decode_attn(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ position
) {{
    const int H = {h};
    const int KV = {kv};
    const int G = {g};
    const int D = {d};
    const int C = {c};
    const float SCALE = {scale}f;

    const int bh = blockIdx.x;          // 0 .. B*H-1
    const int bi = bh / H;
    const int hh = bh % H;
    const int kv = hh / G;              // GQA: query head -> kv head
    const int t  = threadIdx.x;         // 0 .. D-1

    int pos = (int)position[0];
    if (pos < 0) pos = 0;
    int n = pos + 1;                    // valid slots 0..pos
    if (n > C) n = C;

    extern __shared__ float sh[];
    float* qs = sh;                     // D
    float* sc = sh + D;                 // C
    __shared__ float red[{nwarp}];
    __shared__ float smax_s;
    __shared__ float ssum_s;

    // Load this (b,head) query row.
    qs[t] = q[((long)bi * H + hh) * D + t];
    __syncthreads();

    // Scores: each thread handles a strided subset of slots, full dot over D.
    for (int s = t; s < n; s += D) {{
        const float* krow = k + (((long)bi * C + s) * KV + kv) * D;
        float dot = 0.f;
        #pragma unroll 8
        for (int dd = 0; dd < D; ++dd) dot += qs[dd] * krow[dd];
        sc[s] = dot * SCALE;
    }}
    __syncthreads();

    // Block max over sc[0..n).
    float m = -1e30f;
    for (int s = t; s < n; s += D) m = fmaxf(m, sc[s]);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((t & 31) == 0) red[t >> 5] = m;
    __syncthreads();
    if (t == 0) {{ float mm = red[0]; for (int i = 1; i < {nwarp}; ++i) mm = fmaxf(mm, red[i]); smax_s = mm; }}
    __syncthreads();
    float smax = smax_s;

    // exp(score - max) in place, block sum.
    float ssum = 0.f;
    for (int s = t; s < n; s += D) {{ float e = __expf(sc[s] - smax); sc[s] = e; ssum += e; }}
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) ssum += __shfl_down_sync(0xffffffffu, ssum, o);
    if ((t & 31) == 0) red[t >> 5] = ssum;
    __syncthreads();
    if (t == 0) {{ float ss = 0.f; for (int i = 0; i < {nwarp}; ++i) ss += red[i]; ssum_s = ss; }}
    __syncthreads();
    float inv = (ssum_s > 0.f) ? (1.f / ssum_s) : 0.f;

    // Output dim t = inv * sum_s softmax[s] * v[s, t].
    float acc = 0.f;
    for (int s = 0; s < n; ++s) {{
        acc += sc[s] * v[(((long)bi * C + s) * KV + kv) * D + t];
    }}
    out[((long)bi * H + hh) * D + t] = acc * inv;
}}
"#,
            h = self.n_heads,
            kv = self.n_kv_heads,
            g = groups,
            d = self.head_dim,
            c = self.cap,
            scale = scale,
            nwarp = nwarp,
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("fused_decode_attn").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                Expression::from(self.batch * self.n_heads),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            (
                Expression::from(self.head_dim),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            // Dynamic shared: qs[head_dim] + sc[cap], f32.
            Expression::from((self.head_dim + self.cap) * 4),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        Expression::from(self.batch * self.n_heads * self.head_dim)
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::F32
    }

    fn bytes_loaded(&self) -> Expression {
        // ~ K + V over the valid window; cap is the static upper bound.
        Expression::from(self.batch * self.n_heads * self.cap * self.head_dim * 2 * 4)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        Expression::from(self.batch * self.n_heads * self.cap * self.head_dim * 2 * 2)
    }

    fn kernel_name(&self) -> &'static str {
        "FusedDecodeAttn"
    }
}

/// CustomOp wrapper for [`FusedDecodeAttnKernel`].
#[derive(Debug, Clone)]
pub struct FusedDecodeAttnCustom(pub FusedDecodeAttnKernel);

impl CustomOp for FusedDecodeAttnCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Fused seq=1 GQA decode attention.
///
/// * `q` — rotated query, `[batch, 1, n_heads*head_dim]` (F32, contiguous).
/// * `k_full` / `v_full` — the fixed-cap cache `[batch, cap, n_kv_heads*head_dim]`
///   with the new token already selected into slot `position` (F32).
/// * `position` — `[1]` F32 absolute position (shared across the batch).
///
/// Returns `[batch, 1, n_heads*head_dim]` F32. Masking is implicit: only slots
/// `0..=position` are read.
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_attention(
    q: GraphTensor,
    k_full: GraphTensor,
    v_full: GraphTensor,
    position: GraphTensor,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    cap: usize,
    batch: usize,
) -> GraphTensor {
    assert_eq!(q.dtype, DType::F32, "fused_decode_attention expects F32 q");
    assert_eq!(k_full.dtype, DType::F32, "fused_decode_attention expects F32 k");
    assert_eq!(v_full.dtype, DType::F32, "fused_decode_attention expects F32 v");
    assert_eq!(
        head_dim % 32,
        0,
        "fused_decode_attention requires head_dim % 32 == 0 (got {head_dim})"
    );
    let kern = FusedDecodeAttnKernel {
        batch,
        n_heads,
        n_kv_heads,
        head_dim,
        cap,
    };
    let cx = unsafe { &mut *q.graph_ref };
    cx.custom_op(
        FusedDecodeAttnCustom(kern),
        vec![q, k_full, v_full, position],
        (batch, 1usize, n_heads * head_dim),
        DType::F32,
    )
}

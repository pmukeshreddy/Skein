//! On-device argmax over a bf16 logits vector (SKEIN_DEVICE_LOGITS).
//!
//! Why: the default decode tail reads the full f32 vocab logits to the host every
//! token (`read(LOGITS)` → D2H of `vocab` f32) and runs argmax on the CPU. With
//! greedy decode the only thing the host needs is the argmax *index*. This kernel
//! computes that index on the GPU so the per-token host transfer shrinks from
//! `vocab * 4` bytes to 4 bytes (the token id).
//!
//! How: one block of 1024 threads; each thread scans its strided slice of the
//! `vocab` bf16 logits tracking a local (max, index), then a shared-memory tree
//! reduction finds the global argmax and thread 0 writes the index to a
//! persistent 1-element device buffer reused every token. bf16 is widened to f32
//! for the comparison by `bits << 16` (bf16 is the high 16 bits of f32), so the
//! ordering is exact (no rounding). Mirrors the nvrtc-kernel pattern in
//! [`super::shm_allreduce`].

use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};

const BLOCK: u32 = 1024;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void argmax_bf16(
    const unsigned short* logits, unsigned int n, unsigned int* out_idx
) {
    __shared__ float s_val[1024];
    __shared__ unsigned int s_idx[1024];
    unsigned int tid = threadIdx.x;
    unsigned int nthreads = blockDim.x;
    // bf16 is the top 16 bits of an IEEE-754 f32; widen by <<16 for an exact
    // float comparison (no rounding). Start below any finite logit.
    float best = -3.402823e38f;
    unsigned int best_i = 0;
    for (unsigned int i = tid; i < n; i += nthreads) {
        unsigned int bits = ((unsigned int)logits[i]) << 16;
        float v = __uint_as_float(bits);
        if (v > best) { best = v; best_i = i; }
    }
    s_val[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();
    // Tree reduction; on ties keep the lower index (the `>` keeps the lower-tid
    // operand, and lower tid scanned lower indices first).
    for (unsigned int stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) { *out_idx = s_idx[0]; }
}
"#;

/// Persistent on-device argmax: one compiled kernel + a reused 1-element device
/// output buffer. Built once at rank bootstrap, launched once per decode token.
pub struct DeviceArgmax {
    func: CudaFunction,
    stream: Arc<CudaStream>,
    out: CudaSlice<u32>,
    _module: Arc<CudaModule>,
}

impl DeviceArgmax {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, String> {
        let ctx = stream.context().clone();
        ctx.bind_to_thread().map_err(|e| format!("bind_to_thread: {e}"))?;
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC).map_err(|e| format!("nvrtc: {e:?}"))?;
        let module = ctx.load_module(ptx).map_err(|e| format!("load_module: {e:?}"))?;
        let func = module
            .load_function("argmax_bf16")
            .map_err(|e| format!("load_function: {e:?}"))?;
        let out = stream.alloc_zeros::<u32>(1).map_err(|e| format!("alloc out: {e:?}"))?;
        Ok(Self { func, stream, out, _module: module })
    }

    /// Device pointer of the 1-element u32 result buffer (for a 4-byte D2H).
    pub fn out_ptr(&self) -> u64 {
        let (p, _g) = self.out.device_ptr(&self.stream);
        p
    }

    /// Launch the argmax kernel over `n` bf16 logits at device pointer `ptr`.
    /// Writes the argmax index into the persistent result buffer.
    ///
    /// # Safety
    /// `ptr` must be a valid device buffer of `n` bf16 elements on this stream's
    /// device, alive for the launch.
    pub unsafe fn launch(&self, ptr: u64, n: usize) -> Result<(), String> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_u = n as u32;
        let out_ptr = self.out_ptr();
        self.stream
            .launch_builder(&self.func)
            .arg(&ptr)
            .arg(&n_u)
            .arg(&out_ptr)
            .launch(cfg)
            .map_err(|e| format!("launch argmax_bf16: {e:?}"))?;
        Ok(())
    }

    /// Read the argmax index back to the host (4-byte D2H, synchronizes).
    pub fn read_index(&self) -> Result<u32, String> {
        let v = self
            .stream
            .memcpy_dtov(&self.out)
            .map_err(|e| format!("d2h argmax: {e:?}"))?;
        Ok(v[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;
    use half::bf16;

    // Pure-Rust reference: index of the (first) maximum value.
    fn host_argmax(v: &[bf16]) -> usize {
        let mut best = f32::NEG_INFINITY;
        let mut bi = 0usize;
        for (i, &x) in v.iter().enumerate() {
            let f = x.to_f32();
            if f > best {
                best = f;
                bi = i;
            }
        }
        bi
    }

    #[test]
    fn device_argmax_matches_host() {
        let ctx = CudaContext::new(0).expect("cuda ctx");
        let stream = ctx.default_stream();
        let am = DeviceArgmax::new(stream.clone()).expect("argmax build");

        // Deterministic LCG so the test is reproducible without an rng dep.
        let mut state: u64 = 0x243F6A8885A308D3;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // ~[-1,1)
        };

        for &n in &[1usize, 2, 33, 1024, 1025, 16000, 32000] {
            for _seed in 0..4 {
                let host: Vec<bf16> = (0..n).map(|_| bf16::from_f32(next() * 10.0)).collect();
                let dev = stream.memcpy_stod(&host).expect("h2d");
                let (ptr, _g) = dev.device_ptr(&stream);
                unsafe { am.launch(ptr, n) }.expect("launch");
                let got = am.read_index().expect("read") as usize;
                let want = host_argmax(&host);
                // Compare by VALUE so bf16 ties (exact-equal maxima) pass either way.
                assert!(
                    got < n,
                    "n={n}: device index {got} out of range"
                );
                assert_eq!(
                    host[got].to_f32(),
                    host[want].to_f32(),
                    "n={n}: device argmax idx {got} (val {}) != host max val {}",
                    host[got].to_f32(),
                    host[want].to_f32()
                );
            }
        }
    }
}

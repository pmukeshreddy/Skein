//! Custom 2-rank all-reduce over cross-process shared host memory, bypassing
//! NCCL for the small device-bf16 all-reduces in the decode hot loop.
//!
//! Both rank processes mmap the same /dev/shm file (name derived from the
//! shared NCCL id), `cuMemHostRegister(... DEVICEMAP)` it to get a device pointer
//! to the same physical pages, and a one-shot kernel does: write my partial ->
//! `__threadfence_system` -> set my seq flag -> spin on peer flag -> fp32-
//! accumulate sum (round-to-nearest-even -> bf16) -> write back in place. No
//! NCCL call, no proxy thread.

use std::sync::Arc;

use cudarc::driver::{
    CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use memmap2::MmapMut;

/// Custom path handles all-reduces with `elems <= MAX_ELEMS`. The per-step
/// decode all-reduce is `batch * hidden` (hidden=4096); batched decode (batch up
/// to 32) makes that 131072, so size the shm slot for it — otherwise the batched
/// all-reduce exceeds the slot and falls back to NCCL, which cannot be recorded
/// into the full-step CUDA-graph capture (breaking batched+capture). Larger
/// (prefill) still falls back to NCCL. Slot = MAX_ELEMS*2 bytes (bf16) per rank.
pub const MAX_ELEMS: usize = 131072;
const FLAGS_BYTES: usize = 64; // 2 u64 seq flags + pad to a cache line
const SLOT_BYTES: usize = MAX_ELEMS * 2; // bf16
const SHM_BYTES: usize = FLAGS_BYTES + 2 * SLOT_BYTES;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void shm_allreduce2(
    unsigned long long my_data, unsigned long long shm,
    int my_rank, int elems, int slot_bytes
) {
    // flags[rank] is a monotonically increasing generation, self-incremented by
    // the kernel each invocation (it persists in shared host memory across CUDA
    // graph replays). No host-supplied seq is baked in, so this all-reduce is
    // correct as a static node in a replayed full-step graph. Single writer per
    // flag (each rank writes only flags[my_rank]); the peer only reads it.
    volatile unsigned long long* flags = (volatile unsigned long long*)shm;
    char* base = (char*)shm + 64;
    int peer = 1 - my_rank;
    unsigned short* my_slot   = (unsigned short*)(base + (long long)my_rank * slot_bytes);
    unsigned short* peer_slot = (unsigned short*)(base + (long long)peer    * slot_bytes);
    unsigned short* d = (unsigned short*)my_data;
    int tid = threadIdx.x, n = blockDim.x;
    // 1. publish my partial into my slot
    for (int i = tid; i < elems; i += n) my_slot[i] = d[i];
    __threadfence_system();
    __syncthreads();
    // 2. self-increment my generation, signal arrival, wait for peer (thread 0)
    if (tid == 0) {
        unsigned long long g = flags[my_rank] + 1ULL;
        flags[my_rank] = g;
        __threadfence_system();
        while (flags[peer] < g) { }
    }
    __syncthreads();
    __threadfence_system();
    // 3. sum (fp32 accumulate, round-to-nearest-even back to bf16), in place
    for (int i = tid; i < elems; i += n) {
        unsigned int ua = ((unsigned int)d[i]) << 16;
        unsigned int ub = ((unsigned int)peer_slot[i]) << 16;
        float s = __uint_as_float(ua) + __uint_as_float(ub);
        unsigned int us = __float_as_uint(s);
        unsigned int r = us + 0x7FFFu + ((us >> 16) & 1u);
        d[i] = (unsigned short)(r >> 16);
    }
}
"#;

pub struct ShmAllReduce {
    rank: usize,
    shm_dev_ptr: u64,
    func: CudaFunction,
    stream: Arc<CudaStream>,
    _mmap: MmapMut,
    _module: Arc<CudaModule>,
}

impl ShmAllReduce {
    /// Set up the shared segment + kernel. Both ranks must call this with the same
    /// `id_bytes` (the NCCL unique id, already shared at bootstrap). The caller
    /// MUST barrier across ranks after construction (both must have zeroed the
    /// flags) before the first [`all_reduce`](Self::all_reduce).
    pub fn new(stream: Arc<CudaStream>, rank: usize, id_bytes: &[u8]) -> Result<Self, String> {
        // FNV-1a of the shared id -> identical path on both ranks.
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in id_bytes.iter().take(32) {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let path = format!("/dev/shm/skein_ar_{h:016x}");

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|e| format!("open {path}: {e}"))?;
        file.set_len(SHM_BYTES as u64)
            .map_err(|e| format!("set_len {path}: {e}"))?;
        let mut mmap = unsafe {
            MmapMut::map_mut(&file).map_err(|e| format!("mmap {path}: {e}"))?
        };
        // Zero the seq flags (the file may be stale from a prior run).
        for b in mmap[0..FLAGS_BYTES].iter_mut() {
            *b = 0;
        }

        let ctx = stream.context().clone();
        ctx.bind_to_thread().map_err(|e| format!("bind_to_thread: {e}"))?;

        let host_ptr = mmap.as_mut_ptr() as *mut core::ffi::c_void;
        let flags = cudarc::driver::sys::CU_MEMHOSTREGISTER_PORTABLE
            | cudarc::driver::sys::CU_MEMHOSTREGISTER_DEVICEMAP;
        unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(host_ptr, SHM_BYTES, flags)
                .result()
                .map_err(|e| format!("cuMemHostRegister: {e:?}"))?;
        }
        let mut dptr: cudarc::driver::sys::CUdeviceptr = 0;
        unsafe {
            cudarc::driver::sys::cuMemHostGetDevicePointer_v2(&mut dptr, host_ptr, 0)
                .result()
                .map_err(|e| format!("cuMemHostGetDevicePointer: {e:?}"))?;
        }
        let shm_dev_ptr = dptr as u64;

        let ptx = cudarc::nvrtc::compile_ptx(KERNEL_SRC).map_err(|e| format!("nvrtc: {e:?}"))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| format!("load_module: {e:?}"))?;
        let func = module
            .load_function("shm_allreduce2")
            .map_err(|e| format!("load_function: {e:?}"))?;

        tracing::info!(rank, %path, "SHM all-reduce ready (custom kernel-only path)");
        Ok(Self {
            rank,
            shm_dev_ptr,
            func,
            stream,
            _mmap: mmap,
            _module: module,
        })
    }

    /// (shm device pointer, rank, slot_bytes, max_elems) for the luminal-launched
    /// all-reduce under SKEIN_CAPTURE (the kernel must launch from luminal so it
    /// lands on the shared capture stream). The shm region was registered in the
    /// device's primary context, so the pointer is valid from luminal too.
    pub fn info(&self) -> (u64, i32, i32, usize) {
        (self.shm_dev_ptr, self.rank as i32, SLOT_BYTES as i32, MAX_ELEMS)
    }

    /// In-place sum all-reduce of `elems` bf16 at device pointer `ptr`. Both ranks
    /// MUST call with identical `elems` (the schedule guarantees this), else they
    /// deadlock on the seq flags.
    ///
    /// # Safety
    /// `ptr` must be a valid device buffer of `elems` bf16 on this rank's device.
    pub unsafe fn all_reduce(&self, ptr: u64, elems: usize) -> Result<(), String> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        };
        let rank_i = self.rank as i32;
        let elems_i = elems as i32;
        let slot_i = SLOT_BYTES as i32;
        let shm = self.shm_dev_ptr;
        self.stream
            .launch_builder(&self.func)
            .arg(&ptr)
            .arg(&shm)
            .arg(&rank_i)
            .arg(&elems_i)
            .arg(&slot_i)
            .launch(cfg)
            .map_err(|e| format!("launch shm_allreduce2: {e:?}"))?;
        Ok(())
    }
}

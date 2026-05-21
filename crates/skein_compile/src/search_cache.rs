//! On-disk cache for Luminal's per-segment search result.
//!
//! Compiling a segment runs two genuinely expensive stages: building the egglog
//! search space (`build_search_space`) and the search itself, which compiles
//! candidate kernels via NVRTC. The artifact only stores the high-level IR
//! recipe, so `serve`/`verify` re-pay both on every boot (5–8 min).
//!
//! This module caches the *result* of that work so the second and later runs of
//! the same artifact load instead of re-searching. Two layers compose:
//!
//! 1. **egglog search result** — the serialized [`SerializedEGraph`] plus the
//!    winning genome (the `(ClassId, NodeId)` choices). Replaying the genome
//!    against the e-graph reproduces the exact same LLIR — the same kernels —
//!    via [`luminal::prelude::Graph::load_search_result`], skipping both
//!    `build_search_space` and the search loop. Handled here.
//! 2. **NVRTC cubins** — content-addressed cubins keyed by (source, arch),
//!    handled inside `luminal_cuda_lite` and pointed at [`cubin_cache_dir`] via
//!    the `SKEIN_KERNEL_CACHE` env var.
//!
//! The key is a content hash of the deterministic egglog *program* string
//! (`hlir_to_egglog`) plus a backend tag, so it is stable across the `compile`
//! that writes the cache and the `serve`/`verify` that read it, and changes if
//! the segment's source changes. A stale or unreadable entry simply falls back
//! to a full search, so the cache can never produce a wrong result — at worst it
//! is ignored.

use std::path::{Path, PathBuf};

use luminal::egglog_utils::{ClassId, NodeId, SerializedEGraph, hlir_to_egglog};
use luminal::op::Runtime;
use luminal::prelude::Graph;

use crate::CompileError;

/// Subdirectory of an artifact that holds the compile cache.
pub const CACHE_DIR_NAME: &str = "cache";
/// Subdirectory of [`CACHE_DIR_NAME`] that holds NVRTC cubins.
pub const CUBIN_DIR_NAME: &str = "cubins";
/// Env var read by `luminal_cuda_lite` to locate the cubin cache.
pub const KERNEL_CACHE_ENV: &str = "SKEIN_KERNEL_CACHE";

/// The egraph/genome cache directory for an artifact root.
pub fn search_cache_dir(artifact_root: &Path) -> PathBuf {
    artifact_root.join(CACHE_DIR_NAME)
}

/// The cubin cache directory for an artifact root.
pub fn cubin_cache_dir(artifact_root: &Path) -> PathBuf {
    artifact_root.join(CACHE_DIR_NAME).join(CUBIN_DIR_NAME)
}

/// Point `luminal_cuda_lite`'s NVRTC cubin cache at this artifact by exporting
/// `SKEIN_KERNEL_CACHE`. Idempotent; safe to call on every load. One artifact is
/// served per process, so a process-global var is the right scope.
pub fn enable_cubin_cache(artifact_root: &Path) {
    let dir = cubin_cache_dir(artifact_root);
    let _ = std::fs::create_dir_all(&dir);
    // SAFETY: set once, early, before any compilation thread reads it.
    unsafe { std::env::set_var(KERNEL_CACHE_ENV, dir) };
}

/// Run Luminal's search for `cx`, served from / written to the on-disk cache.
///
/// On a cache hit the recorded genome is replayed, reproducing the same kernels
/// without re-running egglog or the search loop. On a miss (or no cache dir, or
/// a bucketed graph, which is not cached) a full `build_search_space` + search
/// runs and the result is persisted.
///
/// - `make_runtime` builds a fresh backend runtime; it may be called more than
///   once (e.g. after a stale-cache fall-through), so it must be repeatable.
/// - `stage_inputs` stages any per-`Input` zero buffers the search needs to
///   profile candidates (CUDA); it runs only on the miss path, before the search.
pub fn cached_search<R: Runtime + 'static>(
    cx: &mut Graph,
    budget: usize,
    cache_dir: Option<&Path>,
    backend_tag: &str,
    make_runtime: impl Fn() -> Result<R, CompileError>,
    stage_inputs: impl Fn(&mut R),
) -> Result<R, CompileError> {
    // Bucketed graphs are not cached (skein does not emit buckets); fall back to
    // the plain build+search path. `cache_dir == None` disables caching too.
    let Some(dir) = cache_dir.filter(|_| cx.dim_buckets.is_empty()) else {
        cx.build_search_space::<R>();
        let mut rt = make_runtime()?;
        stage_inputs(&mut rt);
        return Ok(cx.search(rt, budget));
    };

    let key = segment_cache_key(cx, backend_tag);

    if let Some((egraph, genome)) = read_cached(dir, &key) {
        if let Some(rt) = cx.load_search_result(make_runtime()?, &egraph, &genome) {
            tracing::debug!(%key, "search cache hit");
            return Ok(rt);
        }
        tracing::warn!(%key, "search cache entry did not resolve; re-searching");
    }

    cx.build_search_space::<R>();
    let mut rt = make_runtime()?;
    stage_inputs(&mut rt);
    let (rt, genome) = cx.search_capture(rt, budget);
    if let Some(egraph) = cx.egraph() {
        write_cached(dir, &key, egraph, &genome);
    }
    Ok(rt)
}

/// Content hash of the segment's deterministic egglog program plus a backend
/// tag. The program string is rebuilt identically by `compile` and by
/// `serve`/`verify` (both lower the same graph), so the key is stable across
/// processes and unique per segment source.
fn segment_cache_key(cx: &Graph, backend_tag: &str) -> String {
    let (program, _root) = hlir_to_egglog(cx);
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"skein-search-v1\0");
    hasher.update(backend_tag.as_bytes());
    hasher.update(b"\0");
    hasher.update(program.as_bytes());
    hasher.finalize().to_hex().to_string()
}

type CachedSearch = (SerializedEGraph, Vec<(ClassId, NodeId)>);

fn read_cached(dir: &Path, key: &str) -> Option<CachedSearch> {
    let egraph_bytes = std::fs::read(dir.join(format!("{key}.egraph"))).ok()?;
    let genome_bytes = std::fs::read(dir.join(format!("{key}.genome"))).ok()?;
    let egraph = bincode::deserialize::<SerializedEGraph>(&egraph_bytes).ok()?;
    let genome = bincode::deserialize::<Vec<(ClassId, NodeId)>>(&genome_bytes).ok()?;
    Some((egraph, genome))
}

/// Persist the search result. Best-effort: any IO/serialization failure leaves
/// the cache cold (the next run re-searches) rather than erroring the compile.
fn write_cached(dir: &Path, key: &str, egraph: &SerializedEGraph, genome: &[(ClassId, NodeId)]) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    if let Ok(bytes) = bincode::serialize(egraph) {
        atomic_write(&dir.join(format!("{key}.egraph")), &bytes);
    }
    if let Ok(bytes) = bincode::serialize(&genome.to_vec()) {
        atomic_write(&dir.join(format!("{key}.genome")), &bytes);
    }
}

/// Write via a process-unique temp file plus rename so a concurrent reader never
/// observes a half-written entry.
fn atomic_write(path: &Path, bytes: &[u8]) {
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    if std::fs::write(&tmp, bytes).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

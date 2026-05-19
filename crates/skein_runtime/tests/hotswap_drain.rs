//! Test 8 + 9 — hot-swap drain protocol + incompatible artifact rejection.

// Hot-swap tests synthesize their own artifact dirs and don't share the
// model fixtures `common.rs` exposes.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;

use skein_ir::types::*;
use skein_runtime::RuntimeError;
use skein_runtime::batcher::InflightSet;
use skein_runtime::hotswap::HotSwap;
use skein_runtime::token_stream::TokenStreamer;
use skein_runtime::types::{IncomingRequest, RequestId};

fn write_io_manifest(dir: &std::path::Path, inputs: Vec<(&str, Vec<usize>, Dtype)>) {
    use skein_emit::{IoManifest, IoTensor, IoTensorKind};
    let manifest = IoManifest {
        device_idx: 0,
        tensors: inputs
            .into_iter()
            .map(|(name, shape, dtype)| IoTensor {
                name: name.to_string(),
                shape,
                dtype,
                kind: IoTensorKind::Input,
            })
            .collect(),
    };
    let device0 = dir.join("device_0");
    std::fs::create_dir_all(&device0).unwrap();
    std::fs::write(
        device0.join("io.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

fn tempdir(prefix: &str) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!(
        "{prefix}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

// Test 9 — incompatible artifact rejected; no symlink touched.
#[tokio::test]
async fn hotswap_incompatible_artifact_rejected() {
    let artifacts_dir = tempdir("skein_rt_swap_incompat");

    let old_artifact = artifacts_dir.join("old");
    let new_artifact = artifacts_dir.join("new");
    write_io_manifest(&old_artifact, vec![("tokens", vec![1, 1], Dtype::Bf16)]);
    write_io_manifest(
        &new_artifact,
        // Different *input* dtype — must be rejected.
        vec![("tokens", vec![1, 1], Dtype::Fp16)],
    );

    let latest = artifacts_dir.join("LATEST");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&old_artifact, &latest).unwrap();

    let hotswap = HotSwap::new(artifacts_dir.clone(), Duration::from_secs(1));
    let inflight = Arc::new(RwLock::new(InflightSet::new()));
    let result = hotswap.swap_artifact(&new_artifact, inflight.clone()).await;
    match result {
        Err(RuntimeError::IncompatibleArtifact { reason, .. }) => {
            assert!(reason.contains("dtype changed"));
        }
        other => panic!("expected IncompatibleArtifact, got {other:?}"),
    }
    // Symlink still points to old artifact.
    let target = std::fs::read_link(&latest).unwrap();
    assert_eq!(target, old_artifact);
}

// Test 8 — hot-swap drain with sustained load. We seed the inflight set
// with 5 requests, then concurrently run the swap (drain) while another
// task retires them. Verify the swap completes only after the inflight
// empties and the symlink resolves to the new target.
#[tokio::test]
async fn hotswap_drain_during_sustained_load() {
    let artifacts_dir = tempdir("skein_rt_swap_drain");
    let old_artifact = artifacts_dir.join("old");
    let new_artifact = artifacts_dir.join("new");
    write_io_manifest(&old_artifact, vec![("tokens", vec![1, 1], Dtype::Bf16)]);
    write_io_manifest(&new_artifact, vec![("tokens", vec![1, 1], Dtype::Bf16)]);

    let latest = artifacts_dir.join("LATEST");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&old_artifact, &latest).unwrap();

    let inflight = Arc::new(RwLock::new(InflightSet::new()));
    {
        let mut guard = inflight.write().await;
        for i in 0..5u64 {
            let req = IncomingRequest {
                id: RequestId(i + 1),
                prompt_tokens: vec![],
                max_output_tokens: 1,
                arrival_ms: 0,
            };
            let (_s, sender) = TokenStreamer::paired();
            guard.insert(req, sender, BatchPolicy::Continuous { max_batch: 8 }, 0);
        }
    }

    let hotswap = HotSwap::new(artifacts_dir.clone(), Duration::from_secs(5));

    // Spawn a task that retires inflight at 5 ms intervals.
    let drainer = {
        let inflight = inflight.clone();
        tokio::spawn(async move {
            for i in 1..=5u64 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                inflight.write().await.remove(RequestId(i));
            }
        })
    };

    let swap = hotswap.swap_artifact(&new_artifact, inflight.clone()).await;
    drainer.await.ok();
    swap.expect("swap should succeed after drain");

    // Symlink atomically points to the new target.
    let target = std::fs::read_link(&latest).unwrap();
    assert_eq!(target, new_artifact);
    assert_eq!(inflight.read().await.len(), 0);
}

// Auxiliary — the symlink swap is atomic even without a real swap path:
// `LATEST` always points to a valid existing artifact dir.
#[tokio::test]
async fn hotswap_drain_timeout_when_load_never_clears() {
    let artifacts_dir = tempdir("skein_rt_swap_timeout");
    let old_artifact = artifacts_dir.join("old");
    let new_artifact = artifacts_dir.join("new");
    write_io_manifest(&old_artifact, vec![("tokens", vec![1, 1], Dtype::Bf16)]);
    write_io_manifest(&new_artifact, vec![("tokens", vec![1, 1], Dtype::Bf16)]);
    let latest = artifacts_dir.join("LATEST");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&old_artifact, &latest).unwrap();
    let inflight = Arc::new(RwLock::new(InflightSet::new()));
    {
        let mut guard = inflight.write().await;
        let req = IncomingRequest {
            id: RequestId(1),
            prompt_tokens: vec![],
            max_output_tokens: 1,
            arrival_ms: 0,
        };
        let (_s, sender) = TokenStreamer::paired();
        guard.insert(req, sender, BatchPolicy::Continuous { max_batch: 8 }, 0);
    }
    let hotswap = HotSwap::new(artifacts_dir.clone(), Duration::from_millis(50));
    let err = hotswap
        .swap_artifact(&new_artifact, inflight.clone())
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        RuntimeError::DrainTimeout { remaining: 1, .. }
    ));
    // Symlink unchanged.
    let target = std::fs::read_link(&latest).unwrap();
    assert_eq!(target, old_artifact);
}

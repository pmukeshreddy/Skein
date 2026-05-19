//! Test-only tiny artifact fixture builder smoke test.

#[path = "fixtures/build_tiny_artifact.rs"]
mod build_tiny_artifact;

use build_tiny_artifact::build_tiny_artifact;
use skein_compile::SkeinArtifact;

#[test]
fn tiny_artifact_fixture_builder_writes_loadable_artifact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let tiny = build_tiny_artifact(dir.path());
    let artifact = SkeinArtifact::load(&tiny.artifact_dir).expect("load tiny artifact");
    assert!(tiny.root.exists());
    assert_eq!(tiny.ir.meta.num_layers, 2);
    assert_eq!(tiny.cluster.num_devices, 1);
    assert_eq!(artifact.plan, tiny.plan);
    assert_eq!(artifact.devices.len(), 1);
    assert_eq!(artifact.devices[0].device_idx, 0);
    assert!(artifact.devices[0].weights_path.exists());
    assert_eq!(artifact.devices[0].io_manifest.device_idx, 0);
}

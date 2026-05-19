mod fixtures {
    pub mod build_tiny_artifact;
}

use std::path::PathBuf;

use skein_parity::{HFReference, PythonSubprocessReference, RealSkeinForward, SkeinForward};

#[test]
#[ignore = "requires Python venv with transformers + torch and network/model cache for gpt2"]
fn python_subprocess_reference_gpt2() {
    let r = PythonSubprocessReference::new(PathBuf::from("gpt2"), "float32")
        .expect("construct python reference");
    let tokens = r.tokenize("Hello, world").expect("tokenize");
    let out = r.forward_with_hooks(&tokens).expect("forward");
    assert_eq!(out.per_layer_activations.len(), 12);
    assert!(out.final_logits.iter().all(|x| x.is_finite()));
}

#[test]
#[ignore = "native Luminal compile for the tiny artifact is slow"]
fn real_skein_forward_loads_and_runs_native() {
    let tmp = tempfile::tempdir().unwrap();
    let tiny = fixtures::build_tiny_artifact::build_tiny_artifact(tmp.path());
    assert!(tiny.root.exists());
    assert_eq!(tiny.cluster.num_devices, 1);
    assert_eq!(tiny.plan.model_meta.vocab, tiny.ir.meta.vocab);
    let mut forward = RealSkeinForward::load_native(&tiny.artifact_dir).expect("load native");
    let out = forward
        .forward_with_hooks(&[1, 2, 3, 4, 5])
        .expect("forward");
    assert_eq!(out.per_layer_activations.len(), tiny.ir.meta.num_layers);
    assert_eq!(out.final_logits.len(), tiny.ir.meta.vocab);
    assert!(
        out.per_layer_activations
            .iter()
            .flatten()
            .chain(out.final_logits.iter())
            .all(|x| x.is_finite())
    );
}

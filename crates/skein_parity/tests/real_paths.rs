//! Real-reference path. Ignored by default because it needs a Python venv
//! with `transformers` + `torch` and a cached model.

use std::path::PathBuf;

use skein_parity::{HFReference, PythonSubprocessReference};

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

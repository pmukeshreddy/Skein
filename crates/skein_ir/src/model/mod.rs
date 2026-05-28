//! HF `config.json` → typed IR importer.
//!
//! Dispatches on the `architectures` field. `MixtralForCausalLM` is
//! supported. Unrecognised architectures return
//! `ImportError::ArchitectureNotYetImplemented`.

use serde::Deserialize;
use std::path::Path;

use crate::error::ImportError;
use crate::ir::Graph;

pub mod mixtral;

/// The subset of HF `config.json` fields Skein consults to dispatch.
/// Architecture-specific fields are parsed inside the per-arch module.
#[derive(Debug, Clone, Deserialize)]
struct ArchDispatch {
    #[serde(default)]
    architectures: Vec<String>,
}

/// Import a model from an HF `config.json` file on disk.
pub fn import_from_file(path: &Path) -> Result<Graph, ImportError> {
    let s = std::fs::read_to_string(path).map_err(|source| ImportError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    import_from_str(&s)
}

/// Import a model from a raw `config.json` string.
pub fn import_from_str(s: &str) -> Result<Graph, ImportError> {
    let dispatch: ArchDispatch = serde_json::from_str(s)?;
    let arch = dispatch
        .architectures
        .first()
        .cloned()
        .ok_or(ImportError::NoArchitecture)?;
    match arch.as_str() {
        "MixtralForCausalLM" => mixtral::build_from_str(s),
        "LlamaForCausalLM" | "MistralForCausalLM" | "Qwen2ForCausalLM" => {
            Err(ImportError::ArchitectureNotYetImplemented { arch })
        }
        _ => Err(ImportError::UnsupportedArchitecture(arch)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_architectures() {
        let bad = r#"{"hidden_size": 4096}"#;
        assert!(matches!(
            import_from_str(bad),
            Err(ImportError::NoArchitecture)
        ));
    }

    #[test]
    fn flags_recognized_but_unimplemented_archs() {
        let llama = r#"{"architectures": ["LlamaForCausalLM"]}"#;
        match import_from_str(llama).unwrap_err() {
            ImportError::ArchitectureNotYetImplemented { arch } => {
                assert_eq!(arch, "LlamaForCausalLM")
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_arch() {
        let bad = r#"{"architectures": ["GpuGoBrrrForCausalLM"]}"#;
        assert!(matches!(
            import_from_str(bad),
            Err(ImportError::UnsupportedArchitecture(_))
        ));
    }
}

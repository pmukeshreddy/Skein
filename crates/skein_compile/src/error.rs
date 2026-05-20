//! Error type for `skein_compile`. Minimal — Luminal's compile/search path
//! is currently infallible (returns `R` directly, not `Result<R, _>`), but
//! `Result` shape on `ComputeRuntime::build_and_search` is preserved so a
//! future Luminal version that surfaces errors slots in without a breaking
//! API change.

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("Luminal compile/search failed: {0}")]
    Search(String),

    #[error("could not read {path}: {source}")]
    Io {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("could not parse JSON at {path}: {source}")]
    Json {
        path: std::path::PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("could not parse {what} JSON embedded in artifact recipe: {source}")]
    RecipeJson {
        what: &'static str,
        #[source]
        source: serde_json::Error,
    },

    #[error("could not parse bincode at {path}: {source}")]
    BincodeRead {
        path: std::path::PathBuf,
        #[source]
        source: Box<bincode::ErrorKind>,
    },

    #[error("could not encode bincode for {path}: {source}")]
    BincodeWrite {
        path: std::path::PathBuf,
        #[source]
        source: Box<bincode::ErrorKind>,
    },

    #[error("artifact has {weights} weights files for {devices} devices")]
    ArtifactDeviceCountMismatch { devices: usize, weights: usize },

    #[error("artifact at {path} is missing required file {file}")]
    ArtifactMissingFile {
        path: std::path::PathBuf,
        file: &'static str,
    },

    #[error("artifact has no devices")]
    ArtifactHasNoDevices,

    #[error("artifact rebuild failed: {0}")]
    Emit(#[from] skein_emit::EmitError),

    #[error("runtime tensor {0:?} is not known in this segment")]
    UnknownTensor(String),

    #[error("runtime error: {0}")]
    DynRuntime(#[from] crate::DynRuntimeError),

    #[error("collective execution failed: {0}")]
    Collective(String),

    #[error("luminal CudaRuntime initialization failed")]
    CudaRuntimeInit {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("artifact weight tensor {tensor:?} is missing from {path}")]
    MissingWeight {
        path: std::path::PathBuf,
        tensor: String,
    },

    #[error("could not parse safetensors at {path}: {source}")]
    Safetensors {
        path: std::path::PathBuf,
        #[source]
        source: safetensors::SafeTensorError,
    },

    #[error("unsupported safetensors dtype {dtype:?} for tensor {tensor:?}")]
    UnsupportedWeightDtype {
        tensor: String,
        dtype: safetensors::Dtype,
    },

    #[error("tensor {tensor:?} has {got} values, expected {expected}")]
    TensorSizeMismatch {
        tensor: String,
        expected: usize,
        got: usize,
    },

    #[error("executor could not find device {device_idx} segment {segment_idx}")]
    MissingSegment {
        device_idx: usize,
        segment_idx: usize,
    },

    #[error("executor has no runtime device {0}")]
    MissingDevice(usize),
}

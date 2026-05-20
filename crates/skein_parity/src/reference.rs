//! `HFReference` — the upstream "ground truth" forward pass.
//!
//! [`PythonSubprocessReference`] drives a `transformers` subprocess: it
//! carries the subprocess configuration and a validated reference dtype,
//! caches outputs per token sequence, and executes the forward pass over a
//! JSON-lines protocol on stdin/stdout.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::ParityError;

/// Output of one forward pass through the reference. `per_layer_activations`
/// is one `Vec<f32>` per decoder block (length `num_decoder_blocks`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceOutput {
    pub per_layer_activations: Vec<Vec<f32>>,
    pub final_logits: Vec<f32>,
}

/// Upstream reference. [`PythonSubprocessReference`] invokes `transformers`
/// to produce ground-truth activations and logits.
pub trait HFReference: Send + Sync {
    fn forward_with_hooks(&self, tokens: &[u32]) -> Result<ReferenceOutput, ParityError>;
    fn forward_batch_with_hooks(
        &self,
        prompts: &[Vec<u32>],
    ) -> Result<Vec<ReferenceOutput>, ParityError> {
        prompts
            .iter()
            .map(|tokens| self.forward_with_hooks(tokens))
            .collect()
    }
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, ParityError>;
}

// ---------------------------------------------------------------------------
// PythonSubprocessReference — `transformers` ground-truth reference.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReferenceDtype(String);

impl ReferenceDtype {
    pub fn new(dtype: impl Into<String>) -> Result<Self, ParityError> {
        let dtype = dtype.into();
        match dtype.as_str() {
            "bfloat16" | "float16" | "float32" => Ok(Self(dtype)),
            _ => Err(ParityError::InvalidReferenceDtype { dtype }),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct PythonSubprocessReference {
    pub model_path: PathBuf,
    pub python_executable: PathBuf,
    pub script_path: PathBuf,
    pub reference_dtype: ReferenceDtype,
    cache: Arc<Mutex<HashMap<Vec<u32>, ReferenceOutput>>>,
    timeout: Duration,
}

impl PythonSubprocessReference {
    pub fn new(
        model_path: PathBuf,
        reference_dtype: impl Into<String>,
    ) -> Result<Self, ParityError> {
        let python_executable = std::env::var_os("PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("python3"));
        let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("reference")
            .join("verify_reference.py");
        Self::with_paths(model_path, python_executable, script_path, reference_dtype)
    }

    pub fn with_paths(
        model_path: PathBuf,
        python_executable: PathBuf,
        script_path: PathBuf,
        reference_dtype: impl Into<String>,
    ) -> Result<Self, ParityError> {
        Ok(Self {
            model_path,
            python_executable,
            script_path,
            reference_dtype: ReferenceDtype::new(reference_dtype)?,
            cache: Arc::new(Mutex::new(HashMap::new())),
            timeout: Duration::from_secs(120),
        })
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    fn run_script(
        &self,
        args: &[&str],
        request: &serde_json::Value,
    ) -> Result<String, ParityError> {
        let mut child = Command::new(&self.python_executable)
            .arg(&self.script_path)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| ParityError::PythonIo {
                action: "spawning verify_reference.py",
                source,
            })?;

        let mut stdin = child.stdin.take().ok_or_else(|| {
            ParityError::PythonProtocol("python subprocess stdin was not piped".to_string())
        })?;
        let request_bytes = serde_json::to_vec(request)?;
        stdin
            .write_all(&request_bytes)
            .and_then(|_| stdin.flush())
            .map_err(|source| ParityError::PythonIo {
                action: "writing subprocess stdin",
                source,
            })?;
        drop(stdin);

        let mut stdout = child.stdout.take().ok_or_else(|| {
            ParityError::PythonProtocol("python subprocess stdout was not piped".to_string())
        })?;
        let mut stderr = child.stderr.take().ok_or_else(|| {
            ParityError::PythonProtocol("python subprocess stderr was not piped".to_string())
        })?;
        let stdout_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            stdout.read_to_end(&mut buf).map(|_| buf)
        });
        let stderr_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            stderr.read_to_end(&mut buf).map(|_| buf)
        });

        let start = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait().map_err(|source| ParityError::PythonIo {
                action: "waiting for subprocess",
                source,
            })? {
                break status;
            }
            if start.elapsed() >= self.timeout {
                let _ = child.kill();
                let _ = child.wait();
                let stderr = join_reader(stderr_handle, "reading subprocess stderr")?;
                let _ = stdout_handle.join();
                return Err(ParityError::PythonSubprocessTimeout {
                    timeout_ms: self.timeout.as_millis() as u64,
                    stderr: String::from_utf8_lossy(&stderr).to_string(),
                });
            }
            std::thread::sleep(Duration::from_millis(10));
        };

        let stdout = join_reader(stdout_handle, "reading subprocess stdout")?;
        let stderr = join_reader(stderr_handle, "reading subprocess stderr")?;
        let stderr_text = String::from_utf8_lossy(&stderr).to_string();
        if !status.success() {
            return Err(ParityError::PythonSubprocessFailed {
                exit_code: status.code(),
                stderr: stderr_text,
            });
        }
        String::from_utf8(stdout)
            .map_err(|e| ParityError::PythonProtocol(format!("stdout was not utf-8: {e}")))
    }
}

impl HFReference for PythonSubprocessReference {
    fn forward_with_hooks(&self, tokens: &[u32]) -> Result<ReferenceOutput, ParityError> {
        let mut out = self.forward_batch_with_hooks(&[tokens.to_vec()])?;
        out.pop()
            .ok_or_else(|| ParityError::PythonProtocol("empty forward response".to_string()))
    }

    fn forward_batch_with_hooks(
        &self,
        prompts: &[Vec<u32>],
    ) -> Result<Vec<ReferenceOutput>, ParityError> {
        let mut outputs: Vec<Option<ReferenceOutput>> = vec![None; prompts.len()];
        let mut missing = Vec::new();
        {
            let cache = self.cache.lock().map_err(|_| {
                ParityError::PythonProtocol("reference cache lock poisoned".to_string())
            })?;
            for (idx, tokens) in prompts.iter().enumerate() {
                if let Some(output) = cache.get(tokens) {
                    outputs[idx] = Some(output.clone());
                } else {
                    missing.push((idx, tokens.clone()));
                }
            }
        }

        if !missing.is_empty() {
            let request = serde_json::json!({
                "prompts": missing
                    .iter()
                    .map(|(_, tokens)| serde_json::json!({ "tokens": tokens }))
                    .collect::<Vec<_>>(),
                "model_path": self.model_path,
                "reference_dtype": self.reference_dtype.as_str(),
            });
            let stdout = self.run_script(&[], &request)?;
            let mut cache = self.cache.lock().map_err(|_| {
                ParityError::PythonProtocol("reference cache lock poisoned".to_string())
            })?;
            for line in stdout.lines().filter(|l| !l.trim().is_empty()) {
                let response: ForwardLine = serde_json::from_str(line).map_err(|e| {
                    ParityError::PythonProtocol(format!(
                        "could not parse forward JSON line {line:?}: {e}"
                    ))
                })?;
                let Some((original_idx, tokens)) = missing.get(response.prompt_idx) else {
                    return Err(ParityError::PythonProtocol(format!(
                        "forward response prompt_idx {} out of range",
                        response.prompt_idx
                    )));
                };
                let output = ReferenceOutput {
                    per_layer_activations: response.per_layer_activations,
                    final_logits: response.final_logits,
                };
                cache.insert(tokens.clone(), output.clone());
                outputs[*original_idx] = Some(output);
            }
        }

        outputs
            .into_iter()
            .enumerate()
            .map(|(idx, output)| {
                output.ok_or_else(|| {
                    ParityError::PythonProtocol(format!(
                        "missing forward response for prompt {idx}"
                    ))
                })
            })
            .collect()
    }

    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, ParityError> {
        let request = serde_json::json!({
            "prompts": [{ "text": prompt }],
            "model_path": self.model_path,
            "reference_dtype": self.reference_dtype.as_str(),
        });
        let stdout = self.run_script(&["--tokenize-only"], &request)?;
        let line = stdout
            .lines()
            .find(|l| !l.trim().is_empty())
            .ok_or_else(|| ParityError::PythonProtocol("empty tokenize response".to_string()))?;
        let response: TokenizeLine = serde_json::from_str(line).map_err(|e| {
            ParityError::PythonProtocol(format!("could not parse tokenize JSON line {line:?}: {e}"))
        })?;
        Ok(response.tokens)
    }
}

#[derive(Debug, Deserialize)]
struct ForwardLine {
    prompt_idx: usize,
    per_layer_activations: Vec<Vec<f32>>,
    final_logits: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct TokenizeLine {
    tokens: Vec<u32>,
}

fn join_reader(
    handle: std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
    action: &'static str,
) -> Result<Vec<u8>, ParityError> {
    handle
        .join()
        .map_err(|_| ParityError::PythonProtocol(format!("{action} thread panicked")))?
        .map_err(|source| ParityError::PythonIo { action, source })
}

//! Real prompt tokenization / detokenization for the serving runtime.
//!
//! Wraps the HuggingFace [`tokenizers`] crate so the server encodes prompts
//! and decodes generated tokens with the model's *actual* tokenizer
//! (`tokenizer.json`), replacing the byte-mod-vocab placeholder. The artifact
//! (or a `--tokenizer` path) supplies `tokenizer.json`; when absent the server
//! falls back to the byte tokenizer with a warning so it still runs.

use std::path::Path;

use crate::error::RuntimeError;

/// A loaded model tokenizer.
pub struct SkeinTokenizer {
    inner: tokenizers::Tokenizer,
}

impl SkeinTokenizer {
    /// Load `tokenizer.json` from a file.
    pub fn from_file(path: &Path) -> Result<Self, RuntimeError> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| RuntimeError::Tokenizer(format!("loading {}: {e}", path.display())))?;
        Ok(Self { inner })
    }

    /// Load `tokenizer.json` from an artifact directory, if present.
    /// Returns `Ok(None)` when the file does not exist (caller falls back).
    pub fn from_artifact_dir(dir: &Path) -> Result<Option<Self>, RuntimeError> {
        let path = dir.join("tokenizer.json");
        if path.exists() {
            Ok(Some(Self::from_file(&path)?))
        } else {
            Ok(None)
        }
    }

    /// Encode `text` into token ids (no special tokens — the serving loop adds
    /// any BOS/EOS handling itself).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, RuntimeError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| RuntimeError::Tokenizer(format!("encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids back into text, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> Result<String, RuntimeError> {
        self.inner
            .decode(ids, true)
            .map_err(|e| RuntimeError::Tokenizer(format!("decode: {e}")))
    }

    /// Vocabulary size (with added tokens), for sanity-checking against the
    /// artifact's `model_meta.vocab`.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::Tokenizer;

    /// Build a tiny whitespace WordLevel tokenizer and serialize it to a
    /// `tokenizer.json`, so the test exercises the real `from_file` + encode +
    /// decode path with no network or model download.
    fn write_tiny_tokenizer(path: &Path) {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, w) in ["[UNK]", "hello", "world", "skein"].iter().enumerate() {
            vocab.insert((*w).to_string(), i as u32);
        }
        let wl = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".to_string())
            .build()
            .expect("wordlevel");
        let mut tk = Tokenizer::new(wl);
        tk.with_pre_tokenizer(Some(Whitespace {}));
        tk.save(path.to_str().unwrap(), false).expect("save tokenizer");
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "skein_tok_{name}_{}.json",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn encode_decode_round_trips() {
        let path = tmp("rt");
        write_tiny_tokenizer(&path);
        let tk = SkeinTokenizer::from_file(&path).expect("load");

        let ids = tk.encode("hello world skein").expect("encode");
        assert_eq!(ids, vec![1, 2, 3]);
        let text = tk.decode(&ids).expect("decode");
        assert_eq!(text, "hello world skein");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unknown_words_map_to_unk() {
        let path = tmp("unk");
        write_tiny_tokenizer(&path);
        let tk = SkeinTokenizer::from_file(&path).expect("load");
        let ids = tk.encode("hello nope").expect("encode");
        assert_eq!(ids, vec![1, 0]); // "hello"=1, unknown -> [UNK]=0
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn from_artifact_dir_absent_is_none() {
        let dir = std::env::temp_dir().join(format!(
            "skein_tok_missing_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(SkeinTokenizer::from_artifact_dir(&dir).expect("ok").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

//! HF bundle tokenization through the upstream read-only tokenizer pipeline.
use super::{Result, unsupported};
use std::{collections::BTreeMap, path::Path};
use tokenizers::pipeline::{EncodeOptions, Override, PipelineTokenizer};

pub(super) struct Tokenizer {
    inner: PipelineTokenizer,
    vocab: BTreeMap<String, u32>,
}

impl Tokenizer {
    pub(super) fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(error)?;
        Self::from_json(&json)
    }

    pub(super) fn from_json(json: &str) -> Result<Self> {
        let canonical = tokenizers::canonicalize_str(json).map_err(error)?;
        let inner = tokenizers::from_json(&canonical).map_err(error)?;
        let config: serde_json::Value = serde_json::from_str(&canonical).map_err(error)?;
        let vocab = config["model"]["vocab"].as_object()
            .ok_or_else(|| unsupported("HybridAgent requires a token-to-ID vocabulary"))?
            .iter()
            .map(|(token, id)| {
                let id = id.as_u64().and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| unsupported("invalid tokenizer vocabulary ID"))?;
                Ok((token.clone(), id))
            })
            .collect::<Result<_>>()?;
        Ok(Self { inner, vocab })
    }

    pub(super) fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.get_added_vocabulary().token_to_id(token)
            .or_else(|| self.vocab.get(token).copied())
    }

    pub(super) fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let options = EncodeOptions {
            padding: Override::Off,
            truncation: Override::Off,
            ..EncodeOptions::no_specials()
        };
        let mut ids = Vec::new();
        self.inner.encode_into(text, &options, &mut ids).map_err(error)?;
        Ok(ids.into_iter().map(u32::from).collect())
    }

    pub(super) fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens, false).map_err(error)
    }
}

fn error(error: impl std::fmt::Display) -> crate::Error {
    unsupported(format!("tokenizer: {error}"))
}

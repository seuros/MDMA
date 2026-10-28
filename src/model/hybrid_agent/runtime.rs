//! Owned HybridAgent sessions.
use super::{Backend, Error, HybridAgent, HybridDecoder, LatentLimits, Result, unsupported};
use std::{path::Path, sync::Arc};
use super::tokenizer::Tokenizer;

mod session;
pub use session::{Session, Step};

/// Loaded weights and tokenizer, shared by independently owned sessions.
///
/// Loading and inference are blocking.
pub struct Engine<B: Backend> {
    agent: Arc<HybridAgent<B>>,
    tokenizer: Arc<Tokenizer>,
    controls: Controls,
    backend: Arc<B>,
}

impl<B: Backend + 'static> Engine<B> {
    pub fn load(backend: B, directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let tokenizer = Tokenizer::from_file(directory.join("tokenizer.json"))
            .map_err(tokenizer_error)?;
        let controls = Controls::load(&tokenizer)?;
        let agent = HybridAgent::load_bundle(&backend, directory)?;
        if controls.ids().any(|id| id as usize >= agent.model.token_embd.shape[0])
            || controls.entry != agent.entry_token() || controls.exit != agent.exit_token()
        {
            return Err(unsupported("generation controls do not match the backbone"));
        }
        Ok(Self {
            agent: Arc::new(agent), tokenizer: Arc::new(tokenizer),
            controls, backend: Arc::new(backend),
        })
    }

    /// Create a session with private KV/scratch that can outlive this engine.
    pub fn session(&self, context: usize, limits: LatentLimits) -> Result<Session<B>> {
        let decoder = HybridDecoder::shared(self.agent.clone(), self.backend.clone(), context, limits)?;
        Ok(Session::new(decoder, self.tokenizer.clone(), self.controls))
    }

    /// Encode raw checkpoint-formatted text, without rendering a chat template.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        encode(&self.tokenizer, text)
    }

    /// Decode complete token sequences; individual tokens can split UTF-8.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        decode(&self.tokenizer, tokens)
    }

    pub fn config(&self) -> &crate::model::ModelConfig { self.agent.config() }
    pub fn device_label(&self) -> Option<String> { self.backend.device_label() }
    pub fn entry_token(&self) -> u32 { self.controls.entry }
}

fn tokenizer_error(error: impl std::fmt::Display) -> Error {
    unsupported(format!("tokenizer: {error}"))
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer.encode(text)
}

fn decode(tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String> {
    tokenizer.decode(tokens)
}

#[derive(Clone, Copy)]
struct Controls {
    think: u32,
    think_end: u32,
    entry: u32,
    exit: u32,
    eos: u32,
    im_end: Option<u32>,
}

impl Controls {
    fn load(tokenizer: &Tokenizer) -> Result<Self> {
        let id = |token| tokenizer.token_to_id(token)
            .ok_or_else(|| unsupported(format!("missing generation control {token}")));
        let controls = Self {
            think: id("<think>")?, think_end: id("</think>")?,
            entry: id("<latent>")?, exit: id("</latent>")?,
            eos: id("<|endoftext|>")?, im_end: tokenizer.token_to_id("<|im_end|>"),
        };
        let ids: Vec<_> = controls.ids().collect();
        for (i, id) in ids.iter().enumerate() {
            if ids[..i].contains(id) { return Err(unsupported("generation controls must differ")); }
        }
        Ok(controls)
    }

    fn ids(self) -> impl Iterator<Item = u32> {
        [self.think, self.think_end, self.entry, self.exit, self.eos].into_iter().chain(self.im_end)
    }

    fn allows(self, token: u32, in_think: bool) -> bool {
        token != self.exit && token != if in_think { self.think } else { self.think_end }
            && (in_think || token != self.entry)
    }

    fn is_eos(self, token: u32) -> bool { token == self.eos || Some(token) == self.im_end }
}

#[cfg(test)]
mod tests;

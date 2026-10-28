use super::{Backend, Controls, HybridDecoder, Result, Tokenizer, decode, encode, unsupported};
use crate::model::hybrid_agent::LatentStep;
use std::sync::Arc;

/// Exactly one generated context position; continuous thoughts have no token ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Token { token_id: u32, end_of_turn: bool },
    Latent(LatentStep),
}

/// An owned, resumable trajectory with caller-controlled budgets and stopping.
pub struct Session<B: Backend + 'static> {
    decoder: HybridDecoder<'static, B>,
    tokenizer: Arc<Tokenizer>,
    controls: Controls,
    in_think: bool,
    ended: bool,
}

impl<B: Backend + 'static> Session<B> {
    pub(super) fn new(decoder: HybridDecoder<'static, B>, tokenizer: Arc<Tokenizer>, controls: Controls) -> Self {
        Self { decoder, tokenizer, controls, in_think: false, ended: false }
    }

    /// Append raw text without chat formatting or opening latent spans.
    pub fn append(&mut self, text: &str) -> Result<()> {
        let tokens = encode(&self.tokenizer, text)?;
        self.append_tokens(&tokens)
    }

    pub fn append_tokens(&mut self, tokens: &[u32]) -> Result<()> {
        self.decoder.feed_text(tokens)?;
        for &token in tokens { self.track_think(token); }
        self.ended = false;
        Ok(())
    }

    /// Greedy masked text sampling, or one continuous latent/exit step.
    ///
    /// Returns `None` after EOS until a successful append. Each `Some` consumes
    /// one position, including EOS and latent exits. Context exhaustion is an error.
    pub fn step(&mut self) -> Result<Option<Step>> {
        self.decoder.healthy()?;
        if self.ended { return Ok(None); }
        if self.decoder.in_latent_mode() {
            return Ok(Some(Step::Latent(self.decoder.latent_step()?)));
        }
        let logits = self.decoder.logits()?;
        let token = greedy(&logits, self.controls, self.in_think)?;
        self.token(token).map(Some)
    }

    /// Current logits for a caller-owned sampler. Do not sample tokens while
    /// `in_latent_mode()` is true; use `step()` for continuous reasoning.
    pub fn logits(&self) -> Result<Vec<f32>> { self.decoder.logits() }

    /// Consume a caller-sampled token, applying the same control masks as `step`.
    pub fn token(&mut self, token_id: u32) -> Result<Step> {
        if self.ended || !self.controls.allows(token_id, self.in_think) {
            return Err(unsupported("sampled token violates generation controls"));
        }
        self.decoder.token(token_id)?;
        self.track_think(token_id);
        self.ended = self.controls.is_eos(token_id);
        Ok(Step::Token { token_id, end_of_turn: self.ended })
    }

    pub fn position(&self) -> usize { self.decoder.position() }
    pub fn in_latent_mode(&self) -> bool { self.decoder.in_latent_mode() }
    pub fn is_end_of_turn(&self) -> bool { self.ended }
    pub fn decode(&self, tokens: &[u32]) -> Result<String> { decode(&self.tokenizer, tokens) }
    pub fn finish(&mut self) -> Result<()> { self.decoder.finish() }

    fn track_think(&mut self, token: u32) {
        if token == self.controls.think { self.in_think = true; }
        if token == self.controls.think_end { self.in_think = false; }
    }
}

fn greedy(logits: &[f32], controls: Controls, in_think: bool) -> Result<u32> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err(unsupported("empty or nonfinite generation logits"));
    }
    let mut best = None;
    for (i, &value) in logits.iter().enumerate() {
        let token = u32::try_from(i).map_err(|_| crate::Error::SizeOverflow)?;
        if !controls.allows(token, in_think) { continue; }
        if best.is_none_or(|(_, maximum)| value > maximum) { best = Some((token, value)); }
    }
    best.map(|(token, _)| token).ok_or_else(|| unsupported("no admissible text token"))
}

#[cfg(test)]
mod tests;

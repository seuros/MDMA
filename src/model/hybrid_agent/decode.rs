use super::{Backend, Error, HybridAgent, Result, link, unsupported};
use crate::{backend::Pool, kv_cache::KvCache, model::{Workspace, arch::Decoder}};
use std::{ops::Deref, sync::Arc};

enum Handle<'a, T> {
    Borrowed(&'a T),
    Shared(Arc<T>),
}

impl<T> Deref for Handle<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        match self {
            Self::Borrowed(value) => value,
            Self::Shared(value) => value,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LatentLimits {
    pub min_steps: usize,
    pub max_steps: usize,
}

impl Default for LatentLimits {
    fn default() -> Self { Self { min_steps: 4, max_steps: 200 } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason { Anchor, MaxSteps }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentStep {
    Thought { steps: usize },
    Exit { token: u32, steps: usize, reason: ExitReason },
}

/// A trajectory with backend-resident hidden states, continuous inputs and KV.
pub struct HybridDecoder<'a, B: Backend> {
    kv: KvCache<B>,
    ws: Workspace<B>,
    link_ws: link::Workspace<B>,
    index: B::Buf,
    limits: LatentLimits,
    latent_steps: Option<usize>,
    poisoned: bool,
    // Device buffers must drop before their model/backend owners.
    agent: Handle<'a, HybridAgent<B>>,
    backend: Handle<'a, B>,
}

impl<'a, B: Backend> HybridDecoder<'a, B> {
    pub(super) fn new(agent: &'a HybridAgent<B>, backend: &'a B, context: usize, limits: LatentLimits) -> Result<Self> {
        Self::with_handles(Handle::Borrowed(agent), Handle::Borrowed(backend), context, limits)
    }

    pub(super) fn shared(agent: Arc<HybridAgent<B>>, backend: Arc<B>, context: usize, limits: LatentLimits) -> Result<Self> {
        Self::with_handles(Handle::Shared(agent), Handle::Shared(backend), context, limits)
    }

    fn with_handles(agent: Handle<'a, HybridAgent<B>>, backend: Handle<'a, B>, context: usize, limits: LatentLimits) -> Result<Self> {
        if context == 0 || context > agent.config().n_ctx || context > u32::MAX as usize
            || limits.max_steps == 0 || limits.min_steps > limits.max_steps
        {
            return Err(unsupported("invalid context or latent-step limits"));
        }
        Ok(Self {
            kv: agent.model.new_kv_cache(&backend, context)?,
            ws: agent.model.alloc_workspace(&backend, 1)?,
            link_ws: agent.link.workspace(&backend)?,
            index: backend.alloc(size_of::<u32>(), Pool::Scratch)?,
            limits, latent_steps: None, poisoned: false,
            agent, backend,
        })
    }

    pub fn position(&self) -> usize { self.kv.len() }
    pub fn in_latent_mode(&self) -> bool { self.latent_steps.is_some() }

    pub(super) fn healthy(&self) -> Result<()> {
        if self.poisoned { return Err(Error::Backend("HybridAgent trajectory invalid after backend failure".into())); }
        Ok(())
    }

    fn room(&self, count: usize) -> Result<()> {
        self.healthy()?;
        if count > self.kv.capacity().saturating_sub(self.kv.len()) {
            return Err(Error::ContextLimit { prompt_tokens: self.kv.len().saturating_add(count), n_ctx: self.kv.capacity() });
        }
        Ok(())
    }

    fn validate_tokens(&self, tokens: &[u32]) -> Result<()> {
        self.room(tokens.len())?;
        if tokens.is_empty() || tokens.iter().any(|&t| t as usize >= self.agent.model.token_embd.shape[0]) {
            return Err(Error::ShapeMismatch("HybridAgent requires nonempty, in-vocabulary tokens".into()));
        }
        Ok(())
    }

    fn forward_token(&mut self, token: u32) -> Result<()> {
        let pos = self.kv_position()?;
        self.poisoned = true;
        self.agent.model.forward_device(&self.backend, &mut self.kv, &mut self.ws, token, pos)?;
        self.poisoned = false;
        Ok(())
    }

    fn kv_position(&self) -> Result<u32> {
        u32::try_from(self.kv.len()).map_err(|_| Error::SizeOverflow)
    }

    /// Append external tokens without treating anchors as mode transitions.
    pub fn feed_text(&mut self, tokens: &[u32]) -> Result<()> {
        self.validate_tokens(tokens)?;
        if self.in_latent_mode() { return Err(unsupported("cannot insert external text inside a latent span")); }
        for &token in tokens { self.forward_token(token)?; }
        Ok(())
    }

    /// Consume a sampled token. The caller supplies masks and EOS policy;
    /// only `latent_step` may inject the exit anchor.
    pub fn token(&mut self, token: u32) -> Result<()> {
        self.validate_tokens(&[token])?;
        if self.in_latent_mode() || token == self.agent.exit {
            return Err(unsupported("text token is not valid in the current generation mode"));
        }
        self.forward_token(token)?;
        if token == self.agent.entry { self.latent_steps = Some(0); }
        Ok(())
    }

    pub fn logits(&self) -> Result<Vec<f32>> {
        self.healthy()?;
        if self.kv.len() == 0 { return Err(unsupported("no logits before the first input")); }
        self.agent.model.download_logits(&self.backend, &self.ws)
    }

    /// Join queued device work without downloading the vocabulary-sized output.
    pub fn finish(&mut self) -> Result<()> {
        self.healthy()?;
        if self.kv.len() == 0 { return Err(unsupported("no work before the first input")); }
        self.poisoned = true;
        self.backend.download(&self.ws.logits, 0, &mut [0u8; 4])?;
        self.poisoned = false;
        Ok(())
    }

    /// Decide exit from the *current* unmasked LM logits, before feeding the
    /// next continuous vector. Only a single u32 is read back in latent mode.
    pub fn latent_step(&mut self) -> Result<LatentStep> {
        self.room(1)?;
        let steps = self.latent_steps.ok_or_else(|| unsupported("no open latent span"))?;
        self.poisoned = true;
        self.backend.argmax(&self.ws.logits, &mut self.index)?;
        let mut raw = [0u8; 4];
        self.backend.download(&self.index, 0, &mut raw)?;
        let readout = u32::from_ne_bytes(raw);
        if readout as usize >= self.agent.model.token_embd.shape[0] {
            return Err(Error::Backend("HybridAgent received nonfinite logits".into()));
        }
        let reason = if steps >= self.limits.max_steps { Some(ExitReason::MaxSteps) }
            else if steps >= self.limits.min_steps && readout == self.agent.exit { Some(ExitReason::Anchor) }
            else { None };
        if let Some(reason) = reason {
            self.forward_token(self.agent.exit)?;
            self.latent_steps = None;
            return Ok(LatentStep::Exit { token: self.agent.exit, steps, reason });
        }
        self.agent.link.apply(&self.backend, &self.ws.normed, &mut self.link_ws, &mut self.ws.x)?;
        let Decoder::Qwen(qwen) = &self.agent.model.architecture else {
            return Err(unsupported("continuous inputs require Qwen"));
        };
        let pos = self.kv_position()?;
        qwen.run_embedded(&self.backend, &self.agent.model, &mut self.kv, &mut self.ws, pos)?;
        self.backend.qmatmul(self.agent.model.output_weight(), &self.ws.normed, &mut self.ws.logits)?;
        self.poisoned = false;
        self.latent_steps = Some(steps + 1);
        Ok(LatentStep::Thought { steps: steps + 1 })
    }
}

#[cfg(test)]
mod tests;

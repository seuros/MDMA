use super::{Decoder, Model, ModelConfig};
use crate::backend::{Backend, Error, Result};
use crate::gguf::GgufModel;
use crate::kv_cache::KvCache;
use crate::memory::{NativeMemoryBudget, Reservation, product, sum};
#[cfg(feature = "service")]
use crate::memory::MemoryCategory;
#[cfg(feature = "service")]
use std::sync::Arc;

#[cfg(feature = "service")]
#[derive(Default)]
pub(crate) struct RequestMemory {
    pub kv: Option<Reservation>,
    // Prefill drops its workspace before decode allocates one. Their shared
    // reservation stays live through the worker, including graph teardown.
    pub scratch: Option<Arc<Reservation>>,
}

pub(crate) fn weight_bytes(gguf: &GgufModel) -> Result<usize> {
    let bytes = sum(gguf.tensor_names().map(|name| gguf.weight_bytes(name)))?;
    if gguf.metadata_str("general.architecture") == Some("gemma4") {
        let (config, gemma) = super::gemma4::Config::read(gguf)?;
        return sum([bytes, gemma.generated_weight_bytes(&config)?]);
    }
    if gguf.metadata_str("general.architecture") == Some(super::qwen35::ARCH) {
        // Multi-token prediction blocks after the trunk are never loaded.
        let (config, _) = super::qwen35::Config::read(gguf)?;
        return sum(gguf
            .tensor_names()
            .filter(|&name| super::planning::layer_index(name).is_none_or(|l| l < config.n_layers))
            .map(|name| gguf.weight_bytes(name)));
    }
    if let Some(arch) = gguf.metadata_str("general.architecture")
        && super::qwen::is_arch(arch)
    {
        let config = super::qwen::preflight_config(gguf, arch)?;
        return sum([bytes, product(&[config.head_dim / 2, 4])?]);
    }
    Ok(bytes)
}

pub(super) fn workspace_sizes(
    config: &ModelConfig,
    vocab: usize,
    tokens: usize,
) -> Result<[usize; 13]> {
    let e = product(&[config.n_embd, tokens, 4])?;
    let q = product(&[config.n_heads, config.head_dim, tokens, 4])?;
    let k = product(&[config.n_kv_heads, config.head_dim, tokens, 4])?;
    let f = product(&[config.n_ff, tokens, 4])?;
    Ok([
        e,
        e,
        e,
        q,
        k,
        k,
        q,
        e,
        f,
        f,
        f,
        product(&[config.n_embd, 4])?,
        product(&[vocab, 4])?,
    ])
}

impl<B: Backend> Model<B> {
    pub(super) fn common_workspace_sizes(&self, tokens: usize) -> Result<[usize; 13]> {
        let mut sizes = workspace_sizes(&self.config, self.output_weight().shape[0], tokens)?;
        if matches!(&self.architecture, Decoder::Gemma4(_)) {
            // Gemma4 owns its variable-width attention buffers separately.
            sizes[3] = sizes[0];
            sizes[6] = sizes[0];
        }
        Ok(sizes)
    }

    /// The budget this model was loaded under, if any.
    pub fn memory_budget(&self) -> Option<&NativeMemoryBudget> {
        self.memory.as_ref().map(|memory| &memory.budget)
    }

    /// Weight bytes this model holds in its budget, if it was loaded under one.
    pub fn weight_reservation_bytes(&self) -> Option<usize> {
        self.memory.as_ref().map(Reservation::bytes)
    }

    pub fn kv_bytes(&self, backend: &B, capacity: usize) -> Result<usize> {
        KvCache::<B>::allocation_bytes(
            backend,
            &self.kv_widths()?,
            &self.kv_windows(),
            &self.kv_states()?,
            capacity.min(self.config.n_ctx),
        )
    }

    /// Sliding window of each KV layer, 0 where it keeps every position.
    pub(crate) fn kv_windows(&self) -> Vec<usize> {
        match &self.architecture {
            Decoder::Gemma4(gemma) => gemma.config.kv_windows(),
            Decoder::Qwen35(qwen) => qwen
                .config
                .recurrent
                .iter()
                .filter(|&&r| !r)
                .map(|_| 0)
                .collect(),
            _ => vec![0; self.config.n_layers],
        }
    }

    /// Bytes of each recurrent state buffer, which no position count changes.
    pub(crate) fn kv_states(&self) -> Result<Vec<usize>> {
        match &self.architecture {
            Decoder::Qwen35(qwen) => qwen.config.state_bytes(),
            _ => Ok(Vec::new()),
        }
    }

    /// K (and V) elements per position, per layer.
    pub(crate) fn kv_widths(&self) -> Result<Vec<usize>> {
        match &self.architecture {
            Decoder::Gemma4(gemma) => return gemma.config.kv_widths(&self.config),
            Decoder::Qwen35(qwen) => return qwen.config.kv_widths(&self.config),
            _ => {}
        }
        Ok(vec![
            product(&[
                self.config.n_kv_heads,
                self.config.head_dim
            ])?;
            self.config.n_layers
        ])
    }

    pub fn workspace_bytes(&self, tokens: usize) -> Result<usize> {
        if tokens == 0 || tokens > self.config.n_ctx {
            return Err(Error::ShapeMismatch(
                "workspace token count outside model context".into(),
            ));
        }
        let batch = if self.sequential_prefill() { 1 } else { tokens };
        let bytes = sum(self.common_workspace_sizes(batch)?)?;
        match &self.architecture {
            Decoder::Gemma4(gemma) => sum([bytes, gemma.config.workspace_bytes(&self.config)?]),
            Decoder::Llama(_) => Ok(bytes),
            Decoder::Qwen(_) => sum([
                bytes,
                super::qwen::Qwen::<B>::workspace_bytes(&self.config)?,
            ]),
            Decoder::Qwen35(qwen) => sum([bytes, qwen.config.workspace_bytes(&self.config)?]),
            Decoder::Glm4(_) => sum([
                bytes,
                super::glm4::Glm4::<B>::workspace_bytes(&self.config)?,
            ]),
        }
    }

    #[cfg(feature = "service")]
    pub(crate) fn reserve_request(
        &self,
        backend: &B,
        new_kv: Option<usize>,
        prompt_tokens: usize,
    ) -> Result<RequestMemory> {
        let Some(budget) = self.memory_budget() else {
            return Ok(RequestMemory::default());
        };
        let kv = new_kv
            .map(|capacity| self.kv_bytes(backend, capacity))
            .transpose()?
            .unwrap_or(0);
        // Conservative even on a prefix hit; prompt_tokens >= 1 also covers decode.
        let scratch = self.workspace_bytes(prompt_tokens)?;
        let mut memory = budget.reserve(sum([kv, scratch])?, MemoryCategory::Workspace)?;
        let kv = new_kv.map(|_| memory.split_as(kv, MemoryCategory::Kv));
        Ok(RequestMemory {
            kv,
            scratch: Some(Arc::new(memory)),
        })
    }
}

#[cfg(test)]
mod tests;

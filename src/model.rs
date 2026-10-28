//! Decoder dispatch, reusable workspaces, and shared model loading.
use crate::backend::{Backend, Error, Pool, QWeight, Result};
use crate::gguf::{GgmlDType, GgufModel};
use crate::kv_cache::KvCache;
use crate::memory::{MemoryCategory, NativeMemoryBudget, Reservation};
use std::sync::Arc;

pub mod arch;
pub mod decoder;
pub mod hybrid;
pub mod hybrid_agent;
pub(crate) mod layers;
pub mod planning;
pub mod weights;
pub(crate) use arch::llama::Matmuls as LlamaMatmuls;
use arch::{Decoder, gemma4, glm4, llama, qwen, qwen35};
pub(crate) mod memory;
mod preflight;

pub struct ModelConfig {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub n_ctx: usize,
}

impl ModelConfig {
    /// Read uniform Llama-style geometry with the given metadata prefix.
    pub fn from_gguf(gguf: &GgufModel, arch: &str) -> Option<Self> {
        llama::config_from_gguf(gguf, arch)
    }
}

/// Intermediate buffers reused across layers and tokens.
/// Residual adds alternate between `x` and `x_alt` to avoid aliasing inputs
/// and outputs, without allocating or copying.
pub struct Workspace<B: Backend> {
    n_tokens: usize,
    x: B::Buf,
    x_alt: B::Buf,
    normed: B::Buf,
    q: B::Buf,
    k: B::Buf,
    v: B::Buf,
    attn_out: B::Buf,
    proj: B::Buf,
    gate: B::Buf,
    up: B::Buf,
    act: B::Buf,
    /// Final prefill row, copied from `normed` for the output projection.
    last_row: B::Buf,
    logits: B::Buf,
    gemma: Option<gemma4::Workspace<B>>,
    qwen: Option<qwen::Workspace<B>>,
    qwen35: Option<qwen35::Workspace<B>>,
    glm: Option<glm4::Workspace<B>>,
    _memory: Option<Arc<Reservation>>,
}

pub struct Model<B: Backend> {
    config: ModelConfig,
    token_embd: QWeight<B::Buf>,
    output_norm: B::Buf,
    output: Option<QWeight<B::Buf>>,
    architecture: Decoder<B>,
    /// `tokenizer.ggml.bos_token_id`, when the file names one.
    bos: Option<u32>,
    // Last: refund only after all weight buffers have been dropped.
    memory: Option<Reservation>,
}

fn load_weight<B: Backend>(
    backend: &B,
    gguf: &mut GgufModel,
    name: &str,
) -> Result<QWeight<B::Buf>> {
    let desc = gguf
        .tensor_desc(name)
        .ok_or_else(|| Error::ShapeMismatch(format!("missing tensor {name}")))?
        .clone();
    let (raw, layout) = gguf
        .read_weight_bytes(name)
        .map_err(|e| Error::ShapeMismatch(e.to_string()))?;
    let buf = backend.upload_weight(&raw, layout)?;
    Ok(QWeight {
        dtype: desc.dtype,
        shape: desc.shape,
        buf,
    })
}

fn load_norm<B: Backend>(backend: &B, gguf: &mut GgufModel, name: &str) -> Result<B::Buf> {
    let desc = gguf
        .tensor_desc(name)
        .ok_or_else(|| Error::ShapeMismatch(format!("missing tensor {name}")))?
        .clone();
    if desc.dtype != GgmlDType::F32 {
        return Err(Error::ShapeMismatch(format!(
            "{name}: expected F32 norm weight, got {:?}",
            desc.dtype
        )));
    }
    let raw = gguf
        .read_tensor_bytes(name)
        .map_err(|e| Error::ShapeMismatch(e.to_string()))?;
    let mut buf = backend.alloc(raw.len(), Pool::Weights)?;
    backend.upload(&mut buf, 0, &raw)?;
    Ok(buf)
}

/// `output.weight`, or `None` when the output projection is tied to the embedding.
fn load_output<B: Backend>(backend: &B, gguf: &mut GgufModel) -> Result<Option<QWeight<B::Buf>>> {
    gguf.tensor_desc("output.weight")
        .is_some()
        .then(|| load_weight(backend, gguf, "output.weight"))
        .transpose()
}

impl<B: Backend> Model<B> {
    /// Checks native decoder support using metadata and tensor descriptors only.
    /// Does not read weight payloads or allocate/upload backend buffers.
    pub fn validate(backend: &B, gguf: &GgufModel) -> Result<ModelConfig> {
        preflight::validate(backend, gguf)
    }

    pub fn load(backend: &B, gguf: &mut GgufModel) -> Result<Self> {
        Self::prepare_source(backend, gguf)?;
        Self::load_reserved(backend, gguf, None)
    }

    pub(crate) fn prepare_source(backend: &B, gguf: &mut GgufModel) -> Result<()> {
        Self::validate(backend, gguf)?;
        crate::cache::attach_from_env(backend, gguf)
    }

    /// Reserve native weight bytes before materializing/uploading tensors (after cache lookup).
    pub fn load_with_budget(
        backend: &B,
        gguf: &mut GgufModel,
        budget: &NativeMemoryBudget,
    ) -> Result<Self> {
        Self::prepare_source(backend, gguf)?;
        let memory = budget.reserve(memory::weight_bytes(gguf)?, MemoryCategory::Weights)?;
        Self::load_reserved(backend, gguf, Some(memory))
    }

    pub(crate) fn load_reserved(
        backend: &B,
        gguf: &mut GgufModel,
        memory: Option<Reservation>,
    ) -> Result<Self> {
        let config = Self::validate(backend, gguf)?;
        let mut model = match gguf.metadata_str("general.architecture") {
            Some("llama" | llama::ARCH) => llama::Llama::load(backend, gguf, config)?,
            Some("gemma4") => gemma4::Gemma::load(backend, gguf)?,
            Some("glm4") => glm4::Glm4::load(backend, gguf, config)?,
            Some(arch) if qwen::is_arch(arch) => qwen::Qwen::load(backend, gguf, config)?,
            Some(qwen35::ARCH) => qwen35::Qwen35::load(backend, gguf)?,
            _ => {
                return Err(Error::UnsupportedModel(
                    "unvalidated decoder architecture".into(),
                ));
            }
        };
        model.memory = memory;
        model.bos = gguf.metadata_u32("tokenizer.ggml.bos_token_id");
        gguf.check_cache_source()?;
        Ok(model)
    }

    fn output_weight(&self) -> &QWeight<B::Buf> {
        self.output.as_ref().unwrap_or(&self.token_embd)
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    #[cfg_attr(
        not(all(feature = "coral", any(target_os = "freebsd", target_os = "linux"))),
        expect(dead_code, reason = "read by the Coral backend")
    )]
    pub(crate) fn bos_token(&self) -> Option<u32> {
        self.bos
    }

    #[cfg_attr(
        not(all(feature = "coral", any(target_os = "freebsd", target_os = "linux"))),
        expect(dead_code, reason = "read by the Coral backend")
    )]
    /// Every layer's (attention norm, FFN norm) weights, for a Llama-architecture model.
    pub(crate) fn llama_norms(&self) -> Option<Vec<(&B::Buf, &B::Buf)>> {
        match &self.architecture {
            Decoder::Llama(llama) => Some(llama.layers.iter().map(llama::Layer::norms).collect()),
            _ => None,
        }
    }

    #[cfg_attr(
        not(all(feature = "coral", any(target_os = "freebsd", target_os = "linux"))),
        expect(dead_code, reason = "read by the Coral backend")
    )]
    /// The embedding table, the final norm and the classifier.
    pub(crate) fn head(&self) -> (&QWeight<B::Buf>, &B::Buf, &QWeight<B::Buf>) {
        (&self.token_embd, &self.output_norm, self.output_weight())
    }

    /// Every layer's matmul weights, for a Llama-architecture model.
    #[cfg_attr(
        not(all(feature = "coral", any(target_os = "freebsd", target_os = "linux"))),
        expect(dead_code, reason = "read by the Coral backend")
    )]
    pub(crate) fn llama_matmuls(&self) -> Option<Vec<LlamaMatmuls<'_, B::Buf>>> {
        match &self.architecture {
            Decoder::Llama(llama) => Some(llama.layers.iter().map(llama::Layer::matmuls).collect()),
            _ => None,
        }
    }

    fn sequential_prefill(&self) -> bool {
        !matches!(&self.architecture, Decoder::Llama(_))
    }

    #[cfg(feature = "rocm")]
    pub(crate) fn supports_graph_decode(&self) -> bool {
        matches!(&self.architecture, Decoder::Llama(_))
    }

    /// Allocate up to `n_ctx` positions, capped at the model's context length.
    pub fn new_kv_cache(&self, backend: &B, n_ctx: usize) -> Result<KvCache<B>> {
        self.new_kv_cache_reserved(backend, n_ctx, None)
    }

    pub(crate) fn new_kv_cache_reserved(
        &self,
        backend: &B,
        n_ctx: usize,
        memory: Option<Reservation>,
    ) -> Result<KvCache<B>> {
        let n_ctx = n_ctx.min(self.config.n_ctx);
        let memory = match (memory, self.memory_budget()) {
            (Some(memory), Some(budget))
                if memory.covers(budget, self.kv_bytes(backend, n_ctx)?) =>
            {
                Some(memory)
            }
            (None, Some(budget)) => {
                Some(budget.reserve(self.kv_bytes(backend, n_ctx)?, MemoryCategory::Kv)?)
            }
            (None, None) => None,
            _ => return Err(Error::ShapeMismatch("invalid KV reservation".into())),
        };
        KvCache::with_geometry_reserved(
            backend,
            &self.kv_widths()?,
            &self.kv_windows(),
            &self.kv_states()?,
            n_ctx,
            memory,
        )
    }

    /// Allocate reusable scratch for exactly `n_tokens` positions.
    /// Decode requires one position; prefill requires `tokens.len()`.
    /// Buffer lengths determine batch geometry, so these sizes are not interchangeable.
    pub fn alloc_workspace(&self, backend: &B, n_tokens: usize) -> Result<Workspace<B>> {
        self.alloc_workspace_reserved(backend, n_tokens, None)
    }

    pub(crate) fn alloc_workspace_reserved(
        &self,
        backend: &B,
        n_tokens: usize,
        memory: Option<&Arc<Reservation>>,
    ) -> Result<Workspace<B>> {
        let bytes = self.workspace_bytes(n_tokens)?;
        let batch = if self.sequential_prefill() {
            1
        } else {
            n_tokens
        };
        let sizes = self.common_workspace_sizes(batch)?;
        let memory = match (memory, self.memory_budget()) {
            (Some(memory), Some(budget)) if memory.covers(budget, bytes) => Some(memory.clone()),
            (None, Some(budget)) => {
                Some(Arc::new(budget.reserve(bytes, MemoryCategory::Workspace)?))
            }
            (None, None) => None,
            _ => return Err(Error::ShapeMismatch("invalid workspace reservation".into())),
        };
        let mut workspace = Workspace::basic(backend, n_tokens, sizes)?;
        workspace.gemma = match &self.architecture {
            Decoder::Gemma4(g) => Some(g.workspace(backend, &self.config)?),
            _ => None,
        };
        workspace.qwen = match &self.architecture {
            Decoder::Qwen(q) => Some(q.workspace(backend, &self.config)?),
            _ => None,
        };
        workspace.qwen35 = match &self.architecture {
            Decoder::Qwen35(q) => Some(q.workspace(backend, &self.config)?),
            _ => None,
        };
        workspace.glm = match &self.architecture {
            Decoder::Glm4(g) => Some(g.workspace(backend, &self.config)?),
            _ => None,
        };
        #[expect(
            clippy::used_underscore_binding,
            reason = "the reservation is held for its Drop"
        )]
        {
            workspace._memory = memory;
        }
        Ok(workspace)
    }

    /// One decode step for `token_id` at position `pos`, updating `kv` in
    /// place. Returns vocab-sized logits for the next token. Every
    /// intermediate buffer comes from `ws` - this never calls
    /// `backend.alloc()` itself.
    pub fn forward(
        &self,
        backend: &B,
        kv: &mut KvCache<B>,
        ws: &mut Workspace<B>,
        token_id: u32,
        pos: u32,
    ) -> Result<Vec<f32>> {
        self.forward_device(backend, kv, ws, token_id, pos)?;
        self.download_logits(backend, ws)
    }

    pub(crate) fn validate_decode(
        &self,
        kv: &KvCache<B>,
        ws: &Workspace<B>,
        token: u32,
        pos: u32,
    ) -> Result<()> {
        if ws.n_tokens != 1
            || token as usize >= self.token_embd.shape[0]
            || pos as usize >= kv.capacity()
            || !kv.can_resume(pos as usize)
        {
            return Err(Error::ShapeMismatch("decode requires a T=1 workspace, valid token and resumable KV prefix within capacity".into()));
        }
        Ok(())
    }

    pub(crate) fn forward_device(
        &self,
        backend: &B,
        kv: &mut KvCache<B>,
        ws: &mut Workspace<B>,
        token_id: u32,
        pos: u32,
    ) -> Result<()> {
        self.validate_decode(kv, ws, token_id, pos)?;
        match &self.architecture {
            Decoder::Gemma4(gemma) => {
                gemma.run(backend, self, kv, ws, token_id, pos)?;
                backend.qmatmul(self.output_weight(), &ws.normed, &mut ws.logits)?;
                backend.softcap(&mut ws.logits, gemma.config.softcap)
            }
            Decoder::Llama(llama) => {
                llama.run(backend, self, kv, ws, &[token_id], pos)?;
                backend.qmatmul(self.output_weight(), &ws.normed, &mut ws.logits)
            }
            Decoder::Qwen(qwen) => {
                qwen.run(backend, self, kv, ws, token_id, pos)?;
                backend.qmatmul(self.output_weight(), &ws.normed, &mut ws.logits)
            }
            Decoder::Qwen35(qwen) => {
                qwen.run(backend, self, kv, ws, token_id, pos)?;
                backend.qmatmul(self.output_weight(), &ws.normed, &mut ws.logits)
            }
            Decoder::Glm4(glm) => {
                glm.run(backend, self, kv, ws, token_id, pos)?;
                backend.qmatmul(self.output_weight(), &ws.normed, &mut ws.logits)
            }
        }
    }

    pub(crate) fn download_logits(&self, backend: &B, ws: &Workspace<B>) -> Result<Vec<f32>> {
        let mut logits = vec![0f32; self.output_weight().shape[0]];
        backend.download(&ws.logits, 0, bytemuck::cast_slice_mut(&mut logits))?;
        Ok(logits)
    }

    /// Process a nonempty batch at `pos`, updating every layer's KV rows.
    /// `ws` must be sized for `tokens.len()` and the preceding KV prefix initialized.
    /// Returns only the last position's logits.
    pub fn prefill(
        &self,
        backend: &B,
        kv: &mut KvCache<B>,
        ws: &mut Workspace<B>,
        tokens: &[u32],
        pos: u32,
    ) -> Result<Vec<f32>> {
        let n_tokens = tokens.len();
        if n_tokens == 0 {
            return Err(Error::ShapeMismatch(
                "prefill: tokens must be non-empty".into(),
            ));
        }
        if ws.n_tokens != n_tokens
            || !kv.can_resume(pos as usize)
            || n_tokens > kv.capacity().saturating_sub(pos as usize)
        {
            return Err(Error::ShapeMismatch(
                "prefill requires a matching workspace and resumable KV prefix within capacity"
                    .into(),
            ));
        }
        if tokens
            .iter()
            .any(|&id| id as usize >= self.token_embd.shape[0])
        {
            return Err(Error::ShapeMismatch(
                "prefill token outside vocabulary".into(),
            ));
        }
        match &self.architecture {
            Decoder::Gemma4(gemma) => {
                for (i, &token) in tokens.iter().enumerate() {
                    gemma.run(backend, self, kv, ws, token, pos + i as u32)?;
                }
                backend.qmatmul(self.output_weight(), &ws.normed, &mut ws.logits)?;
                backend.softcap(&mut ws.logits, gemma.config.softcap)?;
                return self.download_logits(backend, ws);
            }
            Decoder::Llama(llama) => llama.run(backend, self, kv, ws, tokens, pos)?,
            Decoder::Qwen(qwen) => {
                for (i, &token) in tokens.iter().enumerate() {
                    qwen.run(backend, self, kv, ws, token, pos + i as u32)?;
                }
                backend.qmatmul(self.output_weight(), &ws.normed, &mut ws.logits)?;
                return self.download_logits(backend, ws);
            }
            Decoder::Qwen35(qwen) => {
                for (i, &token) in tokens.iter().enumerate() {
                    qwen.run(backend, self, kv, ws, token, pos + i as u32)?;
                }
                backend.qmatmul(self.output_weight(), &ws.normed, &mut ws.logits)?;
                return self.download_logits(backend, ws);
            }
            Decoder::Glm4(glm) => {
                for (i, &token) in tokens.iter().enumerate() {
                    glm.run(backend, self, kv, ws, token, pos + i as u32)?;
                }
                backend.qmatmul(self.output_weight(), &ws.normed, &mut ws.logits)?;
                return self.download_logits(backend, ws);
            }
        }

        let n_embd = self.config.n_embd;
        let row_bytes = n_embd * size_of::<f32>();
        let last_row_offset = (n_tokens - 1) * row_bytes;
        backend.copy(&ws.normed, last_row_offset, &mut ws.last_row, 0, row_bytes)?;
        backend.qmatmul(self.output_weight(), &ws.last_row, &mut ws.logits)?;
        self.download_logits(backend, ws)
    }
}

impl<B: Backend> Workspace<B> {
    fn basic(backend: &B, n_tokens: usize, sizes: [usize; 13]) -> Result<Self> {
        Ok(Self {
            n_tokens,
            x: backend.alloc(sizes[0], Pool::Scratch)?,
            x_alt: backend.alloc(sizes[1], Pool::Scratch)?,
            normed: backend.alloc(sizes[2], Pool::Scratch)?,
            q: backend.alloc(sizes[3], Pool::Scratch)?,
            k: backend.alloc(sizes[4], Pool::Scratch)?,
            v: backend.alloc(sizes[5], Pool::Scratch)?,
            attn_out: backend.alloc(sizes[6], Pool::Scratch)?,
            proj: backend.alloc(sizes[7], Pool::Scratch)?,
            gate: backend.alloc(sizes[8], Pool::Scratch)?,
            up: backend.alloc(sizes[9], Pool::Scratch)?,
            act: backend.alloc(sizes[10], Pool::Scratch)?,
            last_row: backend.alloc(sizes[11], Pool::Scratch)?,
            logits: backend.alloc(sizes[12], Pool::Scratch)?,
            gemma: None,
            qwen: None,
            qwen35: None,
            glm: None,
            _memory: None,
        })
    }
}

#[cfg(test)]
mod bonsai_tests;
#[cfg(test)]
mod checkpoint_tests;
#[cfg(test)]
pub(crate) mod compat_tests;
#[cfg(test)]
mod long_context_tests;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
pub(crate) mod tests;

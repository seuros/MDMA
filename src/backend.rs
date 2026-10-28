pub mod cpu;
pub mod dtype_code;
pub mod select;
pub mod shape;
#[cfg(any(feature = "vulkan", test))]
mod submission;
// GPU backends wrap driver FFI; unsafe stays confined to these modules.
#[cfg(feature = "cuda")]
#[allow(unsafe_code)]
pub mod cuda;
#[cfg(feature = "level0")]
#[allow(unsafe_code)]
pub mod level0;
#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(unsafe_code)]
pub mod metal;
#[cfg(feature = "rocm")]
#[allow(unsafe_code)]
pub mod rocm;
#[cfg(feature = "vulkan")]
#[allow(unsafe_code)]
pub mod vulkan;
// The Edge TPU is driven through a character device; only its `pcie::os` module is unsafe.
#[cfg(feature = "coral")]
pub mod coral;

#[cfg(test)]
pub(crate) mod test_support;

// The decoder retains the model, KV and workspace borrows.
#[cfg(any(feature = "cuda", feature = "rocm", test))]
macro_rules! delegate_decoder {
    ($factory:path) => {
        fn decoder<'a>(
            &'a self,
            model: &'a $crate::model::Model<Self>,
            kv: &'a mut $crate::kv_cache::KvCache<Self>,
            ws: &'a mut $crate::model::Workspace<Self>,
        ) -> $crate::backend::Result<Box<dyn $crate::model::decoder::Decoder + 'a>> {
            $factory(self, model, kv, ws)
        }
    };
}
#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) use delegate_decoder;

use crate::gguf::GgmlDType;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "prompt of {prompt_tokens} tokens must fit the {n_ctx}-token session context with room for output"
    )]
    ContextLimit { prompt_tokens: usize, n_ctx: usize },
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),
    #[error("not yet implemented: {0}")]
    NotImplemented(&'static str),
    #[error("unsupported model: {0}")]
    UnsupportedModel(String),
    #[error(
        "native tensor memory budget exhausted: requested {requested} bytes, used/reserved {used}, limit {limit}"
    )]
    MemoryBudget {
        requested: usize,
        used: usize,
        limit: usize,
    },
    #[error(
        "{domain} buffer budget exhausted: requested {requested} bytes, allocated/reserved {used}, limit {limit}"
    )]
    BufferBudget {
        domain: crate::memory::MemoryDomain,
        requested: usize,
        used: usize,
        limit: usize,
    },
    #[error(
        "insufficient device memory: requested {requested} bytes including headroom, available {available}"
    )]
    InsufficientMemory { requested: usize, available: usize },
    #[error("out of memory: {0}")]
    OutOfMemory(String),
    #[error("tensor allocation byte size overflows usize")]
    SizeOverflow,
    /// Driver or device runtime failure.
    #[error("backend error: {0}")]
    Backend(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    pub fn is_memory_exhausted(&self) -> bool {
        matches!(
            self,
            Self::MemoryBudget { .. }
                | Self::BufferBudget { .. }
                | Self::InsufficientMemory { .. }
                | Self::OutOfMemory(_)
        )
    }
}

#[derive(Debug)]
pub enum Pool {
    Weights,
    Kv,
    Scratch,
}

pub struct QWeight<Buf> {
    pub dtype: GgmlDType,
    pub shape: Vec<usize>,
    pub buf: Buf,
}

pub struct RopeSpec {
    pub head_dim: usize,
    pub theta: f32,
}

pub struct KvView<'a, Buf> {
    pub k: &'a Buf,
    pub v: &'a Buf,
    pub len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnSpec {
    pub head_dim: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
}

/// Optional compute primitives. Architectures declare which ones they require.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Scale,
    GeluTanhMul,
    Softcap,
    RopeSplitHalf,
    WindowedAttention,
    AddBias,
    RopePartialInterleaved,
    RopePartialSplitHalf,
    SigmoidMul,
    /// `ssm_conv_silu` and `gated_delta_step`, the recurrent half of a hybrid decoder.
    GatedDeltaNet,
    LayerNorm,
    Gelu,
    Argmax,
}

/// Gated DeltaNet geometry. Value head `h` reads query/key head
/// `h % n_k_heads`; every head is `head_dim` wide, and each value head keeps a
/// `head_dim`×`head_dim` state with one row per value element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaSpec {
    pub n_k_heads: usize,
    pub n_v_heads: usize,
    pub head_dim: usize,
    /// Added to each query/key head's squared norm before L2 normalization.
    pub eps: f32,
}

/// One token's gated DeltaNet inputs: the convolved `[q | k | v]` row, and
/// one `alpha`, `beta`, `dt_bias` and `a` per value head.
pub struct DeltaInputs<'a, Buf> {
    pub qkv: &'a Buf,
    pub alpha: &'a Buf,
    pub beta: &'a Buf,
    pub dt_bias: &'a Buf,
    pub a: &'a Buf,
}

/// Geometry of an encoder (bidirectional, batched, no KV cache) activation:
/// q/k/v/o are `[batch, seq_len, n_heads, head_dim]`, batch inferred from
/// the buffer length. Sequences are right-padded; the per-sequence valid
/// length travels alongside as a host slice, like `embed`'s token ids.
pub struct EncoderSpec {
    pub seq_len: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    /// Keys farther than this from the query (`|i - j| > window`) are
    /// masked; `None` is full attention.
    pub window: Option<usize>,
}

/// Device memory and inference operations.
pub trait Backend {
    type Buf;

    /// Join native device work before a generation job releases its resources
    /// or accounting, including error/cancellation/unwind paths. Concrete GPU
    /// implementations withdraw their profiles if completion cannot be proved.
    fn join_execution(&self) -> Result<()> {
        Ok(())
    }

    /// Instance-local identity for binding uploaded GPU resources to their
    /// execution backend. This token alone cannot confer a native profile.
    fn execution_identity(&self) -> Option<std::sync::Arc<()>> {
        None
    }

    fn device_label(&self) -> Option<String> {
        None
    }
    fn unified_memory(&self) -> bool {
        false
    }

    /// A live free-memory snapshot, not a reservation against other processes.
    fn available_device_bytes(&self) -> Result<Option<usize>> {
        Ok(None)
    }

    /// Device memory in total, when the backend has its own.
    fn total_device_bytes(&self) -> Option<usize> {
        None
    }

    /// Identifies the exact prepared layout, target and kernel build.
    fn weight_cache_profile(&self) -> Option<&str> {
        None
    }

    fn upload_weight(&self, bytes: &[u8], layout: crate::cache::WeightLayout) -> Result<Self::Buf> {
        if layout != crate::cache::WeightLayout::Gguf {
            return Err(Error::NotImplemented(
                "prepared weight layout for this backend",
            ));
        }
        let mut buf = self.alloc(bytes.len(), Pool::Weights)?;
        self.upload(&mut buf, 0, bytes)?;
        Ok(buf)
    }

    fn supports(&self, _operation: Operation) -> bool {
        false
    }

    fn scale(&self, x: &mut Self::Buf, factor: f32) -> Result<()> {
        let _ = (x, factor);
        Err(Error::NotImplemented("scale"))
    }

    /// Tanh-approximated GELU multiplied by `b`, distinct from erf GELU.
    fn gelu_tanh_mul(&self, a: &Self::Buf, b: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let _ = (a, b, y);
        Err(Error::NotImplemented("gelu_tanh_mul"))
    }

    fn softcap(&self, x: &mut Self::Buf, cap: f32) -> Result<()> {
        let _ = (x, cap);
        Err(Error::NotImplemented("softcap"))
    }

    /// Split-half RoPE at one position, with one frequency divisor per pair.
    fn rope_split_half(
        &self,
        x: &mut Self::Buf,
        pos: u32,
        spec: &RopeSpec,
        factors: &Self::Buf,
    ) -> Result<()> {
        let _ = (x, pos, spec, factors);
        Err(Error::NotImplemented("rope_split_half"))
    }

    /// Interleaved RoPE over the first `rotary_dim` values of each head at one
    /// position. Frequencies use `rotary_dim`; the remaining values are unchanged.
    fn rope_partial_interleaved(
        &self,
        x: &mut Self::Buf,
        pos: u32,
        spec: &RopeSpec,
        rotary_dim: usize,
    ) -> Result<()> {
        let _ = (x, pos, spec, rotary_dim);
        Err(Error::NotImplemented("rope_partial_interleaved"))
    }

    /// Split-half RoPE over the first `rotary_dim` values of each head at one
    /// position: value `i` pairs with `i + rotary_dim / 2`, frequencies use
    /// `rotary_dim`, and the remaining values are unchanged.
    fn rope_partial_split_half(
        &self,
        x: &mut Self::Buf,
        pos: u32,
        spec: &RopeSpec,
        rotary_dim: usize,
    ) -> Result<()> {
        let _ = (x, pos, spec, rotary_dim);
        Err(Error::NotImplemented("rope_partial_split_half"))
    }

    /// `y = sigmoid(gate) * x`, elementwise.
    fn sigmoid_mul(&self, gate: &Self::Buf, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let _ = (gate, x, y);
        Err(Error::NotImplemented("sigmoid_mul"))
    }

    /// One token of a causal depthwise convolution, then SiLU. `w` holds each
    /// channel's taps, oldest input first; `state` holds each channel's
    /// previous `taps - 1` inputs, oldest first, and takes `x` in.
    fn ssm_conv_silu(
        &self,
        w: &Self::Buf,
        state: &mut Self::Buf,
        x: &Self::Buf,
        y: &mut Self::Buf,
    ) -> Result<()> {
        let _ = (w, state, x, y);
        Err(Error::NotImplemented("ssm_conv_silu"))
    }

    /// One token of the gated delta rule. Per value head, with its query and
    /// key head L2-normalized: `S ← exp(g)·S`, `S ← S + β(v − S·k)·kᵀ`,
    /// `o = S·q / √head_dim`, where `g = softplus(alpha + dt_bias)·a` and
    /// `β = sigmoid(beta)`.
    fn gated_delta_step(
        &self,
        spec: &DeltaSpec,
        inputs: &DeltaInputs<Self::Buf>,
        state: &mut Self::Buf,
        o: &mut Self::Buf,
    ) -> Result<()> {
        let _ = (spec, inputs, state, o);
        Err(Error::NotImplemented("gated_delta_step"))
    }

    /// Causal attention with an explicit score multiplier and window size.
    /// A zero window includes every causally visible key.
    fn attn_windowed(
        &self,
        q: &Self::Buf,
        kv: &KvView<Self::Buf>,
        spec: &AttnSpec,
        scale: f32,
        window: usize,
        o: &mut Self::Buf,
    ) -> Result<()> {
        let _ = (q, kv, spec, scale, window, o);
        Err(Error::NotImplemented("attn_windowed"))
    }

    /// Install before allocating or sharing the backend. Unsupported backends
    /// reject the quota instead of silently leaving their buffers unaccounted.
    fn set_buffer_budget(&mut self, _budget: crate::memory::BufferBudget) -> Result<()> {
        Err(Error::NotImplemented(
            "backend buffer accounting (requires rocm, cuda or vulkan)",
        ))
    }

    fn buffer_budget(&self) -> Option<&crate::memory::BufferBudget> {
        None
    }

    /// Weight formats supported by both row gather and matrix multiplication.
    /// Defaults to the shared GPU kernel formats; CPU supports more.
    fn supports_weight_dtype(&self, dtype: GgmlDType) -> bool {
        dtype_code::code(dtype).is_ok()
    }

    /// Hard attention/RoPE head-width ceiling, not a fast-path threshold.
    fn max_head_dim(&self) -> Option<usize> {
        None
    }

    fn decoder<'a>(
        &'a self,
        model: &'a crate::model::Model<Self>,
        kv: &'a mut crate::kv_cache::KvCache<Self>,
        ws: &'a mut crate::model::Workspace<Self>,
    ) -> Result<Box<dyn crate::model::decoder::Decoder + 'a>>
    where
        Self: Sized,
    {
        Ok(Box::new(crate::model::decoder::EagerDecoder {
            backend: self,
            model,
            kv,
            ws,
        }))
    }

    fn alloc(&self, bytes: usize, pool: Pool) -> Result<Self::Buf>;
    fn upload(&self, dst: &mut Self::Buf, offset: usize, src: &[u8]) -> Result<()>;
    fn download(&self, src: &Self::Buf, offset: usize, dst: &mut [u8]) -> Result<()>;
    /// Device-to-device (or host-agnostic) byte copy - no round trip
    /// through host memory. `src`/`dst` may be different buffers only;
    /// overlapping-buffer self-copy is not a supported use case here.
    fn copy(
        &self,
        src: &Self::Buf,
        src_off: usize,
        dst: &mut Self::Buf,
        dst_off: usize,
        len: usize,
    ) -> Result<()>;

    /// The q, k and v projections of one input. Backends may fuse the
    /// three into one launch.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the fused QKV kernel launch"
    )]
    fn qmatmul_qkv(
        &self,
        wq: &QWeight<Self::Buf>,
        wk: &QWeight<Self::Buf>,
        wv: &QWeight<Self::Buf>,
        x: &Self::Buf,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        v: &mut Self::Buf,
    ) -> Result<()> {
        self.qmatmul(wq, x, q)?;
        self.qmatmul(wk, x, k)?;
        self.qmatmul(wv, x, v)
    }

    /// Bytes per stored K/V element. Sources (`k`/`v` passed to
    /// `kv_write`) are always f32; a backend that stores f16 returns 2
    /// and converts on the way in.
    fn kv_elem_bytes(&self) -> usize {
        4
    }

    /// Append `n_elems` consecutive K/V elements (f32 in `k`/`v`, from
    /// offset 0) at element `elem_offset` of the caches. Backends may
    /// fuse the two writes.
    fn kv_write(
        &self,
        k: &Self::Buf,
        v: &Self::Buf,
        dst_k: &mut Self::Buf,
        dst_v: &mut Self::Buf,
        elem_offset: usize,
        n_elems: usize,
    ) -> Result<()> {
        self.copy(k, 0, dst_k, elem_offset * 4, n_elems * 4)?;
        self.copy(v, 0, dst_v, elem_offset * 4, n_elems * 4)
    }

    /// RoPE on q and k, then the K/V rows into the caches at element
    /// `elem_offset` (`n_tokens` rows of `k`'s per-token width). Backends
    /// may fuse the rotation with the write; `k` is not required to hold
    /// the rotated rows afterwards.
    #[allow(clippy::too_many_arguments)]
    fn rope_kv_write(
        &self,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        v: &Self::Buf,
        dst_k: &mut Self::Buf,
        dst_v: &mut Self::Buf,
        elem_offset: usize,
        pos: u32,
        n_tokens: usize,
        spec: &RopeSpec,
        n_elems: usize,
    ) -> Result<()> {
        self.rope(q, k, pos, n_tokens, spec)?;
        self.kv_write(k, v, dst_k, dst_v, elem_offset, n_elems)
    }

    fn qmatmul(&self, w: &QWeight<Self::Buf>, x: &Self::Buf, y: &mut Self::Buf) -> Result<()>;
    /// `y = SiLU(gate * x) * (up * x)`. Weights have matching shapes.
    /// Scratch buffers have the output shape; their final contents are unspecified.
    fn ffn_gate_up(
        &self,
        gate: &QWeight<Self::Buf>,
        up: &QWeight<Self::Buf>,
        x: &Self::Buf,
        gate_scratch: &mut Self::Buf,
        up_scratch: &mut Self::Buf,
        y: &mut Self::Buf,
    ) -> Result<()> {
        if gate.shape != up.shape {
            return Err(Error::ShapeMismatch(
                "ffn gate/up weight shapes must match".into(),
            ));
        }
        self.qmatmul(gate, x, gate_scratch)?;
        self.qmatmul(up, x, up_scratch)?;
        self.silu_mul(gate_scratch, up_scratch, y)
    }

    /// Dequantized row gather - `table` is a [vocab, n_embd] weight, `out`
    /// gets `[token_ids.len(), n_embd]` rows in order.
    fn embed(
        &self,
        table: &QWeight<Self::Buf>,
        token_ids: &[u32],
        out: &mut Self::Buf,
    ) -> Result<()>;
    fn rmsnorm(&self, w: &Self::Buf, x: &Self::Buf, y: &mut Self::Buf, eps: f32) -> Result<()>;
    /// `sum = a + b; y = rmsnorm(sum) * w` -- the residual connection and
    /// the norm that always follows it. Backends may fuse the two.
    fn add_rmsnorm(
        &self,
        w: &Self::Buf,
        a: &Self::Buf,
        b: &Self::Buf,
        sum: &mut Self::Buf,
        y: &mut Self::Buf,
        eps: f32,
    ) -> Result<()> {
        self.add(a, b, sum)?;
        self.rmsnorm(w, sum, y, eps)
    }
    /// Elementwise `y = a + b` - residual connections.
    fn add(&self, a: &Self::Buf, b: &Self::Buf, y: &mut Self::Buf) -> Result<()>;
    /// q/k are [n_tokens, n_q_heads/n_kv_heads, head_dim]; token t is
    /// rotated at position `pos + t`.
    fn rope(
        &self,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        pos: u32,
        n_tokens: usize,
        spec: &RopeSpec,
    ) -> Result<()>;
    /// q/o are [n_q, n_q_heads, head_dim]; n_q is inferred from q's
    /// length. kv.len is the total keys already cached (prefix + this
    /// batch); query i attends causally to keys [0, kv.len - n_q + i].
    /// With n_q == 1, the query attends to all cached keys.
    fn attn(
        &self,
        q: &Self::Buf,
        kv: &KvView<Self::Buf>,
        spec: &AttnSpec,
        o: &mut Self::Buf,
    ) -> Result<()>;
    fn silu_mul(&self, gate: &Self::Buf, up: &Self::Buf, y: &mut Self::Buf) -> Result<()>;

    // Encoder ops (Laya / ModernBERT). Default to NotImplemented so a
    // backend without them still serves the decoder path.

    /// `y = layernorm(x) * w + b` per row of `w.len()`; mean and biased
    /// variance over the row.
    fn layernorm(
        &self,
        w: &Self::Buf,
        b: Option<&Self::Buf>,
        x: &Self::Buf,
        y: &mut Self::Buf,
        eps: f32,
    ) -> Result<()> {
        let _ = (w, b, x, y, eps);
        Err(Error::NotImplemented("layernorm"))
    }
    /// In place: every row of `y` (rows of `b.len()`) gets `b` added.
    fn add_bias(&self, b: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let _ = (b, y);
        Err(Error::NotImplemented("add_bias"))
    }
    /// Exact (erf) GELU, elementwise.
    fn gelu(&self, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let _ = (x, y);
        Err(Error::NotImplemented("gelu"))
    }
    /// Lowest-index maximum of a nonempty f32 vector, written as one u32.
    /// Any nonfinite input produces u32::MAX instead of a valid token.
    fn argmax(&self, x: &Self::Buf, index: &mut Self::Buf) -> Result<()> {
        let _ = (x, index);
        Err(Error::NotImplemented("argmax"))
    }
    /// `y = gelu(a) * gate` - the GeGLU half of ModernBERT's MLP.
    fn gelu_mul(&self, a: &Self::Buf, gate: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let _ = (a, gate, y);
        Err(Error::NotImplemented("gelu_mul"))
    }
    fn relu(&self, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let _ = (x, y);
        Err(Error::NotImplemented("relu"))
    }
    /// Split-half (NeoX / HF `rotate_half`) RoPE on q and k laid out per
    /// `spec`; every sequence in the batch is rotated from position 0.
    fn rope_neox(
        &self,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        spec: &EncoderSpec,
        theta: f32,
    ) -> Result<()> {
        let _ = (q, k, spec, theta);
        Err(Error::NotImplemented("rope_neox"))
    }
    /// Bidirectional attention over each sequence of the batch. `lens[b]`
    /// keys are valid in sequence `b` (right padding); padded query rows
    /// produce unspecified output. `spec.window` bounds `|i - j|`.
    fn attn_bidir(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        lens: &[u32],
        spec: &EncoderSpec,
        o: &mut Self::Buf,
    ) -> Result<()> {
        let _ = (q, k, v, lens, spec, o);
        Err(Error::NotImplemented("attn_bidir"))
    }
}

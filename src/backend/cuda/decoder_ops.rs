//! Gemma and Qwen3.5 primitives (kernels/decoder_ops.cu): elementwise ops,
//! split-half RoPE, gated DeltaNet and windowed attention over the f16 KV
//! cache up to 512-wide heads.
use super::*;

/// head_dim ceiling of attn_windowed_f16kv (DW_MAX_HEAD_DIM in decoder_ops.cu).
pub(super) const WINDOWED_MAX_HEAD_DIM: usize = 512;
/// Maximum Qwen3.5 DeltaNet head width supported by the shared-memory kernel.
const DELTA_MAX_HEAD_DIM: usize = 256;

fn partial_rope_freqs(op: &str, spec: &RopeSpec, rotary_dim: usize) -> Result<Freqs> {
    if rotary_dim > 2 * ROPE_PAIRS {
        return Err(Error::ShapeMismatch(format!(
            "{op}: rotary_dim {rotary_dim} exceeds {}",
            2 * ROPE_PAIRS
        )));
    }
    let mut freqs = Freqs([0.0; ROPE_PAIRS]);
    for (i, f) in freqs.0.iter_mut().take(rotary_dim / 2).enumerate() {
        *f = spec.theta.powf(-2.0 * i as f32 / rotary_dim as f32);
    }
    Ok(freqs)
}

impl CudaBackend {
    /// op 0 scales, op 1 softcaps (unary_f32).
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes x on the device"
    )]
    pub(super) fn unary(&self, x: &mut CudaBuf, value: f32, op: i32) -> Result<()> {
        let n = shape::elems("unary", x.len)?;
        if n == 0 {
            return Ok(());
        }
        let (mut x_ptr, mut n_i, mut value, mut op) = (x.ptr, int(n)?, value, op);
        self.launch(
            "unary_f32",
            [(n as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            &mut params!(x_ptr, n_i, value, op),
        )
    }

    /// Q8_0 matvec. Qwen3.5 uses this format for the small alpha/beta
    /// projections; its 32-element blocks do not use the K-quant kernels.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes y on the device"
    )]
    pub(super) fn qmatmul_q8_0(
        &self,
        w: &QWeight<CudaBuf>,
        x: &CudaBuf,
        y: &mut CudaBuf,
        n_out: usize,
        n_in: usize,
        n_tokens: usize,
    ) -> Result<()> {
        if !n_in.is_multiple_of(32) {
            return Err(Error::ShapeMismatch(format!(
                "qmatmul Q8_0 width {n_in} is not a multiple of 32"
            )));
        }
        let (mut w_ptr, mut x_ptr, mut y_ptr) = (w.buf.ptr, x.ptr, y.ptr);
        let (mut n_out_i, mut n_in_i) = (int(n_out)?, int(n_in)?);
        self.launch(
            "gemv_q8_0",
            [(n_out as u32).div_ceil(WARPS), n_tokens as u32, 1],
            [32, WARPS, 1],
            &mut params!(w_ptr, x_ptr, y_ptr, n_out_i, n_in_i),
        )
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes y on the device"
    )]
    pub(super) fn gelu(&self, a: &CudaBuf, b: &CudaBuf, y: &mut CudaBuf) -> Result<()> {
        let n = shape::elementwise(
            "gelu_tanh_mul",
            &[shape::elems("gelu", a.len)?, shape::elems("gelu", b.len)?],
            &[shape::elems("gelu", y.len)?],
        )?;
        if n == 0 {
            return Ok(());
        }
        let (mut a_ptr, mut b_ptr, mut y_ptr, mut n_i) = (a.ptr, b.ptr, y.ptr, int(n)?);
        self.launch(
            "gelu_tanh_mul_f32",
            [(n as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            &mut params!(a_ptr, b_ptr, y_ptr, n_i),
        )
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes x on the device"
    )]
    pub(super) fn split_rope(
        &self,
        x: &mut CudaBuf,
        pos: u32,
        spec: &RopeSpec,
        factors: &CudaBuf,
    ) -> Result<()> {
        self.eager_only("rope_split_half")?;
        let n = shape::elems("rope_split_half", x.len)?;
        shape::rope_split_half(n, shape::elems("rope_split_half", factors.len)?, spec)?;
        let mut fr = rope_freqs("rope_split_half", spec)?;
        let (mut x_ptr, mut f_ptr) = (x.ptr, factors.ptr);
        let (mut n_i, mut hd, mut pos) = (int(n)?, int(spec.head_dim)?, pos);
        self.launch(
            "rope_split_half_f32",
            [(n as u32 / 2).div_ceil(256), 1, 1],
            [256, 1, 1],
            &mut params!(x_ptr, f_ptr, n_i, hd, pos, fr),
        )
    }

    /// Split-half RoPE over the first `rotary_dim` values of each head. The
    /// graph replay parameter buffer supplies the current position when this
    /// operation is captured for a CUDA decoder.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes x on the device"
    )]
    pub(super) fn split_rope_partial(
        &self,
        x: &mut CudaBuf,
        pos: u32,
        spec: &RopeSpec,
        rotary_dim: usize,
    ) -> Result<()> {
        let n = shape::elems("rope_partial_split_half", x.len)?;
        let pairs = shape::rope_partial(n, spec, rotary_dim)?;
        let mut fr = partial_rope_freqs("rope_partial_split_half", spec, rotary_dim)?;
        let (mut x_ptr, mut n_i, mut hd, mut rotary_i, mut pos) =
            (x.ptr, int(n)?, int(spec.head_dim)?, int(rotary_dim)?, pos);
        let mut pp = self.replay.as_ref().map_or(0, |replay| replay.params.ptr);
        self.launch(
            "rope_partial_split_half_f32",
            [(pairs as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            &mut params!(x_ptr, n_i, hd, rotary_i, pos, fr, pp),
        )
    }

    /// `y = sigmoid(gate) * x` for Qwen3.5's gated attention output.
    pub(super) fn sigmoid_mul(&self, gate: &CudaBuf, x: &CudaBuf, y: &mut CudaBuf) -> Result<()> {
        self.launch_binary_f32("sigmoid_mul", "sigmoid_mul_f32", gate, x, y)
    }

    /// One token of Qwen3.5's causal depthwise convolution followed by SiLU.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes state and y on the device"
    )]
    pub(super) fn ssm_conv_silu(
        &self,
        w: &CudaBuf,
        state: &mut CudaBuf,
        x: &CudaBuf,
        y: &mut CudaBuf,
    ) -> Result<()> {
        let (channels, taps) = shape::ssm_conv(
            shape::elems("ssm_conv_silu", w.len)?,
            shape::elems("ssm_conv_silu", state.len)?,
            shape::elems("ssm_conv_silu", x.len)?,
            shape::elems("ssm_conv_silu", y.len)?,
        )?;
        let (mut w_ptr, mut state_ptr, mut x_ptr, mut y_ptr) = (w.ptr, state.ptr, x.ptr, y.ptr);
        let (mut channels_i, mut taps_i) = (int(channels)?, int(taps)?);
        self.launch(
            "ssm_conv_silu_f32",
            [(channels as u32).div_ceil(256), 1, 1],
            [256, 1, 1],
            &mut params!(w_ptr, state_ptr, x_ptr, y_ptr, channels_i, taps_i),
        )
    }

    /// One token of the Qwen3.5 gated DeltaNet recurrence.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes state and o on the device"
    )]
    pub(super) fn gated_delta_step(
        &self,
        spec: &DeltaSpec,
        inputs: &DeltaInputs<CudaBuf>,
        state: &mut CudaBuf,
        o: &mut CudaBuf,
    ) -> Result<()> {
        let qkv = shape::elems("gated_delta", inputs.qkv.len)?;
        let gates = [
            shape::elems("gated_delta", inputs.alpha.len)?,
            shape::elems("gated_delta", inputs.beta.len)?,
            shape::elems("gated_delta", inputs.dt_bias.len)?,
            shape::elems("gated_delta", inputs.a.len)?,
        ];
        let state_elems = shape::elems("gated_delta", state.len)?;
        let o_elems = shape::elems("gated_delta", o.len)?;
        shape::gated_delta(spec, qkv, gates, state_elems, o_elems)?;
        if spec.head_dim > DELTA_MAX_HEAD_DIM {
            return Err(Error::ShapeMismatch(format!(
                "gated_delta: head_dim {} exceeds {DELTA_MAX_HEAD_DIM}",
                spec.head_dim
            )));
        }
        let (
            mut qkv_ptr,
            mut alpha_ptr,
            mut beta_ptr,
            mut dt_bias_ptr,
            mut a_ptr,
            mut state_ptr,
            mut o_ptr,
        ) = (
            inputs.qkv.ptr,
            inputs.alpha.ptr,
            inputs.beta.ptr,
            inputs.dt_bias.ptr,
            inputs.a.ptr,
            state.ptr,
            o.ptr,
        );
        let (mut n_k, mut n_v, mut head_dim) = (
            int(spec.n_k_heads)?,
            int(spec.n_v_heads)?,
            int(spec.head_dim)?,
        );
        let mut eps = spec.eps;
        self.launch(
            "gated_delta_f32",
            [spec.n_v_heads as u32, 1, 1],
            [256, 1, 1],
            &mut params!(
                qkv_ptr,
                alpha_ptr,
                beta_ptr,
                dt_bias_ptr,
                a_ptr,
                state_ptr,
                o_ptr,
                n_k,
                n_v,
                head_dim,
                eps
            ),
        )
    }

    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes o on the device"
    )]
    pub(super) fn attention_windowed(
        &self,
        q: &CudaBuf,
        kv: &KvView<CudaBuf>,
        spec: &AttnSpec,
        scale: f32,
        window: usize,
        o: &mut CudaBuf,
    ) -> Result<()> {
        self.eager_only("attn_windowed")?;
        shape::attention_scale(scale)?;
        if !kv.k.len.is_multiple_of(2) || !kv.v.len.is_multiple_of(2) {
            return Err(Error::ShapeMismatch("attn: partial f16 KV element".into()));
        }
        let shape::Attn { n_q, prefix } = shape::attn(
            spec,
            shape::elems("attn", q.len)?,
            kv.len,
            kv.k.len / 2,
            kv.v.len / 2,
            shape::elems("attn", o.len)?,
        )?;
        if spec.head_dim > WINDOWED_MAX_HEAD_DIM {
            return Err(Error::ShapeMismatch(format!(
                "attn: head_dim {} exceeds {WINDOWED_MAX_HEAD_DIM}",
                spec.head_dim
            )));
        }
        let (mut q_ptr, mut k_ptr, mut v_ptr, mut o_ptr) = (q.ptr, kv.k.ptr, kv.v.ptr, o.ptr);
        let (mut hd, mut qh, mut kh) = (
            int(spec.head_dim)?,
            int(spec.n_q_heads)?,
            int(spec.n_kv_heads)?,
        );
        let (mut prefix, mut scale) = (int(prefix)?, scale);
        let mut window = int(window.min(kv.len))?;
        self.launch(
            "attn_windowed_f16kv",
            [spec.n_q_heads as u32, n_q as u32, 1],
            [128, 1, 1],
            &mut params!(
                q_ptr, k_ptr, v_ptr, o_ptr, hd, qh, kh, prefix, scale, window
            ),
        )
    }

    /// Q5_K matvec (gemv_q5k): never the int8 or gemv_q paths, which only
    /// know Q4_K and Q6_K.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "the kernel writes y on the device"
    )]
    pub(super) fn qmatmul_q5k(
        &self,
        w: &QWeight<CudaBuf>,
        x: &CudaBuf,
        y: &mut CudaBuf,
        n_out: usize,
        n_in: usize,
        n_tokens: usize,
    ) -> Result<()> {
        shape::kquant_aligned("qmatmul", w.dtype, n_in)?;
        let (mut w_ptr, mut x_ptr, mut y_ptr) = (w.buf.ptr, x.ptr, y.ptr);
        let (mut n_out_i, mut n_in_i) = (int(n_out)?, int(n_in)?);
        self.launch(
            "gemv_q5k",
            [(n_out as u32).div_ceil(WARPS), n_tokens as u32, 1],
            [32, WARPS, 1],
            &mut params!(w_ptr, x_ptr, y_ptr, n_out_i, n_in_i),
        )
    }
}

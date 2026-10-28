mod dequant;
mod encoder;

use super::{
    AttnSpec, Backend, DeltaInputs, DeltaSpec, EncoderSpec, Error, KvView, Operation, Pool,
    QWeight, Result, RopeSpec, shape,
};
use crate::gguf::GgmlDType;
use dequant::dequantize_row;
use rayon::prelude::*;
use std::sync::OnceLock;

mod fused;
pub mod isa;
mod packed;

pub use isa::isa;

#[derive(Default)]
pub struct CpuBackend;

pub struct CpuBuf(pub Vec<u8>);

/// How the CPU multiplies quantized weights, fixed for the process.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Matmul {
    /// Each weight row dequantized to f32 and summed in 16 fixed lanes.
    #[default]
    Exact,
    /// Activations quantized once to the weight type's dot type, rows
    /// multiplied in their blocks. Several times faster; changes logits.
    Packed,
}

impl Matmul {
    /// `MDMA_CPU_MATMUL`: unset or `exact`, or `packed`.
    pub fn parse(value: Option<&str>) -> std::result::Result<Self, String> {
        match value {
            None | Some("exact") => Ok(Self::Exact),
            Some("packed") => Ok(Self::Packed),
            Some(other) => Err(format!(
                "MDMA_CPU_MATMUL must be exact or packed, not {other:?}"
            )),
        }
    }

    pub fn from_env() -> std::result::Result<Self, String> {
        match std::env::var("MDMA_CPU_MATMUL") {
            Ok(value) => Self::parse(Some(&value)),
            Err(std::env::VarError::NotPresent) => Self::parse(None),
            Err(error) => Err(format!("MDMA_CPU_MATMUL: {error}")),
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Packed => "packed",
        }
    }
}

static MATMUL: OnceLock<Matmul> = OnceLock::new();

/// Fixes the process's CPU matmul before its first use. Choosing another
/// later is refused: execution profiles name the one in force.
pub fn choose_matmul(mode: Matmul) -> std::result::Result<(), String> {
    match *MATMUL.get_or_init(|| mode) {
        chosen if chosen == mode => Ok(()),
        chosen => Err(format!("the CPU matmul is already {}", chosen.name())),
    }
}

/// The process's CPU matmul; exact unless chosen before first use.
pub fn matmul() -> Matmul {
    *MATMUL.get_or_init(Matmul::default)
}

/// `qmatmul` with the matmul `mode`, so both paths run in one process.
fn qmatmul_in(mode: Matmul, w: &QWeight<CpuBuf>, x_f32: &[f32], y_f32: &mut [f32]) -> Result<()> {
    let shape::Matmul {
        n_out,
        n_in,
        n_tokens,
    } = shape::matmul(&w.shape, x_f32.len(), y_f32.len())?;

    if mode == Matmul::Packed && packed::matmul(w, x_f32, y_f32, n_out, n_in) {
        return Ok(());
    }

    // Decode is the common one-token case. Write each row's dot directly to
    // the output instead of allocating and transposing a [n_out, n_tokens]
    // scratch buffer; types with a fused kernel never write the row out.
    if n_tokens == 1 {
        y_f32.par_iter_mut().enumerate().try_for_each_init(
            || vec![0f32; n_in],
            |row_f32, (row, out)| {
                *out = if let Some(dot) = fused::row_dot(w, row, n_in, x_f32) {
                    dot
                } else {
                    dequant_row_at(w, row, n_in, row_f32)?;
                    kernels().row_dot(row_f32, x_f32)
                };
                Ok::<(), Error>(())
            },
        )?;
        return Ok(());
    }

    // Rows in parallel, each dequantized once and dotted against every
    // token into a row-major [n_out, n_tokens] scratch (contiguous per
    // thread); the transpose into y keeps every element's reduction
    // order independent of thread scheduling.
    let mut yt = vec![0f32; n_out * n_tokens];
    yt.par_chunks_mut(n_tokens).enumerate().try_for_each_init(
        || vec![0f32; n_in],
        |row_f32, (row, out)| {
            dequant_row_at(w, row, n_in, row_f32)?;
            for (t, y) in out.iter_mut().enumerate() {
                let x_t = &x_f32[t * n_in..(t + 1) * n_in];
                *y = kernels().row_dot(row_f32, x_t);
            }
            Ok::<(), Error>(())
        },
    )?;
    for t in 0..n_tokens {
        for row in 0..n_out {
            y_f32[t * n_out + row] = yt[row * n_tokens + t];
        }
    }
    Ok(())
}

fn as_f32(bytes: &[u8]) -> Result<&[f32]> {
    bytemuck::try_cast_slice(bytes).map_err(|e| Error::ShapeMismatch(e.to_string()))
}

fn as_f32_mut(bytes: &mut [u8]) -> Result<&mut [f32]> {
    bytemuck::try_cast_slice_mut(bytes).map_err(|e| Error::ShapeMismatch(e.to_string()))
}

/// Dequantize row `row` of a [n_rows, n_cols] weight into `out`.
fn dequant_row_at(w: &QWeight<CpuBuf>, row: usize, n_cols: usize, out: &mut [f32]) -> Result<()> {
    let row_bytes = w.dtype.type_size() * (n_cols / w.dtype.block_size());
    let start = row * row_bytes;
    let bytes =
        w.buf.0.get(start..start + row_bytes).ok_or_else(|| {
            Error::ShapeMismatch(format!("row {row} out of range for weight buffer"))
        })?;
    dequantize_row(w.dtype, bytes, out)
}

/// Row `row` of a `[n_out, n_in]` weight, dequantized into `out` (`n_in` values).
#[cfg_attr(
    not(all(feature = "coral", any(target_os = "freebsd", target_os = "linux"))),
    expect(dead_code, reason = "read by the Coral backend")
)]
pub(crate) fn dequant_row(w: &QWeight<CpuBuf>, row: usize, out: &mut [f32]) -> Result<()> {
    dequant_row_at(w, row, out.len(), out)
}

/// Sequential sum of products, so every caller reduces in one order.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Lanes a row's products are summed in; fused kernels add to the same ones.
const LANES: usize = 16;

isa::multiversion! {
    /// `row_dot` per level; every level gives the same bits.
    fn row_dot_at(a: &[f32], b: &[f32]) -> f32 = row_dot;
}

/// The exact kernels at one level. One exists only for a level this CPU
/// has, which is what makes calling through its pointers sound.
struct Kernels {
    level: isa::Level,
    row_dot: unsafe fn(&[f32], &[f32]) -> f32,
    q4k: unsafe fn(&[u8], &[f32]) -> f32,
}

impl Kernels {
    fn at(level: isa::Level) -> Option<Self> {
        Some(Self {
            level,
            row_dot: row_dot_at(level)?,
            q4k: fused::q4k_at(level)?,
        })
    }

    #[allow(unsafe_code)]
    fn row_dot(&self, a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: `at` holds a variant only for a level `isa` detected.
        unsafe { (self.row_dot)(a, b) }
    }

    #[allow(unsafe_code)]
    fn q4k(&self, row: &[u8], x: &[f32]) -> f32 {
        // SAFETY: `at` holds a variant only for a level `isa` detected.
        unsafe { (self.q4k)(row, x) }
    }
}

/// The kernels for this CPU's level, chosen by [`init`] at startup.
fn kernels() -> &'static Kernels {
    static KERNELS: OnceLock<Kernels> = OnceLock::new();
    KERNELS.get_or_init(|| Kernels::at(isa().level).expect("the detected level runs on this CPU"))
}

/// Startup: detects the CPU's instruction sets, picks the kernels for them
/// and fixes the matmul mode, once, before any work. A machine does not
/// change under a running process; generation only follows the choice.
pub fn init(matmul: Matmul) -> std::result::Result<&'static isa::Isa, String> {
    choose_matmul(matmul)?;
    debug_assert_eq!(kernels().level, isa().level);
    Ok(isa())
}

/// A weight row against an activation: products summed in 16 fixed lanes,
/// then the lanes and the tail in order. The order does not depend on the
/// machine, the thread count or the vector width the compiler picks.
#[expect(
    clippy::inline_always,
    reason = "each multiversioned variant compiles this body with its own instruction sets"
)]
#[inline(always)]
fn row_dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let ((a_lanes, a_tail), (b_lanes, b_tail)) = (a.as_chunks::<LANES>(), b.as_chunks::<LANES>());
    let mut lanes = [0f32; LANES];
    for (a, b) in a_lanes.iter().zip(b_lanes) {
        for i in 0..LANES {
            lanes[i] += a[i] * b[i];
        }
    }
    let mut sum = 0.0;
    for lane in lanes {
        sum += lane;
    }
    for (a, b) in a_tail.iter().zip(b_tail) {
        sum += a * b;
    }
    sum
}

fn softmax_inplace(xs: &mut [f32]) {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for x in xs.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in xs.iter_mut() {
        *x /= sum;
    }
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * x * (1.0 + 0.044715 * x * x)).tanh())
}

/// Interleaved-pair RoPE on `[n_tokens, n_heads, head_dim]`.
/// Token `t` uses position `pos + t` for every head.
fn apply_rope(buf: &mut [f32], pos: u32, n_tokens: usize, spec: &RopeSpec) {
    let token_stride = buf.len() / n_tokens;
    let half = spec.head_dim / 2;
    for (t, token_buf) in buf.chunks_exact_mut(token_stride).enumerate() {
        let token_pos = pos + t as u32;
        for head in token_buf.chunks_exact_mut(spec.head_dim) {
            for i in 0..half {
                let freq = spec.theta.powf(-2.0 * i as f32 / spec.head_dim as f32);
                let (sin_a, cos_a) = (token_pos as f32 * freq).sin_cos();
                let x0 = head[2 * i];
                let x1 = head[2 * i + 1];
                head[2 * i] = x0 * cos_a - x1 * sin_a;
                head[2 * i + 1] = x0 * sin_a + x1 * cos_a;
            }
        }
    }
}

impl Backend for CpuBackend {
    type Buf = CpuBuf;

    fn device_label(&self) -> Option<String> {
        Some("CPU".into())
    }

    fn supports(&self, operation: Operation) -> bool {
        match operation {
            Operation::Scale
            | Operation::GeluTanhMul
            | Operation::Softcap
            | Operation::RopeSplitHalf
            | Operation::WindowedAttention
            | Operation::AddBias
            | Operation::RopePartialInterleaved
            | Operation::RopePartialSplitHalf
            | Operation::SigmoidMul
            | Operation::GatedDeltaNet
            | Operation::LayerNorm
            | Operation::Gelu
            | Operation::Argmax => true,
        }
    }

    fn argmax(&self, x: &CpuBuf, index: &mut CpuBuf) -> Result<()> {
        let x = as_f32(&x.0)?;
        shape::argmax(x.len(), index.0.len())?;
        let mut best = 0;
        for i in 0..x.len() {
            if !x[i].is_finite() {
                index.0.copy_from_slice(&u32::MAX.to_ne_bytes());
                return Ok(());
            }
            if x[i] > x[best] { best = i; }
        }
        index.0.copy_from_slice(&(best as u32).to_ne_bytes());
        Ok(())
    }

    fn scale(&self, x: &mut CpuBuf, factor: f32) -> Result<()> {
        if !factor.is_finite() {
            return Err(Error::ShapeMismatch("nonfinite scale".into()));
        }
        for v in as_f32_mut(&mut x.0)? {
            *v *= factor;
        }
        Ok(())
    }

    fn softcap(&self, x: &mut CpuBuf, cap: f32) -> Result<()> {
        shape::softcap(cap)?;
        for v in as_f32_mut(&mut x.0)? {
            *v = (*v / cap).tanh() * cap;
        }
        Ok(())
    }

    fn gelu_tanh_mul(&self, a: &CpuBuf, b: &CpuBuf, y: &mut CpuBuf) -> Result<()> {
        let (a, b, y) = (as_f32(&a.0)?, as_f32(&b.0)?, as_f32_mut(&mut y.0)?);
        shape::elementwise("gelu_tanh_mul", &[a.len(), b.len()], &[y.len()])?;
        for ((a, b), y) in a.iter().zip(b).zip(y) {
            *y = gelu_tanh(*a) * b;
        }
        Ok(())
    }

    fn rope_split_half(
        &self,
        x: &mut CpuBuf,
        pos: u32,
        spec: &RopeSpec,
        factors: &CpuBuf,
    ) -> Result<()> {
        let x = as_f32_mut(&mut x.0)?;
        let factors = as_f32(&factors.0)?;
        shape::rope_split_half(x.len(), factors.len(), spec)?;
        for head in x.chunks_exact_mut(spec.head_dim) {
            for i in 0..spec.head_dim / 2 {
                let angle = pos as f32 * spec.theta.powf(-2.0 * i as f32 / spec.head_dim as f32)
                    / factors[i];
                let (s, c) = angle.sin_cos();
                let (a, b) = (head[i], head[i + spec.head_dim / 2]);
                head[i] = a * c - b * s;
                head[i + spec.head_dim / 2] = a * s + b * c;
            }
        }
        Ok(())
    }

    fn attn_windowed(
        &self,
        q: &CpuBuf,
        kv: &KvView<CpuBuf>,
        spec: &AttnSpec,
        scale: f32,
        window: usize,
        o: &mut CpuBuf,
    ) -> Result<()> {
        shape::attention_scale(scale)?;
        self.attention(q, kv, spec, window, scale, o)
    }

    fn rope_partial_interleaved(
        &self,
        x: &mut CpuBuf,
        pos: u32,
        spec: &RopeSpec,
        rotary_dim: usize,
    ) -> Result<()> {
        let x = as_f32_mut(&mut x.0)?;
        shape::rope_partial(x.len(), spec, rotary_dim)?;
        for head in x.chunks_exact_mut(spec.head_dim) {
            for i in (0..rotary_dim).step_by(2) {
                let angle = pos as f32 * spec.theta.powf(-(i as f32) / rotary_dim as f32);
                let (s, c) = angle.sin_cos();
                let (a, b) = (head[i], head[i + 1]);
                head[i] = a * c - b * s;
                head[i + 1] = a * s + b * c;
            }
        }
        Ok(())
    }

    fn rope_partial_split_half(
        &self,
        x: &mut CpuBuf,
        pos: u32,
        spec: &RopeSpec,
        rotary_dim: usize,
    ) -> Result<()> {
        let x = as_f32_mut(&mut x.0)?;
        shape::rope_partial(x.len(), spec, rotary_dim)?;
        let half = rotary_dim / 2;
        for head in x.chunks_exact_mut(spec.head_dim) {
            for i in 0..half {
                let angle = pos as f32 * spec.theta.powf(-2.0 * i as f32 / rotary_dim as f32);
                let (s, c) = angle.sin_cos();
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * c - b * s;
                head[i + half] = a * s + b * c;
            }
        }
        Ok(())
    }

    fn sigmoid_mul(&self, gate: &CpuBuf, x: &CpuBuf, y: &mut CpuBuf) -> Result<()> {
        let (gate, x, y) = (as_f32(&gate.0)?, as_f32(&x.0)?, as_f32_mut(&mut y.0)?);
        shape::elementwise("sigmoid_mul", &[gate.len(), x.len()], &[y.len()])?;
        for ((g, x), y) in gate.iter().zip(x).zip(y) {
            *y = x / (1.0 + (-g).exp());
        }
        Ok(())
    }

    fn ssm_conv_silu(
        &self,
        w: &CpuBuf,
        state: &mut CpuBuf,
        x: &CpuBuf,
        y: &mut CpuBuf,
    ) -> Result<()> {
        let (w, state) = (as_f32(&w.0)?, as_f32_mut(&mut state.0)?);
        let (x, y) = (as_f32(&x.0)?, as_f32_mut(&mut y.0)?);
        let (_, taps) = shape::ssm_conv(w.len(), state.len(), x.len(), y.len())?;
        for (((taps, past), &x), y) in w
            .chunks_exact(taps)
            .zip(state.chunks_exact_mut(taps - 1))
            .zip(x)
            .zip(y)
        {
            let sum = dot(past, taps) + x * taps[taps.len() - 1];
            past.copy_within(1.., 0);
            past[past.len() - 1] = x;
            *y = sum / (1.0 + (-sum).exp());
        }
        Ok(())
    }

    fn gated_delta_step(
        &self,
        spec: &DeltaSpec,
        inputs: &DeltaInputs<CpuBuf>,
        state: &mut CpuBuf,
        o: &mut CpuBuf,
    ) -> Result<()> {
        let qkv = as_f32(&inputs.qkv.0)?;
        let (alpha, beta) = (as_f32(&inputs.alpha.0)?, as_f32(&inputs.beta.0)?);
        let (dt_bias, a) = (as_f32(&inputs.dt_bias.0)?, as_f32(&inputs.a.0)?);
        let (state, o) = (as_f32_mut(&mut state.0)?, as_f32_mut(&mut o.0)?);
        shape::gated_delta(
            spec,
            qkv.len(),
            [alpha.len(), beta.len(), dt_bias.len(), a.len()],
            state.len(),
            o.len(),
        )?;
        let d = spec.head_dim;
        let (q, kv) = qkv.split_at(spec.n_k_heads * d);
        let (k, v) = kv.split_at(spec.n_k_heads * d);
        let l2 = |x: &[f32]| {
            let inv = 1.0 / (dot(x, x) + spec.eps).sqrt();
            x.iter().map(|v| v * inv).collect::<Vec<_>>()
        };
        let q: Vec<_> = q.chunks_exact(d).map(l2).collect();
        let k: Vec<_> = k.chunks_exact(d).map(l2).collect();
        let scale = 1.0 / (d as f32).sqrt();
        state
            .par_chunks_exact_mut(d * d)
            .zip(o[..spec.n_v_heads * d].par_chunks_exact_mut(d))
            .zip(v.par_chunks_exact(d))
            .enumerate()
            .for_each(|(h, ((s, o), v))| {
                let (q, k) = (&q[h % spec.n_k_heads], &k[h % spec.n_k_heads]);
                let x = alpha[h] + dt_bias[h];
                let softplus = if x > 20.0 { x } else { x.exp().ln_1p() };
                let decay = (softplus * a[h]).exp();
                let beta = 1.0 / (1.0 + (-beta[h]).exp());
                // Each value row of the state updates from itself and k alone.
                for ((row, v), o) in s.chunks_exact_mut(d).zip(v).zip(o) {
                    for s in row.iter_mut() {
                        *s *= decay;
                    }
                    let delta = (v - dot(row, k)) * beta;
                    for (s, k) in row.iter_mut().zip(k) {
                        *s += k * delta;
                    }
                    *o = dot(row, q) * scale;
                }
            });
        Ok(())
    }

    fn supports_weight_dtype(&self, _dtype: GgmlDType) -> bool {
        true
    }

    fn alloc(&self, bytes: usize, _pool: Pool) -> Result<Self::Buf> {
        Ok(CpuBuf(vec![0u8; bytes]))
    }

    fn upload(&self, dst: &mut Self::Buf, offset: usize, src: &[u8]) -> Result<()> {
        shape::range("upload", offset, src.len(), dst.0.len())?;
        dst.0[offset..offset + src.len()].copy_from_slice(src);
        Ok(())
    }

    fn download(&self, src: &Self::Buf, offset: usize, dst: &mut [u8]) -> Result<()> {
        shape::range("download", offset, dst.len(), src.0.len())?;
        dst.copy_from_slice(&src.0[offset..offset + dst.len()]);
        Ok(())
    }

    fn copy(
        &self,
        src: &Self::Buf,
        src_off: usize,
        dst: &mut Self::Buf,
        dst_off: usize,
        len: usize,
    ) -> Result<()> {
        shape::range("copy", src_off, len, src.0.len())?;
        shape::range("copy", dst_off, len, dst.0.len())?;
        dst.0[dst_off..dst_off + len].copy_from_slice(&src.0[src_off..src_off + len]);
        Ok(())
    }

    /// x/y are [n_tokens, n_in]/[n_tokens, n_out] row-major; n_tokens is
    /// inferred from x's length.
    fn qmatmul(&self, w: &QWeight<Self::Buf>, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        qmatmul_in(matmul(), w, as_f32(&x.0)?, as_f32_mut(&mut y.0)?)
    }

    fn embed(
        &self,
        table: &QWeight<Self::Buf>,
        token_ids: &[u32],
        out: &mut Self::Buf,
    ) -> Result<()> {
        let out_f32 = as_f32_mut(&mut out.0)?;
        let shape::Embed { n_embd, .. } = shape::embed(&table.shape, token_ids, out_f32.len())?;
        let mut row_f32 = vec![0f32; n_embd];
        for (t, &token_id) in token_ids.iter().enumerate() {
            dequant_row_at(table, token_id as usize, n_embd, &mut row_f32)?;
            out_f32[t * n_embd..(t + 1) * n_embd].copy_from_slice(&row_f32);
        }
        Ok(())
    }

    /// x/y are [n_tokens, dim]; dim = w.len(). RMS is computed per token
    /// over `dim`, never over the whole buffer.
    fn rmsnorm(&self, w: &Self::Buf, x: &Self::Buf, y: &mut Self::Buf, eps: f32) -> Result<()> {
        let w_f32 = as_f32(&w.0)?;
        let x_f32 = as_f32(&x.0)?;
        let y_f32 = as_f32_mut(&mut y.0)?;
        let dim = w_f32.len();
        shape::rows("rmsnorm", dim, x_f32.len(), y_f32.len())?;
        for (row, out) in x_f32.chunks_exact(dim).zip(y_f32.chunks_exact_mut(dim)) {
            let mean_sq = dot(row, row) / dim as f32;
            let scale = 1.0 / (mean_sq + eps).sqrt();
            for i in 0..dim {
                out[i] = row[i] * scale * w_f32[i];
            }
        }
        Ok(())
    }

    fn rope(
        &self,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        pos: u32,
        n_tokens: usize,
        spec: &RopeSpec,
    ) -> Result<()> {
        let q_f32 = as_f32_mut(&mut q.0)?;
        let k_f32 = as_f32_mut(&mut k.0)?;
        shape::rope(spec, q_f32.len(), k_f32.len(), n_tokens)?;
        apply_rope(q_f32, pos, n_tokens, spec);
        apply_rope(k_f32, pos, n_tokens, spec);
        Ok(())
    }

    fn add(&self, a: &Self::Buf, b: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let a_f32 = as_f32(&a.0)?;
        let b_f32 = as_f32(&b.0)?;
        let y_f32 = as_f32_mut(&mut y.0)?;
        shape::elementwise("add", &[a_f32.len(), b_f32.len()], &[y_f32.len()])?;
        for i in 0..a_f32.len() {
            y_f32[i] = a_f32[i] + b_f32[i];
        }
        Ok(())
    }

    /// q/o are [n_q, n_q_heads, head_dim]; n_q is inferred from q's
    /// length. kv.len is the *total* keys already written (prefix +
    /// this batch); prefix = kv.len - n_q. Query i attends to keys
    /// [0, prefix+i] inclusive -- standard causal masking, with the
    /// already-cached prefix always fully visible. n_q==1 (decode):
    /// the one query attends to all cached keys.
    fn attn(
        &self,
        q: &Self::Buf,
        kv: &KvView<Self::Buf>,
        spec: &AttnSpec,
        o: &mut Self::Buf,
    ) -> Result<()> {
        self.attention(q, kv, spec, 0, 1.0 / (spec.head_dim as f32).sqrt(), o)
    }

    fn silu_mul(&self, gate: &Self::Buf, up: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        let gate_f32 = as_f32(&gate.0)?;
        let up_f32 = as_f32(&up.0)?;
        let y_f32 = as_f32_mut(&mut y.0)?;
        shape::elementwise("silu_mul", &[gate_f32.len(), up_f32.len()], &[y_f32.len()])?;
        for i in 0..gate_f32.len() {
            let g = gate_f32[i];
            let silu = g / (1.0 + (-g).exp());
            y_f32[i] = silu * up_f32[i];
        }
        Ok(())
    }

    fn layernorm(
        &self,
        w: &Self::Buf,
        b: Option<&Self::Buf>,
        x: &Self::Buf,
        y: &mut Self::Buf,
        eps: f32,
    ) -> Result<()> {
        let b = b.map(|b| as_f32(&b.0)).transpose()?;
        encoder::layernorm(as_f32(&w.0)?, b, as_f32(&x.0)?, as_f32_mut(&mut y.0)?, eps)
    }

    fn add_bias(&self, b: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        encoder::add_bias(as_f32(&b.0)?, as_f32_mut(&mut y.0)?)
    }

    fn gelu(&self, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        encoder::map(
            "gelu",
            as_f32(&x.0)?,
            as_f32_mut(&mut y.0)?,
            encoder::gelu_scalar,
        )
    }

    fn gelu_mul(&self, a: &Self::Buf, gate: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        encoder::gelu_mul(as_f32(&a.0)?, as_f32(&gate.0)?, as_f32_mut(&mut y.0)?)
    }

    fn relu(&self, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        encoder::map("relu", as_f32(&x.0)?, as_f32_mut(&mut y.0)?, |v| v.max(0.0))
    }

    fn rope_neox(
        &self,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        spec: &EncoderSpec,
        theta: f32,
    ) -> Result<()> {
        let q_f32 = as_f32_mut(&mut q.0)?;
        let k_f32 = as_f32_mut(&mut k.0)?;
        shape::rope_neox(spec, q_f32.len(), k_f32.len())?;
        encoder::rope_neox(q_f32, spec, theta);
        encoder::rope_neox(k_f32, spec, theta);
        Ok(())
    }

    fn attn_bidir(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        lens: &[u32],
        spec: &EncoderSpec,
        o: &mut Self::Buf,
    ) -> Result<()> {
        encoder::attn_bidir(
            as_f32(&q.0)?,
            as_f32(&k.0)?,
            as_f32(&v.0)?,
            lens,
            spec,
            as_f32_mut(&mut o.0)?,
        )
    }
}

impl CpuBackend {
    #[expect(clippy::unused_self, reason = "backend operations are methods")]
    fn attention(
        &self,
        q: &CpuBuf,
        kv: &KvView<CpuBuf>,
        spec: &AttnSpec,
        window: usize,
        scale: f32,
        o: &mut CpuBuf,
    ) -> Result<()> {
        let q_f32 = as_f32(&q.0)?;
        let k_f32 = as_f32(&kv.k.0)?;
        let v_f32 = as_f32(&kv.v.0)?;
        let o_f32 = as_f32_mut(&mut o.0)?;
        let shape::Attn { n_q, prefix } = shape::attn(
            spec,
            q_f32.len(),
            kv.len,
            k_f32.len(),
            v_f32.len(),
            o_f32.len(),
        )?;

        let per_token = spec.n_q_heads * spec.head_dim;
        let group = spec.n_q_heads / spec.n_kv_heads;
        let kv_row = spec.n_kv_heads * spec.head_dim;

        o_f32[..n_q * per_token].fill(0.0);
        for i in 0..n_q {
            let visible = prefix + i + 1; // exclusive upper bound, causal
            let first = if window == 0 {
                0
            } else {
                visible.saturating_sub(window)
            };
            let mut scores = vec![0f32; visible - first];
            for qh in 0..spec.n_q_heads {
                let kvh = qh / group;
                let q_start = (i * spec.n_q_heads + qh) * spec.head_dim;
                let q_vec = &q_f32[q_start..q_start + spec.head_dim];

                for (offset, score) in scores.iter_mut().enumerate() {
                    let t = first + offset;
                    let k_vec = &k_f32
                        [t * kv_row + kvh * spec.head_dim..t * kv_row + (kvh + 1) * spec.head_dim];
                    *score = dot(q_vec, k_vec) * scale;
                }
                softmax_inplace(&mut scores);

                let o_vec = &mut o_f32[q_start..q_start + spec.head_dim];
                for (offset, &weight) in scores.iter().enumerate() {
                    let t = first + offset;
                    let v_vec = &v_f32
                        [t * kv_row + kvh * spec.head_dim..t * kv_row + (kvh + 1) * spec.head_dim];
                    for (o_val, v_val) in o_vec.iter_mut().zip(v_vec) {
                        *o_val += weight * v_val;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

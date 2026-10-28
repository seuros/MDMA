//! Geometry rules shared by every backend. Each function takes buffer
//! lengths (f32 elements unless named `bytes`) plus the op's spec, and
//! returns the numbers a launch needs. A backend validates through here
//! and never restates a rule in its own words, so the CPU oracle and
//! every GPU port reject exactly the same inputs.
//!
//! Inputs must be exactly whole rows or tokens; an output only has to be
//! large enough. Kernels write nothing past the result, and the GPU
//! suites prove that with oversized outputs and a canary tail.
use super::{AttnSpec, DeltaSpec, EncoderSpec, Error, Result, RopeSpec};

fn err(msg: String) -> Error {
    Error::ShapeMismatch(msg)
}

/// f32 elements held by `bytes`.
pub fn elems(op: &str, bytes: usize) -> Result<usize> {
    if !bytes.is_multiple_of(size_of::<f32>()) {
        return Err(err(format!("{op}: {bytes} bytes is not whole f32s")));
    }
    Ok(bytes / size_of::<f32>())
}

pub fn argmax(n: usize, index_bytes: usize) -> Result<()> {
    if n == 0 || n > u32::MAX as usize || index_bytes != size_of::<u32>() {
        return Err(err("argmax requires a nonempty u32-indexable vector and one u32 output".into()));
    }
    Ok(())
}

pub fn norm_epsilon(eps: f32) -> Result<()> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(err("normalization requires positive finite epsilon".into()));
    }
    Ok(())
}

/// `[offset, offset + len)` lies inside `cap` bytes (upload/download/copy).
pub fn range(op: &str, offset: usize, len: usize, cap: usize) -> Result<()> {
    match offset.checked_add(len) {
        Some(end) if end <= cap => Ok(()),
        _ => Err(err(format!(
            "{op}: {len} bytes at offset {offset} exceeds {cap} byte buffer"
        ))),
    }
}

pub fn rank2(op: &str, shape: &[usize]) -> Result<(usize, usize)> {
    match *shape {
        [rows, cols] => Ok((rows, cols)),
        _ => Err(err(format!(
            "{op} expects a rank-2 weight shape, got {shape:?}"
        ))),
    }
}

pub struct Matmul {
    pub n_out: usize,
    pub n_in: usize,
    pub n_tokens: usize,
}

/// `y[n_tokens, n_out] = x[n_tokens, n_in] · w[n_out, n_in]ᵀ`; n_tokens is
/// inferred from `x`, and every row must be complete.
pub fn matmul(shape: &[usize], x_elems: usize, y_elems: usize) -> Result<Matmul> {
    let (n_out, n_in) = rank2("qmatmul", shape)?;
    if n_in == 0 || n_out == 0 {
        return Err(err(format!("qmatmul: empty weight {shape:?}")));
    }
    if x_elems == 0 || !x_elems.is_multiple_of(n_in) {
        return Err(err(format!(
            "qmatmul: x has {x_elems} elems, not whole rows of n_in {n_in}"
        )));
    }
    let n_tokens = x_elems / n_in;
    if y_elems < n_tokens * n_out {
        return Err(err(format!(
            "qmatmul: y has {y_elems} elems, needs {}",
            n_tokens * n_out
        )));
    }
    Ok(Matmul {
        n_out,
        n_in,
        n_tokens,
    })
}

/// K-quant kernels walk 256-wide superblocks along the reduced dimension.
pub fn kquant_aligned(op: &str, dtype: crate::gguf::GgmlDType, n: usize) -> Result<()> {
    use crate::gguf::GgmlDType::{Q4K, Q5K, Q6K};
    if matches!(dtype, Q4K | Q5K | Q6K) && !n.is_multiple_of(256) {
        return Err(err(format!(
            "{op}: K-quant width {n} is not a multiple of 256"
        )));
    }
    Ok(())
}

pub struct Embed {
    pub vocab: usize,
    pub n_embd: usize,
}

/// `out[token_ids.len(), n_embd]` gathered from `table[vocab, n_embd]`.
pub fn embed(shape: &[usize], token_ids: &[u32], out_elems: usize) -> Result<Embed> {
    let (vocab, n_embd) = rank2("embed", shape)?;
    if token_ids.is_empty() {
        return Err(err("embed: no tokens".into()));
    }
    if out_elems < token_ids.len() * n_embd {
        return Err(err(format!(
            "embed: out has {out_elems} elems, needs {}",
            token_ids.len() * n_embd
        )));
    }
    if let Some(id) = token_ids.iter().find(|&&id| id as usize >= vocab) {
        return Err(err(format!(
            "embed: token_id {id} out of range (vocab {vocab})"
        )));
    }
    Ok(Embed { vocab, n_embd })
}

/// `x` is complete rows of `dim` and `y` holds at least as many (norms, a
/// per-row bias). Returns the row count.
pub fn rows(op: &str, dim: usize, x_elems: usize, y_elems: usize) -> Result<usize> {
    if dim == 0 || x_elems == 0 || !x_elems.is_multiple_of(dim) {
        return Err(err(format!(
            "{op}: x has {x_elems} elems, not whole rows of {dim}"
        )));
    }
    if y_elems < x_elems {
        return Err(err(format!("{op}: y has {y_elems} elems, needs {x_elems}")));
    }
    Ok(x_elems / dim)
}

/// Elementwise: every input has the same length `n`, every output holds
/// at least `n`. Returns `n`.
pub fn elementwise(op: &str, inputs: &[usize], outputs: &[usize]) -> Result<usize> {
    let n = inputs.first().copied().unwrap_or(0);
    if inputs.iter().any(|&l| l != n) {
        return Err(err(format!("{op}: input lengths differ: {inputs:?}")));
    }
    if outputs.iter().any(|&l| l < n) {
        return Err(err(format!(
            "{op}: outputs {outputs:?} cannot hold {n} elems"
        )));
    }
    Ok(n)
}

pub struct Rope {
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
}

/// q/k are `[n_tokens, heads, head_dim]` with an even head_dim.
pub fn rope(spec: &RopeSpec, q_elems: usize, k_elems: usize, n_tokens: usize) -> Result<Rope> {
    if n_tokens == 0 || !q_elems.is_multiple_of(n_tokens) || !k_elems.is_multiple_of(n_tokens) {
        return Err(err(format!(
            "rope: q {q_elems} k {k_elems} elems are not whole tokens of n_tokens {n_tokens}"
        )));
    }
    let (q_stride, k_stride) = (q_elems / n_tokens, k_elems / n_tokens);
    if spec.head_dim == 0
        || !spec.head_dim.is_multiple_of(2)
        || !q_stride.is_multiple_of(spec.head_dim)
        || !k_stride.is_multiple_of(spec.head_dim)
    {
        return Err(err(format!(
            "rope: token strides {q_stride}/{k_stride} are not whole even heads of {}",
            spec.head_dim
        )));
    }
    Ok(Rope {
        n_q_heads: q_stride / spec.head_dim,
        n_kv_heads: k_stride / spec.head_dim,
    })
}

/// Complete split-half heads and one frequency divisor per pair.
pub fn rope_split_half(n: usize, factors: usize, spec: &RopeSpec) -> Result<()> {
    if spec.head_dim == 0
        || !spec.head_dim.is_multiple_of(2)
        || n == 0
        || !n.is_multiple_of(spec.head_dim)
        || factors != spec.head_dim / 2
        || !spec.theta.is_finite()
        || spec.theta <= 0.0
    {
        return Err(err(
            "split-half RoPE requires complete heads and one factor per pair".into(),
        ));
    }
    Ok(())
}

/// Complete heads with an even, nonempty rotary prefix. Returns rotated pairs.
pub fn rope_partial(n: usize, spec: &RopeSpec, rotary_dim: usize) -> Result<usize> {
    if spec.head_dim == 0
        || !spec.head_dim.is_multiple_of(2)
        || n == 0
        || !n.is_multiple_of(spec.head_dim)
        || rotary_dim == 0
        || !rotary_dim.is_multiple_of(2)
        || rotary_dim > spec.head_dim
        || !spec.theta.is_finite()
        || spec.theta <= 0.0
    {
        return Err(err(
            "partial RoPE requires complete heads and an even rotary prefix within head width"
                .into(),
        ));
    }
    Ok((n / spec.head_dim) * (rotary_dim / 2))
}

/// One token through `channels` causal depthwise filters of `taps` inputs:
/// `w` holds `[channels, taps]`, `state` the previous `taps - 1` inputs per
/// channel, `x` one input per channel. Returns `(channels, taps)`.
pub fn ssm_conv(
    w_elems: usize,
    state_elems: usize,
    x_elems: usize,
    y_elems: usize,
) -> Result<(usize, usize)> {
    let channels = x_elems;
    if channels == 0 || w_elems == 0 || !w_elems.is_multiple_of(channels) {
        return Err(err(format!(
            "ssm_conv: {w_elems} taps are not whole filters for {channels} channels"
        )));
    }
    let taps = w_elems / channels;
    if taps < 2 || state_elems != channels * (taps - 1) || y_elems < channels {
        return Err(err(format!(
            "ssm_conv: {taps}-tap filters need {} state and {channels} output elems, got {state_elems} and {y_elems}",
            channels * (taps.max(1) - 1)
        )));
    }
    Ok((channels, taps))
}

/// One token of the gated delta rule: `qkv` is `[q | k | v]` for
/// `spec`'s heads, `gates` the `alpha`, `beta`, `dt_bias` and `a` lengths,
/// `state` one `head_dim`² matrix per value head.
pub fn gated_delta(
    spec: &DeltaSpec,
    qkv_elems: usize,
    gates: [usize; 4],
    state_elems: usize,
    o_elems: usize,
) -> Result<()> {
    let DeltaSpec {
        n_k_heads,
        n_v_heads,
        head_dim,
        eps,
    } = *spec;
    let width = (2 * n_k_heads + n_v_heads) * head_dim;
    if n_k_heads == 0
        || head_dim == 0
        || !n_v_heads.is_multiple_of(n_k_heads)
        || !eps.is_finite()
        || eps < 0.0
    {
        return Err(err(format!(
            "gated_delta: {n_v_heads} value heads over {n_k_heads} key heads of {head_dim} with eps {eps}"
        )));
    }
    if qkv_elems != width
        || gates.iter().any(|&n| n != n_v_heads)
        || state_elems != n_v_heads * head_dim * head_dim
        || o_elems < n_v_heads * head_dim
    {
        return Err(err(format!(
            "gated_delta: qkv {qkv_elems} (needs {width}), gates {gates:?} (need {n_v_heads}), state {state_elems}, o {o_elems}"
        )));
    }
    Ok(())
}

pub fn softcap(cap: f32) -> Result<()> {
    if !cap.is_finite() || cap <= 0.0 {
        return Err(err("softcap must be positive and finite".into()));
    }
    Ok(())
}

pub fn attention_scale(scale: f32) -> Result<()> {
    if !scale.is_finite() {
        return Err(err("attention scale must be finite".into()));
    }
    Ok(())
}

pub struct Attn {
    pub n_q: usize,
    pub prefix: usize,
}

/// Causal attention of `q[n_q, n_q_heads, head_dim]` over `kv_len` cached
/// keys. `k_elems`/`v_elems` are counted in the cache's own element size.
pub fn attn(
    spec: &AttnSpec,
    q_elems: usize,
    kv_len: usize,
    k_elems: usize,
    v_elems: usize,
    o_elems: usize,
) -> Result<Attn> {
    let per = spec.n_q_heads * spec.head_dim;
    if per == 0 || q_elems == 0 || !q_elems.is_multiple_of(per) {
        return Err(err(format!(
            "attn: q has {q_elems} elems, not whole tokens of n_q_heads*head_dim {per}"
        )));
    }
    if spec.n_kv_heads == 0 || !spec.n_q_heads.is_multiple_of(spec.n_kv_heads) {
        return Err(err(format!(
            "attn: {} q heads are not a multiple of {} KV heads",
            spec.n_q_heads, spec.n_kv_heads
        )));
    }
    let n_q = q_elems / per;
    if n_q > kv_len {
        return Err(err(format!("attn: n_q {n_q} exceeds kv.len {kv_len}")));
    }
    let need = kv_len * spec.n_kv_heads * spec.head_dim;
    if k_elems < need || v_elems < need {
        return Err(err(format!(
            "attn: K {k_elems} V {v_elems} elems for kv.len {kv_len}, need {need}"
        )));
    }
    if o_elems < q_elems {
        return Err(err(format!("attn: o has {o_elems} elems, needs {q_elems}")));
    }
    Ok(Attn {
        n_q,
        prefix: kv_len - n_q,
    })
}

/// Sequences in a `[batch, seq_len, n_heads, head_dim]` buffer.
pub fn encoder_batch(op: &str, spec: &EncoderSpec, elems: usize) -> Result<usize> {
    let per_seq = spec.seq_len * spec.n_heads * spec.head_dim;
    if per_seq == 0 || elems == 0 || !elems.is_multiple_of(per_seq) {
        return Err(err(format!(
            "{op}: {elems} elems are not whole sequences of {per_seq}"
        )));
    }
    Ok(elems / per_seq)
}

/// `rope_neox` needs the even head_dim `rope` does; positions restart per sequence.
pub fn rope_neox(spec: &EncoderSpec, q_elems: usize, k_elems: usize) -> Result<usize> {
    if !spec.head_dim.is_multiple_of(2) {
        return Err(err(format!("rope_neox: head_dim {} is odd", spec.head_dim)));
    }
    elementwise("rope_neox", &[q_elems, k_elems], &[])?;
    encoder_batch("rope_neox", spec, q_elems)
}

/// q/k/v share one layout and `o` holds it; `lens` gives each sequence's
/// valid prefix.
pub fn attn_bidir(
    spec: &EncoderSpec,
    lens: &[u32],
    q_elems: usize,
    k_elems: usize,
    v_elems: usize,
    o_elems: usize,
) -> Result<usize> {
    elementwise("attn_bidir", &[q_elems, k_elems, v_elems], &[o_elems])?;
    let batch = encoder_batch("attn_bidir", spec, q_elems)?;
    if batch != lens.len() {
        return Err(err(format!(
            "attn_bidir: {batch} sequences but {} lengths",
            lens.len()
        )));
    }
    if let Some(len) = lens.iter().find(|&&l| l as usize > spec.seq_len) {
        return Err(err(format!(
            "attn_bidir: len {len} exceeds seq_len {}",
            spec.seq_len
        )));
    }
    Ok(batch)
}

/// Rows of `dim` with an optional bias of the same width.
pub fn layernorm(dim: usize, bias: Option<usize>, x_elems: usize, y_elems: usize) -> Result<usize> {
    if bias.is_some_and(|b| b != dim) {
        return Err(err(format!(
            "layernorm: bias has {bias:?} elems, weight {dim}"
        )));
    }
    rows("layernorm", dim, x_elems, y_elems)
}

#[cfg(test)]
mod tests;

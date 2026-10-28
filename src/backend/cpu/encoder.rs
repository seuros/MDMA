// Reference encoder ops: LayerNorm, GELU/ReLU, split-half RoPE and
// bidirectional windowed attention over right-padded batches. Scalar,
// f32 activations and accumulation.
use super::softmax_inplace;
use crate::backend::{EncoderSpec, Result, shape};

pub(super) fn layernorm(
    w: &[f32],
    b: Option<&[f32]>,
    x: &[f32],
    y: &mut [f32],
    eps: f32,
) -> Result<()> {
    let dim = w.len();
    shape::norm_epsilon(eps)?;
    shape::layernorm(dim, b.map(<[f32]>::len), x.len(), y.len())?;
    for (row, out) in x.chunks_exact(dim).zip(y.chunks_exact_mut(dim)) {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let scale = 1.0 / (var + eps).sqrt();
        for i in 0..dim {
            out[i] = (row[i] - mean) * scale * w[i] + b.map_or(0.0, |b| b[i]);
        }
    }
    Ok(())
}

pub(super) fn add_bias(b: &[f32], y: &mut [f32]) -> Result<()> {
    shape::rows("add_bias", b.len(), y.len(), y.len())?;
    for row in y.chunks_exact_mut(b.len()) {
        for (v, b) in row.iter_mut().zip(b) {
            *v += b;
        }
    }
    Ok(())
}

/// Exact GELU via a polynomial erf (Abramowitz–Stegun 7.1.26, |err| < 1.5e-7).
pub(super) fn gelu_scalar(x: f32) -> f32 {
    let z = f64::from(x) / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z.abs());
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erf = (1.0 - poly * (-z * z).exp()).copysign(z);
    (0.5 * f64::from(x) * (1.0 + erf)) as f32
}

pub(super) fn map(op: &str, x: &[f32], y: &mut [f32], f: impl Fn(f32) -> f32) -> Result<()> {
    shape::elementwise(op, &[x.len()], &[y.len()])?;
    for (o, &v) in y.iter_mut().zip(x) {
        *o = f(v);
    }
    Ok(())
}

pub(super) fn gelu_mul(a: &[f32], gate: &[f32], y: &mut [f32]) -> Result<()> {
    shape::elementwise("gelu_mul", &[a.len(), gate.len()], &[y.len()])?;
    for i in 0..a.len() {
        y[i] = gelu_scalar(a[i]) * gate[i];
    }
    Ok(())
}

/// Rows of `n_heads * head_dim`; row r sits at position `r % seq_len`.
/// Pair (i, i + half) rotates together (HF `rotate_half`). Geometry is
/// checked by the caller through `shape::rope_neox`.
pub(super) fn rope_neox(buf: &mut [f32], spec: &EncoderSpec, theta: f32) {
    let row = spec.n_heads * spec.head_dim;
    let half = spec.head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| theta.powf(-((2 * i) as f32) / spec.head_dim as f32))
        .collect();
    for (r, token) in buf.chunks_exact_mut(row).enumerate() {
        let pos = (r % spec.seq_len) as f32;
        for head in token.chunks_exact_mut(spec.head_dim) {
            for i in 0..half {
                let (sin, cos) = (pos * inv_freq[i]).sin_cos();
                let (x0, x1) = (head[i], head[i + half]);
                head[i] = x0 * cos - x1 * sin;
                head[i + half] = x1 * cos + x0 * sin;
            }
        }
    }
}

pub(super) fn attn_bidir(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    lens: &[u32],
    spec: &EncoderSpec,
    o: &mut [f32],
) -> Result<()> {
    shape::attn_bidir(spec, lens, q.len(), k.len(), v.len(), o.len())?;
    let row = spec.n_heads * spec.head_dim;
    let per_seq = row * spec.seq_len;
    let scale = 1.0 / (spec.head_dim as f32).sqrt();
    o[..per_seq * lens.len()].fill(0.0);
    let mut scores = vec![0f32; spec.seq_len];
    for (b, &len) in lens.iter().enumerate() {
        let len = len as usize;
        let base = b * per_seq;
        for i in 0..len {
            let (lo, hi) = match spec.window {
                Some(w) => (i.saturating_sub(w), (i + w + 1).min(len)),
                None => (0, len),
            };
            for h in 0..spec.n_heads {
                let at = |t: usize| base + t * row + h * spec.head_dim;
                let q_vec = &q[at(i)..at(i) + spec.head_dim];
                let scores = &mut scores[lo..hi];
                for (t, s) in (lo..hi).zip(scores.iter_mut()) {
                    *s = q_vec
                        .iter()
                        .zip(&k[at(t)..at(t) + spec.head_dim])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale;
                }
                softmax_inplace(scores);
                let o_vec = &mut o[at(i)..at(i) + spec.head_dim];
                for (t, &w) in (lo..hi).zip(scores.iter()) {
                    for (o_val, v_val) in o_vec.iter_mut().zip(&v[at(t)..at(t) + spec.head_dim]) {
                        *o_val += w * v_val;
                    }
                }
            }
        }
    }
    Ok(())
}

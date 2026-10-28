//! Quantized weights multiplied in their own blocks: activations are
//! quantized once per matmul to the weight type's dot type (Q8_K for the
//! k-quants, Q8_0 or Q8_1 otherwise) and Candle's packed kernels take each
//! row. Faster than dequantizing every row to f32, and not the same numbers.
use super::CpuBuf;
use crate::backend::QWeight;
use crate::gguf::GgmlDType;
use candle_core::quantized::GgmlType;
use candle_core::quantized::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8_1, BlockQ8K,
};
use rayon::prelude::*;

/// `y = w x` for `n_tokens` rows of `x`, when `w`'s type has a packed dot and
/// its buffer holds whole aligned blocks; `false` leaves it to the exact path.
pub(super) fn matmul(
    w: &QWeight<CpuBuf>,
    x: &[f32],
    y: &mut [f32],
    n_out: usize,
    n_in: usize,
) -> bool {
    let shape = (n_out, n_in);
    match w.dtype {
        GgmlDType::Q4_0 => rows::<BlockQ4_0>(&w.buf.0, x, y, shape),
        GgmlDType::Q4_1 => rows::<BlockQ4_1>(&w.buf.0, x, y, shape),
        GgmlDType::Q5_0 => rows::<BlockQ5_0>(&w.buf.0, x, y, shape),
        GgmlDType::Q5_1 => rows::<BlockQ5_1>(&w.buf.0, x, y, shape),
        GgmlDType::Q8_0 => rows::<BlockQ8_0>(&w.buf.0, x, y, shape),
        GgmlDType::Q8_1 => rows::<BlockQ8_1>(&w.buf.0, x, y, shape),
        GgmlDType::Q2K => rows::<BlockQ2K>(&w.buf.0, x, y, shape),
        GgmlDType::Q3K => rows::<BlockQ3K>(&w.buf.0, x, y, shape),
        GgmlDType::Q4K => rows::<BlockQ4K>(&w.buf.0, x, y, shape),
        GgmlDType::Q5K => rows::<BlockQ5K>(&w.buf.0, x, y, shape),
        GgmlDType::Q6K => rows::<BlockQ6K>(&w.buf.0, x, y, shape),
        GgmlDType::Q8K => rows::<BlockQ8K>(&w.buf.0, x, y, shape),
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q1_0 => false,
    }
}

fn rows<T: GgmlType>(
    bytes: &[u8],
    x: &[f32],
    y: &mut [f32],
    (n_out, n_in): (usize, usize),
) -> bool {
    let dot = T::VecDotType::BLCK_SIZE;
    if n_in == 0 || !n_in.is_multiple_of(T::BLCK_SIZE) || !n_in.is_multiple_of(dot) {
        return false;
    }
    let Some(weights) = blocks::<T>(bytes, n_out * (n_in / T::BLCK_SIZE)) else {
        return false;
    };
    let n_tokens = x.len() / n_in;
    let (per_row, per_token) = (n_in / T::BLCK_SIZE, n_in / dot);
    let mut quantized = vec![T::VecDotType::zeros(); per_token * n_tokens];
    for (x, q) in x
        .chunks_exact(n_in)
        .zip(quantized.chunks_exact_mut(per_token))
    {
        T::VecDotType::from_float(x, q);
    }
    let row = |r: usize| &weights[r * per_row..(r + 1) * per_row];
    if n_tokens == 1 {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(r, out)| *out = T::vec_dot(n_in, row(r), &quantized));
        return true;
    }
    // [n_out, n_tokens] per row, then transposed, as the exact path does.
    let mut yt = vec![0f32; n_out * n_tokens];
    yt.par_chunks_mut(n_tokens)
        .enumerate()
        .for_each(|(r, out)| {
            for (out, q) in out.iter_mut().zip(quantized.chunks_exact(per_token)) {
                *out = T::vec_dot(n_in, row(r), q);
            }
        });
    for t in 0..n_tokens {
        for r in 0..n_out {
            y[t * n_out + r] = yt[r * n_tokens + t];
        }
    }
    true
}

/// The buffer's first `count` blocks, when it holds them and is aligned.
#[allow(unsafe_code, clippy::cast_ptr_alignment)]
fn blocks<T: GgmlType>(bytes: &[u8], count: usize) -> Option<&[T]> {
    let bytes = bytes.get(..count.checked_mul(size_of::<T>())?)?;
    if !bytes.as_ptr().cast::<T>().is_aligned() {
        return None;
    }
    // SAFETY: Candle's quantized blocks are `repr(C)` structs of f16, u8 and
    // i8 fields, so any bytes form a valid value; the length and alignment
    // are checked above, and the slice borrows `bytes` for its lifetime.
    Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<T>(), count) })
}

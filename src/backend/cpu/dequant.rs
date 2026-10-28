//! GGUF block dequantization.
use crate::backend::{Error, Result};
use crate::gguf::GgmlDType;
use candle_core::quantized::GgmlType;
use candle_core::quantized::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1, BlockQ5K, BlockQ6K,
    BlockQ8_0, BlockQ8_1, BlockQ8K,
};
use half::{bf16, f16};

// Alignment is checked before each block reinterpretation below.
#[allow(unsafe_code, clippy::cast_ptr_alignment)]
pub fn dequantize_row(dtype: GgmlDType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    macro_rules! dequant {
        ($block:ty) => {{
            let block_bytes = std::mem::size_of::<$block>();
            let n = bytes.len() / block_bytes;
            if n * block_bytes != bytes.len() {
                return Err(Error::ShapeMismatch(format!(
                    "{dtype:?} row is {} bytes, not a multiple of block size {block_bytes}",
                    bytes.len()
                )));
            }
            // Row slices start at arbitrary offsets of a Vec<u8>, which only
            // guarantees byte alignment.
            if !bytes.as_ptr().cast::<$block>().is_aligned() {
                return Err(Error::ShapeMismatch(format!(
                    "{dtype:?} row is not aligned to {} bytes",
                    std::mem::align_of::<$block>()
                )));
            }
            // SAFETY: $block is #[repr(C)] plain old data, the pointer is
            // aligned for it (checked above), and bytes.len() is exactly
            // n * size_of::<$block>(), so the slice covers only borrowed bytes.
            let blocks: &[$block] = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), n) };
            <$block as GgmlType>::to_float(blocks, out);
        }};
    }

    match dtype {
        GgmlDType::Q1_0 => {
            if !out.len().is_multiple_of(128) || bytes.len() != out.len() / 128 * 18 {
                return Err(Error::ShapeMismatch(
                    "Q1_0 requires 18 bytes per 128 weights".into(),
                ));
            }
            for (block, row) in bytes
                .as_chunks::<18>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<128>().0)
            {
                let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                if !scale.is_finite() {
                    return Err(Error::ShapeMismatch("Q1_0 scale must be finite".into()));
                }
                for (index, value) in row.iter_mut().enumerate() {
                    *value = if block[2 + index / 8] & (1 << (index % 8)) == 0 {
                        -scale
                    } else {
                        scale
                    };
                }
            }
        }
        GgmlDType::F32 => dequant!(f32),
        GgmlDType::F16 => dequant!(f16),
        GgmlDType::BF16 => dequant!(bf16),
        GgmlDType::Q4_0 => dequant!(BlockQ4_0),
        GgmlDType::Q4_1 => dequant!(BlockQ4_1),
        GgmlDType::Q5_0 => dequant!(BlockQ5_0),
        GgmlDType::Q5_1 => dequant!(BlockQ5_1),
        GgmlDType::Q8_0 => dequant!(BlockQ8_0),
        GgmlDType::Q8_1 => dequant!(BlockQ8_1),
        GgmlDType::Q2K => dequant!(BlockQ2K),
        GgmlDType::Q3K => dequant!(BlockQ3K),
        GgmlDType::Q4K => dequant!(BlockQ4K),
        GgmlDType::Q5K => dequant!(BlockQ5K),
        GgmlDType::Q6K => dequant!(BlockQ6K),
        GgmlDType::Q8K => dequant!(BlockQ8K),
    }
    Ok(())
}

#[cfg(test)]
mod tests;

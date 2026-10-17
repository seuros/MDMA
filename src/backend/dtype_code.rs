//! The weight dtype tag GPU kernels switch on. One numbering for every
//! backend (`kernels.cu`, `kernels.hip`, the GLSL and the OpenCL sources
//! share the first four values; decoder-capable ports add BF16 and Q5_K).
use super::{Error, Result};
use crate::gguf::GgmlDType;

mod tags;
pub use tags::*;

/// Tags supported by the original four-format GPU kernels.
pub fn code(dtype: GgmlDType) -> Result<i32> {
    match dtype {
        GgmlDType::F32 => Ok(F32),
        GgmlDType::F16 => Ok(F16),
        GgmlDType::Q4K => Ok(Q4K),
        GgmlDType::Q6K => Ok(Q6K),
        GgmlDType::BF16 => Err(Error::NotImplemented("BF16 weights on GPU")),
        GgmlDType::Q4_0 => Err(Error::NotImplemented("Q4_0 weights on GPU")),
        GgmlDType::Q4_1 => Err(Error::NotImplemented("Q4_1 weights on GPU")),
        GgmlDType::Q5_0 => Err(Error::NotImplemented("Q5_0 weights on GPU")),
        GgmlDType::Q5_1 => Err(Error::NotImplemented("Q5_1 weights on GPU")),
        GgmlDType::Q8_0 => Err(Error::NotImplemented("Q8_0 weights on GPU")),
        GgmlDType::Q8_1 => Err(Error::NotImplemented("Q8_1 weights on GPU")),
        GgmlDType::Q2K => Err(Error::NotImplemented("Q2K weights on GPU")),
        GgmlDType::Q3K => Err(Error::NotImplemented("Q3K weights on GPU")),
        GgmlDType::Q5K => Err(Error::NotImplemented("Q5K weights on GPU")),
        GgmlDType::Q8K => Err(Error::NotImplemented("Q8K weights on GPU")),
    }
}

pub fn is_kquant(code: i32) -> bool {
    matches!(code, Q4K | Q5K | Q6K)
}

/// Extended tags for ports implementing Gemma's weight formats.
pub fn decoder_code(dtype: GgmlDType) -> Result<i32> {
    match dtype {
        GgmlDType::BF16 => Ok(BF16),
        GgmlDType::Q5K => Ok(Q5K),
        GgmlDType::Q8_0 => Ok(Q8_0),
        _ => code(dtype),
    }
}

/// Bytes per 256-element superblock of a K-quant, 0 for float weights.
pub fn block_bytes(dtype: GgmlDType) -> i32 {
    match dtype {
        GgmlDType::Q4K => 144,
        GgmlDType::Q5K => 176,
        GgmlDType::Q6K => 210,
        _ => 0,
    }
}

#[cfg(test)]
mod tests;

//! Static Qwen3 YaRN, matching Prism ggml c96baf4
//! (beta_fast=32, beta_slow=1).
use crate::backend::Result;
use crate::gguf::GgufModel;
use crate::model::ModelConfig;
use crate::model::preflight::{positive, positive_f32, unsupported};

const PREFIX: &str = "rope.scaling.";
const KEYS: &[&str] = &[
    "rope.scaling.type",
    "rope.scaling.factor",
    "rope.scaling.original_context_length",
];

pub(super) fn is_key(suffix: &str) -> bool {
    KEYS.contains(&suffix)
}

pub(super) struct Yarn {
    factor: f32,
    original_context: usize,
}

impl Yarn {
    pub fn read(g: &GgufModel, arch: &str, c: &ModelConfig) -> Result<Option<Self>> {
        let prefix = format!("{arch}.{PREFIX}");
        if !g.metadata_keys().any(|k| k.starts_with(&prefix)) {
            return Ok(None);
        }
        let key = format!("{arch}.{PREFIX}type");
        if arch != "qwen3" || g.metadata_str(&key) != Some("yarn") {
            return Err(unsupported(format!(
                "{key}: only static Qwen3 yarn is implemented"
            )));
        }
        let factor = positive_f32(g, arch, "rope.scaling.factor")?;
        let original_context = positive(g, arch, "rope.scaling.original_context_length")?;
        if factor < 1.0 || original_context > c.n_ctx || c.rope_theta <= 1.0 {
            return Err(unsupported(
                "invalid Qwen3 YaRN factor, original context or RoPE base",
            ));
        }
        Ok(Some(Self {
            factor,
            original_context,
        }))
    }

    pub fn amplitude(&self) -> f32 {
        1.0 + 0.1 * self.factor.ln()
    }

    pub fn divisors(&self, c: &ModelConfig) -> Vec<f32> {
        let correction = |rotations: f32| {
            c.head_dim as f32
                * (self.original_context as f32 / (rotations * std::f32::consts::TAU)).ln()
                / (2.0 * c.rope_theta.ln())
        };
        let low = correction(32.0).floor().max(0.0);
        let high = correction(1.0).ceil().min((c.head_dim - 1) as f32);
        (0..c.head_dim / 2)
            .map(|i| {
                let extrapolate =
                    1.0 - ((i as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0);
                let frequency_scale = (1.0 - extrapolate) / self.factor + extrapolate;
                1.0 / frequency_scale
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;

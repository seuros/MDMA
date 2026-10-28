//! HybridAgent continuous-input decoding.
use super::{Model, arch::{Decoder, qwen}};
use crate::backend::{Backend, Error, Operation, Result};
use crate::gguf::GgufModel;
use std::path::Path;

mod link;
mod weights;
mod decode;
mod runtime;
pub use decode::{HybridDecoder, LatentLimits, LatentStep, ExitReason};
pub use runtime::{Engine, Session, Step};

pub struct HybridAgent<B: Backend> {
    model: Model<B>,
    link: link::HybridLink<B>,
    entry: u32,
    exit: u32,
}

fn unsupported(message: impl Into<String>) -> Error {
    Error::UnsupportedModel(format!("HybridAgent: {}", message.into()))
}

fn validate_backend<B: Backend>(backend: &B) -> Result<()> {
    for operation in [Operation::LayerNorm, Operation::Gelu, Operation::AddBias, Operation::Argmax] {
        if !backend.supports(operation) {
            return Err(unsupported(format!("backend lacks {operation:?}")));
        }
    }
    if !backend.supports_weight_dtype(crate::gguf::GgmlDType::F32) {
        return Err(unsupported("backend lacks F32 adapter weights"));
    }
    Ok(())
}

impl<B: Backend> HybridAgent<B> {
    /// Load the released HF bundle directly; matrices retain their source dtype.
    pub fn load_bundle(backend: &B, directory: &Path) -> Result<Self> {
        raw_cache_policy(std::env::var_os("MDMA_CACHE").as_deref())?;
        validate_backend(backend)?;
        let json: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(directory.join("config.json"))?
        ).map_err(|e| unsupported(e.to_string()))?;
        let bytes = std::fs::read(directory.join("model.safetensors"))?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes).map_err(|e| unsupported(e.to_string()))?;
        let config = qwen::hf::validate(backend, &json, &tensors)?;
        let (entry, exit) = anchors(directory, config.vocab)?;
        let weights = weights::Weights::load(directory, config.model.n_embd)?;
        if let Some(available) = backend.available_device_bytes()? {
            let need = crate::memory::sum([
                bytes.len(), crate::memory::product(&[config.model.n_embd, config.model.n_embd, 8])?,
                256 * 1024 * 1024,
            ])?;
            if need > available { return Err(Error::InsufficientMemory { requested: need, available }); }
        }
        let model = qwen::hf::load(backend, &tensors, config)?;
        let link = link::HybridLink::upload(backend, weights)?;
        Ok(Self { model, link, entry, exit })
    }

    /// Load a GGUF conversion of the trained bundle, not an ordinary Qwen model.
    pub fn load(backend: &B, backbone: &mut GgufModel, directory: &Path) -> Result<Self> {
        if backbone.metadata_str("general.architecture") != Some("qwen3") {
            return Err(unsupported("only the released Qwen3 backbone is supported"));
        }
        validate_backend(backend)?;
        let config = Model::validate(backend, backbone)?;
        let vocab = backbone.tensor_desc("token_embd.weight").unwrap().shape[0];
        let bundle: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(directory.join("config.json"))?
        ).map_err(|e| unsupported(e.to_string()))?;
        if bundle["model_type"] != "qwen3" {
            return Err(unsupported("backbone/bundle architecture mismatch"));
        }
        for (name, expected) in [
            ("hidden_size", config.n_embd), ("vocab_size", vocab),
            ("num_hidden_layers", config.n_layers), ("intermediate_size", config.n_ff),
            ("num_attention_heads", config.n_heads), ("num_key_value_heads", config.n_kv_heads),
            ("head_dim", config.head_dim),
        ] {
            if bundle[name].as_u64() != Some(expected as u64) {
                return Err(unsupported(format!("backbone/bundle {name} mismatch")));
            }
        }
        let rope = &bundle["rope_parameters"];
        if bundle["rms_norm_eps"].as_f64().map(|v| v as f32) != Some(config.rms_eps)
            || rope["rope_type"] != "default"
            || rope["rope_theta"].as_f64().map(|v| v as f32) != Some(config.rope_theta)
        {
            return Err(unsupported("backbone/bundle normalization or RoPE mismatch"));
        }
        let (entry, exit) = anchors(directory, vocab)?;
        let weights = weights::Weights::load(directory, config.n_embd)?;
        let link = link::HybridLink::upload(backend, weights)?;
        let model = Model::load(backend, backbone)?;
        Ok(Self { model, link, entry, exit })
    }

    pub fn config(&self) -> &super::ModelConfig { self.model.config() }
    pub fn entry_token(&self) -> u32 { self.entry }
    pub fn exit_token(&self) -> u32 { self.exit }

    pub fn decoder<'a>(&'a self, backend: &'a B, context: usize, limits: LatentLimits) -> Result<HybridDecoder<'a, B>> {
        if !matches!(self.model.architecture, Decoder::Qwen(_)) {
            return Err(unsupported("continuous inputs require the Qwen decoder"));
        }
        HybridDecoder::new(self, backend, context, limits)
    }
}

fn raw_cache_policy(value: Option<&std::ffi::OsStr>) -> Result<()> {
    match value.map(std::ffi::OsStr::to_str) {
        None | Some(Some("auto" | "off")) => Ok(()),
        Some(Some("required")) => Err(Error::NotImplemented("prepared cache for HF bundles")),
        _ => Err(unsupported("MDMA_CACHE must be auto, off or required")),
    }
}

mod tokenizer;

fn anchors(directory: &Path, vocab: usize) -> Result<(u32, u32)> {
    let tokenizer = tokenizer::Tokenizer::from_file(directory.join("tokenizer.json"))
        .map_err(|e| unsupported(e.to_string()))?;
    let anchor = |token| tokenizer.token_to_id(token)
        .filter(|&id| (id as usize) < vocab)
        .ok_or_else(|| unsupported(format!("missing or out-of-range anchor {token}")));
    let (entry, exit) = (anchor("<latent>")?, anchor("</latent>")?);
    if entry == exit { return Err(unsupported("entry and exit anchors must differ")); }
    Ok((entry, exit))
}

#[cfg(test)]
pub(crate) mod tests;

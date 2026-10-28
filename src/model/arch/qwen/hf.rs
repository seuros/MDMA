//! Direct loading of the released, unquantized Qwen3 HF backbone.
use super::*;
use crate::backend::{Error, Operation};
use crate::gguf::GgmlDType;
use safetensors::{Dtype, SafeTensors, tensor::TensorView};
use serde_json::Value;
use std::collections::BTreeMap;

pub(in crate::model) struct Config {
    pub model: ModelConfig,
    pub vocab: usize,
}

fn err(message: impl Into<String>) -> Error {
    Error::UnsupportedModel(format!("Qwen3 HF: {}", message.into()))
}

pub(in crate::model) fn validate<B: Backend>(b: &B, json: &Value, tensors: &SafeTensors<'_>) -> Result<Config> {
    let integer = |key: &str| json[key].as_u64().and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v > 0).ok_or_else(|| err(format!("invalid {key}")));
    let float = |value: &Value| value.as_f64().map(|v| v as f32)
        .filter(|v| v.is_finite() && *v > 0.0).ok_or_else(|| err("invalid normalization or RoPE scalar"));
    if json["model_type"] != "qwen3" || json["hidden_act"] != "silu"
        || json["attention_bias"] != false || json["tie_word_embeddings"] != false
        || json["use_sliding_window"] != false || !json["sliding_window"].is_null()
        || !json["quantization_config"].is_null() || !json["rope_scaling"].is_null()
    {
        return Err(err("requires dense, untied, bias-free, full-attention Qwen3 with SiLU"));
    }
    let rope = &json["rope_parameters"];
    if rope["rope_type"] != "default" || rope.as_object().is_none_or(|v| v.len() != 2) {
        return Err(err("only default split-half RoPE is supported"));
    }
    let c = ModelConfig {
        n_layers: integer("num_hidden_layers")?, n_embd: integer("hidden_size")?,
        n_ff: integer("intermediate_size")?, n_heads: integer("num_attention_heads")?,
        n_kv_heads: integer("num_key_value_heads")?, head_dim: integer("head_dim")?,
        n_ctx: integer("max_position_embeddings")?, rms_eps: float(&json["rms_norm_eps"])?,
        rope_theta: float(&rope["rope_theta"])?,
    };
    let vocab = integer("vocab_size")?;
    if c.head_dim % 2 != 0 || c.n_heads % c.n_kv_heads != 0 || c.n_ctx > u32::MAX as usize
        || vocab > u32::MAX as usize || b.max_head_dim().is_some_and(|max| c.head_dim > max)
        || !b.supports(Operation::RopeSplitHalf)
    { return Err(err("unsupported attention geometry, context or backend")); }
    if json["layer_types"].as_array().is_none_or(|types|
        types.len() != c.n_layers || types.iter().any(|v| v != "full_attention"))
    { return Err(err("requires one full_attention layer type per layer")); }
    if c.n_layers.checked_mul(11).and_then(|v| v.checked_add(3)) != Some(tensors.len()) {
        return Err(err("unexpected tensor count (missing weights or unsupported extra tensors)"));
    }
    let config = Config { model: c, vocab };
    let expected = shapes(&config)?;
    for (name, tensor) in tensors.iter() {
        if expected.get(name).is_none_or(|shape| shape != tensor.shape())
            || !b.supports_weight_dtype(dtype(tensor.dtype())?)
        { return Err(err(format!("unsupported shape or dtype for {name}"))); }
        if tensor.shape().len() == 1 { floats(&tensor)?; }
    }
    Ok(config)
}

fn shapes(config: &Config) -> Result<BTreeMap<String, Vec<usize>>> {
    let c = &config.model;
    let q = product(&[c.n_heads, c.head_dim])?;
    let kv = product(&[c.n_kv_heads, c.head_dim])?;
    let mut out = BTreeMap::from([
        ("model.embed_tokens.weight".into(), vec![config.vocab, c.n_embd]),
        ("model.norm.weight".into(), vec![c.n_embd]),
        ("lm_head.weight".into(), vec![config.vocab, c.n_embd]),
    ]);
    for i in 0..c.n_layers {
        for (suffix, shape) in [
            ("input_layernorm", vec![c.n_embd]), ("post_attention_layernorm", vec![c.n_embd]),
            ("self_attn.q_proj", vec![q, c.n_embd]), ("self_attn.k_proj", vec![kv, c.n_embd]),
            ("self_attn.v_proj", vec![kv, c.n_embd]), ("self_attn.o_proj", vec![c.n_embd, q]),
            ("self_attn.q_norm", vec![c.head_dim]), ("self_attn.k_norm", vec![c.head_dim]),
            ("mlp.gate_proj", vec![c.n_ff, c.n_embd]), ("mlp.up_proj", vec![c.n_ff, c.n_embd]),
            ("mlp.down_proj", vec![c.n_embd, c.n_ff]),
        ] { out.insert(format!("model.layers.{i}.{suffix}.weight"), shape); }
    }
    for shape in out.values() { product(shape)?; }
    Ok(out)
}

fn dtype(dtype: Dtype) -> Result<GgmlDType> {
    match dtype {
        Dtype::F32 => Ok(GgmlDType::F32), Dtype::F16 => Ok(GgmlDType::F16),
        Dtype::BF16 => Ok(GgmlDType::BF16), _ => Err(err("only F32/F16/BF16 weights are supported")),
    }
}

fn floats(tensor: &TensorView<'_>) -> Result<Vec<f32>> {
    let data = tensor.data();
    let values = match tensor.dtype() {
        Dtype::F32 => data.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>(),
        Dtype::F16 | Dtype::BF16 => data.chunks_exact(2).map(|v| {
            let bits = u16::from_le_bytes(v.try_into().unwrap());
            if tensor.dtype() == Dtype::BF16 { half::bf16::from_bits(bits).to_f32() }
            else { half::f16::from_bits(bits).to_f32() }
        }).collect(),
        _ => return Err(err("unsupported norm dtype")),
    };
    if values.iter().any(|v| !v.is_finite()) { return Err(err("nonfinite norm weight")); }
    Ok(values)
}

pub(in crate::model) fn load<B: Backend>(b: &B, tensors: &SafeTensors<'_>, config: Config) -> Result<Model<B>> {
    let matrix = |name: &str| {
        let tensor = tensors.tensor(name).map_err(|e| err(e.to_string()))?;
        let mut buf = b.alloc(tensor.data().len(), Pool::Weights)?;
        b.upload(&mut buf, 0, tensor.data())?;
        Ok::<_, Error>(QWeight { dtype: dtype(tensor.dtype())?, shape: tensor.shape().to_vec(), buf })
    };
    let norm = |name: &str| {
        let tensor = tensors.tensor(name).map_err(|e| err(e.to_string()))?;
        let values = floats(&tensor)?;
        let mut buf = b.alloc(product(&[values.len(), 4])?, Pool::Weights)?;
        b.upload(&mut buf, 0, bytemuck::cast_slice(&values))?;
        Ok::<_, Error>(buf)
    };
    let c = config.model;
    let mut layers = Vec::with_capacity(c.n_layers);
    for i in 0..c.n_layers {
        let name = |suffix| format!("model.layers.{i}.{suffix}.weight");
        layers.push(Layer {
            attn_norm: norm(&name("input_layernorm"))?, ffn_norm: norm(&name("post_attention_layernorm"))?,
            q: matrix(&name("self_attn.q_proj"))?, k: matrix(&name("self_attn.k_proj"))?,
            v: matrix(&name("self_attn.v_proj"))?, out: matrix(&name("self_attn.o_proj"))?,
            biases: None,
            head_norms: Some([norm(&name("self_attn.q_norm"))?, norm(&name("self_attn.k_norm"))?]),
            gate: matrix(&name("mlp.gate_proj"))?, up: matrix(&name("mlp.up_proj"))?,
            down: matrix(&name("mlp.down_proj"))?,
        });
    }
    let mut factors = b.alloc(product(&[c.head_dim / 2, 4])?, Pool::Weights)?;
    b.upload(&mut factors, 0, bytemuck::cast_slice(&vec![1f32; c.head_dim / 2]))?;
    Ok(Model {
        config: c, token_embd: matrix("model.embed_tokens.weight")?,
        output_norm: norm("model.norm.weight")?, output: Some(matrix("lm_head.weight")?),
        architecture: Decoder::Qwen(Qwen { layers, factors, rope_amplitude: 1.0 }), bos: None, memory: None,
    })
}

#[cfg(test)]
mod tests;

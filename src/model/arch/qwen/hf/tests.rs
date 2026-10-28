use super::*;
use crate::backend::{cpu::CpuBackend, test_support::{NoUpload, close}};
use crate::gguf::GgufModel;
use candle_core::Device;

fn config() -> Value {
    serde_json::json!({
        "model_type":"qwen3", "hidden_act":"silu", "attention_bias":false,
        "tie_word_embeddings":false, "use_sliding_window":false, "sliding_window":null,
        "hidden_size":12, "intermediate_size":24, "num_attention_heads":2,
        "num_key_value_heads":1, "head_dim":8, "num_hidden_layers":2,
        "max_position_embeddings":64, "vocab_size":128, "rms_norm_eps":1e-5,
        "rope_parameters":{"rope_type":"default","rope_theta":10000.0},
        "layer_types":["full_attention","full_attention"],
    })
}

fn gguf_name(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => "token_embd.weight".into(),
        "model.norm.weight" => "output_norm.weight".into(),
        "lm_head.weight" => "output.weight".into(),
        _ => {
            let (layer, suffix) = name.strip_prefix("model.layers.").unwrap().split_once('.').unwrap();
            let suffix = match suffix {
                "input_layernorm.weight" => "attn_norm", "post_attention_layernorm.weight" => "ffn_norm",
                "self_attn.q_proj.weight" => "attn_q", "self_attn.k_proj.weight" => "attn_k",
                "self_attn.v_proj.weight" => "attn_v", "self_attn.o_proj.weight" => "attn_output",
                "self_attn.q_norm.weight" => "attn_q_norm", "self_attn.k_norm.weight" => "attn_k_norm",
                "mlp.gate_proj.weight" => "ffn_gate", "mlp.up_proj.weight" => "ffn_up",
                "mlp.down_proj.weight" => "ffn_down", _ => unreachable!(),
            };
            format!("blk.{layer}.{suffix}.weight")
        }
    }
}

fn fixture_bytes() -> Vec<u8> {
    let f = super::super::tests::fixture("qwen3", 12, 8);
    let c = Config { model: ModelConfig {
        n_layers:2, n_embd:12, n_ff:24, n_heads:2, n_kv_heads:1, head_dim:8,
        n_ctx:64, rms_eps:1e-5, rope_theta:10000.0,
    }, vocab:128 };
    let storage = shapes(&c).unwrap().into_iter().map(|(name, shape)| {
        let tensor = &f.tensors.iter().find(|(n, _)| n == &gguf_name(&name)).unwrap().1;
        let values = tensor.dequantize(&Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        (name, (shape, bytemuck::cast_slice::<f32, u8>(&values).to_vec()))
    }).collect::<BTreeMap<_, _>>();
    let views = storage.iter().map(|(name, (shape, data))| {
        (name, TensorView::new(Dtype::F32, shape.clone(), data).unwrap())
    }).collect::<Vec<_>>();
    safetensors::serialize(views, None).unwrap()
}

#[test]
fn hf_qwen3_preserves_weights_norms_and_split_half_layout() {
    let bytes = fixture_bytes();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    let c = validate(&CpuBackend, &config(), &tensors).unwrap();
    let hf = load(&CpuBackend, &tensors, c).unwrap();
    let file = super::super::tests::fixture("qwen3", 12, 8).write();
    let gguf = Model::load(&CpuBackend, &mut GgufModel::open(file.path()).unwrap()).unwrap();
    let (mut hk, mut gk) = (hf.new_kv_cache(&CpuBackend, 8).unwrap(), gguf.new_kv_cache(&CpuBackend, 8).unwrap());
    let (mut hw, mut gw) = (hf.alloc_workspace(&CpuBackend, 1).unwrap(), gguf.alloc_workspace(&CpuBackend, 1).unwrap());
    for (pos, token) in [1, 3, 5, 7].into_iter().enumerate() {
        let got = hf.forward(&CpuBackend, &mut hk, &mut hw, token, pos as u32).unwrap();
        let want = gguf.forward(&CpuBackend, &mut gk, &mut gw, token, pos as u32).unwrap();
        close(&got, &want, 1e-6);
    }
}

#[test]
fn unsupported_hf_semantics_and_geometry_fail_before_upload() {
    let bytes = fixture_bytes();
    let tensors = SafeTensors::deserialize(&bytes).unwrap();
    for (key, value) in [
        ("model_type", serde_json::json!("qwen2")), ("head_dim", serde_json::json!(7)),
        ("num_hidden_layers", serde_json::json!(3)), ("attention_bias", serde_json::json!(true)),
        ("tie_word_embeddings", serde_json::json!(true)), ("rms_norm_eps", serde_json::json!(0)),
        ("hidden_size", serde_json::json!(16)), ("vocab_size", serde_json::json!(64)),
        ("layer_types", serde_json::json!(["full_attention","sliding_attention"])),
        ("quantization_config", serde_json::json!({"bits":4})),
    ] {
        let mut json = config();
        json[key] = value;
        assert!(validate(&NoUpload, &json, &tensors).is_err(), "{key}");
    }
    let mut json = config();
    json["rope_parameters"]["rope_type"] = serde_json::json!("yarn");
    assert!(validate(&NoUpload, &json, &tensors).is_err());
}

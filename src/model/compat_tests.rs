//! Always-on compatibility checks. All weights are generated, never downloaded.
use super::*;
use crate::{backend::cpu::CpuBackend, tokenizer::Tokenizer};
use candle_core::{
    Device, Tensor,
    quantized::{
        QTensor,
        gguf_file::{self, Value},
    },
};
use std::{
    assert_matches,
    fs::File,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

pub(crate) struct Fixture(pub(crate) PathBuf);

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

struct Builder {
    metadata: Vec<(String, Value)>,
    tensors: Vec<(String, QTensor)>,
}

impl Builder {
    fn set(&mut self, key: &str, value: Value) {
        self.metadata.retain(|(k, _)| k != key);
        self.metadata.push((key.into(), value));
    }

    fn tensor(&mut self, name: &str, shape: &[usize], values: Option<Vec<f32>>) {
        let seed = self.tensors.len() + 1;
        let data = values.unwrap_or_else(|| {
            (0..shape.iter().product())
                .map(|i| ((i * 17 + seed * 13) as f32 * 0.17).sin() * 0.15)
                .collect()
        });
        let tensor = Tensor::from_vec(data, shape, &Device::Cpu).unwrap();
        self.tensors.push((
            name.into(),
            QTensor::quantize(&tensor, GgmlDType::F32.candle().unwrap()).unwrap(),
        ));
    }

    fn norm(&mut self, name: &str, width: usize) {
        self.tensor(name, &[width], Some(vec![1.0; width]));
    }

    fn write(&self) -> Fixture {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "mdma-compat-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = File::create(&path).unwrap();
        let metadata: Vec<_> = self.metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let tensors: Vec<_> = self.tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        gguf_file::write(&mut file, &metadata, &tensors).unwrap();
        Fixture(path)
    }
}

fn fixture(gemma: bool) -> Builder {
    let arch = if gemma { "gemma4" } else { "llama" };
    let mut b = Builder {
        metadata: Vec::new(),
        tensors: Vec::new(),
    };
    b.set("general.architecture", Value::String(arch.into()));
    for (name, value) in [
        ("block_count", 3),
        ("embedding_length", 8),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 1),
        ("context_length", 16),
    ] {
        b.set(&format!("{arch}.{name}"), Value::U32(value));
    }
    b.set(&format!("{arch}.rope.freq_base"), Value::F32(10000.0));
    b.set(
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
        Value::F32(1e-6),
    );
    b.set("tokenizer.ggml.model", Value::String(arch.into()));
    let mut pieces: Vec<String> = [
        "<unk>",
        "<s>",
        "</s>",
        "a",
        "b",
        "ab",
        "▁",
        "▁a",
        "\n",
        "\n\n",
        "<turn|>",
        "<|tool_response>",
        "<eos>",
    ]
    .iter()
    .map(std::string::ToString::to_string)
    .collect();
    pieces.extend((pieces.len()..32).map(|i| format!("<unused{i}>")));
    b.set(
        "tokenizer.ggml.tokens",
        Value::Array(pieces.into_iter().map(Value::String).collect()),
    );
    b.set(
        "tokenizer.ggml.scores",
        Value::Array((0..32).map(|i| Value::F32(i as f32)).collect()),
    );
    b.set(
        "tokenizer.ggml.token_type",
        Value::Array(
            (0..32)
                .map(|i| {
                    Value::I32(if [1, 2, 10, 11, 12].contains(&i) {
                        3
                    } else {
                        1
                    })
                })
                .collect(),
        ),
    );
    b.set("tokenizer.ggml.bos_token_id", Value::U32(1));
    b.set("tokenizer.ggml.eos_token_id", Value::U32(2));
    b.set("tokenizer.ggml.unknown_token_id", Value::U32(0));
    b.tensor("token_embd.weight", &[32, 8], None);
    b.norm("output_norm.weight", 8);
    if gemma {
        for (name, value) in [
            ("embedding_length_per_layer_input", 4),
            ("attention.shared_kv_layers", 1),
            ("attention.key_length_swa", 4),
            ("attention.value_length_swa", 4),
            ("attention.key_length", 8),
            ("attention.value_length", 8),
            ("attention.sliding_window", 2),
        ] {
            b.set(&format!("{arch}.{name}"), Value::U32(value));
        }
        b.set(
            "gemma4.feed_forward_length",
            Value::Array([16, 24, 16].map(Value::I32).to_vec()),
        );
        b.set(
            "gemma4.attention.sliding_window_pattern",
            Value::Array([true, false, true].map(Value::Bool).to_vec()),
        );
        b.set("gemma4.rope.freq_base_swa", Value::F32(10000.0));
        b.set("gemma4.final_logit_softcapping", Value::F32(30.0));
        b.set("tokenizer.ggml.add_space_prefix", Value::Bool(false));
        b.set(
            "tokenizer.ggml.merges",
            Value::Array(["a b", "▁ a"].map(|s| Value::String(s.into())).to_vec()),
        );
        b.tensor("per_layer_token_embd.weight", &[32, 12], None);
        b.tensor("per_layer_model_proj.weight", &[12, 8], None);
        b.norm("per_layer_proj_norm.weight", 4);
        b.tensor(
            "rope_freqs.weight",
            &[4],
            Some(vec![1.0, 1.0, f32::INFINITY, f32::INFINITY]),
        );
    } else {
        b.set("llama.feed_forward_length", Value::U32(16));
        b.tensor("output.weight", &[32, 8], None);
    }
    for i in 0..3 {
        let name = |s: &str| format!("blk.{i}.{s}.weight");
        let d = if gemma && i == 1 { 8 } else { 4 };
        let ff = if gemma && i == 1 { 24 } else { 16 };
        b.norm(&name("attn_norm"), 8);
        b.tensor(&name("attn_q"), &[2 * d, 8], None);
        if !gemma || i < 2 {
            b.tensor(&name("attn_k"), &[d, 8], None);
            // Exercise Gemma's V=K alias on the global producer.
            if !gemma || i != 1 {
                b.tensor(&name("attn_v"), &[d, 8], None);
            }
        }
        b.tensor(&name("attn_output"), &[8, 2 * d], None);
        b.norm(&name("ffn_norm"), 8);
        b.tensor(&name("ffn_gate"), &[ff, 8], None);
        b.tensor(&name("ffn_up"), &[ff, 8], None);
        b.tensor(&name("ffn_down"), &[8, ff], None);
        if gemma {
            b.norm(&name("attn_q_norm"), d);
            if i < 2 {
                b.norm(&name("attn_k_norm"), d);
            }
            for n in ["post_attention_norm", "post_ffw_norm", "post_norm"] {
                b.norm(&name(n), 8);
            }
            b.tensor(&name("inp_gate"), &[4, 8], None);
            b.tensor(&name("proj"), &[8, 4], None);
            b.tensor(&name("layer_output_scale"), &[1], Some(vec![0.9]));
        }
    }
    b
}

/// Generated architecture fixture for transport tests; no external checkpoint.
pub(crate) fn decoder_fixture(gemma: bool) -> Fixture {
    fixture(gemma).write()
}

fn window_boundary<B: Backend>(backend: &B, f: &Fixture) -> Vec<f32> {
    let model = Model::load(backend, &mut GgufModel::open(&f.0).unwrap()).unwrap();
    let input: Vec<u32> = (0..528).map(|i| 3 + i % 6).collect();
    let mut kv = model.new_kv_cache(backend, 536).unwrap();
    // Full-context reference for the sliding-window rings, which wrap at 512.
    let mut seq = KvCache::with_geometry(backend, &model.kv_widths().unwrap(), 536).unwrap();
    let mut decode = model.alloc_workspace(backend, 1).unwrap();
    let mut start = 0;
    let mut last = Vec::new();
    for end in [511, 512, 513, 528] {
        let mut ws = model.alloc_workspace(backend, end - start).unwrap();
        let batched = model
            .prefill(backend, &mut kv, &mut ws, &input[start..end], start as u32)
            .unwrap();
        for (pos, &id) in input.iter().enumerate().take(end).skip(start) {
            last = model
                .forward(backend, &mut seq, &mut decode, id, pos as u32)
                .unwrap();
        }
        close(&batched, &last, 1e-4);
        start = end;
    }
    for pos in 528..536 {
        let id = crate::sampler::greedy(&last);
        last = model
            .forward(backend, &mut seq, &mut decode, id, pos)
            .unwrap();
        let batched = model
            .forward(backend, &mut kv, &mut decode, id, pos)
            .unwrap();
        close(&batched, &last, 1e-4);
    }
    last
}

fn window_fixture() -> Fixture {
    let mut b = fixture(true);
    b.set("gemma4.context_length", Value::U32(536));
    b.set("gemma4.attention.sliding_window", Value::U32(512));
    b.write()
}

#[test]
fn gemma_compat_actual_window_boundary_and_continuation() {
    window_boundary(&CpuBackend, &window_fixture());
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_compat_actual_window_boundary_matches_cpu() {
    let f = window_fixture();
    let cpu = window_boundary(&CpuBackend, &f);
    let metal = crate::backend::metal::MetalBackend::new().unwrap();
    close(&window_boundary(&metal, &f), &cpu, 1e-4);
}

fn close(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() < tolerance,
            "logit {i}: {a} != {b}"
        );
    }
}

fn inference<B: Backend>(backend: &B, f: &Fixture) -> Vec<f32> {
    let mut gguf = GgufModel::open(&f.0).unwrap();
    let model = Model::load(backend, &mut gguf).unwrap();
    let tokens = [1, 3, 4, 5, 6];
    let mut kv = model.new_kv_cache(backend, 8).unwrap();
    let mut ws = model.alloc_workspace(backend, tokens.len()).unwrap();
    let batched = model
        .prefill(backend, &mut kv, &mut ws, &tokens, 0)
        .unwrap();
    assert_eq!(kv.len(), tokens.len());
    // Full-context reference: Gemma's window of 2 wraps its rings repeatedly.
    let widths = model.kv_widths().unwrap();
    let mut sequential = KvCache::with_geometry_reserved(
        backend,
        &widths,
        &vec![0; widths.len()],
        &model.kv_states().unwrap(),
        8,
        None,
    )
    .unwrap();
    let mut decode = model.alloc_workspace(backend, 1).unwrap();
    let mut logits = Vec::new();
    for (pos, &token) in tokens.iter().enumerate() {
        logits = model
            .forward(backend, &mut sequential, &mut decode, token, pos as u32)
            .unwrap();
    }
    close(&batched, &logits, 1e-4);
    assert!(model.forward(backend, &mut kv, &mut decode, 32, 5).is_err());
    assert!(model.forward(backend, &mut kv, &mut decode, 1, 8).is_err());
    assert!(model.forward(backend, &mut kv, &mut decode, 1, 6).is_err());
    assert!(model.alloc_workspace(backend, 0).is_err());
    let mut empty = model.new_kv_cache(backend, 0).unwrap();
    assert!(
        model
            .forward(backend, &mut empty, &mut decode, 1, 0)
            .is_err()
    );
    // Reused prefix and fresh sequential decode must agree, including shared KV.
    let suffix = [7, 8];
    let mut suffix_ws = model.alloc_workspace(backend, 2).unwrap();
    let cached = model
        .prefill(backend, &mut kv, &mut suffix_ws, &suffix, 5)
        .unwrap();
    for (pos, &token) in suffix.iter().enumerate() {
        logits = model
            .forward(
                backend,
                &mut sequential,
                &mut decode,
                token,
                (5 + pos) as u32,
            )
            .unwrap();
    }
    close(&cached, &logits, 1e-4);
    cached
}

#[test]
fn llama_compat_load_prefill_decode() {
    inference(&CpuBackend, &fixture(false).write());
}

#[test]
fn gemma4_compat_tied_embeddings_shared_kv_and_variable_heads() {
    inference(&CpuBackend, &fixture(true).write());
    // Uniform FFN widths arrive as one scalar (E4B), not a per-layer array.
    let mut b = fixture(true);
    b.set("gemma4.feed_forward_length", Value::U32(16));
    for (suffix, shape) in [
        ("ffn_gate", [16, 8]),
        ("ffn_up", [16, 8]),
        ("ffn_down", [8, 16]),
    ] {
        let name = format!("blk.1.{suffix}.weight");
        b.tensors.retain(|(n, _)| *n != name);
        b.tensor(&name, &shape, None);
    }
    inference(&CpuBackend, &b.write());
}

/// Two trunk layers, DeltaNet then gated attention, and one multi-token
/// prediction block that is never run.
fn qwen35_fixture() -> Builder {
    let mut b = Builder {
        metadata: Vec::new(),
        tensors: Vec::new(),
    };
    b.set("general.architecture", Value::String("qwen35".into()));
    for (name, value) in [
        ("block_count", 3),
        ("nextn_predict_layers", 1),
        ("full_attention_interval", 2),
        ("embedding_length", 8),
        ("feed_forward_length", 16),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 8),
        ("attention.value_length", 8),
        ("rope.dimension_count", 4),
        ("context_length", 16),
        ("ssm.conv_kernel", 4),
        ("ssm.state_size", 4),
        ("ssm.group_count", 1),
        ("ssm.time_step_rank", 2),
        ("ssm.inner_size", 8),
    ] {
        b.set(&format!("qwen35.{name}"), Value::U32(value));
    }
    b.set(
        "qwen35.rope.dimension_sections",
        Value::Array([1, 1, 0, 0].map(Value::I32).to_vec()),
    );
    b.set("qwen35.rope.freq_base", Value::F32(10000.0));
    b.set("qwen35.attention.layer_norm_rms_epsilon", Value::F32(1e-6));
    b.tensor("token_embd.weight", &[32, 8], None);
    b.norm("output_norm.weight", 8);
    for i in 0..2 {
        let name = |s: &str| format!("blk.{i}.{s}");
        for n in ["attn_norm.weight", "post_attention_norm.weight"] {
            b.norm(&name(n), 8);
        }
        b.tensor(&name("ffn_gate.weight"), &[16, 8], None);
        b.tensor(&name("ffn_up.weight"), &[16, 8], None);
        b.tensor(&name("ffn_down.weight"), &[8, 16], None);
    }
    b.tensor("blk.0.attn_qkv.weight", &[16, 8], None);
    b.tensor("blk.0.attn_gate.weight", &[8, 8], None);
    b.tensor("blk.0.ssm_alpha.weight", &[2, 8], None);
    b.tensor("blk.0.ssm_beta.weight", &[2, 8], None);
    b.tensor("blk.0.ssm_conv1d.weight", &[16, 4], None);
    b.tensor("blk.0.ssm_dt.bias", &[2], Some(vec![0.5, -1.0]));
    b.tensor("blk.0.ssm_a", &[2], Some(vec![-0.5, -0.05]));
    b.norm("blk.0.ssm_norm.weight", 4);
    b.tensor("blk.0.ssm_out.weight", &[8, 8], None);
    b.tensor("blk.1.attn_q.weight", &[32, 8], None);
    b.tensor("blk.1.attn_k.weight", &[8, 8], None);
    b.tensor("blk.1.attn_v.weight", &[8, 8], None);
    b.tensor("blk.1.attn_output.weight", &[8, 16], None);
    b.norm("blk.1.attn_q_norm.weight", 8);
    b.norm("blk.1.attn_k_norm.weight", 8);
    b.norm("blk.2.attn_norm.weight", 8);
    b.tensor("blk.2.nextn.eh_proj.weight", &[8, 16], None);
    b
}

#[test]
fn qwen35_compat_recurrent_state_resumes_only_where_it_stopped() {
    let f = qwen35_fixture().write();
    inference(&CpuBackend, &f);
    let model = Model::load(&CpuBackend, &mut GgufModel::open(&f.0).unwrap()).unwrap();
    let mut kv = model.new_kv_cache(&CpuBackend, 8).unwrap();
    let mut ws = model.alloc_workspace(&CpuBackend, 3).unwrap();
    let first = model
        .prefill(&CpuBackend, &mut kv, &mut ws, &[1, 3, 4], 0)
        .unwrap();
    // The state has absorbed all three positions, so it resumes at 3 alone.
    assert!(kv.can_resume(3));
    assert!(!kv.can_resume(2));
    assert_eq!(kv.rewind(2), 0);
    // Starting over clears the state rather than folding the first run in.
    let again = model
        .prefill(&CpuBackend, &mut kv, &mut ws, &[1, 3, 4], 0)
        .unwrap();
    assert_eq!(first, again);
    for (key, value) in [
        ("qwen35.ssm.inner_size", Value::U32(4)),
        // The second rotated pair would fall to the fourth section.
        (
            "qwen35.rope.dimension_sections",
            Value::Array([1, 0, 0, 1].map(Value::I32).to_vec()),
        ),
        ("qwen35.rope.scaling.type", Value::String("yarn".into())),
    ] {
        let mut b = qwen35_fixture();
        b.set(key, value);
        let f = b.write();
        assert_matches!(
            Model::load(&CpuBackend, &mut GgufModel::open(&f.0).unwrap()).map(|_| ()),
            Err(Error::UnsupportedModel(_)),
            "{key}"
        );
    }
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_decoder_compat_matches_cpu() {
    let backend = crate::backend::vulkan::VulkanBackend::new()
        .unwrap()
        .with_f16_weights(false);
    for gemma in [false, true] {
        let f = fixture(gemma).write();
        close(&inference(&backend, &f), &inference(&CpuBackend, &f), 2e-3);
    }
}

#[cfg(feature = "rocm")]
#[test]
fn rocm_f32_decoder_compat_matches_cpu() {
    let backend = crate::backend::rocm::HipBackend::new_f32().unwrap();
    for gemma in [false, true] {
        let f = fixture(gemma).write();
        close(&inference(&backend, &f), &inference(&CpuBackend, &f), 2e-3);
    }
}

#[test]
fn tokenizer_compat_bos_specials_ranked_merges_and_newlines() {
    let llama = fixture(false).write();
    let tokenizer = Tokenizer::from_gguf(&GgufModel::open(&llama.0).unwrap()).unwrap();
    assert_eq!(tokenizer.encode("a"), [1, 7]); // missing add_bos defaults true
    assert_eq!(tokenizer.encode("<s>a</s>a"), [1, 1, 7, 2, 7]);
    let gemma = fixture(true).write();
    let tokenizer = Tokenizer::from_gguf(&GgufModel::open(&gemma.0).unwrap()).unwrap();
    assert_eq!(tokenizer.encode("ab\n\n<turn|>a"), [1, 5, 9, 10, 3]);
    assert_eq!(tokenizer.decode(&[5, 9, 3]), "ab\n\na");
    for id in [2, 10, 11, 12] {
        assert!(tokenizer.is_eos(id));
    }
    assert!(!tokenizer.is_eos(3));
}

#[test]
fn compat_rejects_bad_metadata_and_missing_tensors() {
    for (gemma, key, value) in [
        (
            false,
            "general.architecture",
            Value::String("not-a-decoder".into()),
        ),
        (false, "llama.attention.head_count", Value::U32(0)),
        (true, "gemma4.attention.head_count_kv", Value::U32(0)),
        (
            true,
            "gemma4.attention.sliding_window_pattern",
            Value::Array(vec![Value::Bool(true)]),
        ),
        (true, "gemma4.attention.shared_kv_layers", Value::U32(3)),
    ] {
        let mut b = fixture(gemma);
        b.set(key, value);
        let f = b.write();
        assert!(
            Model::load(&CpuBackend, &mut GgufModel::open(&f.0).unwrap()).is_err(),
            "{key}"
        );
    }
    let mut b = fixture(true);
    b.tensors
        .retain(|(name, _)| name != "blk.1.attn_q_norm.weight");
    let f = b.write();
    assert!(Model::load(&CpuBackend, &mut GgufModel::open(&f.0).unwrap()).is_err());
}

#[test]
#[cfg(feature = "service")]
fn registry_compat_caps_reported_context_to_model_limit() {
    use crate::registry::{Options, Registry, Source};
    for gemma in [false, true] {
        let f = fixture(gemma).write();
        let registry = Registry::new(
            std::sync::Arc::new(CpuBackend),
            vec![("m".into(), Source::Gguf(f.0.clone()))],
            None,
            Options::new(2048),
        )
        .unwrap();
        assert_eq!(registry.info("").unwrap().n_ctx, 16);
    }
}

#[test]
fn compat_memory_budgets_cover_gemma_shared_kv_and_sequential_scratch() {
    use crate::memory::NativeMemoryBudget;
    for gemma in [false, true] {
        let f = fixture(gemma).write();
        let mut gguf = GgufModel::open(&f.0).unwrap();
        let weights = memory::weight_bytes(&gguf).unwrap();
        let too_small = NativeMemoryBudget::new(weights - 1);
        assert_matches!(
            Model::load_with_budget(&CpuBackend, &mut gguf, &too_small).map(|_| ()),
            Err(Error::MemoryBudget { .. })
        );
        assert_eq!(too_small.used(), 0);

        let budget = NativeMemoryBudget::new(weights + 100_000);
        let model = Model::load_with_budget(&CpuBackend, &mut gguf, &budget).unwrap();
        let kv_bytes = model.kv_bytes(&CpuBackend, 8).unwrap();
        let scratch = model.workspace_bytes(5).unwrap();
        if gemma {
            // The local producer keeps its 2-row window, the global one all 8.
            assert_eq!(kv_bytes, (4 * 2 + 8 * 8) * 4 * 2);
            assert_eq!(scratch, model.workspace_bytes(1).unwrap());
            // Base scratch plus per-layer Q/K/V, FFN and input scratch.
            assert_eq!(scratch, 704 + 192 + 2 * 400 + 656);
            let payload: usize = gguf
                .tensor_names()
                .map(|n| gguf.tensor_desc(n).unwrap().n_bytes)
                .sum();
            assert_eq!(weights, payload + (2 + 4 + 8 + 4) * 4);
        }
        let mut kv = model.new_kv_cache(&CpuBackend, 8).unwrap();
        let stored: usize = (0..model.kv_widths().unwrap().len())
            .map(|layer| kv.view(layer).k.0.len() + kv.view(layer).v.0.len())
            .sum();
        assert_eq!(stored, kv_bytes);
        let mut ws = model.alloc_workspace(&CpuBackend, 5).unwrap();
        assert_eq!(budget.used(), weights + kv_bytes + scratch);
        model
            .prefill(&CpuBackend, &mut kv, &mut ws, &[1, 3, 4, 5, 6], 0)
            .unwrap();
        drop(model);
        assert_eq!(budget.used(), kv_bytes + scratch);
        drop(ws);
        drop(kv);
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn gemma_preflight_rejects_late_tensor_errors_without_reading_weights() {
    let mut b = fixture(true);
    b.tensors
        .retain(|(name, _)| name != "blk.2.post_norm.weight");
    let f = b.write();
    let error = Model::validate(&CpuBackend, &GgufModel::open(&f.0).unwrap())
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("blk.2.post_norm.weight"),
        "{error}"
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_compat_models_match_cpu() {
    let metal = crate::backend::metal::MetalBackend::new().unwrap();
    for gemma in [false, true] {
        let f = fixture(gemma).write();
        close(&inference(&metal, &f), &inference(&CpuBackend, &f), 2e-3);
    }
}

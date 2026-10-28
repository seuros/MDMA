use super::*;
use crate::backend::test_support::close;
use crate::backend::{Error, cpu::CpuBackend};
use crate::gguf::{
    GgmlDType,
    tests::{Fixture, weight},
};
use crate::memory::NativeMemoryBudget;
use candle_core::{
    Device, Tensor,
    quantized::{QTensor, gguf_file::Value},
};
use std::assert_matches;
use std::collections::BTreeMap;

pub(in crate::model) fn fixture(arch: &str, width: usize, head: usize) -> Fixture {
    let mut f = Fixture::new(2, width);
    f.metadata = f
        .metadata
        .into_iter()
        .map(|(k, v)| (k.replace("internlm2.", &format!("{arch}.")), v))
        .collect();
    f.metadata
        .insert("general.architecture".into(), Value::String(arch.into()));
    f.metadata.insert(
        format!("{arch}.attention.key_length"),
        Value::U32(head as u32),
    );
    f.metadata.insert(
        format!("{arch}.attention.value_length"),
        Value::U32(head as u32),
    );
    f.metadata.insert(
        format!("{arch}.rope.dimension_count"),
        Value::U32(head as u32),
    );
    if arch == "qwen2vl" {
        f.metadata.insert(
            format!("{arch}.rope.dimension_sections"),
            Value::Array(
                [head / 4, head / 8, head / 8, 0]
                    .map(|n| Value::I32(i32::try_from(n).unwrap()))
                    .into(),
            ),
        );
    }
    for i in 0..2 {
        f.replace(
            &format!("blk.{i}.attn_q.weight"),
            &[2 * head, width],
            GgmlDType::F32,
        );
        f.replace(
            &format!("blk.{i}.attn_k.weight"),
            &[head, width],
            GgmlDType::F32,
        );
        f.replace(
            &format!("blk.{i}.attn_v.weight"),
            &[head, width],
            GgmlDType::F32,
        );
        f.replace(
            &format!("blk.{i}.attn_output.weight"),
            &[width, 2 * head],
            GgmlDType::F32,
        );
        if arch == "qwen3" {
            for name in ["attn_q_norm", "attn_k_norm"] {
                let values: Vec<_> = (0..head).map(|j| 0.7 + j as f32 * 0.03).collect();
                let t = Tensor::from_vec(values, &[head], &Device::Cpu).unwrap();
                f.tensors.push((
                    format!("blk.{i}.{name}.weight"),
                    QTensor::quantize(&t, GgmlDType::F32.candle().unwrap()).unwrap(),
                ));
            }
        } else {
            for (name, n) in [("attn_q", 2 * head), ("attn_k", head), ("attn_v", head)] {
                let values: Vec<_> = (0..n).map(|j| (j as f32 * 0.7).sin() * 0.1).collect();
                let t = Tensor::from_vec(values, &[n], &Device::Cpu).unwrap();
                f.tensors.push((
                    format!("blk.{i}.{name}.bias"),
                    QTensor::quantize(&t, GgmlDType::F32.candle().unwrap()).unwrap(),
                ));
            }
        }
    }
    f
}

pub(crate) use crate::backend::test_support::NoUpload;

fn rejected(f: &Fixture, expected: &str) {
    let file = f.write();
    let mut g = GgufModel::open(file.path()).unwrap();
    let e = Model::load(&NoUpload, &mut g).err().expect("must reject");
    assert_matches!(e, Error::UnsupportedModel(_), "{e}");
    assert!(
        e.to_string().contains(expected),
        "expected {expected:?}: {e}"
    );
}

#[test]
fn qwen_preflight_handles_independent_heads_and_tied_embeddings() {
    for arch in ["qwen2", "qwen2vl", "qwen3"] {
        let mut f = fixture(arch, 12, 8);
        f.tensors.retain(|(k, _)| k != "output.weight");
        let file = f.write();
        let g = GgufModel::open(file.path()).unwrap();
        let c = Model::validate(&NoUpload, &g).unwrap();
        assert_eq!(
            (c.n_embd, c.n_heads, c.n_kv_heads, c.head_dim),
            (12, 2, 1, 8)
        );
    }
}

#[test]
fn qwen_rejects_unimplemented_metadata_and_task_heads_before_upload() {
    for (suffix, value) in [
        ("attention.causal", Value::Bool(false)),
        ("rope.scaling.type", Value::String("linear".into())),
        ("pooling_type", Value::U32(4)),
        ("attention.value_length", Value::U32(4)),
        ("attention.key_length", Value::U32(0)),
    ] {
        let mut f = fixture("qwen3", 12, 8);
        f.metadata.insert(format!("qwen3.{suffix}"), value);
        rejected(&f, suffix);
    }
    let mut f = fixture("qwen3", 12, 8);
    f.metadata.remove("qwen3.attention.key_length");
    rejected(&f, "attention.key_length");
    f = fixture("qwen3", 12, 8);
    f.tensors.push((
        "cls.output.weight".into(),
        weight(&[2, 12], GgmlDType::F32, 1),
    ));
    rejected(&f, "cls.output.weight");
}

#[test]
fn qwen_rejects_missing_or_wrong_head_norms_biases_and_mrope_sections() {
    for (arch, name) in [
        ("qwen3", "blk.1.attn_k_norm.weight"),
        ("qwen2", "blk.1.attn_q.bias"),
    ] {
        let mut f = fixture(arch, 12, 8);
        f.replace(name, &[4], GgmlDType::F32);
        rejected(&f, name);
        f.tensors.retain(|(k, _)| k != name);
        rejected(&f, name);
    }
    let mut f = fixture("qwen2vl", 12, 8);
    f.metadata.insert(
        "qwen2vl.rope.dimension_sections".into(),
        Value::Array(vec![Value::I32(8); 3]),
    );
    rejected(&f, "rope.dimension_sections");
}

fn rms(x: &[f32], w: &[f32]) -> Vec<f32> {
    x.chunks_exact(w.len())
        .flat_map(|row| {
            let s = (row.iter().map(|v| v * v).sum::<f32>() / w.len() as f32 + 1e-5)
                .sqrt()
                .recip();
            row.iter().zip(w).map(move |(x, w)| x * s * w)
        })
        .collect()
}

fn mat(w: &[f32], x: &[f32]) -> Vec<f32> {
    w.chunks_exact(x.len())
        .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

fn rope(x: &mut [f32], head: usize, pos: usize) {
    for h in x.chunks_exact_mut(head) {
        for i in 0..head / 2 {
            let angle = pos as f32 / 10000f32.powf(2.0 * i as f32 / head as f32);
            let (s, c) = angle.sin_cos();
            let (a, b) = (h[i], h[i + head / 2]);
            h[i] = a * c - b * s;
            h[i + head / 2] = a * s + b * c;
        }
    }
}

// Independent scalar decoder: no MDMA backend primitives or execution code.
fn reference(f: &Fixture, tokens: &[u32], head: usize) -> Vec<f32> {
    let w: BTreeMap<_, _> = f
        .tensors
        .iter()
        .map(|(k, t)| {
            (
                k.as_str(),
                t.dequantize(&Device::Cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            )
        })
        .collect();
    let width = f.metadata[&format!(
        "{}.embedding_length",
        f.metadata["general.architecture"].to_string().unwrap()
    )]
        .to_u32()
        .unwrap() as usize;
    let mut keys = [Vec::<Vec<f32>>::new(), Vec::new()];
    let mut values = keys.clone();
    let mut last = Vec::new();
    for (pos, &token) in tokens.iter().enumerate() {
        let mut x =
            w["token_embd.weight"][token as usize * width..(token as usize + 1) * width].to_vec();
        for layer in 0..2 {
            let get = |s: &str| w[format!("blk.{layer}.{s}").as_str()].as_slice();
            let normed = rms(&x, get("attn_norm.weight"));
            let mut q = mat(get("attn_q.weight"), &normed);
            let mut k = mat(get("attn_k.weight"), &normed);
            let mut v = mat(get("attn_v.weight"), &normed);
            if w.contains_key(format!("blk.{layer}.attn_q_norm.weight").as_str()) {
                q = rms(&q, get("attn_q_norm.weight"));
                k = rms(&k, get("attn_k_norm.weight"));
            } else {
                for (data, name) in [
                    (&mut q, "attn_q.bias"),
                    (&mut k, "attn_k.bias"),
                    (&mut v, "attn_v.bias"),
                ] {
                    for (x, b) in data.iter_mut().zip(get(name)) {
                        *x += b;
                    }
                }
            }
            rope(&mut q, head, pos);
            rope(&mut k, head, pos);
            keys[layer].push(k);
            values[layer].push(v);
            let mut attn = vec![0.0; 2 * head];
            for h in 0..2 {
                let q = &q[h * head..(h + 1) * head];
                let mut scores: Vec<f32> = keys[layer]
                    .iter()
                    .map(|k| {
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() / (head as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for score in &mut scores {
                    *score = (*score - max).exp();
                }
                let sum: f32 = scores.iter().sum();
                for (score, v) in scores.iter().zip(&values[layer]) {
                    for j in 0..head {
                        attn[h * head + j] += score / sum * v[j];
                    }
                }
            }
            for (x, a) in x.iter_mut().zip(mat(get("attn_output.weight"), &attn)) {
                *x += a;
            }
            let normed = rms(&x, get("ffn_norm.weight"));
            let gate = mat(get("ffn_gate.weight"), &normed);
            let up = mat(get("ffn_up.weight"), &normed);
            let act: Vec<_> = gate
                .iter()
                .zip(up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            for (x, a) in x.iter_mut().zip(mat(get("ffn_down.weight"), &act)) {
                *x += a;
            }
        }
        last = mat(
            w.get("output.weight")
                .unwrap_or_else(|| &w["token_embd.weight"]),
            &rms(&x, &w["output_norm.weight"]),
        );
    }
    last
}

fn inference<B: Backend>(b: &B, f: &Fixture) -> Vec<f32> {
    let file = f.write();
    let model = Model::load(b, &mut GgufModel::open(file.path()).unwrap()).unwrap();
    let mut kv = model.new_kv_cache(b, 8).unwrap();
    let mut seq = model.new_kv_cache(b, 8).unwrap();
    let tokens = [1, 3, 5, 7, 9, 11, 13, 15];
    let mut decode = model.alloc_workspace(b, 1).unwrap();
    let mut last = Vec::new();
    let mut start = 0;
    for end in [2, 5, 8] {
        let mut ws = model.alloc_workspace(b, end - start).unwrap();
        let batch = model
            .prefill(b, &mut kv, &mut ws, &tokens[start..end], start as u32)
            .unwrap();
        for (pos, &token) in tokens.iter().enumerate().take(end).skip(start) {
            last = model
                .forward(b, &mut seq, &mut decode, token, pos as u32)
                .unwrap();
        }
        close(&batch, &last, 1e-5);
        start = end;
    }
    assert_eq!(kv.len(), 8);
    assert!(model.forward(b, &mut kv, &mut decode, 1, 8).is_err());
    last
}

#[test]
fn qwen_cpu_matches_independent_scalar_decoder_and_nonzero_prefix_prefill() {
    for arch in ["qwen2", "qwen2vl", "qwen3"] {
        for tied in [false, true] {
            let mut f = fixture(arch, 12, 8);
            if tied {
                f.tensors.retain(|(k, _)| k != "output.weight");
            }
            let expected = reference(&f, &[1, 3, 5, 7, 9, 11, 13, 15], 8);
            close(&inference(&CpuBackend, &f), &expected, 1e-5);
        }
    }
}

#[test]
fn qwen_memory_counts_independent_q_width_and_generated_weights() {
    let f = fixture("qwen3", 12, 8);
    let file = f.write();
    let mut g = GgufModel::open(file.path()).unwrap();
    let weights = crate::model::memory::weight_bytes(&g).unwrap();
    let actual_weights: usize = g
        .tensor_names()
        .map(|n| g.tensor_desc(n).unwrap().n_bytes)
        .sum();
    assert_eq!(weights, actual_weights + 16);
    let budget = NativeMemoryBudget::new(weights + 100_000);
    let model = Model::load_with_budget(&CpuBackend, &mut g, &budget).unwrap();
    assert_eq!(budget.used(), weights);
    let ws = model.alloc_workspace(&CpuBackend, 5).unwrap();
    assert_eq!(ws.q.0.len(), 2 * 8 * 4);
    let scratch: usize = [
        &ws.x,
        &ws.x_alt,
        &ws.normed,
        &ws.q,
        &ws.k,
        &ws.v,
        &ws.attn_out,
        &ws.proj,
        &ws.gate,
        &ws.up,
        &ws.act,
        &ws.last_row,
        &ws.logits,
        &ws.qwen.as_ref().unwrap().kn,
    ]
    .iter()
    .map(|b| b.0.len())
    .sum();
    assert_eq!(model.workspace_bytes(5).unwrap(), scratch);
    assert_eq!(model.workspace_bytes(1).unwrap(), scratch);
    drop(model);
    assert_eq!(budget.used(), scratch);
    drop(ws);
    assert_eq!(budget.used(), 0);
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires ROCm GPU"]
fn qwen_rocm_f32_and_fast_decoder_match_cpu_and_use_eager_decode() {
    for arch in ["qwen2", "qwen3"] {
        let f = fixture(arch, 256, 128);
        let expected = inference(&CpuBackend, &f);
        for b in [
            crate::backend::rocm::HipBackend::new_f32().unwrap(),
            crate::backend::rocm::HipBackend::new().unwrap(),
        ] {
            close(
                &inference(&b, &f),
                &expected,
                if b.kv_elem_bytes() == 4 { 1e-3 } else { 0.02 },
            );
            let file = f.write();
            let model = Model::load(&b, &mut GgufModel::open(file.path()).unwrap()).unwrap();
            let mut kv = model.new_kv_cache(&b, 8).unwrap();
            let mut ws = model.alloc_workspace(&b, 1).unwrap();
            assert!(!b.decoder(&model, &mut kv, &mut ws).unwrap().is_graph());
        }
    }
}

#[test]
#[ignore = "requires MDMA_TEST_QWEN_GGUF"]
fn qwen_real_model_preflight() {
    let path = std::env::var("MDMA_TEST_QWEN_GGUF").expect("set MDMA_TEST_QWEN_GGUF");
    let g = GgufModel::open(path).unwrap();
    let c = Model::validate(&CpuBackend, &g).expect("supported dense Qwen");
    eprintln!(
        "Qwen: {} layers, embd {}, {} Q / {} KV heads of {}",
        c.n_layers, c.n_embd, c.n_heads, c.n_kv_heads, c.head_dim
    );
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires MDMA_TEST_QWEN_GGUF and ROCm; teacher-forced checkpoint diagnostics"]
fn qwen_real_model_teacher_forced_logprob_diagnostic() {
    use crate::{sampler::greedy, tokenizer::Tokenizer};
    let path = std::env::var("MDMA_TEST_QWEN_GGUF").expect("set MDMA_TEST_QWEN_GGUF");
    let prompt = std::env::var("MDMA_TEST_QWEN_PROMPT")
        .unwrap_or_else(|_| "The capital of France is".into());
    let forced: Vec<u32> = serde_json::from_str(
        &std::env::var("MDMA_TEST_QWEN_SUFFIX_IDS").unwrap_or_else(|_| "[]".into()),
    )
    .unwrap();
    for b in [
        crate::backend::rocm::HipBackend::new_f32().unwrap(),
        crate::backend::rocm::HipBackend::new().unwrap(),
    ] {
        let mut g = GgufModel::open(&path).unwrap();
        let tok = Tokenizer::from_gguf(&g).unwrap();
        let mut ids = tok.try_encode(&prompt).unwrap();
        ids.extend_from_slice(&forced);
        assert!(!ids.is_empty());
        let model = Model::load(&b, &mut g).unwrap();
        drop(g);
        let mut kv = model.new_kv_cache(&b, ids.len()).unwrap();
        let mut ws = model.alloc_workspace(&b, ids.len()).unwrap();
        let logits = model.prefill(&b, &mut kv, &mut ws, &ids, 0).unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let log_z = max
            + logits
                .iter()
                .map(|&v| (v as f64 - max).exp())
                .sum::<f64>()
                .ln();
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let top: Vec<_> = order
            .iter()
            .take(10)
            .map(|&id| {
                serde_json::json!({
                    "id": id, "logit": logits[id], "logprob": logits[id] as f64 - log_z,
                })
            })
            .collect();
        eprintln!(
            "QWEN_LOGITS: {}",
            serde_json::json!({
                "kv_bytes": b.kv_elem_bytes(), "prefix": ids, "argmax": greedy(&logits), "top": top,
            })
        );
    }
}

pub(super) fn trace<B: Backend>(
    b: &B,
    buf: &B::Buf,
    n: usize,
    name: &str,
    layer: usize,
    pos: u32,
) -> Result<()> {
    let Ok(selected) = std::env::var("MDMA_TEST_QWEN_TRACE_POS") else {
        return Ok(());
    };
    if selected.parse::<u32>().ok() != Some(pos) {
        return Ok(());
    }
    let mut values = vec![0f32; n];
    b.download(buf, 0, bytemuck::cast_slice_mut(&mut values))?;
    assert!(
        values.iter().all(|v| v.is_finite()),
        "{name}-{layer} at {pos}"
    );
    let edge = n.min(3);
    let comparison = std::env::var("MDMA_TEST_QWEN_REFERENCE_DIR")
        .ok()
        .and_then(|directory| {
            let (tensor, op) = match name {
                "Qcur_raw" => ("Qcur", "MUL_MAT"),
                "Kcur_raw" => ("Kcur", "MUL_MAT"),
                "Qcur" | "Kcur" => (name, "ROPE"),
                "Vcur" | "ffn_out" => (name, "MUL_MAT"),
                "ffn_swiglu" => (name, "SWIGLU"),
                "ffn_inp" | "l_out" => (name, "ADD"),
                "attn_norm" | "ffn_norm" | "result_norm" => (name, "MUL"),
                _ => return None,
            };
            let label = if tensor == "result_norm" {
                tensor.into()
            } else {
                format!("{tensor}-{layer}")
            };
            let bytes = std::fs::read(
                std::path::Path::new(&directory)
                    .join(pos.to_string())
                    .join(format!("{label}.{op}.f32")),
            )
            .expect("missing reference tensor");
            let expected: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect();
            assert_eq!(bytes.len(), n * 4);
            assert!(expected.iter().all(|v| v.is_finite()));
            let (mut squared_error, mut squared_reference, mut max_error) = (0f64, 0f64, 0f32);
            for (&got, &want) in values.iter().zip(&expected) {
                squared_error += f64::from(got - want).powi(2);
                squared_reference += f64::from(want).powi(2);
                max_error = max_error.max((got - want).abs());
            }
            Some(serde_json::json!({
                "relative_rms": (squared_error / squared_reference.max(1e-30)).sqrt(),
                "max_abs": max_error,
            }))
        });
    eprintln!(
        "QWEN_LAYER: {}",
        serde_json::json!({
            "kv_bytes": b.kv_elem_bytes(), "name": name, "layer": layer, "pos": pos,
            "first": &values[..edge], "last": &values[n - edge..], "sum": values.iter().sum::<f32>(),
            "reference": comparison,
        })
    );
    Ok(())
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires pinned MDMA_TEST_NUMARKDOWN_GGUF and ROCm; greedy/EOS checkpoint gate"]
fn numarkdown_rocm_greedy_checkpoint() {
    use crate::{model::test_support::assert_greedy, tokenizer::Tokenizer};
    // llama.cpp 6f41ac59e0a49a00483a316a22ada6b04edd2950, HIP, ctx128,
    // ubatch1, f16 KV, flash attention off, raw prompts, greedy/no penalties.
    let mut g = GgufModel::open(
        std::env::var("MDMA_TEST_NUMARKDOWN_GGUF").expect("set MDMA_TEST_NUMARKDOWN_GGUF"),
    )
    .unwrap();
    assert_eq!(g.metadata_str("general.architecture"), Some("qwen2vl"));
    assert_eq!(
        crate::cache::source_fingerprint(&mut g.file).unwrap().0,
        "1212e4e1df30b31891a43cf793a966bcfd236756ce1013a0233b648e3beafa4d"
    );
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let b = crate::backend::rocm::HipBackend::new().unwrap();
    eprintln!(
        "ROCm checkpoint profile: {}",
        b.weight_cache_profile().unwrap()
    );
    let model = Model::load(&b, &mut g).unwrap();
    drop(g);
    let cases: [(&str, &[u32]); 3] = [
        (
            "The capital of France is",
            &[
                12095, 13, 576, 6722, 315, 9856, 374, 19846, 13, 576, 6722, 315, 15344, 374, 21718,
                13, 576, 6722, 315, 17689, 374, 24081, 13, 576, 6722, 315, 279, 25662, 374, 37741,
                13, 576,
            ],
        ),
        (
            "<|im_start|>user\nWhat is 2+2? Answer briefly.<|im_end|>\n<|im_start|>assistant\n",
            &[19, 151645],
        ),
        (
            "中文：法国的首都是",
            &[
                106004, 3837, 104328, 9370, 100210, 20412, 20, 20, 31207, 107231, 3837, 104328,
                9370, 59975, 100132, 106004, 3837, 104328, 9370, 100210, 20412, 20, 20, 31207,
                107231, 3837, 104328, 9370, 59975, 100132, 106004, 3837,
            ],
        ),
    ];
    for (prompt, expected) in cases {
        let (ids, logits) = assert_greedy(&b, &model, &tok, prompt, expected);
        // Sequential decode reproduces the prefill, eagerly.
        let mut seq = model.new_kv_cache(&b, ids.len()).unwrap();
        let mut decode = model.alloc_workspace(&b, 1).unwrap();
        assert!(!b.decoder(&model, &mut seq, &mut decode).unwrap().is_graph());
        let mut other = Vec::new();
        for (pos, &id) in ids.iter().enumerate() {
            other = model
                .forward(&b, &mut seq, &mut decode, id, pos as u32)
                .unwrap();
        }
        close(&logits, &other, 1e-5);
    }
}

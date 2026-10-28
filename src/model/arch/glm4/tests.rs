use super::*;
use crate::backend::test_support::close;
use crate::backend::{Error, cpu::CpuBackend};
use crate::gguf::{
    GgmlDType,
    tests::{Fixture, weight},
};
use crate::memory::NativeMemoryBudget;
use crate::model::arch::qwen::tests::NoUpload;
use candle_core::{
    Device, Tensor,
    quantized::{QTensor, gguf_file::Value},
};
use std::assert_matches;
use std::collections::BTreeMap;

fn fixture(width: usize, head: usize, rot: usize) -> Fixture {
    let mut f = Fixture::new(2, width);
    f.metadata = f
        .metadata
        .into_iter()
        .map(|(k, v)| (k.replace("internlm2.", "glm4."), v))
        .collect();
    f.metadata
        .insert("general.architecture".into(), Value::String("glm4".into()));
    f.metadata
        .insert("glm4.attention.key_length".into(), Value::U32(head as u32));
    f.metadata.insert(
        "glm4.attention.value_length".into(),
        Value::U32(head as u32),
    );
    f.metadata
        .insert("glm4.rope.dimension_count".into(), Value::U32(rot as u32));
    f.tensors.retain(|(k, _)| !k.ends_with("ffn_gate.weight"));
    for i in 0..2 {
        for (name, shape) in [
            ("attn_q", vec![2 * head, width]),
            ("attn_k", vec![head, width]),
            ("attn_v", vec![head, width]),
            ("attn_output", vec![width, 2 * head]),
            ("ffn_up", vec![4 * width, width]),
        ] {
            f.replace(&format!("blk.{i}.{name}.weight"), &shape, GgmlDType::F32);
        }
        for (name, seed) in [("post_attention_norm", 0.6), ("post_ffw_norm", 1.1)] {
            let values: Vec<_> = (0..width).map(|j| seed + j as f32 * 0.02).collect();
            let t = Tensor::from_vec(values, &[width], &Device::Cpu).unwrap();
            f.tensors.push((
                format!("blk.{i}.{name}.weight"),
                QTensor::quantize(&t, GgmlDType::F32.candle().unwrap()).unwrap(),
            ));
        }
    }
    f
}

fn rejected(f: &Fixture, expected: &str) {
    let file = f.write();
    let error = Model::load(&NoUpload, &mut GgufModel::open(file.path()).unwrap())
        .err()
        .unwrap();
    assert_matches!(error, Error::UnsupportedModel(_), "{error}");
    assert!(error.to_string().contains(expected), "{error}");
}

#[test]
fn glm4_rejects_unknown_scaling_multimodal_mtp_and_task_metadata_before_upload() {
    for (suffix, value) in [
        ("rope.scaling.type", Value::String("yarn".into())),
        (
            "rope.dimension_sections",
            Value::Array(vec![Value::I32(2); 4]),
        ),
        ("nextn_predict_layers", Value::U32(1)),
        ("pooling_type", Value::U32(4)),
        ("attention.causal", Value::Bool(false)),
        ("attention.value_length", Value::U32(4)),
    ] {
        let mut f = fixture(12, 8, 4);
        f.metadata.insert(format!("glm4.{suffix}"), value);
        rejected(&f, suffix);
    }
    for rot in [0, 3, 10] {
        rejected(&fixture(12, 8, rot), "rope.dimension_count");
    }
}

#[test]
fn glm4_rejects_wrong_packed_layout_missing_post_norms_and_extra_heads_before_upload() {
    let mut f = fixture(12, 8, 4);
    f.replace("blk.1.ffn_up.weight", &[24, 12], GgmlDType::F32);
    rejected(&f, "ffn_up.weight");
    for name in [
        "blk.1.post_attention_norm.weight",
        "blk.0.post_ffw_norm.weight",
    ] {
        let mut f = fixture(12, 8, 4);
        f.tensors.retain(|(key, _)| key != name);
        rejected(&f, name);
    }
    let mut f = fixture(12, 8, 4);
    f.tensors.push((
        "cls.output.weight".into(),
        weight(&[2, 12], GgmlDType::F32, 1),
    ));
    rejected(&f, "cls.output.weight");
}

fn rms(x: &[f32], w: &[f32]) -> Vec<f32> {
    let scale = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5)
        .sqrt()
        .recip();
    x.iter().zip(w).map(|(x, w)| x * scale * w).collect()
}

fn mat(w: &[f32], x: &[f32]) -> Vec<f32> {
    w.chunks_exact(x.len())
        .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
        .collect()
}

// Independent scalar oracle; no backend operations or decoder calls.
fn reference(f: &Fixture, tokens: &[u32], head: usize, rot: usize) -> Vec<f32> {
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
    let width = f.metadata["glm4.embedding_length"].to_u32().unwrap() as usize;
    let mut keys = [Vec::<Vec<f32>>::new(), Vec::new()];
    let mut values = keys.clone();
    let mut last = Vec::new();
    for (pos, &token) in tokens.iter().enumerate() {
        let mut x =
            w["token_embd.weight"][token as usize * width..(token as usize + 1) * width].to_vec();
        for layer in 0..2 {
            let get = |s: &str| w[format!("blk.{layer}.{s}.weight").as_str()].as_slice();
            let normed = rms(&x, get("attn_norm"));
            let mut q = mat(get("attn_q"), &normed);
            let mut k = mat(get("attn_k"), &normed);
            let v = mat(get("attn_v"), &normed);
            for data in [&mut q, &mut k] {
                for h in data.chunks_exact_mut(head) {
                    for j in (0..rot).step_by(2) {
                        let angle = pos as f32 / 10000f32.powf(j as f32 / rot as f32);
                        let (s, c) = angle.sin_cos();
                        let (a, b) = (h[j], h[j + 1]);
                        h[j] = a * c - b * s;
                        h[j + 1] = a * s + b * c;
                    }
                }
            }
            keys[layer].push(k);
            values[layer].push(v);
            let mut attn = vec![0.0; 2 * head];
            for h in 0..2 {
                let q = &q[h * head..(h + 1) * head];
                let mut scores: Vec<_> = keys[layer]
                    .iter()
                    .map(|k| {
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() / (head as f32).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum: f32 = scores.iter().sum();
                for (score, v) in scores.iter().zip(&values[layer]) {
                    for j in 0..head {
                        attn[h * head + j] += score / sum * v[j];
                    }
                }
            }
            let projected = rms(&mat(get("attn_output"), &attn), get("post_attention_norm"));
            for (x, a) in x.iter_mut().zip(projected) {
                *x += a;
            }
            let packed = mat(get("ffn_up"), &rms(&x, get("ffn_norm")));
            let half = packed.len() / 2;
            let act: Vec<_> = packed[..half]
                .iter()
                .zip(&packed[half..])
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let ffn = rms(&mat(get("ffn_down"), &act), get("post_ffw_norm"));
            for (x, a) in x.iter_mut().zip(ffn) {
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
    let mut decode = model.alloc_workspace(b, 1).unwrap();
    let tokens = [1, 3, 5, 7, 9, 11, 13, 15];
    let mut last = Vec::new();
    let mut start = 0;
    for end in [2, 5, 8] {
        let mut ws = model.alloc_workspace(b, end - start).unwrap();
        let batch = model
            .prefill(b, &mut kv, &mut ws, &tokens[start..end], start as u32)
            .unwrap();
        for (pos, &id) in tokens.iter().enumerate().take(end).skip(start) {
            last = model
                .forward(b, &mut seq, &mut decode, id, pos as u32)
                .unwrap();
        }
        close(&batch, &last, 1e-5);
        start = end;
    }
    assert!(last.iter().all(|v| v.is_finite()));
    assert!(model.forward(b, &mut kv, &mut decode, 1, 8).is_err());
    last
}

#[test]
fn glm4_cpu_matches_independent_scalar_decoder_with_partial_rope_and_tied_output() {
    for rot in [2, 4, 8] {
        for tied in [false, true] {
            let mut f = fixture(12, 8, rot);
            if tied {
                f.tensors.retain(|(k, _)| k != "output.weight");
            }
            close(
                &inference(&CpuBackend, &f),
                &reference(&f, &[1, 3, 5, 7, 9, 11, 13, 15], 8, rot),
                1e-5,
            );
        }
    }
}

#[test]
fn glm4_memory_accounts_for_packed_ffn_scratch() {
    let f = fixture(12, 8, 4);
    let file = f.write();
    let mut g = GgufModel::open(file.path()).unwrap();
    let weights = crate::model::memory::weight_bytes(&g).unwrap();
    let budget = NativeMemoryBudget::new(weights + 100_000);
    let model = Model::load_with_budget(&CpuBackend, &mut g, &budget).unwrap();
    assert_eq!(budget.used(), weights);
    let ws = model.alloc_workspace(&CpuBackend, 5).unwrap();
    let extra = &ws.glm.as_ref().unwrap().packed_ffn;
    assert_eq!(extra.0.len(), 2 * 24 * 4);
    let actual: usize = [
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
        extra,
    ]
    .iter()
    .map(|b| b.0.len())
    .sum();
    assert_eq!(model.workspace_bytes(1).unwrap(), actual);
    assert_eq!(model.workspace_bytes(5).unwrap(), actual);
    drop(model);
    assert_eq!(budget.used(), actual);
    drop(ws);
    assert_eq!(budget.used(), 0);
}

const CHECKPOINT_CASES: &[(&str, &[u32], &[u32])] = &[
    (
        "The capital of France is",
        &[785, 6722, 315, 9621, 374],
        &[
            12089, 13, 1084, 374, 279, 7772, 3283, 304, 9621, 323, 279, 3146, 594, 4948, 11, 6955,
            11, 323, 12746, 4126, 13, 12089, 374, 3881, 369, 1181, 26195, 59508, 1741, 438, 279,
            468,
        ],
    ),
    (
        "[gMASK]<sop><|user|>\nWhat is 2 + 2? Reply with one number only.<|assistant|>\n",
        &[
            151331, 151333, 151336, 198, 3838, 374, 220, 17, 488, 220, 17, 30, 17815, 448, 825,
            1372, 1172, 13, 151337, 198,
        ],
        &[19, 151336],
    ),
    (
        "中国的首都是",
        &[101881, 98861, 99041],
        &[
            99334, 3837, 99334, 98316, 101881, 99764, 5373, 98917, 5373, 99274, 104137, 98327,
            106535, 99134, 1773, 99334, 100469, 111854, 109789, 98314, 107526, 3837, 99352, 99607,
            101702, 98611, 3837, 119989, 108203, 98327, 105523, 1773,
        ],
    ),
];

#[test]
#[ignore = "requires MDMA_TEST_GLM4_GGUF; metadata/tokenizer only"]
fn glm4_real_model_preflight() {
    let g = GgufModel::open(std::env::var("MDMA_TEST_GLM4_GGUF").expect("set MDMA_TEST_GLM4_GGUF"))
        .unwrap();
    let c = Model::validate(&CpuBackend, &g).unwrap();
    let tok = crate::tokenizer::Tokenizer::from_gguf(&g).expect("GLM byte BPE");
    for text in [
        "The capital of France is",
        "中文 🦀 12345\n",
        "[gMASK]<sop><|user|>\nHi",
    ] {
        assert_eq!(tok.decode(&tok.try_encode(text).unwrap()), text);
    }
    for &(prompt, ids, _) in CHECKPOINT_CASES {
        assert_eq!(
            tok.try_encode(prompt).unwrap(),
            ids,
            "llama.cpp tokenizer parity"
        );
    }
    eprintln!(
        "GLM: {} layers, embd {}, {} Q / {} KV heads of {}, rotary {}",
        c.n_layers,
        c.n_embd,
        c.n_heads,
        c.n_kv_heads,
        c.head_dim,
        preflight::rotary_dim(&g, &c).unwrap()
    );
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires ROCm; small synthetic Q5_K/Q6_K decoder"]
fn glm4_rocm_matches_cpu_and_uses_eager_decode() {
    let mut f = fixture(256, 128, 64);
    for (i, (_, t)) in f.tensors.iter_mut().enumerate() {
        if t.shape().rank() == 2 {
            *t = QTensor::quantize(
                &t.dequantize(&Device::Cpu).unwrap(),
                if i % 2 == 0 {
                    GgmlDType::Q5K
                } else {
                    GgmlDType::Q6K
                }
                .candle()
                .unwrap(),
            )
            .unwrap();
        }
    }
    let expected = inference(&CpuBackend, &f);
    for b in [
        crate::backend::rocm::HipBackend::new_f32().unwrap(),
        crate::backend::rocm::HipBackend::new().unwrap(),
    ] {
        close(
            &inference(&b, &f),
            &expected,
            if b.kv_elem_bytes() == 4 { 1e-3 } else { 0.03 },
        );
        let file = f.write();
        let model = Model::load(&b, &mut GgufModel::open(file.path()).unwrap()).unwrap();
        let mut kv = model.new_kv_cache(&b, 8).unwrap();
        let mut ws = model.alloc_workspace(&b, 1).unwrap();
        assert!(!b.decoder(&model, &mut kv, &mut ws).unwrap().is_graph());
    }
}

#[cfg(feature = "rocm")]
#[test]
#[ignore = "requires pinned MDMA_TEST_GLM4_GGUF and ROCm; 32B greedy/EOS checkpoint gate"]
fn glm4_rocm_greedy_checkpoint() {
    use crate::{sampler::greedy, tokenizer::Tokenizer};
    // llama.cpp 6f41ac59e0a49a00483a316a22ada6b04edd2950, HIP, ctx128,
    // ubatch1, f16 KV, flash attention off, exact prompt IDs, no penalties.
    let mut g =
        GgufModel::open(std::env::var("MDMA_TEST_GLM4_GGUF").expect("set MDMA_TEST_GLM4_GGUF"))
            .unwrap();
    assert_eq!(g.metadata_str("general.architecture"), Some("glm4"));
    assert_eq!(
        crate::cache::source_fingerprint(&mut g.file).unwrap().0,
        "73d94979af37a34d94081b28894cc19d75571f1c18507464157f6d3d588598a3"
    );
    let tok = Tokenizer::from_gguf(&g).unwrap();
    for b in [
        crate::backend::rocm::HipBackend::new_f32().unwrap(),
        crate::backend::rocm::HipBackend::new().unwrap(),
    ] {
        eprintln!(
            "ROCm checkpoint profile: {}, KV element bytes: {}",
            b.weight_cache_profile().unwrap(),
            b.kv_elem_bytes()
        );
        let budget = NativeMemoryBudget::new(25 * 1024 * 1024 * 1024);
        let model = Model::load_with_budget(&b, &mut g, &budget).unwrap();
        for &(prompt, ids, expected) in CHECKPOINT_CASES {
            assert_eq!(tok.try_encode(prompt).unwrap(), ids);
            let capacity = ids.len() + expected.len();
            let mut kv = model.new_kv_cache(&b, capacity).unwrap();
            let mut ws = model.alloc_workspace(&b, ids.len()).unwrap();
            let mut logits = model.prefill(&b, &mut kv, &mut ws, ids, 0).unwrap();
            drop(ws);
            let mut decode = model.alloc_workspace(&b, 1).unwrap();
            assert!(!b.decoder(&model, &mut kv, &mut decode).unwrap().is_graph());
            let mut seq = model.new_kv_cache(&b, capacity).unwrap();
            let mut other = Vec::new();
            for (pos, &id) in ids.iter().enumerate() {
                other = model
                    .forward(&b, &mut seq, &mut decode, id, pos as u32)
                    .unwrap();
            }
            close(&logits, &other, 1e-5);
            drop(seq);
            let mut generated = Vec::new();
            for (i, &target) in expected.iter().enumerate() {
                assert!(logits.iter().all(|v| v.is_finite()));
                let id = greedy(&logits);
                assert_eq!(
                    id,
                    target,
                    "{prompt:?}: output position {i}, KV element bytes {}",
                    b.kv_elem_bytes()
                );
                generated.push(id);
                if tok.is_eos(id) {
                    assert_eq!(i + 1, expected.len(), "unexpected early EOS");
                    break;
                }
                if i + 1 < expected.len() {
                    logits = model
                        .forward(&b, &mut kv, &mut decode, id, (ids.len() + i) as u32)
                        .unwrap();
                }
            }
            assert_eq!(generated, expected);
            eprintln!("{prompt:?}: {generated:?}");
        }
        drop(model);
        assert_eq!(budget.used(), 0);
    }
}

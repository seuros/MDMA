use super::*;
use crate::backend::cpu::{CpuBackend, CpuBuf};
use crate::backend::{AttnSpec, Error, KvView, Pool, QWeight, RopeSpec};
use crate::gguf::tests::{Fixture, weight};
use crate::gguf::{GgmlDType, TensorDesc};
use crate::model::Model;
use crate::model::arch::llama::ARCH;
use crate::model::preflight::validate_tensor;
#[cfg(feature = "service")]
use crate::registry::{Options, Registry, Source};
use candle_core::quantized::gguf_file::Value;
use std::assert_matches;
use std::cell::Cell;

// CPU execution with the default GPU dtype contract, without a GPU context.
#[derive(Default)]
struct CountingBackend {
    allocations: Cell<usize>,
    uploads: Cell<usize>,
    head_limit: Option<usize>,
}

impl CountingBackend {
    fn assert_unused(&self) {
        assert_eq!(self.allocations.get(), 0, "allocated before rejection");
        assert_eq!(self.uploads.get(), 0, "uploaded before rejection");
    }
}

impl Backend for CountingBackend {
    type Buf = CpuBuf;

    fn max_head_dim(&self) -> Option<usize> {
        self.head_limit
    }
    fn alloc(&self, bytes: usize, pool: Pool) -> Result<Self::Buf> {
        self.allocations.set(self.allocations.get() + 1);
        CpuBackend.alloc(bytes, pool)
    }
    fn upload(&self, dst: &mut Self::Buf, offset: usize, src: &[u8]) -> Result<()> {
        self.uploads.set(self.uploads.get() + 1);
        CpuBackend.upload(dst, offset, src)
    }
    fn download(&self, src: &Self::Buf, offset: usize, dst: &mut [u8]) -> Result<()> {
        CpuBackend.download(src, offset, dst)
    }
    fn copy(
        &self,
        src: &Self::Buf,
        src_off: usize,
        dst: &mut Self::Buf,
        dst_off: usize,
        len: usize,
    ) -> Result<()> {
        CpuBackend.copy(src, src_off, dst, dst_off, len)
    }
    fn qmatmul(&self, w: &QWeight<Self::Buf>, x: &Self::Buf, y: &mut Self::Buf) -> Result<()> {
        CpuBackend.qmatmul(w, x, y)
    }
    fn embed(
        &self,
        table: &QWeight<Self::Buf>,
        token_ids: &[u32],
        out: &mut Self::Buf,
    ) -> Result<()> {
        CpuBackend.embed(table, token_ids, out)
    }
    fn rmsnorm(&self, w: &Self::Buf, x: &Self::Buf, y: &mut Self::Buf, eps: f32) -> Result<()> {
        CpuBackend.rmsnorm(w, x, y, eps)
    }
    fn add(&self, a: &Self::Buf, b: &Self::Buf, out: &mut Self::Buf) -> Result<()> {
        CpuBackend.add(a, b, out)
    }
    fn rope(
        &self,
        q: &mut Self::Buf,
        k: &mut Self::Buf,
        pos: u32,
        n_tokens: usize,
        spec: &RopeSpec,
    ) -> Result<()> {
        CpuBackend.rope(q, k, pos, n_tokens, spec)
    }
    fn attn(
        &self,
        q: &Self::Buf,
        kv: &KvView<Self::Buf>,
        spec: &AttnSpec,
        out: &mut Self::Buf,
    ) -> Result<()> {
        CpuBackend.attn(q, kv, spec, out)
    }
    fn silu_mul(&self, gate: &Self::Buf, up: &Self::Buf, out: &mut Self::Buf) -> Result<()> {
        CpuBackend.silu_mul(gate, up, out)
    }
}

impl Fixture {
    fn assert_rejected(&self, expected: &str) {
        let file = self.write();
        let mut gguf = GgufModel::open(file.path()).unwrap();
        let backend = CountingBackend::default();
        assert_rejected(&backend, &mut gguf, expected);
    }
}

fn assert_rejected(backend: &CountingBackend, gguf: &mut GgufModel, expected: &str) {
    let error = Model::load(backend, gguf)
        .err()
        .expect("must reject unsupported model");
    assert_matches!(error, Error::UnsupportedModel(_), "{error}");
    assert!(
        error.to_string().contains(expected),
        "expected {expected:?}: {error}"
    );
    backend.assert_unused();
}

#[test]
fn supported_load_preserves_tiny_model_logits() {
    let file = Fixture::new(1, 8).write();
    let mut gguf = GgufModel::open(file.path()).unwrap();
    let backend = CountingBackend::default();
    let config = Model::validate(&backend, &gguf).unwrap();
    assert_eq!((config.n_layers, config.head_dim), (1, 4));
    backend.assert_unused();
    let loaded = Model::load(&backend, &mut gguf).unwrap();
    assert_eq!((backend.allocations.get(), backend.uploads.get()), (12, 12));
    let reference = crate::model::tests::tiny_model(&CpuBackend);
    let mut kv = loaded.new_kv_cache(&backend, 3).unwrap();
    let mut ws = loaded.alloc_workspace(&backend, 1).unwrap();
    let mut ref_kv = reference.new_kv_cache(&CpuBackend, 3).unwrap();
    let mut ref_ws = reference.alloc_workspace(&CpuBackend, 1).unwrap();
    for (pos, token) in [1, 5, 9].into_iter().enumerate() {
        let actual = loaded
            .forward(&backend, &mut kv, &mut ws, token, pos as u32)
            .unwrap();
        let expected = reference
            .forward(&CpuBackend, &mut ref_kv, &mut ref_ws, token, pos as u32)
            .unwrap();
        assert!(actual.iter().all(|v| v.is_finite()));
        assert_eq!(actual, expected);
    }
}

#[test]
fn rejects_other_architectures_even_with_compatible_tensor_names() {
    for arch in ["qwen3moe", "gemma2", "clip", "unknown"] {
        let mut fixture = Fixture::new(2, 8);
        fixture.metadata = fixture
            .metadata
            .into_iter()
            .map(|(key, value)| (key.replace("internlm2.", &format!("{arch}.")), value))
            .collect();
        fixture
            .metadata
            .insert("general.architecture".into(), Value::String(arch.into()));
        fixture.assert_rejected(&format!("architecture {arch:?}"));
    }
}

#[test]
fn rejects_missing_or_malformed_architecture_and_non_model_artifacts() {
    let mut fixture = Fixture::new(2, 8);
    fixture.metadata.remove("general.architecture");
    fixture.assert_rejected("general.architecture");
    fixture
        .metadata
        .insert("general.architecture".into(), Value::U32(1));
    fixture.assert_rejected("general.architecture");
    fixture
        .metadata
        .insert("general.architecture".into(), Value::String(ARCH.into()));
    for value in [
        Value::String("adapter".into()),
        Value::String("mmproj".into()),
        Value::U32(1),
    ] {
        fixture.metadata.insert("general.type".into(), value);
        fixture.assert_rejected("general.type");
    }
}

#[test]
fn rejects_missing_late_tensor_and_tied_output() {
    for name in ["blk.1.ffn_down.weight", "output.weight"] {
        let mut fixture = Fixture::new(2, 8);
        fixture.tensors.retain(|(k, _)| k != name);
        fixture.assert_rejected(&format!("missing tensor {name}"));
    }
}

#[test]
fn rejects_wrong_shapes_and_non_f32_norms_before_upload() {
    for (name, shape) in [
        ("token_embd.weight", vec![128, 4]),
        ("output.weight", vec![127, 8]),
        ("blk.1.attn_k.weight", vec![8, 8]),
        ("blk.1.ffn_down.weight", vec![16, 8]),
    ] {
        let mut fixture = Fixture::new(2, 8);
        fixture.replace(name, &shape, GgmlDType::F32);
        fixture.assert_rejected(name);
    }
    let mut fixture = Fixture::new(2, 8);
    fixture.replace("blk.1.ffn_norm.weight", &[8], GgmlDType::F16);
    fixture.assert_rejected("norm weight must be F32");
}

#[test]
fn rejects_ignored_bias_norm_adapter_and_classifier_tensors() {
    for name in [
        "blk.1.attn_q.bias",
        "blk.1.post_attention_norm.weight",
        "cls.output.weight",
        "output.lora_a",
    ] {
        let mut fixture = Fixture::new(2, 8);
        fixture
            .tensors
            .push((name.into(), weight(&[8], GgmlDType::F32, 0)));
        fixture.assert_rejected(&format!("unexpected tensor {name}"));
    }
}

#[test]
fn rejects_invalid_geometry_without_panicking() {
    for suffix in INTEGER_KEYS {
        for value in [Value::U32(0), Value::String("8".into())] {
            let mut fixture = Fixture::new(2, 8);
            let key = format!("{ARCH}.{suffix}");
            fixture.metadata.insert(key.clone(), value);
            fixture.assert_rejected(&key);
        }
    }
    for (suffix, value) in [
        ("embedding_length", 7),
        ("embedding_length", 6),
        ("embedding_length", 1),
        ("attention.head_count_kv", 3),
    ] {
        let mut fixture = Fixture::new(2, 8);
        fixture
            .metadata
            .insert(format!("{ARCH}.{suffix}"), Value::U32(value));
        fixture.assert_rejected("invalid geometry");
    }
    let mut fixture = Fixture::new(2, 8);
    fixture
        .metadata
        .insert("internlm2.block_count".into(), Value::U32(u32::MAX));
    fixture.assert_rejected("block_count exceeds tensor count");
    fixture
        .metadata
        .insert("internlm2.attention.head_count".into(), Value::U32(0));
    let file = fixture.write();
    assert!(ModelConfig::from_gguf(&GgufModel::open(file.path()).unwrap(), ARCH).is_none());
}

#[test]
fn rejects_invalid_rope_and_norm_constants() {
    for suffix in FLOAT_KEYS {
        for value in [
            Value::F32(0.0),
            Value::F32(-1.0),
            Value::F32(f32::NAN),
            Value::F32(f32::INFINITY),
            Value::String("1".into()),
        ] {
            let mut fixture = Fixture::new(2, 8);
            let key = format!("{ARCH}.{suffix}");
            fixture.metadata.insert(key.clone(), value);
            fixture.assert_rejected(&key);
        }
    }
}

#[test]
fn rejects_unimplemented_semantics_and_malformed_optional_metadata() {
    for suffix in [
        "rope.scaling.type",
        "attention.sliding_window",
        "pooling_type",
        "attention.causal",
        "vocab_size",
    ] {
        let mut fixture = Fixture::new(2, 8);
        let key = format!("{ARCH}.{suffix}");
        fixture.metadata.insert(key.clone(), Value::U32(4));
        fixture.assert_rejected(&key);
    }
    for suffix in HEAD_KEYS {
        for value in [Value::U32(2), Value::String("4".into())] {
            let mut fixture = Fixture::new(2, 8);
            let key = format!("{ARCH}.{suffix}");
            fixture.metadata.insert(key.clone(), value);
            fixture.assert_rejected(&key);
        }
    }
}

#[test]
fn accepts_supported_optional_metadata_and_descriptive_task_tags() {
    let mut fixture = Fixture::new(2, 8);
    fixture.metadata.remove("general.type"); // Older GGUF exports omit it.
    for suffix in HEAD_KEYS {
        fixture
            .metadata
            .insert(format!("{ARCH}.{suffix}"), Value::U32(4));
    }
    fixture
        .metadata
        .insert("internlm2.attention.causal".into(), Value::Bool(true));
    fixture
        .metadata
        .insert("internlm2.vocab_size".into(), Value::U32(128));
    fixture.metadata.insert(
        "general.tags".into(),
        Value::Array(vec![Value::String("text-classification".into())]),
    );
    let file = fixture.write();
    let backend = CountingBackend::default();
    Model::validate(&backend, &GgufModel::open(file.path()).unwrap()).unwrap();
    backend.assert_unused();
}

#[test]
fn rejects_backend_dtypes_early_but_preserves_cpu_coverage() {
    for (dtype, width) in [
        (GgmlDType::BF16, 8),
        (GgmlDType::Q8_0, 32),
        (GgmlDType::Q5K, 256),
    ] {
        let mut fixture = Fixture::new(2, width);
        fixture.replace("blk.1.ffn_down.weight", &[width, width * 2], dtype);
        fixture.assert_rejected(&format!("backend does not support {dtype:?}"));
        let file = fixture.write();
        Model::validate(&CpuBackend, &GgufModel::open(file.path()).unwrap()).unwrap();
    }
}

#[test]
fn accepts_each_shared_gpu_weight_dtype() {
    let mut fixture = Fixture::new(1, 256);
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::Q4K,
        GgmlDType::Q6K,
    ] {
        fixture.replace("output.weight", &[128, 256], dtype);
        let file = fixture.write();
        let backend = CountingBackend::default();
        Model::validate(&backend, &GgufModel::open(file.path()).unwrap()).unwrap();
        backend.assert_unused();
    }
}

#[test]
fn cpu_advertised_dtypes_execute_both_weight_ops() {
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ] {
        if !CpuBackend.supports_weight_dtype(dtype) {
            continue;
        }
        let tensor = weight(&[1, 256], dtype, 1);
        let w = QWeight {
            dtype,
            shape: vec![1, 256],
            buf: CpuBuf(tensor.data().unwrap().to_vec()),
        };
        let x = CpuBuf(bytemuck::cast_slice(&[1.0f32; 256]).to_vec());
        let mut y = CpuBuf(vec![0; 4]);
        CpuBackend.qmatmul(&w, &x, &mut y).unwrap();
        assert!(
            f32::from_ne_bytes(y.0.try_into().unwrap()).is_finite(),
            "{dtype:?}"
        );
        let mut row = CpuBuf(vec![0; 256 * 4]);
        CpuBackend.embed(&w, &[0], &mut row).unwrap();
        assert!(
            bytemuck::cast_slice::<u8, f32>(&row.0)
                .iter()
                .all(|v| v.is_finite()),
            "{dtype:?}"
        );
    }
}

#[test]
fn quantization_must_align_rows_not_just_total_elements() {
    let desc = TensorDesc {
        dtype: GgmlDType::Q4K,
        shape: vec![32, 8],
        offset: 0,
        n_bytes: 144,
    };
    let backend = CountingBackend::default();
    let error = validate_tensor(&backend, "token_embd.weight", &desc, &[32, 8], false).unwrap_err();
    assert!(error.to_string().contains("row width"), "{error}");
    backend.assert_unused();
}

#[test]
fn respects_backend_hard_head_limit() {
    let file = Fixture::new(1, 8).write();
    let mut gguf = GgufModel::open(file.path()).unwrap();
    let backend = CountingBackend {
        head_limit: Some(2),
        ..Default::default()
    };
    assert_rejected(&backend, &mut gguf, "head_dim 4 exceeds backend limit 2");
    let backend = CountingBackend {
        head_limit: Some(4),
        ..Default::default()
    };
    Model::validate(&backend, &gguf).unwrap();
    backend.assert_unused();
}

#[test]
fn rejects_truncated_last_payload_before_any_upload() {
    let file = Fixture::new(2, 8).write();
    let mut gguf = GgufModel::open(file.path()).unwrap();
    let len = file.as_file().metadata().unwrap().len();
    file.as_file().set_len(len - 1).unwrap();
    assert_rejected(
        &CountingBackend::default(),
        &mut gguf,
        "output.weight: tensor byte range",
    );
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the counting fake uses Cell; a registry owns its backend"
)]
#[cfg(feature = "service")]
fn registry_preflights_later_models_before_loading_the_first() {
    let first = Fixture::new(2, 8).with_tokenizer().write();
    let mut invalid = Fixture::new(2, 8);
    invalid.metadata.insert(
        "general.architecture".into(),
        Value::String("qwen3moe".into()),
    );
    let second = invalid.write();
    let models = [("first", &first), ("second", &second)]
        .into_iter()
        .map(|(name, file)| (name.to_owned(), Source::Gguf(file.path().into())))
        .collect();
    let backend = std::sync::Arc::new(CountingBackend::default());
    let error = Registry::new(backend.clone(), models, None, Options::new(64))
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("model second: unsupported model: architecture \"qwen3moe\""),
        "{error}"
    );
    backend.assert_unused();
    backend.assert_unused();
}

#[test]
#[ignore = "local MiniCheck file; metadata/descriptors only, no weight uploads"]
fn real_minicheck_metadata_preflight() {
    let gguf = GgufModel::open(crate::backend::test_support::fixture()).unwrap();
    let backend = CountingBackend::default();
    let config = Model::validate(&backend, &gguf).unwrap();
    assert_eq!(
        (config.n_layers, config.n_embd, config.head_dim),
        (32, 4096, 128)
    );
    backend.assert_unused();
}

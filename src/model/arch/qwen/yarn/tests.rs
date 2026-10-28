use super::*;
use crate::backend::cpu::CpuBackend;
use crate::model::Model;
use candle_core::quantized::gguf_file::Value;

fn fixture(factor: f32) -> crate::gguf::tests::Fixture {
    let mut f = super::super::tests::fixture("qwen3", 12, 8);
    f.metadata.insert(
        "qwen3.rope.scaling.type".into(),
        Value::String("yarn".into()),
    );
    f.metadata
        .insert("qwen3.rope.scaling.factor".into(), Value::F32(factor));
    f.metadata.insert(
        "qwen3.rope.scaling.original_context_length".into(),
        Value::U32(16),
    );
    f
}

#[test]
fn yarn_identity_and_bonsai_correction_range_match_reference_formula() {
    let mut c = ModelConfig {
        n_layers: 1,
        n_embd: 2048,
        n_ff: 8192,
        n_heads: 16,
        n_kv_heads: 8,
        head_dim: 128,
        rope_theta: 1_000_000.0,
        rms_eps: 1e-6,
        n_ctx: 32768,
    };
    let yarn = Yarn {
        factor: 4.0,
        original_context: 8192,
    };
    let divisors = yarn.divisors(&c);
    assert_eq!(divisors.len(), 64);
    assert_eq!(divisors[0], 1.0);
    assert_eq!(divisors[63], 4.0);
    assert_eq!(divisors[17], 1.0);
    assert!((divisors[25] - 17.0 / 11.0).abs() < 1e-6);
    assert_eq!(divisors[34], 4.0);
    assert!((yarn.amplitude() - 1.1386294).abs() < 1e-7);
    c.head_dim = 8;
    let identity = Yarn {
        factor: 1.0,
        original_context: 8192,
    };
    assert_eq!(identity.amplitude(), 1.0);
    assert_eq!(identity.divisors(&c), vec![1.0; 4]);
}

#[test]
fn yarn_rejects_unsupported_scaling_before_upload() {
    let file = fixture(4.0).with_tokenizer().write();
    let g = GgufModel::open(file.path()).unwrap();
    Model::validate(&super::super::tests::NoUpload, &g).unwrap();
    for factor in [0.5, f32::NAN, f32::INFINITY] {
        let file = fixture(factor).write();
        let g = GgufModel::open(file.path()).unwrap();
        assert!(Model::validate(&super::super::tests::NoUpload, &g).is_err());
    }
    let mut f = fixture(4.0);
    f.metadata
        .remove("qwen3.rope.scaling.original_context_length");
    let file = f.write();
    assert!(Model::validate(&CpuBackend, &GgufModel::open(file.path()).unwrap()).is_err());
    f.metadata.insert(
        "qwen3.rope.scaling.original_context_length".into(),
        Value::U32(16),
    );
    f.metadata
        .insert("qwen3.rope.scaling.yarn_beta_fast".into(), Value::F32(99.0));
    let file = f.write();
    assert!(Model::validate(&CpuBackend, &GgufModel::open(file.path()).unwrap()).is_err());
}

use super::{GgmlDType, GgufModel, TensorDesc, heaviest, tensor_n_bytes, tensor_start};
use candle_core::quantized::{
    QTensor,
    gguf_file::{self, Value},
};
use candle_core::{Device, Tensor};
use std::collections::BTreeMap;
use tempfile::NamedTempFile;

pub(crate) struct Fixture {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<(String, QTensor)>,
}

pub(crate) fn weight(shape: &[usize], dtype: GgmlDType, seed: usize) -> QTensor {
    let values: Vec<f32> = (0..shape.iter().product())
        .map(|i| {
            if shape.len() == 1 {
                1.0
            } else {
                ((i + seed) as f32 * 0.37).sin() * 0.2
            }
        })
        .collect();
    QTensor::quantize(
        &Tensor::from_vec(values, shape, &Device::Cpu).unwrap(),
        dtype
            .candle()
            .expect("fixture quantizer supports this dtype"),
    )
    .unwrap()
}

impl Fixture {
    pub fn new(layers: usize, width: usize) -> Self {
        let mut metadata = BTreeMap::from([
            (
                "general.architecture".into(),
                Value::String("internlm2".into()),
            ),
            ("general.type".into(), Value::String("model".into())),
            ("internlm2.rope.freq_base".into(), Value::F32(10000.0)),
            (
                "internlm2.attention.layer_norm_rms_epsilon".into(),
                Value::F32(1e-5),
            ),
        ]);
        for (suffix, value) in [
            ("block_count", layers),
            ("embedding_length", width),
            ("feed_forward_length", width * 2),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("context_length", 64),
        ] {
            metadata.insert(format!("internlm2.{suffix}"), Value::U32(value as u32));
        }
        let mut tensors = vec![(
            "token_embd.weight".into(),
            weight(&[128, width], GgmlDType::F32, 3),
        )];
        for i in 0..layers {
            for (suffix, shape, seed) in [
                ("attn_norm", vec![width], 0),
                ("attn_q", vec![width, width], 5),
                ("attn_k", vec![width / 2, width], 7),
                ("attn_v", vec![width / 2, width], 11),
                ("attn_output", vec![width, width], 13),
                ("ffn_norm", vec![width], 0),
                ("ffn_gate", vec![width * 2, width], 17),
                ("ffn_up", vec![width * 2, width], 19),
                ("ffn_down", vec![width, width * 2], 23),
            ] {
                tensors.push((
                    format!("blk.{i}.{suffix}.weight"),
                    weight(&shape, GgmlDType::F32, seed),
                ));
            }
        }
        tensors.push((
            "output_norm.weight".into(),
            weight(&[width], GgmlDType::F32, 0),
        ));
        tensors.push((
            "output.weight".into(),
            weight(&[128, width], GgmlDType::F32, 29),
        ));
        Self { metadata, tensors }
    }

    pub fn with_tokenizer(mut self) -> Self {
        self.metadata
            .insert("tokenizer.ggml.model".into(), Value::String("llama".into()));
        self.metadata.insert(
            "tokenizer.ggml.tokens".into(),
            Value::Array(
                (0..128)
                    .map(|i| Value::String(format!("<0x{i:02X}>")))
                    .collect(),
            ),
        );
        self.metadata.insert(
            "tokenizer.ggml.scores".into(),
            Value::Array(vec![Value::F32(0.0); 128]),
        );
        self.metadata
            .insert("tokenizer.ggml.add_space_prefix".into(), Value::Bool(false));
        self
    }

    pub fn write(&self) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        let metadata: Vec<_> = self.metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let tensors: Vec<_> = self.tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
        gguf_file::write(file.as_file_mut(), &metadata, &tensors).unwrap();
        file
    }

    pub fn replace(&mut self, name: &str, shape: &[usize], dtype: GgmlDType) {
        self.tensors.iter_mut().find(|(k, _)| k == name).unwrap().1 = weight(shape, dtype, 1);
    }

    /// Elements across every tensor.
    pub fn parameters(&self) -> u64 {
        self.tensors
            .iter()
            .map(|(_, tensor)| tensor.shape().elem_count() as u64)
            .sum()
    }
}

use crate::backend::test_support::fixture;

#[test]
#[ignore = "requires MDMA_TEST_GGUF (Bespoke-MiniCheck-7B-Q4_K_M.gguf)"]
fn reads_raw_tensor_bytes() {
    let mut model = GgufModel::open(fixture()).expect("open gguf");
    let name = model
        .tensor_names()
        .next()
        .expect("has tensors")
        .to_string();
    let desc = model.tensor_desc(&name).expect("has desc").clone();
    let bytes = model.read_tensor_bytes(&name).expect("read bytes");
    assert_eq!(bytes.len(), desc.n_bytes);
}

#[test]
fn tensor_byte_sizes_reject_overflow_and_partial_blocks() {
    assert_eq!(
        tensor_n_bytes("weight", GgmlDType::Q4K, &[2, 256]).unwrap(),
        288
    );
    for (dtype, shape) in [
        (GgmlDType::F32, vec![usize::MAX, 2]),
        (GgmlDType::F32, vec![usize::MAX / 4 + 1]),
        (GgmlDType::Q4K, vec![257]),
    ] {
        let error = tensor_n_bytes("bad.weight", dtype, &shape).unwrap_err();
        assert!(error.to_string().contains("bad.weight"), "{error}");
    }
}

#[test]
fn tensor_ranges_reject_overflow_and_truncation() {
    let mut desc = TensorDesc {
        dtype: GgmlDType::F32,
        shape: vec![8],
        offset: 16,
        n_bytes: 32,
    };
    assert_eq!(tensor_start("weight", &desc, 64, 112).unwrap(), 80);
    assert!(tensor_start("weight", &desc, 64, 111).is_err());
    assert!(tensor_start("weight", &desc, u64::MAX, u64::MAX).is_err());
    desc.offset = u64::MAX - 16;
    assert!(tensor_start("weight", &desc, 0, u64::MAX).is_err());
}

#[test]
fn header_names_the_file_type_else_the_type_holding_the_most_bytes() {
    let mut fixture = Fixture::new(1, 8);
    let open = |fixture: &Fixture| GgufModel::open(fixture.write().path()).unwrap();
    let all_f32 = open(&fixture);
    assert_eq!(all_f32.parameters(), 2648);
    assert_eq!(all_f32.parameters(), fixture.parameters());
    assert_eq!(all_f32.quantization(), "F32");
    // Embeddings and output in F16 outweigh the F32 rest.
    for name in ["token_embd.weight", "output.weight"] {
        fixture.replace(name, &[128, 8], GgmlDType::F16);
    }
    for (file_type, expected) in [
        (None, "F16"),
        (Some(Value::U32(15)), "Q4_K_M"),
        (Some(Value::U32(0)), "F32"),
        (Some(Value::U32(32)), "BF16"),
        (Some(Value::U32(38)), "MXFP4_MOE"),
        (Some(Value::U32(41)), "Q2_0"),
        // Removed from llama.cpp, unknown, or not a u32.
        (Some(Value::U32(4)), "F16"),
        (Some(Value::U32(1024)), "F16"),
        (Some(Value::String("Q4_K_M".into())), "F16"),
    ] {
        match &file_type {
            Some(value) => fixture
                .metadata
                .insert("general.file_type".into(), value.clone()),
            None => fixture.metadata.remove("general.file_type"),
        };
        let model = open(&fixture);
        assert_eq!(model.quantization(), expected, "{file_type:?}");
        assert_eq!(model.parameters(), 2648, "{file_type:?}");
    }
    assert_eq!(
        heaviest([("Q6_K", 10), ("Q4_K", 30), ("F32", 5), ("Q6_K", 25)]),
        Some("Q6_K")
    );
    assert_eq!(
        heaviest([("F32", 4), ("F16", 4)]),
        Some("F32"),
        "ties go last"
    );
    assert_eq!(heaviest::<&str>([]), None);
}

use super::dequantize_row;
use crate::gguf::GgmlDType;
use candle_core::Device;
use candle_core::quantized::gguf_file::Content;
use std::fs::File;

#[test]
fn q1_rejects_incomplete_rows_and_nonfinite_scales() {
    for (bytes, count) in [(vec![0; 17], 128), (vec![0; 18], 127), (vec![0; 36], 128)] {
        assert!(super::dequantize_row(GgmlDType::Q1_0, &bytes, &mut vec![0.; count]).is_err());
    }
    let mut bytes = vec![0; 18];
    bytes[..2].copy_from_slice(&half::f16::INFINITY.to_le_bytes());
    assert!(super::dequantize_row(GgmlDType::Q1_0, &bytes, &mut [0.; 128]).is_err());
}

#[test]
fn q1_matmul_and_embedding_match_independently_expanded_f32_weights() {
    use crate::backend::{
        Backend, Pool, QWeight,
        cpu::{CpuBackend, Matmul, qmatmul_in},
    };
    let mut packed = Vec::new();
    let mut expanded = Vec::new();
    for (bits, scale) in [(0xa5u8, 0.5f32), (0x3c, 2.0), (0x69, 0.25), (0xf0, 1.5)] {
        packed.extend(half::f16::from_f32(scale).to_le_bytes());
        packed.extend([bits; 16]);
        for i in 0..128 {
            expanded.push(if bits & (1 << (i % 8)) == 0 {
                -scale
            } else {
                scale
            });
        }
    }
    let b = CpuBackend;
    let weight = |dtype, bytes: &[u8]| {
        let mut buf = b.alloc(bytes.len(), Pool::Weights).unwrap();
        b.upload(&mut buf, 0, bytes).unwrap();
        QWeight {
            dtype,
            shape: vec![2, 256],
            buf,
        }
    };
    let q = weight(GgmlDType::Q1_0, &packed);
    let f = weight(GgmlDType::F32, bytemuck::cast_slice(&expanded));
    for count in [1, 3] {
        let input = (0..256 * count)
            .map(|i| (i as f32 * 0.03).sin())
            .collect::<Vec<_>>();
        let mut x = b.alloc(input.len() * 4, Pool::Scratch).unwrap();
        b.upload(&mut x, 0, bytemuck::cast_slice(&input)).unwrap();
        let mut want = b.alloc(2 * count * 4, Pool::Scratch).unwrap();
        let mut got = b.alloc(2 * count * 4, Pool::Scratch).unwrap();
        b.qmatmul(&f, &x, &mut want).unwrap();
        b.qmatmul(&q, &x, &mut got).unwrap();
        assert_eq!(got.0, want.0);
        let mut fallback = vec![0.; 2 * count];
        qmatmul_in(Matmul::Packed, &q, &input, &mut fallback).unwrap();
        assert_eq!(bytemuck::cast_slice::<f32, u8>(&fallback), want.0);
    }
    let mut want = b.alloc(512 * 4, Pool::Scratch).unwrap();
    let mut got = b.alloc(512 * 4, Pool::Scratch).unwrap();
    b.embed(&f, &[1, 0], &mut want).unwrap();
    b.embed(&q, &[1, 0], &mut got).unwrap();
    assert_eq!(got.0, want.0);
}

use crate::backend::test_support::fixture;

#[test]
#[ignore = "requires MDMA_TEST_GGUF (Bespoke-MiniCheck-7B-Q4_K_M.gguf)"]
fn matches_candles_own_dequant_for_a_q4k_tensor() {
    let mut file = File::open(fixture()).expect("open");
    let content = Content::read(&mut file).expect("read content");
    let (name, info) = content
        .tensor_infos
        .iter()
        .find(|(_, info)| info.ggml_dtype == GgmlDType::Q4K)
        .expect("fixture has a Q4_K tensor");
    let name = name.clone();

    let qtensor = content
        .tensor(&mut file, &name, &Device::Cpu)
        .expect("candle load");
    let expected = qtensor.dequantize(&Device::Cpu).expect("candle dequant");
    let expected: Vec<f32> = expected.flatten_all().unwrap().to_vec1().unwrap();

    let mut model = crate::gguf::GgufModel::open(fixture()).expect("open gguf");
    let raw = model.read_tensor_bytes(&name).expect("read raw bytes");
    let mut ours = vec![0f32; expected.len()];
    dequantize_row(info.ggml_dtype.into(), &raw, &mut ours).expect("dequantize_row");

    assert_eq!(ours.len(), expected.len());
    for (a, b) in ours.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-3, "mismatch: {a} vs {b}");
    }
}

#[test]
fn dequantize_row_rejects_misaligned_rows_and_decodes_aligned_ones() {
    let values = [1.5f32, -2.0, 0.25];
    let mut storage = vec![0u8; values.len() * 4 + 4];
    let offset = (0..4)
        .find(|&o| storage[o..].as_ptr().align_offset(align_of::<f32>()) == 0)
        .expect("an aligned offset");
    let misaligned = if offset == 0 { 1 } else { offset - 1 };
    for (i, v) in values.iter().enumerate() {
        storage[offset + i * 4..offset + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let mut out = [0f32; 3];
    dequantize_row(GgmlDType::F32, &storage[offset..offset + 12], &mut out).expect("aligned row");
    assert_eq!(out, values);
    let err = dequantize_row(
        GgmlDType::F32,
        &storage[misaligned..misaligned + 12],
        &mut out,
    );
    assert!(
        err.is_err(),
        "misaligned f32 row must be rejected, not reinterpreted"
    );
}

use super::*;
use crate::gguf::GgufModel;
use std::io::Write;
use tempfile::NamedTempFile;

fn parse(bytes: &[u8]) -> Result<Header> {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(bytes).unwrap();
    read(file.as_file_mut())
}

fn length(bytes: &mut Vec<u8>, version: u32, value: u64) {
    if version == 1 {
        bytes.extend((value as u32).to_le_bytes());
    } else {
        bytes.extend(value.to_le_bytes());
    }
}

fn string(bytes: &mut Vec<u8>, version: u32, value: &str) {
    length(bytes, version, value.len() as u64);
    bytes.extend(value.as_bytes());
}

fn start(version: u32, tensors: u64, metadata: u64) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(version.to_le_bytes());
    length(&mut bytes, version, tensors);
    length(&mut bytes, version, metadata);
    bytes
}

fn descriptor(bytes: &mut Vec<u8>, version: u32, dtype: u32) {
    string(bytes, version, "weight");
    bytes.extend(2u32.to_le_bytes());
    length(bytes, version, 128);
    length(bytes, version, 1);
    bytes.extend(dtype.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
}

#[test]
fn native_q1_descriptors_and_payload_remain_packed_in_all_container_versions() {
    for version in 1..=3 {
        let mut bytes = start(version, 1, 1);
        string(&mut bytes, version, "general.architecture");
        bytes.extend(8u32.to_le_bytes());
        string(&mut bytes, version, "qwen3");
        descriptor(&mut bytes, version, 41);
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        let payload = [&[0, 0x3c][..], &[0xaa; 16][..]].concat();
        bytes.extend(&payload);
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        let mut model = GgufModel::open(file.path()).unwrap();
        let desc = model.tensor_desc("weight").unwrap();
        assert_eq!(desc.dtype, GgmlDType::Q1_0);
        assert_eq!(desc.shape, [1, 128]);
        assert_eq!(desc.n_bytes, 18);
        assert_eq!(model.quantization(), "Q1_0");
        assert_eq!(model.metadata_str("general.architecture"), Some("qwen3"));
        model.validate_tensor_ranges().unwrap();
        assert_eq!(model.read_tensor_bytes("weight").unwrap(), payload);
    }
}

#[test]
fn metadata_matches_candles_writer_and_reader_for_every_value_kind() {
    use candle_core::quantized::gguf_file;
    let mut fixture = crate::gguf::tests::Fixture::new(1, 8);
    let values = [
        Value::U8(255),
        Value::I8(-127),
        Value::U16(65535),
        Value::I16(-1234),
        Value::U32(u32::MAX),
        Value::I32(-123456),
        Value::F32(0.125),
        Value::Bool(true),
        Value::String("é🦀".into()),
        Value::Array(vec![Value::String("α".into()), Value::String("β".into())]),
        Value::U64(u64::MAX),
        Value::I64(i64::MIN),
        Value::F64(1.25),
        Value::Array(vec![Value::Array(vec![Value::U32(7)])]),
    ];
    for (i, value) in values.into_iter().enumerate() {
        fixture.metadata.insert(format!("test.{i}"), value);
    }
    let mut file = fixture.write();
    let ours = read(file.as_file_mut()).unwrap();
    file.as_file_mut().seek(SeekFrom::Start(0)).unwrap();
    let candle = gguf_file::Content::read(file.as_file_mut()).unwrap();
    for (key, expected) in &candle.metadata {
        assert_eq!(
            format!("{:?}", ours.metadata[key]),
            format!("{expected:?}"),
            "{key}"
        );
    }
    assert_eq!(ours.data_offset, candle.tensor_data_offset);
    for (name, expected) in candle.tensor_infos {
        let actual = &ours.tensors[&name];
        assert_eq!(actual.dtype, expected.ggml_dtype);
        assert_eq!(actual.shape, expected.shape.dims());
        assert_eq!(actual.offset, expected.offset);
    }
}

#[test]
fn types_match_candle_sizes_and_ids_without_reinterpreting_q1() {
    for code in [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 30] {
        let dtype = GgmlDType::from_code(code).unwrap();
        let candle = dtype.candle().unwrap();
        assert_eq!(dtype.type_size(), candle.type_size());
        assert_eq!(dtype.block_size(), candle.block_size());
        assert_eq!(GgmlDType::from(candle), dtype);
    }
    assert_eq!(GgmlDType::Q1_0.block_size(), 128);
    assert_eq!(GgmlDType::Q1_0.type_size(), 18);
    assert_eq!(GgmlDType::Q1_0.candle(), None);
    assert!(GgmlDType::from_code(42).is_err(), "Q2_0 is not Q1_0");
}

#[test]
fn rejects_truncated_or_impossible_headers_without_panicking() {
    for bytes in [
        vec![],
        b"nope".to_vec(),
        start(4, 0, 0),
        start(3, u64::MAX, 0),
        start(3, 0, u64::MAX),
    ] {
        assert!(parse(&bytes).is_err());
    }
    let mut bytes = start(3, 1, 0);
    descriptor(&mut bytes, 3, 41);
    for end in 0..bytes.len() {
        assert!(parse(&bytes[..end]).is_err(), "truncation at {end}");
    }
    let mut unsupported = start(3, 1, 0);
    descriptor(&mut unsupported, 3, 9999);
    assert!(parse(&unsupported).is_err());
}

#[test]
fn rejects_zero_non_power_of_two_and_oversized_alignment() {
    for value in [0, 3, u32::MAX] {
        let mut bytes = start(3, 0, 1);
        string(&mut bytes, 3, "general.alignment");
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
        assert!(parse(&bytes).is_err(), "alignment={value}");
    }
}

#[test]
fn rejects_duplicate_keys_and_tensor_names() {
    let mut bytes = start(3, 0, 2);
    for _ in 0..2 {
        string(&mut bytes, 3, "same");
        bytes.extend(7u32.to_le_bytes());
        bytes.push(1);
    }
    assert!(parse(&bytes).is_err());
    let mut bytes = start(3, 2, 0);
    descriptor(&mut bytes, 3, 41);
    descriptor(&mut bytes, 3, 41);
    assert!(parse(&bytes).is_err());
}

#[test]
fn rejects_invalid_booleans_utf8_arrays_and_deep_nesting() {
    let mut boolean = start(3, 0, 1);
    string(&mut boolean, 3, "bool");
    boolean.extend(7u32.to_le_bytes());
    boolean.push(2);
    assert!(parse(&boolean).is_err());

    let mut utf8 = start(3, 0, 1);
    length(&mut utf8, 3, 1);
    utf8.push(0xff);
    utf8.extend(7u32.to_le_bytes());
    utf8.push(1);
    assert!(parse(&utf8).is_err());

    let mut array = start(3, 0, 1);
    string(&mut array, 3, "array");
    array.extend(9u32.to_le_bytes());
    array.extend(0u32.to_le_bytes());
    length(&mut array, 3, u64::MAX);
    assert!(parse(&array).is_err());

    let mut deep = start(3, 0, 1);
    string(&mut deep, 3, "deep");
    deep.extend(9u32.to_le_bytes());
    for _ in 0..66 {
        deep.extend(9u32.to_le_bytes());
        length(&mut deep, 3, 1);
    }
    deep.extend(0u32.to_le_bytes());
    length(&mut deep, 3, 1);
    deep.push(0);
    assert!(parse(&deep).is_err());
}

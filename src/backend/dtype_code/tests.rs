use super::*;
use std::assert_matches;

#[test]
fn kernel_tags_and_superblock_sizes_are_stable() {
    for (dtype, tag, bytes) in [
        (GgmlDType::F32, 0, 0),
        (GgmlDType::F16, 1, 0),
        (GgmlDType::Q4K, 2, 144),
        (GgmlDType::Q6K, 3, 210),
    ] {
        assert_eq!(code(dtype).unwrap(), tag);
        assert_eq!(block_bytes(dtype), bytes);
        assert_eq!(is_kquant(tag), bytes != 0);
    }
}

#[test]
fn unsupported_formats_never_get_a_kernel_tag() {
    for dtype in [
        GgmlDType::BF16,
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q5K,
        GgmlDType::Q8K,
    ] {
        assert_matches!(code(dtype), Err(Error::NotImplemented(_)));
    }
}

#[test]
fn decoder_formats_do_not_change_legacy_capabilities() {
    for (dtype, tag, quant, size) in [
        (GgmlDType::BF16, 4, false, 0),
        (GgmlDType::Q5K, 5, true, 176),
        (GgmlDType::Q8_0, 6, false, 0),
    ] {
        assert!(code(dtype).is_err());
        assert_eq!(decoder_code(dtype).unwrap(), tag);
        assert_eq!(is_kquant(tag), quant);
        assert_eq!(block_bytes(dtype), size);
    }
}

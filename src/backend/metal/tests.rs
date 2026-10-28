use super::*;
use crate::backend::cpu::CpuBackend;
use crate::backend::test_support::{cpu_qmatmul, parity, qmatmul as gpu_qmatmul};
use crate::gguf::GgmlDType;
use std::{assert_matches, sync::OnceLock};

mod batching;
mod benchmarks;
mod fusion;
mod prefill;
mod hybrid;

#[test]
fn host_and_shader_use_shared_dtype_tags() {
    let source = abi::source(true);
    for (dtype, name) in [
        (GgmlDType::F32, "F32"),
        (GgmlDType::F16, "F16"),
        (GgmlDType::BF16, "BF16"),
        (GgmlDType::Q4K, "Q4K"),
        (GgmlDType::Q5K, "Q5K"),
        (GgmlDType::Q6K, "Q6K"),
    ] {
        let raw = packed(dtype, 1, 256);
        let w = weight(gpu(), dtype, 1, 256, &raw);
        let code = dtype_code::decoder_code(dtype).unwrap();
        assert_eq!(weight_dims(&w).unwrap().2, code.cast_unsigned());
        assert!(source.contains(&format!("#define DTYPE_{name} {code}\n")));
    }
}

#[test]
fn binary_dispatch_boundaries_match_cpu() {
    parity::binary_dispatch_tails(gpu());
}

#[test]
fn elementwise_oversized_outputs_preserve_canaries() {
    let gpu = gpu();
    let cpu = CpuBackend;
    for n in [0, 1, 31, 32, 33, 255, 256, 257] {
        let a = data(n, 3);
        let b = data(n, 7);
        let (ac, bc) = (fbuf(&cpu, &a), fbuf(&cpu, &b));
        let (ag, bg) = (fbuf(gpu, &a), fbuf(gpu, &b));
        // Empty dispatches test Metal's no-op behavior. The scalar CPU
        // oracle uses nonempty, aligned activation buffers.
        for op in 0..3 {
            let mut yc = fbuf(&cpu, &vec![12345.0; n + 7]);
            let mut yg = fbuf(gpu, &vec![12345.0; n + 7]);
            match op {
                0 => {
                    if n > 0 {
                        cpu.add(&ac, &bc, &mut yc).unwrap();
                    }
                    gpu.add(&ag, &bg, &mut yg).unwrap();
                }
                1 => {
                    if n > 0 {
                        cpu.silu_mul(&ac, &bc, &mut yc).unwrap();
                    }
                    gpu.silu_mul(&ag, &bg, &mut yg).unwrap();
                }
                _ => {
                    if n > 0 {
                        cpu.gelu_tanh_mul(&ac, &bc, &mut yc).unwrap();
                    }
                    gpu.gelu_tanh_mul(&ag, &bg, &mut yg).unwrap();
                }
            }
            let got = read(gpu, &yg, n + 7);
            close(&got[..n], &read(&cpu, &yc, n + 7)[..n], 2e-6, 2e-6);
            assert_eq!(&got[n..], &[12345.0; 7]);
        }
    }
}

#[test]
fn shared_decoder_scenarios_match_cpu() {
    let gpu = gpu();
    parity::asymmetric_matmul(gpu);
    parity::matmul_row_tail(gpu);
    parity::elementwise_norm_rope(gpu);
    parity::attention_f32_prefix(gpu);
    parity::embed_f32(gpu);
}

// Share immutable device/pipeline state; every test owns its buffers. Do not
// silently skip on machines without a GPU: --features metal is an explicit
// request to exercise Metal, including compiling the embedded shader source.
fn gpu() -> &'static MetalBackend {
    static GPU: OnceLock<MetalBackend> = OnceLock::new();
    GPU.get_or_init(|| MetalBackend::new().expect("unified-memory Metal GPU"))
}

fn data(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 17 + seed * 13) % 97) as f32 / 31.0 - 1.3)
        .collect()
}

fn upload<B: Backend>(backend: &B, bytes: &[u8]) -> B::Buf {
    let mut buf = backend.alloc(bytes.len(), Pool::Scratch).unwrap();
    backend.upload(&mut buf, 0, bytes).unwrap();
    buf
}

fn fbuf<B: Backend>(backend: &B, values: &[f32]) -> B::Buf {
    upload(backend, bytemuck::cast_slice(values))
}

fn read<B: Backend>(backend: &B, buf: &B::Buf, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    backend
        .download(buf, 0, bytemuck::cast_slice_mut(&mut out))
        .unwrap();
    out
}

fn close(got: &[f32], want: &[f32], atol: f32, rtol: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
        assert!(
            got.is_finite() && want.is_finite(),
            "nonfinite result at {i}: {got} vs {want}"
        );
        assert!(
            (got - want).abs() <= atol + rtol * want.abs(),
            "index {i}: {got} vs {want}"
        );
    }
}

fn weight<B: Backend>(
    backend: &B,
    dtype: GgmlDType,
    rows: usize,
    cols: usize,
    raw: &[u8],
) -> QWeight<B::Buf> {
    QWeight {
        dtype,
        shape: vec![rows, cols],
        buf: upload(backend, raw),
    }
}

// Exercise Q4 scale/min high bits, Q6 quadrants, signed scales and adjacent blocks.
fn packed(dtype: GgmlDType, rows: usize, cols: usize) -> Vec<u8> {
    match dtype {
        GgmlDType::F32 => bytemuck::cast_slice(&data(rows * cols, 7)).to_vec(),
        GgmlDType::F16 => data(rows * cols, 7)
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
            .collect(),
        GgmlDType::BF16 => data(rows * cols, 7)
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_bits().to_le_bytes())
            .collect(),
        GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => {
            let size = dtype.type_size();
            let mut bytes = vec![0u8; rows * (cols / 256) * size];
            for (block, raw) in bytes.chunks_exact_mut(size).enumerate() {
                for (i, byte) in raw.iter_mut().enumerate() {
                    *byte = ((i * 37 + block * 53 + 19) % 256) as u8;
                }
                let d = half::f16::from_f32(0.002 + (block % 7) as f32 * 0.001)
                    .to_bits()
                    .to_le_bytes();
                if matches!(dtype, GgmlDType::Q4K | GgmlDType::Q5K) {
                    raw[..2].copy_from_slice(&d);
                    raw[2..4].copy_from_slice(&half::f16::from_f32(0.003).to_bits().to_le_bytes());
                } else {
                    raw[208..210].copy_from_slice(&d);
                }
            }
            bytes
        }
        _ => unreachable!(),
    }
}

fn matmul_case(dtype: GgmlDType, rows: usize, cols: usize) {
    let gpu = gpu();
    let raw = packed(dtype, rows, cols);
    for tokens in [1, 3] {
        let x = data(tokens * cols, 11);
        let want = cpu_qmatmul(dtype, &raw, &[rows, cols], &x);
        let got = gpu_qmatmul(gpu, dtype, &raw, &[rows, cols], &x);
        close(&got, &want, 2e-3, 2e-5);
    }
}

#[test]
fn qmatmul_f32_decode_and_prefill() {
    matmul_case(GgmlDType::F32, 67, 37);
}

#[test]
fn qmatmul_f16_decode_and_prefill() {
    matmul_case(GgmlDType::F16, 67, 39);
}

#[test]
fn qmatmul_q4k_decode_and_prefill() {
    matmul_case(GgmlDType::Q4K, 67, 512);
}

#[test]
fn qmatmul_q6k_decode_and_prefill() {
    matmul_case(GgmlDType::Q6K, 67, 512);
}

#[test]
fn tinyllama_projection_geometry_matches_cpu() {
    matmul_case(GgmlDType::Q4K, 2048, 2048);
    matmul_case(GgmlDType::Q6K, 2048, 5632);
}

#[test]
fn tinyllama_embedding_rope_and_attention_geometry_matches_cpu() {
    let gpu = gpu();
    let cpu = CpuBackend;
    let raw = packed(GgmlDType::Q6K, 17, 2048);
    let wc = weight(&cpu, GgmlDType::Q6K, 17, 2048, &raw);
    let wg = weight(gpu, GgmlDType::Q6K, 17, 2048, &raw);
    let ids = [16, 0, 7];
    let n = ids.len() * 2048;
    let mut yc = cpu.alloc(n * 4, Pool::Scratch).unwrap();
    let mut yg = gpu.alloc(n * 4, Pool::Scratch).unwrap();
    cpu.embed(&wc, &ids, &mut yc).unwrap();
    gpu.embed(&wg, &ids, &mut yg).unwrap();
    close(&read(gpu, &yg, n), &read(&cpu, &yc, n), 2e-6, 2e-6);
    let spec = RopeSpec {
        head_dim: 64,
        theta: 10000.0,
    };
    let q = data(3 * 32 * 64, 3);
    let k = data(3 * 4 * 64, 5);
    let (mut qc, mut kc) = (fbuf(&cpu, &q), fbuf(&cpu, &k));
    let (mut qg, mut kg) = (fbuf(gpu, &q), fbuf(gpu, &k));
    cpu.rope(&mut qc, &mut kc, 2045, 3, &spec).unwrap();
    gpu.rope(&mut qg, &mut kg, 2045, 3, &spec).unwrap();
    close(
        &read(gpu, &qg, q.len()),
        &read(&cpu, &qc, q.len()),
        1e-4,
        1e-4,
    );
    close(
        &read(gpu, &kg, k.len()),
        &read(&cpu, &kc, k.len()),
        1e-4,
        1e-4,
    );
    for tokens in [1, 3] {
        attention_with_spec(
            tokens,
            2048,
            1.0,
            AttnSpec {
                head_dim: 64,
                n_q_heads: 32,
                n_kv_heads: 4,
            },
        );
    }
}

#[test]
fn bf16_and_q5k_matmul_compat() {
    matmul_case(GgmlDType::BF16, 67, 39);
    matmul_case(GgmlDType::Q5K, 67, 512);
}

#[test]
fn cooperative_matmul_handles_long_rows_and_column_tails() {
    // Five rows leave a partial final workgroup; 4097 leaves a partial
    // SIMD-width column strip. Keep the independent CPU oracle.
    matmul_case(GgmlDType::F32, 5, 4097);
    matmul_case(GgmlDType::F16, 5, 4097);
    matmul_case(GgmlDType::Q4K, 5, 4096);
    matmul_case(GgmlDType::Q6K, 5, 4096);
}

#[test]
fn embeddings_all_supported_formats_and_repeated_ids() {
    let cpu = CpuBackend;
    let gpu = gpu();
    for dtype in [
        GgmlDType::F32,
        GgmlDType::F16,
        GgmlDType::BF16,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        let (rows, cols) = (17, 512);
        let raw = packed(dtype, rows, cols);
        let wc = weight(&cpu, dtype, rows, cols, &raw);
        let wg = weight(gpu, dtype, rows, cols, &raw);
        let ids = [16, 0, 7, 16, 3];
        let n = ids.len() * cols;
        let mut yc = cpu.alloc(n * 4, Pool::Scratch).unwrap();
        let mut yg = gpu.alloc(n * 4, Pool::Scratch).unwrap();
        cpu.embed(&wc, &ids, &mut yc).unwrap();
        gpu.embed(&wg, &ids, &mut yg).unwrap();
        close(&read(gpu, &yg, n), &read(&cpu, &yc, n), 2e-6, 2e-6);
    }
}

#[test]
fn shared_buffers_zero_init_byte_offsets_and_bounds() {
    let gpu = gpu();
    let mut a = gpu.alloc(19, Pool::Weights).unwrap();
    let mut b = gpu.alloc(23, Pool::Kv).unwrap();
    let mut got = [0xff; 23];
    gpu.download(&b, 0, &mut got).unwrap();
    assert_eq!(got, [0; 23]);
    gpu.upload(&mut a, 3, &[11, 22, 33, 44, 55]).unwrap();
    gpu.copy(&a, 4, &mut b, 7, 4).unwrap();
    gpu.download(&b, 0, &mut got).unwrap();
    assert_eq!(&got[7..11], &[22, 33, 44, 55]);
    assert!(got[..7].iter().chain(&got[11..]).all(|&v| v == 0));
    gpu.copy(&a, 19, &mut b, 23, 0).unwrap();
    assert!(gpu.copy(&a, 18, &mut b, 0, 2).is_err());
    assert!(gpu.copy(&a, 0, &mut b, 22, 2).is_err());
    assert!(gpu.copy(&a, usize::MAX, &mut b, 0, 1).is_err());
    assert!(gpu.upload(&mut a, usize::MAX, &[1]).is_err());
    assert!(gpu.download(&a, usize::MAX, &mut [0]).is_err());
    assert!(gpu.download(&a, 19, &mut [0]).is_err());
    assert!(gpu.alloc(usize::MAX, Pool::Scratch).is_err());
}

#[test]
fn empty_buffers_and_empty_embedding_are_safe_noops() {
    let gpu = gpu();
    let a = gpu.alloc(0, Pool::Scratch).unwrap();
    let b = gpu.alloc(0, Pool::Scratch).unwrap();
    let mut out = gpu.alloc(0, Pool::Scratch).unwrap();
    gpu.upload(&mut out, 0, &[]).unwrap();
    gpu.download(&out, 0, &mut []).unwrap();
    gpu.copy(&a, 0, &mut out, 0, 0).unwrap();
    gpu.add(&a, &b, &mut out).unwrap();
    gpu.silu_mul(&a, &b, &mut out).unwrap();
    let table = weight(gpu, GgmlDType::F32, 1, 1, &1.0f32.to_le_bytes());
    gpu.embed(&table, &[], &mut out).unwrap();
}

#[test]
fn add_silu_and_gpu_copy_chain_match_cpu() {
    let cpu = CpuBackend;
    let gpu = gpu();
    let n = 257;
    let mut a = data(n, 3);
    a[0] = -100.0;
    a[1] = 100.0;
    let b = data(n, 5);
    let ac = fbuf(&cpu, &a);
    let bc = fbuf(&cpu, &b);
    let ag = fbuf(gpu, &a);
    let bg = fbuf(gpu, &b);
    let mut yc = cpu.alloc(n * 4, Pool::Scratch).unwrap();
    let mut yg = gpu.alloc(n * 4, Pool::Scratch).unwrap();
    cpu.add(&ac, &bc, &mut yc).unwrap();
    gpu.add(&ag, &bg, &mut yg).unwrap();
    close(&read(gpu, &yg, n), &read(&cpu, &yc, n), 1e-6, 1e-6);
    cpu.silu_mul(&ac, &bc, &mut yc).unwrap();
    gpu.silu_mul(&ag, &bg, &mut yg).unwrap();
    let mut copied = gpu.alloc(n * 4, Pool::Scratch).unwrap();
    gpu.copy(&yg, 0, &mut copied, 0, n * 4).unwrap();
    close(&read(gpu, &copied, n), &read(&cpu, &yc, n), 2e-6, 2e-6);
}

#[test]
fn rmsnorm_is_per_token_including_zero_rows() {
    let cpu = CpuBackend;
    let gpu = gpu();
    let dim = 37;
    let mut x = data(3 * dim, 5);
    x[dim..2 * dim].fill(0.0);
    let w = data(dim, 13);
    let wc = fbuf(&cpu, &w);
    let wg = fbuf(gpu, &w);
    let xc = fbuf(&cpu, &x);
    let xg = fbuf(gpu, &x);
    let mut yc = cpu.alloc(x.len() * 4, Pool::Scratch).unwrap();
    let mut yg = gpu.alloc(x.len() * 4, Pool::Scratch).unwrap();
    cpu.rmsnorm(&wc, &xc, &mut yc, 1e-5).unwrap();
    gpu.rmsnorm(&wg, &xg, &mut yg, 1e-5).unwrap();
    close(
        &read(gpu, &yg, x.len()),
        &read(&cpu, &yc, x.len()),
        2e-6,
        2e-6,
    );
}

#[test]
fn cooperative_rmsnorm_handles_lane_boundaries_and_long_rows() {
    let cpu = CpuBackend;
    let gpu = gpu();
    let lanes = gpu.pipeline("rmsnorm_f32").unwrap().threadExecutionWidth();
    assert!(lanes > 1);
    for width in [1, lanes - 1, lanes, lanes + 1, 4096, 4097] {
        let mut x = data(5 * width, 3);
        x[4 * width..].fill(0.0);
        let w = data(width, 17);
        let wc = fbuf(&cpu, &w);
        let wg = fbuf(gpu, &w);
        let xc = fbuf(&cpu, &x);
        let xg = fbuf(gpu, &x);
        let mut yc = cpu.alloc(x.len() * 4, Pool::Scratch).unwrap();
        let mut yg = gpu.alloc(x.len() * 4, Pool::Scratch).unwrap();
        cpu.rmsnorm(&wc, &xc, &mut yc, 1e-5).unwrap();
        gpu.rmsnorm(&wg, &xg, &mut yg, 1e-5).unwrap();
        close(
            &read(gpu, &yg, x.len()),
            &read(&cpu, &yc, x.len()),
            1e-5,
            1e-5,
        );
    }
}

#[test]
fn rope_and_kv_write_match_cpu_at_nonzero_position() {
    let cpu = CpuBackend;
    let gpu = gpu();
    let spec = RopeSpec {
        head_dim: 10,
        theta: 10000.0,
    };
    let tokens = 3;
    let q = data(tokens * 3 * spec.head_dim, 3);
    let k = data(tokens * spec.head_dim, 5);
    let v = data(k.len(), 7);
    let mut qc = fbuf(&cpu, &q);
    let mut kc = fbuf(&cpu, &k);
    let vc = fbuf(&cpu, &v);
    let mut qg = fbuf(gpu, &q);
    let mut kg = fbuf(gpu, &k);
    let vg = fbuf(gpu, &v);
    let offset = 7 * spec.head_dim;
    let capacity = offset + k.len() + spec.head_dim;
    let mut ckc = cpu.alloc(capacity * 4, Pool::Kv).unwrap();
    let mut cvc = cpu.alloc(capacity * 4, Pool::Kv).unwrap();
    let mut ckg = gpu.alloc(capacity * 4, Pool::Kv).unwrap();
    let mut cvg = gpu.alloc(capacity * 4, Pool::Kv).unwrap();
    cpu.rope_kv_write(
        &mut qc,
        &mut kc,
        &vc,
        &mut ckc,
        &mut cvc,
        offset,
        7,
        tokens,
        &spec,
        k.len(),
    )
    .unwrap();
    gpu.rope_kv_write(
        &mut qg,
        &mut kg,
        &vg,
        &mut ckg,
        &mut cvg,
        offset,
        7,
        tokens,
        &spec,
        k.len(),
    )
    .unwrap();
    close(
        &read(gpu, &qg, q.len()),
        &read(&cpu, &qc, q.len()),
        2e-5,
        2e-5,
    );
    close(
        &read(gpu, &kg, k.len()),
        &read(&cpu, &kc, k.len()),
        2e-5,
        2e-5,
    );
    close(
        &read(gpu, &ckg, capacity),
        &read(&cpu, &ckc, capacity),
        2e-5,
        2e-5,
    );
    close(
        &read(gpu, &cvg, capacity),
        &read(&cpu, &cvc, capacity),
        0.0,
        0.0,
    );
}

fn attention_case(tokens: usize, kv_len: usize, magnitude: f32) {
    attention_with_spec(
        tokens,
        kv_len,
        magnitude,
        AttnSpec {
            head_dim: 7,
            n_q_heads: 6,
            n_kv_heads: 2,
        },
    );
}

fn attention_with_spec(tokens: usize, kv_len: usize, magnitude: f32, spec: AttnSpec) {
    let cpu = CpuBackend;
    let gpu = gpu();
    let q: Vec<f32> = data(tokens * spec.n_q_heads * spec.head_dim, 7)
        .iter()
        .map(|x| x * magnitude)
        .collect();
    let mut k = data((kv_len + 2) * spec.n_kv_heads * spec.head_dim, 3);
    let mut v = data(k.len(), 11);
    // Uninitialized cache capacity must not participate in attention.
    k[kv_len * spec.n_kv_heads * spec.head_dim..].fill(1000.0);
    v[kv_len * spec.n_kv_heads * spec.head_dim..].fill(-1000.0);
    let qc = fbuf(&cpu, &q);
    let kc = fbuf(&cpu, &k);
    let vc = fbuf(&cpu, &v);
    let qg = fbuf(gpu, &q);
    let kg = fbuf(gpu, &k);
    let vg = fbuf(gpu, &v);
    let mut oc = cpu.alloc(q.len() * 4, Pool::Scratch).unwrap();
    let mut og = gpu.alloc(q.len() * 4, Pool::Scratch).unwrap();
    cpu.attn(
        &qc,
        &KvView {
            k: &kc,
            v: &vc,
            len: kv_len,
        },
        &spec,
        &mut oc,
    )
    .unwrap();
    gpu.attn(
        &qg,
        &KvView {
            k: &kg,
            v: &vg,
            len: kv_len,
        },
        &spec,
        &mut og,
    )
    .unwrap();
    close(
        &read(gpu, &og, q.len()),
        &read(&cpu, &oc, q.len()),
        5e-5,
        5e-5,
    );
}

#[test]
fn causal_gqa_prefill_with_and_without_cached_prefix() {
    attention_case(3, 3, 1.0);
    attention_case(3, 8, 1.0);
}

#[test]
fn attention_decode_and_stable_softmax() {
    attention_case(1, 37, 1.0);
    attention_case(3, 8, 100.0);
}

#[test]
fn online_attention_handles_lane_tails_and_scalar_fallback() {
    let lanes = gpu().pipeline("attn_simd").unwrap().threadExecutionWidth();
    assert!(lanes > 1);
    for head_dim in [
        1,
        lanes - 1,
        lanes,
        lanes + 1,
        4 * lanes,
        8 * lanes,
        8 * lanes + 1,
    ] {
        attention_with_spec(
            3,
            17,
            1.0,
            AttnSpec {
                head_dim,
                n_q_heads: 6,
                n_kv_heads: 2,
            },
        );
    }
    attention_with_spec(
        3,
        17,
        100.0,
        AttnSpec {
            head_dim: 128,
            n_q_heads: 6,
            n_kv_heads: 2,
        },
    );
}

#[test]
fn online_attention_matches_cpu_with_a_long_prefix() {
    attention_with_spec(
        1,
        1025,
        1.0,
        AttnSpec {
            head_dim: 128,
            n_q_heads: 6,
            n_kv_heads: 2,
        },
    );
}

#[test]
fn malformed_shapes_are_rejected_before_dispatch() {
    let gpu = gpu();
    let bad = gpu.alloc(3, Pool::Scratch).unwrap();
    let x = fbuf(gpu, &[1.0; 8]);
    let mut y = fbuf(gpu, &[0.0; 8]);
    assert!(gpu.add(&bad, &x, &mut y).is_err());
    assert!(gpu.silu_mul(&x, &bad, &mut y).is_err());
    assert!(gpu.rmsnorm(&bad, &x, &mut y, 1e-5).is_err());
    assert!(gpu.rmsnorm(&x, &x, &mut y, f32::NAN).is_err());
    let zero = gpu.alloc(0, Pool::Weights).unwrap();
    assert!(gpu.rmsnorm(&zero, &x, &mut y, 1e-5).is_err());
    let short = fbuf(gpu, &[0.0]);
    assert!(gpu.add(&x, &short, &mut y).is_err());

    let mut w = weight(gpu, GgmlDType::F32, 2, 4, &[0; 32]);
    assert!(gpu.qmatmul(&w, &x, &mut y).is_err()); // output too long
    w.shape = vec![8];
    assert!(gpu.qmatmul(&w, &x, &mut y).is_err());
    w.shape = vec![2, 0];
    assert!(gpu.qmatmul(&w, &x, &mut y).is_err());
    w.shape = vec![2, 8];
    assert!(gpu.qmatmul(&w, &x, &mut y).is_err()); // truncated weight
    w.shape = vec![2, 4];
    w.dtype = GgmlDType::Q4K;
    assert!(gpu.qmatmul(&w, &x, &mut y).is_err()); // not block-aligned
    w.dtype = GgmlDType::Q8K;
    assert_matches!(gpu.qmatmul(&w, &x, &mut y), Err(Error::NotImplemented(_)));
    w.dtype = GgmlDType::F32;
    assert!(gpu.embed(&w, &[0, 2], &mut y).is_err()); // invalid id
    let mut empty = gpu.alloc(0, Pool::Scratch).unwrap();
    assert!(gpu.qmatmul(&w, &zero, &mut empty).is_err());
}

#[test]
fn invalid_rope_and_attention_dimensions_are_rejected() {
    let gpu = gpu();
    let mut q = fbuf(gpu, &[1.0; 16]);
    let mut k = fbuf(gpu, &[1.0; 8]);
    let spec = RopeSpec {
        head_dim: 4,
        theta: 10000.0,
    };
    assert!(gpu.rope(&mut q, &mut k, 0, 0, &spec).is_err());
    assert!(gpu.rope(&mut q, &mut k, u32::MAX, 2, &spec).is_err());
    for head_dim in [0, 3, 6] {
        assert!(
            gpu.rope(
                &mut q,
                &mut k,
                0,
                1,
                &RopeSpec {
                    head_dim,
                    theta: 10000.0
                }
            )
            .is_err()
        );
    }
    assert!(
        gpu.rope(
            &mut q,
            &mut k,
            0,
            1,
            &RopeSpec {
                head_dim: 4,
                theta: 0.0
            }
        )
        .is_err()
    );
    let v = fbuf(gpu, &[1.0; 8]);
    let kv = KvView {
        k: &k,
        v: &v,
        len: 1,
    };
    let mut out = fbuf(gpu, &[0.0; 16]);
    for (head_dim, n_q_heads, n_kv_heads) in [(0, 4, 2), (4, 0, 2), (4, 4, 0), (4, 4, 3)] {
        assert!(
            gpu.attn(
                &q,
                &kv,
                &AttnSpec {
                    head_dim,
                    n_q_heads,
                    n_kv_heads
                },
                &mut out
            )
            .is_err()
        );
    }
    let spec = AttnSpec {
        head_dim: 4,
        n_q_heads: 2,
        n_kv_heads: 1,
    };
    assert!(gpu.attn(&q, &kv, &spec, &mut out).is_err()); // 2 queries, 1 key
    let kv = KvView {
        k: &k,
        v: &v,
        len: 3,
    };
    assert!(gpu.attn(&q, &kv, &spec, &mut out).is_err()); // undersized KV
}

fn tiny_logits<B: Backend>(backend: &B) -> Vec<Vec<f32>> {
    let model = crate::model::tests::tiny_model(backend);
    let mut kv = model.new_kv_cache(backend, 16).unwrap();
    let mut prefill = model.alloc_workspace(backend, 3).unwrap();
    let mut decode = model.alloc_workspace(backend, 1).unwrap();
    let mut outputs = vec![
        model
            .prefill(backend, &mut kv, &mut prefill, &[1, 7, 13], 0)
            .unwrap(),
    ];
    for (i, token) in [19, 23, 29].into_iter().enumerate() {
        outputs.push(
            model
                .forward(backend, &mut kv, &mut decode, token, (i + 3) as u32)
                .unwrap(),
        );
    }
    outputs
}

#[test]
fn full_tiny_model_prefill_then_decode_matches_cpu() {
    let want = tiny_logits(&CpuBackend);
    let got = tiny_logits(gpu());
    for (got, want) in got.iter().zip(&want) {
        close(got, want, 2e-5, 2e-5);
        assert_eq!(crate::sampler::greedy(got), crate::sampler::greedy(want));
    }
}

#[test]
fn tiny_model_batched_prefill_matches_sequential_decode() {
    let gpu = gpu();
    let model = crate::model::tests::tiny_model(gpu);
    let mut kv = model.new_kv_cache(gpu, 16).unwrap();
    let mut ws = model.alloc_workspace(gpu, 1).unwrap();
    let mut logits = Vec::new();
    for (i, token) in [1, 7, 13, 19, 23, 29].into_iter().enumerate() {
        logits = model
            .forward(gpu, &mut kv, &mut ws, token, i as u32)
            .unwrap();
    }
    close(&logits, tiny_logits(gpu).last().unwrap(), 2e-5, 2e-5);
}

#[test]
fn backend_and_buffers_are_send_sync_and_dispatches_are_independent() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<MetalBackend>();
    send_sync::<MetalBuf>();
    std::thread::scope(|scope| {
        for seed in 0..4 {
            scope.spawn(move || {
                let gpu = gpu();
                let values = data(129, seed);
                let a = fbuf(gpu, &values);
                let mut b = gpu.alloc(values.len() * 4, Pool::Scratch).unwrap();
                for _ in 0..3 {
                    gpu.add(&a, &a, &mut b).unwrap();
                    close(
                        &read(gpu, &b, values.len()),
                        &values.iter().map(|v| 2.0 * v).collect::<Vec<_>>(),
                        0.0,
                        0.0,
                    );
                }
            });
        }
    });
}

#[test]
fn buffers_can_outlive_their_backend() {
    let buffer = {
        let backend = MetalBackend::new().unwrap();
        assert!(!backend.device_name().is_empty());
        fbuf(&backend, &[1.0, -2.0])
    };
    // No pending GPU work or borrowed context is needed when dropping it.
    drop(buffer);
}

// Every kernel against CpuBackend on asymmetric data (the Level Zero
// backend's suite, plus synthetic K-quant blocks and warp-boundary
// shapes for the warp-per-row kernels). A transposed index
// or a swapped nibble does not survive these shapes.
use super::CudaBackend;
use crate::backend::cpu::{CpuBackend, CpuBuf};
use crate::backend::test_support::fixture;
use crate::backend::test_support::{
    activation_row, bytes, close, cpu_qmatmul, f16_round, f32s, qmatmul as gpu_qmatmul,
    real_tensor_rows,
};
use crate::backend::test_support::{
    generation::{benchmark_prefix, greedy_continuation},
    parity,
};
use crate::backend::{AttnSpec, Backend, DeltaInputs, DeltaSpec, KvView, Pool, QWeight, RopeSpec};
use crate::gguf::{GgmlDType, GgufModel};

fn gpu() -> CudaBackend {
    CudaBackend::new().expect("CUDA GPU")
}

/// The KV cache stores f16, so the CPU reference attends over the same
/// rounded keys and values: any gap left is the kernel's.
fn cpu_attn(q: &[f32], k: &[f32], v: &[f32], kv_len: usize, spec: &AttnSpec) -> Vec<f32> {
    let cpu = CpuBackend;
    let q_cpu = crate::backend::cpu::CpuBuf(bytes(q));
    let k_cpu = crate::backend::cpu::CpuBuf(bytes(&f16_round(k)));
    let v_cpu = crate::backend::cpu::CpuBuf(bytes(&f16_round(v)));
    let mut o_cpu = cpu.alloc(q.len() * 4, Pool::Scratch).unwrap();
    let kv = KvView {
        k: &k_cpu,
        v: &v_cpu,
        len: kv_len,
    };
    cpu.attn(&q_cpu, &kv, spec, &mut o_cpu).unwrap();
    f32s(&o_cpu.0)
}

/// K/V go into the f16 cache through kv_write, as the model writes them.
fn gpu_attn(
    gpu: &CudaBackend,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kv_len: usize,
    spec: &AttnSpec,
) -> Vec<f32> {
    let mut q_g = gpu.alloc(q.len() * 4, Pool::Scratch).unwrap();
    let mut k_src = gpu.alloc(k.len() * 4, Pool::Scratch).unwrap();
    let mut v_src = gpu.alloc(v.len() * 4, Pool::Scratch).unwrap();
    gpu.upload(&mut q_g, 0, &bytes(q)).unwrap();
    gpu.upload(&mut k_src, 0, &bytes(k)).unwrap();
    gpu.upload(&mut v_src, 0, &bytes(v)).unwrap();
    let mut k_g = gpu.alloc(k.len() * 2, Pool::Kv).unwrap();
    let mut v_g = gpu.alloc(v.len() * 2, Pool::Kv).unwrap();
    gpu.kv_write(&k_src, &v_src, &mut k_g, &mut v_g, 0, k.len())
        .unwrap();
    let mut o_g = gpu.alloc(q.len() * 4, Pool::Scratch).unwrap();
    let kv = KvView {
        k: &k_g,
        v: &v_g,
        len: kv_len,
    };
    gpu.attn(&q_g, &kv, spec, &mut o_g).unwrap();
    let mut out = vec![0u8; q.len() * 4];
    gpu.download(&o_g, 0, &mut out).unwrap();
    f32s(&out)
}

fn run_qmatmul(dtype: GgmlDType, w: &[u8], shape: &[usize], x: &[f32], tol: f32) {
    let want = cpu_qmatmul(dtype, w, shape, x);
    // f32 activations: gemv_q, exact against the CPU. The int8 paths have
    // their own requantized-reference tests.
    close(
        &gpu_qmatmul(&gpu().without_int8(), dtype, w, shape, x),
        &want,
        tol,
    );
}

#[test]
fn qmatmul_matches_cpu_on_asymmetric_f32() {
    parity::asymmetric_matmul(&gpu().without_int8());
}

#[test]
fn qmatmul_matches_cpu_when_rows_are_not_a_multiple_of_the_group() {
    parity::matmul_row_tail(&gpu().without_int8());
}

fn upload_new(gpu: &CudaBackend, v: &[f32]) -> super::CudaBuf {
    let mut b = gpu.alloc(v.len() * 4, Pool::Scratch).unwrap();
    gpu.upload(&mut b, 0, &bytes(v)).unwrap();
    b
}

fn download_all(gpu: &CudaBackend, b: &super::CudaBuf) -> Vec<u8> {
    let mut out = vec![0u8; b.len];
    gpu.download(b, 0, &mut out).unwrap();
    out
}

#[test]
fn qwen35_partial_rope_matches_cpu() {
    let gpu = gpu();
    let cpu = CpuBackend;
    let spec = RopeSpec {
        head_dim: 32,
        theta: 10_000.0,
    };
    let values: Vec<f32> = (0..3 * spec.head_dim)
        .map(|i| ((i * 19 % 97) as f32 - 48.0) / 17.0)
        .collect();
    let mut want = CpuBuf(bytes(&values));
    cpu.rope_partial_split_half(&mut want, 7, &spec, 24)
        .unwrap();
    let mut got = upload_new(&gpu, &values);
    gpu.rope_partial_split_half(&mut got, 7, &spec, 24).unwrap();
    close(&f32s(&download_all(&gpu, &got)), &f32s(&want.0), 2e-5);
}

#[test]
fn qwen35_sigmoid_gate_matches_cpu() {
    let gpu = gpu();
    let cpu = CpuBackend;
    let gate: Vec<f32> = (0..257).map(|i| (i as f32 - 128.0) / 31.0).collect();
    let x: Vec<f32> = (0..257)
        .map(|i| ((i * 23 % 101) as f32 - 50.0) / 19.0)
        .collect();
    let (gate_cpu, x_cpu) = (CpuBuf(bytes(&gate)), CpuBuf(bytes(&x)));
    let mut want = CpuBuf(bytes(&vec![0.0f32; gate.len()]));
    cpu.sigmoid_mul(&gate_cpu, &x_cpu, &mut want).unwrap();
    let (gate_gpu, x_gpu) = (upload_new(&gpu, &gate), upload_new(&gpu, &x));
    let mut got = gpu.alloc(gate.len() * 4, Pool::Scratch).unwrap();
    gpu.sigmoid_mul(&gate_gpu, &x_gpu, &mut got).unwrap();
    close(&f32s(&download_all(&gpu, &got)), &f32s(&want.0), 2e-5);
}

#[test]
fn qwen35_ssm_conv_silu_matches_cpu_and_updates_state() {
    let gpu = gpu();
    let cpu = CpuBackend;
    let (channels, taps) = (257usize, 4usize);
    let weights: Vec<f32> = (0..channels * taps)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) / 29.0)
        .collect();
    let initial: Vec<f32> = (0..channels * (taps - 1))
        .map(|i| ((i * 7 % 31) as f32 - 15.0) / 41.0)
        .collect();
    let mut state_cpu = CpuBuf(bytes(&initial));
    let weights_cpu = CpuBuf(bytes(&weights));
    let mut state_gpu = upload_new(&gpu, &initial);
    let weights_gpu = upload_new(&gpu, &weights);
    for step in 0..2 {
        let x: Vec<f32> = (0..channels)
            .map(|i| ((i * 13 + step * 17) % 47) as f32 / 23.0 - 1.0)
            .collect();
        let x_cpu = CpuBuf(bytes(&x));
        let mut want = CpuBuf(bytes(&vec![0.0f32; channels]));
        let x_gpu = upload_new(&gpu, &x);
        let mut got = gpu.alloc(channels * 4, Pool::Scratch).unwrap();
        cpu.ssm_conv_silu(&weights_cpu, &mut state_cpu, &x_cpu, &mut want)
            .unwrap();
        gpu.ssm_conv_silu(&weights_gpu, &mut state_gpu, &x_gpu, &mut got)
            .unwrap();
        close(&f32s(&download_all(&gpu, &got)), &f32s(&want.0), 2e-5);
        close(
            &f32s(&download_all(&gpu, &state_gpu)),
            &f32s(&state_cpu.0),
            2e-5,
        );
    }
}

#[test]
fn qwen35_gated_delta_matches_cpu_and_updates_state() {
    let gpu = gpu();
    let cpu = CpuBackend;
    let spec = DeltaSpec {
        n_k_heads: 2,
        n_v_heads: 4,
        head_dim: 32,
        eps: 1e-5,
    };
    let qkv_len = (2 * spec.n_k_heads + spec.n_v_heads) * spec.head_dim;
    let qkv0: Vec<f32> = (0..qkv_len)
        .map(|i| ((i * 17 % 113) as f32 - 56.0) / 89.0)
        .collect();
    let qkv1: Vec<f32> = (0..qkv_len)
        .map(|i| ((i * 29 % 127) as f32 - 63.0) / 97.0)
        .collect();
    let alpha: Vec<f32> = (0..spec.n_v_heads)
        .map(|i| (i as f32 - 1.5) / 7.0)
        .collect();
    let beta: Vec<f32> = (0..spec.n_v_heads)
        .map(|i| (i as f32 - 1.0) / 5.0)
        .collect();
    let dt_bias: Vec<f32> = (0..spec.n_v_heads)
        .map(|i| (i as f32 - 2.0) / 9.0)
        .collect();
    let a: Vec<f32> = (0..spec.n_v_heads)
        .map(|i| -(i as f32 + 1.0) / 11.0)
        .collect();
    let state_len = spec.n_v_heads * spec.head_dim * spec.head_dim;
    let initial: Vec<f32> = (0..state_len)
        .map(|i| ((i * 5 % 43) as f32 - 21.0) / 500.0)
        .collect();
    let (mut state_cpu, state_gpu) = (CpuBuf(bytes(&initial)), upload_new(&gpu, &initial));
    let (alpha_cpu, beta_cpu, dt_cpu, a_cpu) = (
        CpuBuf(bytes(&alpha)),
        CpuBuf(bytes(&beta)),
        CpuBuf(bytes(&dt_bias)),
        CpuBuf(bytes(&a)),
    );
    let (alpha_gpu, beta_gpu, dt_gpu, a_gpu) = (
        upload_new(&gpu, &alpha),
        upload_new(&gpu, &beta),
        upload_new(&gpu, &dt_bias),
        upload_new(&gpu, &a),
    );
    let mut state_gpu = state_gpu;
    for qkv_values in [&qkv0, &qkv1] {
        let qkv_cpu = CpuBuf(bytes(qkv_values));
        let qkv_gpu = upload_new(&gpu, qkv_values);
        let inputs_cpu = DeltaInputs {
            qkv: &qkv_cpu,
            alpha: &alpha_cpu,
            beta: &beta_cpu,
            dt_bias: &dt_cpu,
            a: &a_cpu,
        };
        let inputs_gpu = DeltaInputs {
            qkv: &qkv_gpu,
            alpha: &alpha_gpu,
            beta: &beta_gpu,
            dt_bias: &dt_gpu,
            a: &a_gpu,
        };
        let mut want = CpuBuf(bytes(&vec![0.0f32; spec.n_v_heads * spec.head_dim]));
        let mut got = gpu
            .alloc(spec.n_v_heads * spec.head_dim * 4, Pool::Scratch)
            .unwrap();
        cpu.gated_delta_step(&spec, &inputs_cpu, &mut state_cpu, &mut want)
            .unwrap();
        gpu.gated_delta_step(&spec, &inputs_gpu, &mut state_gpu, &mut got)
            .unwrap();
        close(&f32s(&download_all(&gpu, &got)), &f32s(&want.0), 2e-3);
        close(
            &f32s(&download_all(&gpu, &state_gpu)),
            &f32s(&state_cpu.0),
            2e-3,
        );
    }
}

/// The fused kernel is add_f32 then rmsnorm_f32 in one launch, with the
/// same sums in the same order: bit-identical, not merely close. 300
/// elements per token run past one 256-thread pass.
#[test]
fn add_rmsnorm_is_bit_identical_to_add_then_rmsnorm() {
    let gpu = gpu();
    let (dim, n_tokens) = (300usize, 3usize);
    let n = dim * n_tokens;
    let a: Vec<f32> = (0..n)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 16.0)
        .collect();
    let b: Vec<f32> = (0..n)
        .map(|i| ((i * 13 % 67) as f32 - 30.0) / 8.0)
        .collect();
    let w: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 11) as f32 * 0.07).collect();
    let (a_g, b_g, w_g) = (
        upload_new(&gpu, &a),
        upload_new(&gpu, &b),
        upload_new(&gpu, &w),
    );

    let mut sum_want = gpu.alloc(n * 4, Pool::Scratch).unwrap();
    let mut y_want = gpu.alloc(n * 4, Pool::Scratch).unwrap();
    gpu.add(&a_g, &b_g, &mut sum_want).unwrap();
    gpu.rmsnorm(&w_g, &sum_want, &mut y_want, 1e-5).unwrap();

    let mut sum = gpu.alloc(n * 4, Pool::Scratch).unwrap();
    let mut y = gpu.alloc(n * 4, Pool::Scratch).unwrap();
    gpu.add_rmsnorm(&w_g, &a_g, &b_g, &mut sum, &mut y, 1e-5)
        .unwrap();
    assert_eq!(download_all(&gpu, &sum), download_all(&gpu, &sum_want));
    assert_eq!(download_all(&gpu, &y), download_all(&gpu, &y_want));

    let mut short = gpu.alloc(n * 4 - 4, Pool::Scratch).unwrap();
    assert!(
        gpu.add_rmsnorm(&w_g, &a_g, &b_g, &mut short, &mut y, 1e-5)
            .is_err()
    );
}

/// rope_kv_f16 against rope_f32 + kv_write_f16: q and both caches match
/// bit for bit, rows land at the offset, and the rows before it stay
/// untouched. GQA widths (4 q heads, 2 kv heads) and 3 tokens.
#[test]
fn rope_kv_write_is_bit_identical_to_rope_then_kv_write() {
    let gpu = gpu();
    let (head_dim, n_q, n_kv, n_tokens) = (72usize, 4usize, 2usize, 3usize);
    let spec = RopeSpec {
        head_dim,
        theta: 10000.0,
    };
    let row = n_kv * head_dim;
    let (offset, n_elems) = (2 * row, n_tokens * row);
    let cache_bytes = (offset + n_elems + row) * 2;
    let q: Vec<f32> = (0..n_tokens * n_q * head_dim)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) / 5.0)
        .collect();
    let k: Vec<f32> = (0..n_elems)
        .map(|i| ((i * 5 % 19) as f32 - 9.0) / 4.0)
        .collect();
    let v: Vec<f32> = (0..n_elems)
        .map(|i| ((i * 3 % 17) as f32 - 8.0) / 3.0)
        .collect();

    let mut q_want = upload_new(&gpu, &q);
    let mut k_want = upload_new(&gpu, &k);
    let v_g = upload_new(&gpu, &v);
    let mut dk_want = gpu.alloc(cache_bytes, Pool::Scratch).unwrap();
    let mut dv_want = gpu.alloc(cache_bytes, Pool::Scratch).unwrap();
    gpu.rope(&mut q_want, &mut k_want, 5, n_tokens, &spec)
        .unwrap();
    gpu.kv_write(&k_want, &v_g, &mut dk_want, &mut dv_want, offset, n_elems)
        .unwrap();

    let mut q_g = upload_new(&gpu, &q);
    let mut k_g = upload_new(&gpu, &k);
    let mut dk = gpu.alloc(cache_bytes, Pool::Scratch).unwrap();
    let mut dv = gpu.alloc(cache_bytes, Pool::Scratch).unwrap();
    gpu.rope_kv_write(
        &mut q_g, &mut k_g, &v_g, &mut dk, &mut dv, offset, 5, n_tokens, &spec, n_elems,
    )
    .unwrap();
    assert_eq!(download_all(&gpu, &q_g), download_all(&gpu, &q_want));
    let (dk_got, dk_ref) = (download_all(&gpu, &dk), download_all(&gpu, &dk_want));
    assert_eq!(dk_got, dk_ref);
    assert_eq!(download_all(&gpu, &dv), download_all(&gpu, &dv_want));
    assert!(
        dk_got[..offset * 2].iter().all(|&b| b == 0),
        "rows before the offset"
    );
    assert!(
        dk_got[offset * 2..(offset + n_elems) * 2]
            .iter()
            .any(|&b| b != 0)
    );

    assert!(
        gpu.rope_kv_write(
            &mut q_g,
            &mut k_g,
            &v_g,
            &mut dk,
            &mut dv,
            offset + 2 * row,
            5,
            n_tokens,
            &spec,
            n_elems
        )
        .is_err(),
        "rows past the cache end"
    );
}

fn upload_weight(
    gpu: &CudaBackend,
    dtype: GgmlDType,
    n_out: usize,
    n_in: usize,
) -> QWeight<super::CudaBuf> {
    let raw = synthetic_kquant(dtype, n_out, n_in);
    let mut buf = gpu.alloc(raw.len(), Pool::Weights).unwrap();
    gpu.upload(&mut buf, 0, &raw).unwrap();
    QWeight {
        dtype,
        shape: vec![n_out, n_in],
        buf,
    }
}

/// gemv_q3 and gemv_q_swiglu reuse gemv_q's warp body, so the fused
/// decode launches equal the separate ones bit for bit. Mixed dtypes (a
/// Q6_K v, a Q6_K up, as Q4_K_M files have), row counts that are not a
/// multiple of the 16-row block. Only single-token decode fuses: more
/// tokens take the GEMM.
#[test]
fn fused_decode_matmuls_are_bit_identical_to_separate_ones() {
    let gpu = gpu();
    let n_in = 512;
    let wq = upload_weight(&gpu, GgmlDType::Q4K, 100, n_in);
    let wk = upload_weight(&gpu, GgmlDType::Q4K, 36, n_in);
    let wv = upload_weight(&gpu, GgmlDType::Q6K, 36, n_in);
    let wg = upload_weight(&gpu, GgmlDType::Q4K, 70, n_in);
    let wu = upload_weight(&gpu, GgmlDType::Q6K, 70, n_in);
    let x3 = upload_new(&gpu, &vec![0.5; 3 * n_in]);
    assert!(gpu.gemv_q_fusable(&[&wq, &wk, &wv], &x3).is_none());
    {
        let n_tokens = 1;
        let x: Vec<f32> = (0..n_tokens * n_in)
            .map(|i| ((i * 29 % 83) as f32 - 41.0) / 32.0)
            .collect();
        let x_g = upload_new(&gpu, &x);
        assert!(gpu.gemv_q_fusable(&[&wq, &wk, &wv], &x_g).is_some());
        let out = |n_out: usize| gpu.alloc(n_tokens * n_out * 4, Pool::Scratch).unwrap();

        let (mut q, mut k, mut v) = (out(100), out(36), out(36));
        gpu.qmatmul_qkv(&wq, &wk, &wv, &x_g, &mut q, &mut k, &mut v)
            .unwrap();
        for (w, got) in [(&wq, &q), (&wk, &k), (&wv, &v)] {
            let mut want = out(w.shape[0]);
            gpu.qmatmul(w, &x_g, &mut want).unwrap();
            assert_eq!(download_all(&gpu, got), download_all(&gpu, &want));
        }

        let (mut gs, mut us, mut y) = (out(70), out(70), out(70));
        gpu.ffn_gate_up(&wg, &wu, &x_g, &mut gs, &mut us, &mut y)
            .unwrap();
        let (mut g_want, mut u_want, mut y_want) = (out(70), out(70), out(70));
        gpu.qmatmul(&wg, &x_g, &mut g_want).unwrap();
        gpu.qmatmul(&wu, &x_g, &mut u_want).unwrap();
        gpu.silu_mul(&g_want, &u_want, &mut y_want).unwrap();
        assert_eq!(download_all(&gpu, &y), download_all(&gpu, &y_want));

        let mut short = out(35);
        assert!(
            gpu.qmatmul_qkv(&wq, &wk, &wv, &x_g, &mut q, &mut short, &mut v)
                .is_err()
        );
        assert!(
            gpu.ffn_gate_up(&wg, &wu, &x_g, &mut gs, &mut us, &mut short)
                .is_err()
        );
    }
}

#[test]
fn add_rmsnorm_silu_and_rope_match_cpu() {
    parity::elementwise_norm_rope(&gpu());
}

#[test]
fn binary_dispatch_boundaries_match_cpu() {
    parity::binary_dispatch_tails(&gpu());
}

#[test]
fn attn_matches_cpu_with_a_cached_prefix() {
    let spec = AttnSpec {
        head_dim: 8,
        n_q_heads: 4,
        n_kv_heads: 2,
    };
    let n_q = 3;
    let kv_len = 5;
    let q: Vec<f32> = (0..n_q * spec.n_q_heads * spec.head_dim)
        .map(|i| ((i % 11) as f32) * 0.13 - 0.5)
        .collect();
    let k: Vec<f32> = (0..kv_len * spec.n_kv_heads * spec.head_dim)
        .map(|i| ((i % 7) as f32) * 0.17 - 0.3)
        .collect();
    let v: Vec<f32> = (0..k.len()).map(|i| ((i % 5) as f32) * 0.2 - 0.2).collect();
    let want = cpu_attn(&q, &k, &v, kv_len, &spec);
    close(&gpu_attn(&gpu(), &q, &k, &v, kv_len, &spec), &want, 1e-4);
    // Chunks of 2 keys: query 0 sees keys 0..=2, so chunk 2 (key 4) is
    // causally empty for it and must merge as nothing.
    let split = gpu().with_attn_chunk(2);
    close(&gpu_attn(&split, &q, &k, &v, kv_len, &spec), &want, 1e-4);
}

/// Every group-size path: G = 1 without GQA, G = 2/4/8 blocks that serve
/// a whole group, groups of 3 and 6 that only G = 1 and G = 2 divide (a
/// block then maps to kv head x G / group), unsplit and in 3-key chunks.
#[test]
fn attn_matches_cpu_for_every_gqa_group() {
    let n_q = 4;
    let kv_len = 9;
    for (n_q_heads, n_kv_heads) in [(4, 4), (4, 2), (8, 2), (16, 2), (6, 2), (12, 2), (3, 1)] {
        let spec = AttnSpec {
            head_dim: 72,
            n_q_heads,
            n_kv_heads,
        };
        let q: Vec<f32> = (0..n_q * n_q_heads * spec.head_dim)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.04)
            .collect();
        let k: Vec<f32> = (0..kv_len * n_kv_heads * spec.head_dim)
            .map(|i| ((i * 5 % 19) as f32 - 9.0) * 0.05)
            .collect();
        let v: Vec<f32> = (0..k.len())
            .map(|i| ((i * 3 % 13) as f32 - 6.0) * 0.1)
            .collect();
        let want = cpu_attn(&q, &k, &v, kv_len, &spec);
        close(&gpu_attn(&gpu(), &q, &k, &v, kv_len, &spec), &want, 1e-4);
        let split = gpu().with_attn_chunk(3);
        close(&gpu_attn(&split, &q, &k, &v, kv_len, &spec), &want, 1e-4);
    }
}

#[test]
fn embed_matches_cpu_on_f32_table() {
    parity::embed_f32(&gpu());
}

/// Deterministic xorshift bytes; the f16 scale fields are overwritten
/// with finite values so every block dequantizes to real numbers.
fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u8
        })
        .collect()
}

fn set_f16(buf: &mut [u8], at: usize, v: f32) {
    buf[at..at + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
}

fn synthetic_kquant(dtype: GgmlDType, n_out: usize, n_in: usize) -> Vec<u8> {
    let (bb, scale_offs): (usize, &[usize]) = match dtype {
        GgmlDType::Q4K => (144, &[0, 2]),
        GgmlDType::Q6K => (210, &[208]),
        _ => unreachable!(),
    };
    let n_blocks = n_out * n_in / 256;
    let mut w = noise(n_blocks * bb, 0x9e37_79b9_7f4a_7c15 ^ bb as u64);
    for b in 0..n_blocks {
        for (k, off) in scale_offs.iter().enumerate() {
            // Real 7B K-quant super-block scales sit around 1e-3; larger
            // ones push dot products past what a 1e-3 f32 tolerance means.
            set_f16(&mut w, b * bb + off, 0.0005 + 0.0002 * ((b + k) % 7) as f32);
        }
    }
    w
}

fn synthetic_q8_0(n_out: usize, n_in: usize) -> Vec<u8> {
    assert!(n_in.is_multiple_of(32));
    let block_bytes = 34;
    let blocks = n_out * n_in / 32;
    let mut w = vec![0u8; blocks * block_bytes];
    for (block, bytes) in w.chunks_exact_mut(block_bytes).enumerate() {
        set_f16(bytes, 0, 0.125 + 0.01 * (block % 5) as f32);
        for (i, q) in bytes[2..].iter_mut().enumerate() {
            *q = (i as i8).wrapping_mul(3).wrapping_sub(45).cast_unsigned();
        }
    }
    w
}

// 37 rows: not a multiple of the 4-warp block. n_in 768: three blocks per
// row, so a lane-to-element mapping error cannot hide in a single block.
fn kquant_matches_cpu(dtype: GgmlDType) {
    let (n_out, n_in) = (37, 768);
    let w = synthetic_kquant(dtype, n_out, n_in);
    let x: Vec<f32> = (0..n_in * 3)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
        .collect();
    run_qmatmul(dtype, &w, &[n_out, n_in], &x, 1e-3);
}

#[test]
fn qmatmul_matches_cpu_on_synthetic_q4k_blocks() {
    kquant_matches_cpu(GgmlDType::Q4K);
}

#[test]
fn qmatmul_matches_cpu_on_synthetic_q6k_blocks() {
    kquant_matches_cpu(GgmlDType::Q6K);
}

#[test]
fn qmatmul_matches_cpu_on_synthetic_q8_0_blocks() {
    let (n_out, n_in) = (37, 160);
    let w = synthetic_q8_0(n_out, n_in);
    let x: Vec<f32> = (0..n_in)
        .map(|i| ((i * 29 % 101) as f32 - 50.0) / 64.0)
        .collect();
    run_qmatmul(GgmlDType::Q8_0, &w, &[n_out, n_in], &x, 1e-5);
}

/// x as the GEMMs see it: Q8_1 blocks of 32, d rounded to f16, q =
/// round(x * 127 / amax) -- quantize_q8_1's arithmetic on the host.
fn requantize_q8_1(x: &[f32]) -> Vec<f32> {
    x.chunks(32)
        .flat_map(|blk| {
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = half::f16::from_f32(amax / 127.0).to_f32();
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            blk.iter().map(move |v| d * (v * inv).round())
        })
        .collect()
}

/// Compare identical Q8_1 inputs for kernel error, and unquantized inputs
/// separately for activation-rounding error.
fn int8_matches_requantized_cpu(dtype: GgmlDType, w: &[u8], shape: &[usize], n_tokens: usize) {
    let gpu = gpu().with_int8_decode();
    let (path, available) = if n_tokens >= super::GEMM_MIN_TOKENS {
        ("int8 mma.sync", gpu.int8_gemm())
    } else {
        ("dp4a", gpu.int8_gemv())
    };
    if !available {
        eprintln!("skipped: {} has no {path}", gpu.device_name());
        return;
    }
    let n_in = shape[1];
    // Per-token offsets and magnitudes, so a token/column mixup shows.
    let x: Vec<f32> = (0..n_tokens * n_in)
        .map(|e| {
            let (t, i) = (e / n_in, e % n_in);
            (((i * 37 + t * 11) % 101) as f32 - 50.0) / 64.0 * (1.0 + 0.1 * t as f32)
        })
        .collect();
    let got = gpu_qmatmul(&gpu, dtype, w, shape, &x);
    close(
        &got,
        &cpu_qmatmul(dtype, w, shape, &requantize_q8_1(&x)),
        1e-3,
    );

    let want = cpu_qmatmul(dtype, w, shape, &x);
    let err: f32 = got.iter().zip(&want).map(|(g, w)| (g - w).powi(2)).sum();
    let norm: f32 = want.iter().map(|w| w * w).sum();
    let rel = (err / norm).sqrt();
    assert!(rel < 0.02, "int8 activations: relative RMS error {rel}");
}

// 100 rows: two 64-row GEMM tiles, the second partial, and not a multiple
// of the matvec's 16. 1 and 3 tokens: dp4a decode. 8, 33 and 64: the
// GEMM threshold, a partial 32-token tile, two full tiles. 768 inputs are
// three superblocks, so gemm_q4k refills both of its stage buffers.
fn int8_matches_on_synthetic(dtype: GgmlDType) {
    let (n_out, n_in) = (100, 768);
    let w = synthetic_kquant(dtype, n_out, n_in);
    for n_tokens in [1, 3, 8, 33, 64] {
        int8_matches_requantized_cpu(dtype, &w, &[n_out, n_in], n_tokens);
    }
}

#[test]
fn int8_matches_requantized_cpu_on_synthetic_q4k() {
    int8_matches_on_synthetic(GgmlDType::Q4K);
}

#[test]
fn int8_matches_requantized_cpu_on_synthetic_q6k() {
    int8_matches_on_synthetic(GgmlDType::Q6K);
}

/// The Q8_1 cache: two weights on one x reuse its quantization (q/k/v in
/// decode); a write to x, by upload or by a kernel, must requantize.
#[test]
fn int8_cache_follows_writes_to_x() {
    let gpu = gpu().with_int8_decode();
    if !gpu.int8_gemv() {
        eprintln!("skipped: {} has no dp4a", gpu.device_name());
        return;
    }
    let (n_out, n_in) = (48, 512);
    let shape = [n_out, n_in];
    let wa = synthetic_kquant(GgmlDType::Q4K, n_out, n_in);
    let wb = synthetic_kquant(GgmlDType::Q6K, n_out, n_in);
    let upload_w = |dtype, w: &[u8]| {
        let mut buf = gpu.alloc(w.len(), Pool::Weights).unwrap();
        gpu.upload(&mut buf, 0, w).unwrap();
        QWeight {
            dtype,
            shape: shape.to_vec(),
            buf,
        }
    };
    let (qa, qb) = (upload_w(GgmlDType::Q4K, &wa), upload_w(GgmlDType::Q6K, &wb));
    let x1: Vec<f32> = (0..n_in)
        .map(|i| ((i * 29 % 97) as f32 - 48.0) / 50.0)
        .collect();
    let x2: Vec<f32> = (0..n_in)
        .map(|i| ((i * 53 % 89) as f32 - 44.0) / 30.0)
        .collect();
    let mut x = gpu.alloc(n_in * 4, Pool::Scratch).unwrap();
    let mut y = gpu.alloc(n_out * 4, Pool::Scratch).unwrap();
    let run = |q: &QWeight<_>, x: &super::CudaBuf, y: &mut super::CudaBuf| {
        gpu.qmatmul(q, x, y).unwrap();
        let mut out = vec![0u8; n_out * 4];
        gpu.download(y, 0, &mut out).unwrap();
        f32s(&out)
    };
    let want = |dtype, w: &[u8], xs: &[f32]| cpu_qmatmul(dtype, w, &shape, &requantize_q8_1(xs));

    gpu.upload(&mut x, 0, &bytes(&x1)).unwrap();
    close(&run(&qa, &x, &mut y), &want(GgmlDType::Q4K, &wa, &x1), 1e-3);
    close(&run(&qb, &x, &mut y), &want(GgmlDType::Q6K, &wb, &x1), 1e-3);
    gpu.upload(&mut x, 0, &bytes(&x2)).unwrap();
    close(&run(&qa, &x, &mut y), &want(GgmlDType::Q4K, &wa, &x2), 1e-3);
    // A kernel writing x: x = x2 + x2.
    let mut src = gpu.alloc(n_in * 4, Pool::Scratch).unwrap();
    gpu.upload(&mut src, 0, &bytes(&x2)).unwrap();
    gpu.add(&src, &src, &mut x).unwrap();
    let doubled: Vec<f32> = x2.iter().map(|v| v + v).collect();
    close(
        &run(&qb, &x, &mut y),
        &want(GgmlDType::Q6K, &wb, &doubled),
        1e-3,
    );
}

#[test]
fn qmatmul_matches_cpu_on_f16_weights() {
    let (n_out, n_in) = (9, 100);
    let w: Vec<u8> = (0..n_out * n_in)
        .flat_map(|i| half::f16::from_f32(((i % 23) as f32 - 11.0) * 0.07).to_le_bytes())
        .collect();
    let x: Vec<f32> = (0..n_in * 2)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.3)
        .collect();
    run_qmatmul(GgmlDType::F16, &w, &[n_out, n_in], &x, 1e-3);
}

fn embed_matches_cpu(dtype: GgmlDType, table: &[u8], vocab: usize, n: usize) {
    let ids = [vocab as u32 - 1, 0, 2, 2];
    let cpu = CpuBackend;
    let w = QWeight {
        dtype,
        shape: vec![vocab, n],
        buf: crate::backend::cpu::CpuBuf(table.to_vec()),
    };
    let mut out_cpu = cpu.alloc(ids.len() * n * 4, Pool::Scratch).unwrap();
    cpu.embed(&w, &ids, &mut out_cpu).unwrap();
    let gpu = gpu();
    let mut buf = gpu.alloc(table.len(), Pool::Weights).unwrap();
    gpu.upload(&mut buf, 0, table).unwrap();
    let w_g = QWeight {
        dtype,
        shape: vec![vocab, n],
        buf,
    };
    let mut out = gpu.alloc(ids.len() * n * 4, Pool::Scratch).unwrap();
    gpu.embed(&w_g, &ids, &mut out).unwrap();
    let mut got = vec![0u8; ids.len() * n * 4];
    gpu.download(&out, 0, &mut got).unwrap();
    close(&f32s(&got), &f32s(&out_cpu.0), 1e-5);
}

#[test]
fn embed_matches_cpu_on_synthetic_kquant_tables() {
    for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
        embed_matches_cpu(dtype, &synthetic_kquant(dtype, 5, 512), 5, 512);
    }
    embed_matches_cpu(GgmlDType::Q8_0, &synthetic_q8_0(5, 512), 5, 512);
}

#[test]
fn embed_rejects_out_of_range_tokens() {
    let gpu = gpu();
    let buf = gpu.alloc(4 * 4 * 4, Pool::Weights).unwrap();
    let w = QWeight {
        dtype: GgmlDType::F32,
        shape: vec![4, 4],
        buf,
    };
    let mut out = gpu.alloc(4 * 4, Pool::Scratch).unwrap();
    assert!(gpu.embed(&w, &[4], &mut out).is_err());
}

/// Llama-shaped: head_dim 128 (4 elements per lane), GQA 8:2, a prefix
/// long enough that the online softmax rescales many times, and RoPE at
/// a position where a naive sin.approx would drift.
#[test]
fn rope_and_attn_match_cpu_at_llama_shapes() {
    let cpu = CpuBackend;
    let gpu = gpu();
    let rope = RopeSpec {
        head_dim: 128,
        theta: 1_000_000.0,
    };
    let q: Vec<f32> = (0..3 * 8 * 128)
        .map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..3 * 2 * 128)
        .map(|i| ((i * 5 % 31) as f32 - 15.0) * 0.04)
        .collect();
    let mut q_cpu = crate::backend::cpu::CpuBuf(bytes(&q));
    let mut k_cpu = crate::backend::cpu::CpuBuf(bytes(&k));
    cpu.rope(&mut q_cpu, &mut k_cpu, 30_000, 3, &rope).unwrap();
    let mut q_g = gpu.alloc(q.len() * 4, Pool::Scratch).unwrap();
    let mut k_g = gpu.alloc(k.len() * 4, Pool::Scratch).unwrap();
    gpu.upload(&mut q_g, 0, &bytes(&q)).unwrap();
    gpu.upload(&mut k_g, 0, &bytes(&k)).unwrap();
    gpu.rope(&mut q_g, &mut k_g, 30_000, 3, &rope).unwrap();
    let mut q_out = vec![0u8; q.len() * 4];
    gpu.download(&q_g, 0, &mut q_out).unwrap();
    close(&f32s(&q_out), &f32s(&q_cpu.0), 1e-5);

    let spec = AttnSpec {
        head_dim: 128,
        n_q_heads: 8,
        n_kv_heads: 2,
    };
    let kv_len = 300;
    let kc: Vec<f32> = (0..kv_len * 2 * 128)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) * 0.03)
        .collect();
    let vc: Vec<f32> = (0..kc.len())
        .map(|i| ((i * 3 % 17) as f32 - 8.0) * 0.1)
        .collect();
    let want = cpu_attn(&q, &kc, &vc, kv_len, &spec);
    close(&gpu_attn(&gpu, &q, &kc, &vc, kv_len, &spec), &want, 1e-4);
    // 37-key chunks: 9 chunks, the last one partial.
    let split = super::CudaBackend::new().unwrap().with_attn_chunk(37);
    close(&gpu_attn(&split, &q, &kc, &vc, kv_len, &spec), &want, 1e-4);
}

/// Decode at depth: one query over 2048 keys with 8 heads is 8 blocks,
/// so the heuristic splits keys across the SMs and attn_merge combines
/// them.
#[test]
fn attn_decode_splits_keys_at_depth() {
    let gpu = gpu();
    let spec = AttnSpec {
        head_dim: 128,
        n_q_heads: 8,
        n_kv_heads: 2,
    };
    let kv_len = 2048;
    // One block per kv head here (G = 4).
    assert!(gpu.attn_chunk(spec.n_kv_heads, 1, kv_len) < kv_len);
    let q: Vec<f32> = (0..8 * 128)
        .map(|i| ((i * 7 % 29) as f32 - 14.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..kv_len * 2 * 128)
        .map(|i| ((i * 11 % 37) as f32 - 18.0) * 0.03)
        .collect();
    let v: Vec<f32> = (0..k.len())
        .map(|i| ((i * 3 % 17) as f32 - 8.0) * 0.1)
        .collect();
    let want = cpu_attn(&q, &k, &v, kv_len, &spec);
    close(&gpu_attn(&gpu, &q, &k, &v, kv_len, &spec), &want, 1e-4);
}

#[test]
fn rmsnorm_matches_cpu_across_block_boundaries() {
    let cpu = CpuBackend;
    let gpu = gpu();
    let dim = 4099;
    let w: Vec<f32> = (0..dim).map(|i| 0.5 + (i % 9) as f32 * 0.07).collect();
    let x: Vec<f32> = (0..dim * 2)
        .map(|i| ((i * 13 % 41) as f32 - 20.0) * 0.11)
        .collect();
    let mut y_cpu = cpu.alloc(x.len() * 4, Pool::Scratch).unwrap();
    cpu.rmsnorm(
        &crate::backend::cpu::CpuBuf(bytes(&w)),
        &crate::backend::cpu::CpuBuf(bytes(&x)),
        &mut y_cpu,
        1e-6,
    )
    .unwrap();
    let mut w_g = gpu.alloc(w.len() * 4, Pool::Weights).unwrap();
    let mut x_g = gpu.alloc(x.len() * 4, Pool::Scratch).unwrap();
    gpu.upload(&mut w_g, 0, &bytes(&w)).unwrap();
    gpu.upload(&mut x_g, 0, &bytes(&x)).unwrap();
    let mut y_g = gpu.alloc(x.len() * 4, Pool::Scratch).unwrap();
    gpu.rmsnorm(&w_g, &x_g, &mut y_g, 1e-6).unwrap();
    let mut got = vec![0u8; x.len() * 4];
    gpu.download(&y_g, 0, &mut got).unwrap();
    close(&f32s(&got), &f32s(&y_cpu.0), 1e-5);
}

#[test]
fn copy_moves_bytes_device_to_device() {
    let gpu = gpu();
    let src: Vec<u8> = (0..64).collect();
    let mut a = gpu.alloc(64, Pool::Scratch).unwrap();
    gpu.upload(&mut a, 0, &src).unwrap();
    let mut b = gpu.alloc(64, Pool::Scratch).unwrap();
    gpu.copy(&a, 8, &mut b, 16, 32).unwrap();
    let mut got = vec![0u8; 64];
    gpu.download(&b, 0, &mut got).unwrap();
    assert_eq!(&got[16..48], &src[8..40]);
    assert!(got[..16].iter().chain(&got[48..]).all(|&v| v == 0));
    assert!(gpu.copy(&a, 40, &mut b, 0, 32).is_err());
}

/// One 7B model at a time: two in parallel tests overflow 8 GB of VRAM.
pub(super) static GPU_MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The gemv_q path without int8, on 32 real rows.
fn qmatmul_matches_cpu_on_real_tensor(dtype: GgmlDType) {
    let (raw, shape) = real_tensor_rows(dtype, 32);
    run_qmatmul(dtype, &raw, &shape, &activation_row(shape[1]), 1e-3);
}

#[test]
#[ignore = "requires MDMA_TEST_GGUF and CUDA hardware"]
fn qmatmul_matches_cpu_on_real_q4k_rows() {
    qmatmul_matches_cpu_on_real_tensor(GgmlDType::Q4K);
}

#[test]
#[ignore = "requires MDMA_TEST_GGUF and CUDA hardware"]
fn qmatmul_matches_cpu_on_real_q6k_rows() {
    qmatmul_matches_cpu_on_real_tensor(GgmlDType::Q6K);
}

fn int8_matches_on_real_tensor(dtype: GgmlDType) {
    let (raw, shape) = real_tensor_rows(dtype, 200);
    for n_tokens in [1, 40] {
        int8_matches_requantized_cpu(dtype, &raw, &shape, n_tokens);
    }
}

#[test]
#[ignore = "requires MDMA_TEST_GGUF and CUDA hardware"]
fn int8_matches_requantized_cpu_on_real_q4k_rows() {
    int8_matches_on_real_tensor(GgmlDType::Q4K);
}

#[test]
#[ignore = "requires MDMA_TEST_GGUF and CUDA hardware"]
fn int8_matches_requantized_cpu_on_real_q6k_rows() {
    int8_matches_on_real_tensor(GgmlDType::Q6K);
}

#[test]
#[ignore = "requires MDMA_TEST_GGUF and CUDA hardware"]
fn greedy_continuation_matches_llama_cpp() {
    use crate::model::Model;
    const PROMPT: &[u32] = &[1, 918, 6872, 446, 9760, 505];
    const EXPECTED: &[u32] = &[12247, 281, 12247, 505, 395];
    let backend = gpu();
    let _guard = GPU_MODEL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut gguf = GgufModel::open(fixture()).expect("open gguf");
    let model = Model::load(&backend, &mut gguf).expect("load model");
    let mut kv = model.new_kv_cache(&backend, 64).expect("kv");
    let mut ws = model.alloc_workspace(&backend, 1).expect("workspace");
    let mut pos = 0u32;
    let mut logits = Vec::new();
    for &tok in PROMPT {
        logits = model
            .forward(&backend, &mut kv, &mut ws, tok, pos)
            .expect("forward");
        pos += 1;
    }
    let generated = greedy_continuation(
        &backend,
        &model,
        &mut kv,
        &mut ws,
        logits,
        pos,
        EXPECTED.len(),
    );
    assert_eq!(generated, EXPECTED);
}

#[test]
#[ignore = "requires MDMA_TEST_GGUF and CUDA hardware"]
fn gemm_prefill_continues_like_per_token_decode() {
    use crate::model::Model;
    use crate::sampler::greedy;
    let backend = gpu();
    assert!(
        backend.int8_gemm(),
        "{} has no int8 mma.sync",
        backend.device_name()
    );
    let _guard = GPU_MODEL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut gguf = GgufModel::open(fixture()).expect("open gguf");
    let model = Model::load(&backend, &mut gguf).expect("load model");
    let prompt: Vec<u32> = (0..40).map(|i| 1 + (i * 2749 + 918) % 30000).collect();
    let run = |batched: bool| {
        let mut kv = model.new_kv_cache(&backend, 128).expect("kv");
        let mut ws = model.alloc_workspace(&backend, 1).expect("workspace");
        let mut logits = if batched {
            let mut pws = model
                .alloc_workspace(&backend, prompt.len())
                .expect("prefill ws");
            model
                .prefill(&backend, &mut kv, &mut pws, &prompt, 0)
                .expect("prefill")
        } else {
            let mut l = Vec::new();
            for (pos, &tok) in prompt.iter().enumerate() {
                l = model
                    .forward(&backend, &mut kv, &mut ws, tok, pos as u32)
                    .expect("forward");
            }
            l
        };
        let mut generated = Vec::new();
        for pos in (prompt.len() as u32..).take(16) {
            let next = greedy(&logits);
            generated.push(next);
            logits = model
                .forward(&backend, &mut kv, &mut ws, next, pos)
                .expect("forward");
        }
        generated
    };
    assert_eq!(run(true), run(false));
}

fn bench_env(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or_else(|| panic!("{name} must be a positive integer")),
        Err(_) => default,
    }
}

/// Load, prefill and decode wall time on the 7B fixture, the same
/// shapes as the ROCm benches. MDMA_BENCH_CTX (default 4096) sizes the
/// KV cache; MDMA_BENCH_PREFIX lists decode context depths and
/// MDMA_BENCH_STEPS (default 32) the tokens decoded at each.
///
/// Run explicitly: `cargo test --release --lib --features cuda
/// -- --ignored --nocapture bench_wall_time`.
#[test]
#[ignore = "requires MDMA_TEST_GGUF and an idle CUDA GPU"]
fn bench_wall_time() {
    use crate::model::Model;
    use crate::sampler::greedy;
    use std::time::Instant;
    let backend = gpu();
    let start = Instant::now();
    let _guard = GPU_MODEL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut gguf = GgufModel::open(fixture()).expect("open gguf");
    let model = Model::load(&backend, &mut gguf).expect("load model");
    eprintln!(
        "BENCH kind=load device={:?} elapsed_ms={}",
        backend.device_name(),
        start.elapsed().as_millis()
    );
    let n_ctx = bench_env("MDMA_BENCH_CTX", 4096).min(model.config().n_ctx);

    for n_tokens in [4usize, 16, 64, 256] {
        let tokens: Vec<u32> = (0..n_tokens as u32).map(|i| i % 128 + 1).collect();
        let mut kv = model.new_kv_cache(&backend, n_ctx).expect("kv");
        let mut ws = model
            .alloc_workspace(&backend, n_tokens)
            .expect("workspace");
        model
            .prefill(&backend, &mut kv, &mut ws, &tokens, 0)
            .expect("prefill (cold)");
        kv.set_len(0);
        let start = Instant::now();
        model
            .prefill(&backend, &mut kv, &mut ws, &tokens, 0)
            .expect("prefill");
        let warm = start.elapsed();
        eprintln!(
            "BENCH kind=prefill tokens={n_tokens} elapsed_ms={:.1} tok_s={:.1}",
            warm.as_secs_f64() * 1e3,
            n_tokens as f64 / warm.as_secs_f64()
        );
    }

    let prefixes: Vec<u32> = match std::env::var("MDMA_BENCH_PREFIX") {
        Ok(v) => v
            .split(',')
            .map(|s| s.parse().expect("MDMA_BENCH_PREFIX: comma-separated u32"))
            .collect(),
        Err(_) => vec![0, 256, 2048],
    };
    let n_steps = bench_env("MDMA_BENCH_STEPS", 32) as u32;
    for prefix in prefixes {
        assert!(
            (prefix + n_steps + 1) as usize <= n_ctx,
            "decode must fit MDMA_BENCH_CTX"
        );
        let mut kv = model.new_kv_cache(&backend, n_ctx).expect("kv");
        benchmark_prefix(&backend, &model, &mut kv, prefix);
        let mut ws = model.alloc_workspace(&backend, 1).expect("workspace");
        let mut decoder = backend.decoder(&model, &mut kv, &mut ws).expect("decoder");
        let mut logits = decoder.forward(1, prefix).expect("warmup forward");
        let start = Instant::now();
        for pos in prefix + 1..=prefix + n_steps {
            logits = decoder.forward(greedy(&logits), pos).expect("forward");
        }
        let elapsed = start.elapsed();
        eprintln!(
            "BENCH kind=decode prefix={prefix} tokens={n_steps} elapsed_ms={:.1} tok_s={:.2}",
            elapsed.as_secs_f64() * 1e3,
            f64::from(n_steps) / elapsed.as_secs_f64()
        );
    }
}

/// Effective weight bandwidth of the decode matvec at 7B shapes
/// (T = 1). The GPU's DRAM bandwidth is the ceiling.
///
/// Run explicitly: `cargo test --release --lib --features cuda
/// -- --ignored --nocapture bench_qmatmul_bandwidth`.
#[test]
#[ignore = "benchmark: cargo test --release --features cuda bench_qmatmul_bandwidth -- --ignored --nocapture"]
fn bench_qmatmul_bandwidth() {
    use std::time::Instant;
    let gpu = gpu();
    for (dtype, n_out, n_in) in [
        (GgmlDType::Q4K, 4096, 4096),
        (GgmlDType::Q4K, 14336, 4096),
        (GgmlDType::Q6K, 4096, 14336),
        (GgmlDType::Q6K, 92544, 4096),
    ] {
        let w = synthetic_kquant(dtype, n_out, n_in);
        let mut w_buf = gpu.alloc(w.len(), Pool::Weights).unwrap();
        gpu.upload(&mut w_buf, 0, &w).unwrap();
        let w_gpu = QWeight {
            dtype,
            shape: vec![n_out, n_in],
            buf: w_buf,
        };
        let x_buf = gpu.alloc(n_in * 4, Pool::Scratch).unwrap();
        let mut y_buf = gpu.alloc(n_out * 4, Pool::Scratch).unwrap();
        let mut sink = [0u8; 4];
        gpu.qmatmul(&w_gpu, &x_buf, &mut y_buf).unwrap();
        gpu.download(&y_buf, 0, &mut sink).unwrap();
        let reps = 50;
        let start = Instant::now();
        for _ in 0..reps {
            gpu.qmatmul(&w_gpu, &x_buf, &mut y_buf).unwrap();
        }
        gpu.download(&y_buf, 0, &mut sink).unwrap();
        let per = start.elapsed().as_secs_f64() / f64::from(reps);
        eprintln!(
            "BENCH kind=qmatmul dtype={dtype:?} shape={n_out}x{n_in} us={:.1} gb_s={:.1}",
            per * 1e6,
            w.len() as f64 / per / 1e9
        );
    }
}

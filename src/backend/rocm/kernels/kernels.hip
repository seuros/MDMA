// HIP kernels for mdma's ROCm backend, gfx1201 (RDNA4) target.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hybrid.hip"

// Dtype codes must match backend/dtype_code.rs.
#define DT_F32 0
#define DT_F16 1
#define DT_Q4K 2
#define DT_Q6K 3
#define DT_BF16 4
#define DT_Q5K 5

#define QK_K 256
#define K_SCALE_SIZE 12
#define BLOCK_Q4K_BYTES 144  // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs)
#define BLOCK_Q6K_BYTES 210  // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
#define BLOCK_Q5K_BYTES 176

__device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d_out, uint8_t* m_out) {
    if (j < 4) {
        *d_out = q[j] & 63;
        *m_out = q[j + 4] & 63;
    } else {
        *d_out = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m_out = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template<bool Packed>
__device__ __forceinline__ void q4k_scale_min(int j, const uint8_t* block, uint8_t* sc, uint8_t* mn) {
    if constexpr (Packed) {
        uint32_t planes[3];
        memcpy(planes, block + 4, sizeof(planes));
        const int shift = (j / 2) * 8, parity = j % 2;
        *sc = ((planes[0] >> (shift + parity * 4)) & 15)
            | (((planes[2] >> (shift + parity * 2)) & 3) << 4);
        *mn = ((planes[1] >> (shift + parity * 4)) & 15)
            | (((planes[2] >> (shift + 4 + parity * 2)) & 3) << 4);
    } else {
        get_scale_min_k4(j, block + 4, sc, mn);
    }
}

// Dequantize element `elem` (0..255) of one Q4_K superblock at `block`.
template<bool Packed = false>
__device__ __forceinline__ float dequant_q4k_elem(const uint8_t* block, int elem) {
    __half d_h, dmin_h;
    memcpy(&d_h, block, 2);
    memcpy(&dmin_h, block + 2, 2);
    float d = __half2float(d_h);
    float dmin = __half2float(dmin_h);
    const uint8_t* qs = block + (Packed ? Q4K_PACKED_QUANTS_OFFSET : 16);

    int outer = elem / 64;       // 0..4
    int inner = elem % 64;       // 0..64
    int is_high = inner >= 32;
    int q_byte_idx = outer * 32 + (inner % 32);
    int is = outer * 2 + is_high;

    uint8_t sc, m;
    q4k_scale_min<Packed>(is, block, &sc, &m);
    float dv = d * (float)sc;
    float mv = dmin * (float)m;

    uint8_t qbyte = qs[q_byte_idx];
    uint8_t q4 = is_high ? (qbyte >> 4) : (qbyte & 0xF);
    return dv * (float)q4 - mv;
}

__device__ __forceinline__ float dequant_q5k_elem(const uint8_t* block, int elem) {
    __half d, dmin;
    memcpy(&d, block, 2);
    memcpy(&dmin, block + 2, 2);
    int j = elem / 32, l = elem % 32;
    uint8_t sc, mn;
    get_scale_min_k4(j, block + 4, &sc, &mn);
    int lo = (block[48 + (j / 2) * 32 + l] >> (4 * (j & 1))) & 15;
    int hi = (block[16 + l] >> j) & 1;
    return __half2float(d) * (float)sc * (float)(lo + 16 * hi) - __half2float(dmin) * (float)mn;
}

// Dequantize element `elem` (0..255) of one Q6_K superblock at `block`.
__device__ __forceinline__ float dequant_q6k_elem(const uint8_t* block, int elem) {
    const uint8_t* ql_all = block;
    const uint8_t* qh_all = block + 128;
    const int8_t* sc_all = (const int8_t*)(block + 128 + 64);
    __half d_h;
    memcpy(&d_h, block + 128 + 64 + 16, 2);
    float d = __half2float(d_h);

    int idx = elem / 128;        // 0 or 1
    int local = elem % 128;      // 0..128
    int quadrant = local / 32;   // 0..4 -> which of q1/q2/q3/q4
    int l = local % 32;
    int is = l / 16;

    const uint8_t* ql = ql_all + 64 * idx;
    const uint8_t* qh = qh_all + 32 * idx;
    const int8_t* sc = sc_all + 8 * idx;

    int8_t qv;
    int sc_idx;
    switch (quadrant) {
    case 0:
        qv = (int8_t)((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
        sc_idx = is;
        break;
    case 1:
        qv = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        sc_idx = is + 2;
        break;
    case 2:
        qv = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        sc_idx = is + 4;
        break;
    default:
        qv = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        sc_idx = is + 6;
        break;
    }
    return d * (float)sc[sc_idx] * (float)qv;
}

// Each 256-element header is decoded once. At step g, all lanes share
// the same Q4_K sub-block and scale/min pair.
template<bool Packed = false>
__device__ __forceinline__ void gemv_q4k(const uint8_t* row, int n_in, const float* x, int lane, float& acc) {
    const int n_blocks = n_in / QK_K;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t* block = row + (size_t)b * (Packed ? BLOCK_Q4K_PACKED_BYTES : BLOCK_Q4K_BYTES);
        __half d_h, dmin_h;
        memcpy(&d_h, block, 2);
        memcpy(&dmin_h, block + 2, 2);
        const float d = __half2float(d_h);
        const float dmin = __half2float(dmin_h);
        const uint8_t* qs = block + (Packed ? Q4K_PACKED_QUANTS_OFFSET : 16);
        const int base = b * QK_K;

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            uint8_t sc0, m0, sc1, m1;
            q4k_scale_min<Packed>(2 * g, block, &sc0, &m0);
            q4k_scale_min<Packed>(2 * g + 1, block, &sc1, &m1);
            const uint8_t qbyte = qs[g * 32 + lane]; // coalesced 32B across the warp
            const int i0 = base + g * 64 + lane;
            acc += (d * (float)sc0 * (float)(qbyte & 0xF) - dmin * (float)m0) * x[i0];
            acc += (d * (float)sc1 * (float)(qbyte >> 4) - dmin * (float)m1) * x[i0 + 32];
        }
    }
}

// Q6_K scales differ between the two 16-lane halves; sc[] cannot be
// broadcast across the whole warp.
__device__ __forceinline__ void gemv_q6k(const uint8_t* row, int n_in, const float* x, int lane, float& acc) {
    const int n_blocks = n_in / QK_K;
    const int is = lane / 16; // NOT warp-uniform -- differs for lanes 0-15 vs 16-31
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t* block = row + (size_t)b * BLOCK_Q6K_BYTES;
        const uint8_t* ql_all = block;
        const uint8_t* qh_all = block + 128;
        const int8_t* sc_all = (const int8_t*)(block + 128 + 64);
        __half d_h;
        memcpy(&d_h, block + 128 + 64 + 16, 2);
        const float d = __half2float(d_h);
        const int base = b * QK_K;

        #pragma unroll
        for (int idx = 0; idx < 2; idx++) {
            const uint8_t* ql = ql_all + 64 * idx;
            const uint8_t* qh = qh_all + 32 * idx;
            const int8_t* sc = sc_all + 8 * idx;
            const uint8_t qh_b = qh[lane];
            const uint8_t ql0 = ql[lane];
            const uint8_t ql1 = ql[lane + 32];
            const int i0 = base + idx * 128 + lane;
            const int8_t qv0 = (int8_t)((ql0 & 0xF) | ((qh_b & 3) << 4)) - 32;
            const int8_t qv1 = (int8_t)((ql1 & 0xF) | (((qh_b >> 2) & 3) << 4)) - 32;
            const int8_t qv2 = (int8_t)((ql0 >> 4) | (((qh_b >> 4) & 3) << 4)) - 32;
            const int8_t qv3 = (int8_t)((ql1 >> 4) | (((qh_b >> 6) & 3) << 4)) - 32;
            acc += d * (float)sc[is] * (float)qv0 * x[i0];
            acc += d * (float)sc[is + 2] * (float)qv1 * x[i0 + 32];
            acc += d * (float)sc[is + 4] * (float)qv2 * x[i0 + 64];
            acc += d * (float)sc[is + 6] * (float)qv3 * x[i0 + 96];
        }
    }
}

// Preserve GEMV's memory accesses while reducing arithmetic.
// Every load contributes to the result to prevent dead-code elimination.
__device__ __forceinline__ void gemv_q4k_membound_probe(const uint8_t* row, int n_in, const float* x, int lane, float& acc) {
    const int n_blocks = n_in / QK_K;
    for (int b = 0; b < n_blocks; b++) {
        const uint8_t* block = row + (size_t)b * BLOCK_Q4K_BYTES;
        __half d_h, dmin_h;
        memcpy(&d_h, block, 2);
        memcpy(&dmin_h, block + 2, 2);
        const float d = __half2float(d_h);
        const float dmin = __half2float(dmin_h);
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 4 + K_SCALE_SIZE;
        const int base = b * QK_K;

        #pragma unroll
        for (int g = 0; g < 4; g++) {
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(2 * g, scales, &sc0, &m0);
            get_scale_min_k4(2 * g + 1, scales, &sc1, &m1);
            const uint8_t qbyte = qs[g * 32 + lane];
            const int i0 = base + g * 64 + lane;
            acc += (float)(qbyte & 0xF) * x[i0] + (float)(qbyte >> 4) * x[i0 + 32]
                 + d * 1e-9f + dmin * 1e-9f + (float)sc0 * 1e-9f + (float)m0 * 1e-9f
                 + (float)sc1 * 1e-9f + (float)m1 * 1e-9f;
        }
    }
}

// Q4_K memory-traffic probe, one warp per row.
extern "C" __global__ void qmatmul_gemv_membound_probe(
    const uint8_t* w, int n_out, int n_in, const float* x, float* y
) {
    const int row  = (int)((blockDim.x * blockIdx.x + threadIdx.x) / 32);
    const int lane = threadIdx.x & 31;
    const int rb   = (n_in / QK_K) * BLOCK_Q4K_BYTES;
    const uint8_t* row_ptr = (row < n_out) ? w + (size_t)row * rb : w;
    float acc = 0.0f;
    if (row < n_out) {
        gemv_q4k_membound_probe(row_ptr, n_in, x, lane, acc);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        acc += __shfl_xor(acc, mask);
    }
    if (lane == 0 && row < n_out) {
        y[row] = acc;
    }
}

// Dequantize element `i` of a row given its dtype. `row` points at the
// start of that row's raw quantized bytes.
__device__ __forceinline__ float dequant_elem(int dtype, const uint8_t* row, int i) {
    switch (dtype) {
    case DT_F32: {
        float v;
        memcpy(&v, row + i * 4, 4);
        return v;
    }
    case DT_F16: {
        __half h;
        memcpy(&h, row + i * 2, 2);
        return __half2float(h);
    }
    case DT_BF16: {
        uint16_t bits;
        memcpy(&bits, row + i * 2, 2);
        return __uint_as_float((uint32_t)bits << 16);
    }
    case DT_Q5K:
        return dequant_q5k_elem(row + (i / QK_K) * BLOCK_Q5K_BYTES, i % QK_K);
    case DT_Q4K:
        return dequant_q4k_elem(row + (i / QK_K) * BLOCK_Q4K_BYTES, i % QK_K);
    case DT_Q4K_PACKED:
        return dequant_q4k_elem<true>(row + (i / QK_K) * BLOCK_Q4K_PACKED_BYTES, i % QK_K);
    case DT_Q6K:
        return dequant_q6k_elem(row + (i / QK_K) * BLOCK_Q6K_BYTES, i % QK_K);
    default:
        return NAN; // Host rejects unsupported dtype tags before launch.
    }
}

__device__ __forceinline__ int row_bytes_for(int dtype, int n_cols) {
    switch (dtype) {
    case DT_F32: return n_cols * 4;
    case DT_F16:
    case DT_BF16: return n_cols * 2;
    case DT_Q4K: return (n_cols / QK_K) * BLOCK_Q4K_BYTES;
    case DT_Q4K_PACKED: return (n_cols / QK_K) * BLOCK_Q4K_PACKED_BYTES;
    case DT_Q5K: return (n_cols / QK_K) * BLOCK_Q5K_BYTES;
    case DT_Q6K: return (n_cols / QK_K) * BLOCK_Q6K_BYTES;
    default: return 0;
    }
}

// One warp per output row; lane l accumulates columns l, l+32, l+64, ...
// before a shuffle reduction. All lanes must reach the shuffle, including
// padded rows; mask the final store rather than individual lanes.
extern "C" __global__ void qmatmul_gemv(
    const uint8_t* w, int dtype, int n_out, int n_in, const float* x, float* y
) {
    const int row  = (int)((blockDim.x * blockIdx.x + threadIdx.x) / 32);
    const int lane = threadIdx.x & 31;
    const int rb   = row_bytes_for(dtype, n_in);
    const uint8_t* row_ptr = (row < n_out) ? w + (size_t)row * rb : w;
    float acc = 0.0f;
    if (row < n_out) {
        // Q4_K/Q6_K: header (d/dmin/scales) decoded once per 256-element
        // superblock instead of dequant_elem's per-element re-derivation.
        // GGUF guarantees n_in % QK_K == 0 for these dtypes, so no scalar
        // tail is needed on this path.
        switch (dtype) {
        case DT_Q4K:
            gemv_q4k(row_ptr, n_in, x, lane, acc);
            break;
        case DT_Q4K_PACKED:
            gemv_q4k<true>(row_ptr, n_in, x, lane, acc);
            break;
        case DT_Q6K:
            gemv_q6k(row_ptr, n_in, x, lane, acc);
            break;
        default:
            for (int i = lane; i < n_in; i += 32) {
                acc += dequant_elem(dtype, row_ptr, i) * x[i];
            }
            break;
        }
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        acc += __shfl_xor(acc, mask);
    }
    if (lane == 0 && row < n_out) {
        y[row] = acc;
    }
}

// Vector-of-half types used by the attention and GEMM kernels below and
// by the included files.
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half4_t __attribute__((ext_vector_type(4)));

// Emit the Q8_1 quantization of one float4 chunk of a normalized row:
// eight consecutive lanes hold one 32-value block (as in
// quantize_q8_1_kernel), so amax and the block sum are three xor
// shuffles each. d is rounded to f16 as ggml's block_q8_1 defines it.
__device__ __forceinline__ void q8_emit_chunk(const float4 v, int chunk, int8_t* q8, float2* ys) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) amax = fmaxf(amax, __shfl_xor(amax, mask));
    const float d = __half2float(__float2half(amax / 127.0f));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q0 = (int)roundf(v.x * inv), q1 = (int)roundf(v.y * inv);
    const int q2 = (int)roundf(v.z * inv), q3 = (int)roundf(v.w * inv);
    const unsigned packed = (unsigned)(q0 & 0xFF) | ((unsigned)(q1 & 0xFF) << 8) | ((unsigned)(q2 & 0xFF) << 16) | ((unsigned)(q3 & 0xFF) << 24);
    memcpy(q8 + chunk * 4, &packed, 4);
    int sum_q = q0 + q1 + q2 + q3;
    #pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) sum_q += __shfl_xor(sum_q, mask);
    if ((threadIdx.x & 7) == 0) ys[chunk / 8] = make_float2(d, (float)sum_q);
}

#include "decode.hip"
#include "attention.hip"
#include "decoder_ops.hip"

// One thread per output row, looping tokens inside: y[t, row] =
// dot(dequant(w[row, :]), x[t, :]). x/y are [n_tokens, n_in]/[n_tokens, n_out]
// row-major.
extern "C" __global__ void qmatmul_gemm(
    const uint8_t* w, int dtype, int n_out, int n_in, int n_tokens, const float* x, float* y
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_out) return;
    int rb = row_bytes_for(dtype, n_in);
    const uint8_t* row_ptr = w + (size_t)row * rb;
    for (int t = 0; t < n_tokens; t++) {
        const float* x_t = x + (size_t)t * n_in;
        float acc = 0.0f;
        for (int i = 0; i < n_in; i++) {
            acc += dequant_elem(dtype, row_ptr, i) * x_t[i];
        }
        y[(size_t)t * n_out + row] = acc;
    }
}

// Single-row dequantized gather, for embedding lookup.
extern "C" __global__ void embed_gather(
    const uint8_t* table, int dtype, int n_embd, unsigned int token_id, float* out
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_embd) return;
    int rb = row_bytes_for(dtype, n_embd);
    const uint8_t* row_ptr = table + (size_t)token_id * rb;
    out[i] = dequant_elem(dtype, row_ptr, i);
}

extern "C" __global__ void embed_gather_graph(
    const uint8_t* table, int dtype, int n_embd, unsigned int, float* out, const unsigned int* params
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n_embd)
        out[i] = dequant_elem(dtype, table + (size_t)params[0] * row_bytes_for(dtype, n_embd), i);
}

// Round f32 rows to f16 on entry to the KV cache.
extern "C" __global__ void kv_write_f16(
    const float* k, const float* v, _Float16* dst_k, _Float16* dst_v, int n, int offset
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        dst_k[(size_t)offset + i] = (_Float16)k[i];
        dst_v[(size_t)offset + i] = (_Float16)v[i];
    }
}

extern "C" __global__ void kv_write_graph(
    const float* k, const float* v, _Float16* dst_k, _Float16* dst_v, int n, const unsigned int* params
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        size_t offset = (size_t)params[1] * n + i;
        dst_k[offset] = (_Float16)k[i];
        dst_v[offset] = (_Float16)v[i];
    }
}

extern "C" __global__ void add_kernel(const float* a, const float* b, float* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

extern "C" __global__ void silu_mul_kernel(const float* gate, const float* up, float* y, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        y[i] = silu * up[i];
    }
}

// One block per row (n_embd elements), block-wide reduction for mean(x^2).
// blockIdx.x selects the row (token): x/y are [n_tokens][n], one launch
// covers the whole batch.
extern "C" __global__ void rmsnorm_kernel(const float* w, const float* x, float* y, int n, float eps) {
    __shared__ float wave_sums[8]; // Host launches up to eight full 32-lane groups.
    x += (size_t)blockIdx.x * n;
    y += (size_t)blockIdx.x * n;
    int tid = threadIdx.x;
    float local = 0.0f;
    if (n % 4 == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        for (int i = tid; i < n / 4; i += blockDim.x) {
            const float4 v = x4[i];
            local += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = tid; i < n; i += blockDim.x) {
            float v = x[i];
            local += v * v;
        }
    }
    const int lane = tid & 31;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) local += __shfl_xor(local, mask, 32);
    if (lane == 0) wave_sums[tid / 32] = local;
    __syncthreads();
    local = lane < blockDim.x / 32 ? wave_sums[lane] : 0.0f;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) local += __shfl_xor(local, mask, 32);
    float mean_sq = local / (float)n;
    float scale = rsqrtf(mean_sq + eps);
    if (n % 4 == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* w4 = reinterpret_cast<const float4*>(w);
        float4* y4 = reinterpret_cast<float4*>(y);
        for (int i = tid; i < n / 4; i += blockDim.x) {
            const float4 v = x4[i], weight = w4[i];
            y4[i] = make_float4(v.x * scale * weight.x, v.y * scale * weight.y,
                               v.z * scale * weight.z, v.w * scale * weight.w);
        }
    } else {
        for (int i = tid; i < n; i += blockDim.x) {
            y[i] = x[i] * scale * w[i];
        }
    }
}

// RMSNorm variants: ADD folds the residual add in (sum = a + b is
// written, then normalized); Q8 also emits the row's Q8_1 quantization
// so the matvecs that follow skip their quantize launch (decode, one
// row). One block per row; requires n % 4 == 0 and a block that is a
// multiple of 8 lanes -- the host falls back to the plain ops otherwise.
template <bool ADD, bool Q8>
__device__ __forceinline__ void rmsnorm_body(
    const float* w, const float* a, const float* b, float* sum, float* y, int n, float eps, int8_t* q8, float2* ys
) {
    __shared__ float wave_sums[8];
    const size_t row = (size_t)blockIdx.x * n;
    const float4* a4 = reinterpret_cast<const float4*>(a + row);
    const float4* b4 = ADD ? reinterpret_cast<const float4*>(b + row) : nullptr;
    float4* s4 = ADD ? reinterpret_cast<float4*>(sum + row) : nullptr;
    const float4* src4 = ADD ? s4 : a4;
    float4* y4 = reinterpret_cast<float4*>(y + row);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < n / 4; i += blockDim.x) {
        float4 v = a4[i];
        if constexpr (ADD) {
            const float4 q = b4[i];
            v = make_float4(v.x + q.x, v.y + q.y, v.z + q.z, v.w + q.w);
            s4[i] = v;
        }
        local += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    const int lane = tid & 31;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) local += __shfl_xor(local, mask, 32);
    if (lane == 0) wave_sums[tid / 32] = local;
    __syncthreads();
    local = lane < blockDim.x / 32 ? wave_sums[lane] : 0.0f;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) local += __shfl_xor(local, mask, 32);
    const float scale = rsqrtf(local / (float)n + eps);
    for (int i = tid; i < n / 4; i += blockDim.x) {
        const float4 v = src4[i], weight = w4[i];
        const float4 out = make_float4(v.x * scale * weight.x, v.y * scale * weight.y, v.z * scale * weight.z, v.w * scale * weight.w);
        y4[i] = out;
        if constexpr (Q8) q8_emit_chunk(out, i, q8 + row, ys + row / 32);
    }
}

extern "C" __global__ void add_rmsnorm_kernel(const float* w, const float* a, const float* b, float* sum, float* y, int n, float eps) {
    rmsnorm_body<true, false>(w, a, b, sum, y, n, eps, nullptr, nullptr);
}

extern "C" __global__ void rmsnorm_q8_kernel(const float* w, const float* x, float* y, int n, float eps, int8_t* q8, float2* ys) {
    rmsnorm_body<false, true>(w, x, nullptr, nullptr, y, n, eps, q8, ys);
}

extern "C" __global__ void add_rmsnorm_q8_kernel(const float* w, const float* a, const float* b, float* sum, float* y, int n, float eps, int8_t* q8, float2* ys) {
    rmsnorm_body<true, true>(w, a, b, sum, y, n, eps, q8, ys);
}

// Interleaved-pair RoPE ((2i, 2i+1)). One thread per
// (head, pair-index) in q and k combined: threads 0..half*n_q_heads do
// q, the rest do k.
// blockIdx.y selects the token: q/k advance by their per-token strides
// and pos by the token index, so one launch covers a whole prefill
// batch.
__device__ __forceinline__ void rope_impl(
    float* q, float* k, unsigned int pos, int head_dim, int n_q_heads, int n_kv_heads, float theta,
    int q_stride, int k_stride
) {
    q += (size_t)blockIdx.y * q_stride;
    k += (size_t)blockIdx.y * k_stride;
    pos += blockIdx.y;
    int half = head_dim / 2;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_q = n_q_heads * half;
    int total_k = n_kv_heads * half;
    float* buf;
    int head, i;
    if (tid < total_q) {
        buf = q;
        head = tid / half;
        i = tid % half;
    } else if (tid < total_q + total_k) {
        int t = tid - total_q;
        buf = k;
        head = t / half;
        i = t % half;
    } else {
        return;
    }
    float freq = powf(theta, -2.0f * (float)i / (float)head_dim);
    float angle = (float)pos * freq;
    float sin_a, cos_a;
    sincosf(angle, &sin_a, &cos_a);
    float* h = buf + head * head_dim;
    float x0 = h[2 * i];
    float x1 = h[2 * i + 1];
    h[2 * i] = x0 * cos_a - x1 * sin_a;
    h[2 * i + 1] = x0 * sin_a + x1 * cos_a;
}

// RoPE fused with the KV write: q rotates in place; k rotates and lands
// straight in the f16 cache, and the same threads copy v.
// dst offsets are in elements; token t writes row
// elem_offset + t * k_stride.
__device__ __forceinline__ void rope_kv_impl(
    float* q, const float* k, const float* v, _Float16* dst_k, _Float16* dst_v, size_t elem_offset,
    unsigned int pos, int head_dim, int n_q_heads, int n_kv_heads, float theta, int q_stride, int k_stride
) {
    q += (size_t)blockIdx.y * q_stride;
    k += (size_t)blockIdx.y * k_stride;
    v += (size_t)blockIdx.y * k_stride;
    const size_t row = elem_offset + (size_t)blockIdx.y * k_stride;
    pos += blockIdx.y;
    const int half = head_dim / 2;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int total_q = n_q_heads * half;
    const int total_k = n_kv_heads * half;
    if (tid >= total_q + total_k) return;
    const bool is_q = tid < total_q;
    const int t = is_q ? tid : tid - total_q;
    const int head = t / half, i = t % half;
    const float freq = powf(theta, -2.0f * (float)i / (float)head_dim);
    float sin_a, cos_a;
    sincosf((float)pos * freq, &sin_a, &cos_a);
    const int e = head * head_dim + 2 * i;
    if (is_q) {
        const float x0 = q[e], x1 = q[e + 1];
        q[e] = x0 * cos_a - x1 * sin_a;
        q[e + 1] = x0 * sin_a + x1 * cos_a;
    } else {
        const float x0 = k[e], x1 = k[e + 1];
        dst_k[row + e] = (_Float16)(x0 * cos_a - x1 * sin_a);
        dst_k[row + e + 1] = (_Float16)(x0 * sin_a + x1 * cos_a);
        dst_v[row + e] = (_Float16)v[e];
        dst_v[row + e + 1] = (_Float16)v[e + 1];
    }
}

extern "C" __global__ void rope_kv_kernel(
    float* q, const float* k, const float* v, _Float16* dst_k, _Float16* dst_v, int elem_offset,
    unsigned int pos, int head_dim, int n_q_heads, int n_kv_heads, float theta, int q_stride, int k_stride
) {
    rope_kv_impl(q, k, v, dst_k, dst_v, (size_t)elem_offset, pos, head_dim, n_q_heads, n_kv_heads, theta, q_stride, k_stride);
}

extern "C" __global__ void rope_kv_graph(
    float* q, const float* k, const float* v, _Float16* dst_k, _Float16* dst_v, int,
    unsigned int, int head_dim, int n_q_heads, int n_kv_heads, float theta, int q_stride, int k_stride, const unsigned int* params
) {
    rope_kv_impl(q, k, v, dst_k, dst_v, (size_t)params[1] * k_stride, params[1], head_dim, n_q_heads, n_kv_heads, theta, q_stride, k_stride);
}

extern "C" __global__ void rope_kernel(
    float* q, float* k, unsigned int pos, int head_dim, int n_q_heads, int n_kv_heads, float theta,
    int q_stride, int k_stride
) {
    rope_impl(q, k, pos, head_dim, n_q_heads, n_kv_heads, theta, q_stride, k_stride);
}

extern "C" __global__ void rope_graph(
    float* q, float* k, unsigned int, int head_dim, int n_q_heads, int n_kv_heads, float theta,
    int q_stride, int k_stride, const unsigned int* params
) {
    rope_impl(q, k, params[1], head_dim, n_q_heads, n_kv_heads, theta, q_stride, k_stride);
}

// One block per query head. Scores computed into shared memory, softmax
// reduced within the block, then weighted sum over V. GQA uses kvh = qh / group.
extern "C" __device__ __forceinline__ void attn_decode_impl(
    const float* q, const _Float16* k, const _Float16* v, float* o,
    int head_dim, int n_q_heads, int n_kv_heads, int kv_len
) {
    extern __shared__ float scores[];  // kv_len floats
    int qh = blockIdx.x;
    if (qh >= n_q_heads) return;
    int group = n_q_heads / n_kv_heads;
    int kvh = qh / group;
    int kv_row = n_kv_heads * head_dim;
    float scale = rsqrtf((float)head_dim);
    const float* q_vec = q + qh * head_dim;

    int tid = threadIdx.x;
    for (int t = tid; t < kv_len; t += blockDim.x) {
        const _Float16* k_vec = k + (size_t)t * kv_row + kvh * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q_vec[d] * (float)k_vec[d];
        scores[t] = acc * scale;
    }
    __syncthreads();

    if (tid == 0) {
        float max_v = -INFINITY;
        for (int t = 0; t < kv_len; t++) max_v = fmaxf(max_v, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < kv_len; t++) {
            scores[t] = expf(scores[t] - max_v);
            sum += scores[t];
        }
        for (int t = 0; t < kv_len; t++) scores[t] /= sum;
    }
    __syncthreads();

    float* o_vec = o + qh * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < kv_len; t++) {
            const _Float16* v_vec = v + (size_t)t * kv_row + kvh * head_dim;
            acc += scores[t] * (float)v_vec[d];
        }
        o_vec[d] = acc;
    }
}

extern "C" __global__ void attn_decode_kernel(
    const float* q, const _Float16* k, const _Float16* v, float* o,
    int head_dim, int n_q_heads, int n_kv_heads, int kv_len
) {
    attn_decode_impl(q, k, v, o, head_dim, n_q_heads, n_kv_heads, kv_len);
}

extern "C" __global__ void attn_decode_graph(
    const float* q, const _Float16* k, const _Float16* v, float* o,
    int head_dim, int n_q_heads, int n_kv_heads, int, const unsigned int* params
) {
    attn_decode_impl(q, k, v, o, head_dim, n_q_heads, n_kv_heads, params[1] + 1);
}

// One block per (query_idx, query_head) pair, folded into blockIdx.x as
// i*n_q_heads+qh. Causal: query i sees keys [0, prefix+i] inclusive
// with prefix = kv_len - n_q.
extern "C" __global__ void attn_prefill_kernel(
    const float* q, const _Float16* k, const _Float16* v, float* o,
    int head_dim, int n_q_heads, int n_kv_heads, int n_q, int prefix
) {
    extern __shared__ float scores[];  // up to (prefix+n_q) floats
    int block_id = blockIdx.x;
    int i = block_id / n_q_heads;
    int qh = block_id % n_q_heads;
    if (i >= n_q) return;

    int group = n_q_heads / n_kv_heads;
    int kvh = qh / group;
    int kv_row = n_kv_heads * head_dim;
    float scale = rsqrtf((float)head_dim);
    int visible = prefix + i + 1;  // exclusive upper bound, causal

    const float* q_vec = q + (size_t)(i * n_q_heads + qh) * head_dim;

    int tid = threadIdx.x;
    for (int t = tid; t < visible; t += blockDim.x) {
        const _Float16* k_vec = k + (size_t)t * kv_row + kvh * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < head_dim; d++) acc += q_vec[d] * (float)k_vec[d];
        scores[t] = acc * scale;
    }
    __syncthreads();

    if (tid == 0) {
        float max_v = -INFINITY;
        for (int t = 0; t < visible; t++) max_v = fmaxf(max_v, scores[t]);
        float sum = 0.0f;
        for (int t = 0; t < visible; t++) {
            scores[t] = expf(scores[t] - max_v);
            sum += scores[t];
        }
        for (int t = 0; t < visible; t++) scores[t] /= sum;
    }
    __syncthreads();

    float* o_vec = o + (size_t)(i * n_q_heads + qh) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < visible; t++) {
            const _Float16* v_vec = v + (size_t)t * kv_row + kvh * head_dim;
            acc += scores[t] * (float)v_vec[d];
        }
        o_vec[d] = acc;
    }
}

#if defined(__gfx1200__) || defined(__gfx1201__)

typedef float    float8_t __attribute__((ext_vector_type(8)));

// One WMMA instruction on row-major 16x16 inputs.
// Output acc[j] maps to row (lane/16)*8+j, column lane%16.
extern "C" __global__ void wmma_probe_kernel(const float* a_in, const float* b_in, float* d_out) {
    __shared__ _Float16 As[16][16];  // As[m][k]
    __shared__ _Float16 Bs[16][16];  // Bs[k][n]

    int tid = threadIdx.x;
    for (int e = tid; e < 256; e += 32) {
        int r = e / 16, c = e % 16;
        As[r][c] = (_Float16)a_in[r * 16 + c];
        Bs[r][c] = (_Float16)b_in[r * 16 + c];
    }
    __syncthreads();

    int lane = tid & 31;
    int col = lane % 16;
    int row_base = (lane / 16) * 8;

    // A is K-major: eight contiguous K values per lane, M = lane%16.
    // B and C/D are column-distributed.
    half8_t a, b;
    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        a[j] = As[col][row_base + j];  // A[m][k], m=col, k=row_base+j
        b[j] = Bs[row_base + j][col];  // B[k][n], k=row_base+j, n=col
    }
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int row = row_base + j;
        d_out[row * 16 + col] = acc[j];
    }
}

// WMMA computes Y^T = dequant(W) @ X^T, stored with row/token axes swapped.
// Each 128-thread block covers 64 output rows x 16 tokens.
// A is K-major: As[lane%16][8*(lane/16)+j]. B and C/D are
// column-distributed, with column lane%16 and row 8*(lane/16)+j.
extern "C" __global__ void qmatmul_gemm_wmma(
    const uint8_t* w, int dtype, int n_out, int n_in, int n_tokens, const float* x, float* y
) {
    const int m0 = blockIdx.x * 64;   // output-row (M) tile origin
    const int n0 = blockIdx.y * 16;   // token (N) tile origin
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;
    const int row_base = (lane / 16) * 8;
    const int rb = row_bytes_for(dtype, n_in);

    __shared__ _Float16 As[64][16];  // As[m][k], m local to this block's M-tile
    __shared__ _Float16 Bs[16][16];  // Bs[k][n]

    // Per-row superblock coefficient cache, coef[j][r] for local row r
    // in 0..64. One dtype per launch, so both K-quant layouts share the
    // same 4KB of LDS:
    //   Q4_K: coef[s][r]   = d[r]*scale[r][s]  (s in 0..8)
    //         coef[8+s][r] = dmin[r]*min[r][s]
    //   Q6_K: coef[j][r]   = d[r]*sc[r][j]     (j in 0..16, signed sc)
    // One producer thread decodes each row's coefficients into shared storage.
    __shared__ float coef[16][64];

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const bool kquant = (dtype == DT_Q4K) || (dtype == DT_Q6K);
    const int block_bytes = (dtype == DT_Q4K) ? BLOCK_Q4K_BYTES : BLOCK_Q6K_BYTES;

    for (int k0 = 0; k0 < n_in; k0 += 16) {
        if (kquant && (k0 % QK_K) == 0) {
            // 64 dedicated producer threads, one row each -- uniform
            // across the block (k0 and dtype are loop-invariant/kernel-
            // wide, not per-thread-divergent), so the barrier below is
            // safe.
            if (threadIdx.x < 64) {
                int rm = threadIdx.x;
                int row = m0 + rm;
                if (row < n_out) {
                    const uint8_t* block = w + (size_t)row * rb + (size_t)(k0 / QK_K) * block_bytes;
                    if (dtype == DT_Q4K) {
                        __half d_h, dmin_h;
                        memcpy(&d_h, block, 2);
                        memcpy(&dmin_h, block + 2, 2);
                        const float d = __half2float(d_h);
                        const float dmin = __half2float(dmin_h);
                        const uint8_t* scales = block + 4;
                        #pragma unroll
                        for (int s = 0; s < 8; s++) {
                            uint8_t sc, m;
                            get_scale_min_k4(s, scales, &sc, &m);
                            coef[s][rm] = d * (float)sc;
                            coef[8 + s][rm] = dmin * (float)m;
                        }
                    } else {
                        // Q6_K: 16 signed int8 scales at +192, d at +208
                        // (see dequant_q6k_elem's layout).
                        const int8_t* sc = (const int8_t*)(block + 128 + 64);
                        __half d_h;
                        memcpy(&d_h, block + 128 + 64 + 16, 2);
                        const float d = __half2float(d_h);
                        #pragma unroll
                        for (int j = 0; j < 16; j++) {
                            coef[j][rm] = d * (float)sc[j];
                        }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        coef[j][rm] = 0.0f;
                    }
                }
            }
            __syncthreads();
        }
        // W tile: 64 rows x 16 K, one dequant per (row,k), reused by all
        // 16 tokens below instead of redequantizing per token. K-quants
        // read the LDS coefficient cache above instead of re-deriving
        // the superblock header per element; F32/F16 stay on
        // dequant_elem's per-element path.
        for (int e = threadIdx.x; e < 64 * 16; e += 128) {
            int rm = e / 16, kk = e % 16;
            int row = m0 + rm, k = k0 + kk;
            float v = 0.0f;
            if (row < n_out && k < n_in) {
                switch (dtype) {
                case DT_Q4K: {
                    int elem = k % QK_K;
                    int s = elem / 32;
                    int q_byte_idx = (elem / 64) * 32 + (elem % 32);
                    bool high = (elem % 64) >= 32;
                    const uint8_t* qs = w + (size_t)row * rb + (size_t)(k0 / QK_K) * BLOCK_Q4K_BYTES + 4 + K_SCALE_SIZE;
                    uint8_t qbyte = qs[q_byte_idx];
                    uint8_t q4 = high ? (qbyte >> 4) : (qbyte & 0xF);
                    v = coef[s][rm] * (float)q4 - coef[8 + s][rm];
                    break;
                }
                case DT_Q6K: {
                    // Same (ql-nibble, qh-bit-pair, sc-offset) tuples as
                    // dequant_q6k_elem, minus the per-element d/sc reads.
                    const uint8_t* block = w + (size_t)row * rb + (size_t)(k0 / QK_K) * BLOCK_Q6K_BYTES;
                    int elem = k % QK_K;
                    int idx = elem / 128;
                    int local = elem % 128;
                    int quadrant = local / 32;
                    int l = local % 32;
                    int is = l / 16;
                    const uint8_t* ql = block + 64 * idx;
                    const uint8_t* qh = block + 128 + 32 * idx;
                    int8_t qv;
                    switch (quadrant) {
                    case 0:
                        qv = (int8_t)((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                        break;
                    case 1:
                        qv = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        break;
                    case 2:
                        qv = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        break;
                    default:
                        qv = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        break;
                    }
                    v = coef[8 * idx + 2 * quadrant + is][rm] * (float)qv;
                    break;
                }
                default:
                    v = dequant_elem(dtype, w + (size_t)row * rb, k);
                    break;
                }
            }
            As[rm][kk] = (_Float16)v;
        }
        // X tile, transposed into Bs[k][t] on load (X is [n_tokens, n_in]
        // row-major; B = X^T is [n_in, n_tokens]).
        for (int e = threadIdx.x; e < 16 * 16; e += 128) {
            int kk = e / 16, tt = e % 16;
            int k = k0 + kk, t = n0 + tt;
            float v = 0.0f;
            if (k < n_in && t < n_tokens) {
                v = x[(size_t)t * n_in + k];
            }
            Bs[kk][tt] = (_Float16)v;
        }
        __syncthreads();

        half8_t a, b;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            a[j] = As[warp * 16 + col][row_base + j];  // A[m][k], K-major
            b[j] = Bs[row_base + j][col];              // B[k][n]
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int row = m0 + warp * 16 + row_base + j;
        int t = n0 + col;
        if (row < n_out && t < n_tokens) {
            y[(size_t)t * n_out + row] = acc[j];
        }
    }
}

// Reduced-arithmetic Q4_K probe. Every loaded value contributes to the
// result, preserving the memory accesses through optimization.
__device__ __forceinline__ float dequant_q4k_elem_membound_probe(const uint8_t* block, int elem) {
    __half d_h, dmin_h;
    memcpy(&d_h, block, 2);
    memcpy(&dmin_h, block + 2, 2);
    float d = __half2float(d_h);
    float dmin = __half2float(dmin_h);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 4 + K_SCALE_SIZE;

    int outer = elem / 64;
    int inner = elem % 64;
    int is_high = inner >= 32;
    int q_byte_idx = outer * 32 + (inner % 32);
    int is = outer * 2 + is_high; // touch the same scales byte real dequant would use
    uint8_t sc_touch = scales[is % K_SCALE_SIZE];

    uint8_t qbyte = qs[q_byte_idx];
    uint8_t q4 = is_high ? (qbyte >> 4) : (qbyte & 0xF);
    return (float)q4 + d * 1e-9f + dmin * 1e-9f + (float)sc_touch * 1e-9f;
}

// WMMA memory-traffic probe using reduced-arithmetic Q4_K staging.
extern "C" __global__ void qmatmul_gemm_wmma_membound_probe(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const float* x, float* y
) {
    const int m0 = blockIdx.x * 64;
    const int n0 = blockIdx.y * 16;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;
    const int row_base = (lane / 16) * 8;
    const int rb = row_bytes_for(DT_Q4K, n_in);

    __shared__ _Float16 As[64][16];
    __shared__ _Float16 Bs[16][16];

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int k0 = 0; k0 < n_in; k0 += 16) {
        for (int e = threadIdx.x; e < 64 * 16; e += 128) {
            int rm = e / 16, kk = e % 16;
            int row = m0 + rm, k = k0 + kk;
            float v = 0.0f;
            if (row < n_out && k < n_in) {
                v = dequant_q4k_elem_membound_probe(w + (size_t)row * rb + (k / QK_K) * BLOCK_Q4K_BYTES, k % QK_K);
            }
            As[rm][kk] = (_Float16)v;
        }
        for (int e = threadIdx.x; e < 16 * 16; e += 128) {
            int kk = e / 16, tt = e % 16;
            int k = k0 + kk, t = n0 + tt;
            float v = 0.0f;
            if (k < n_in && t < n_tokens) {
                v = x[(size_t)t * n_in + k];
            }
            Bs[kk][tt] = (_Float16)v;
        }
        __syncthreads();

        half8_t a, b;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            a[j] = As[warp * 16 + col][row_base + j];
            b[j] = Bs[row_base + j][col];
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        int row = m0 + warp * 16 + row_base + j;
        int t = n0 + col;
        if (row < n_out && t < n_tokens) {
            y[(size_t)t * n_out + row] = acc[j];
        }
    }
}

// Integer WMMA operates on unscaled weight values and Q8_1 activations.
// Per-row weight coefficients and per-token activation scales are applied
// to the int32 result in the float epilogue.

typedef int v2i32_t __attribute__((ext_vector_type(2)));
typedef int v8i32_t __attribute__((ext_vector_type(8)));

// Little-endian byte packing, byte 0 = lowest K. Goes through uint8_t so
// negative int8 values never get sign-extended into the neighbouring
// bytes (shifting a sign-extended int would smear 0xFF across the word).
__device__ __forceinline__ int pack4_i8(int8_t b0, int8_t b1, int8_t b2, int8_t b3) {
    return (int)((unsigned)(uint8_t)b0
               | ((unsigned)(uint8_t)b1 << 8)
               | ((unsigned)(uint8_t)b2 << 16)
               | ((unsigned)(uint8_t)b3 << 24));
}

// iu8 fragment packing: A byte j = A[lane%16][8*(lane/16)+j], B byte j =
// B[8*(lane/16)+j][lane%16], C[j] = C[8*(lane/16)+j][lane%16], with the
// two packed ints holding consecutive groups of four K elements. Inputs
// arrive as int32 so the host side never has to reason about int8 byte
// packing; only this kernel does.
extern "C" __global__ void wmma_i8_probe_kernel(const int* a_in, const int* b_in, const int* c_in, int* d_out) {
    __shared__ int8_t As[16][16];  // A[m][k]
    __shared__ int8_t Bs[16][16];  // B[k][n]

    int tid = threadIdx.x;
    for (int e = tid; e < 256; e += 32) {
        int r = e / 16, c = e % 16;
        As[r][c] = (int8_t)a_in[r * 16 + c];
        Bs[r][c] = (int8_t)b_in[r * 16 + c];
    }
    __syncthreads();

    int lane = tid & 31;
    int col = lane % 16;
    int row_base = (lane / 16) * 8;

    v2i32_t a, b;
    a[0] = pack4_i8(As[col][row_base + 0], As[col][row_base + 1], As[col][row_base + 2], As[col][row_base + 3]);
    a[1] = pack4_i8(As[col][row_base + 4], As[col][row_base + 5], As[col][row_base + 6], As[col][row_base + 7]);
    b[0] = pack4_i8(Bs[row_base + 0][col], Bs[row_base + 1][col], Bs[row_base + 2][col], Bs[row_base + 3][col]);
    b[1] = pack4_i8(Bs[row_base + 4][col], Bs[row_base + 5][col], Bs[row_base + 6][col], Bs[row_base + 7][col]);

    v8i32_t acc;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        acc[j] = c_in[(row_base + j) * 16 + col];
    }
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc, false);

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        d_out[(row_base + j) * 16 + col] = acc[j];
    }
}

// Q8_1: d = f16(amax/127), q = roundf(x * 127/amax).
// Zero blocks produce d=0, q=0. Store row-major q8[t][n_in] and
// ys[t*(n_in/32)+b] = {d, sum(q)}; the integer sum is exact in f32.
// Consumers reconstruct s = d * sum(q), not sum of the unquantized inputs.
// Eight lanes share each 32-element block, reducing amax and sum(q) with
// three XOR shuffles. Requires n_in % 32 == 0.
#define Q8_1_LANES 8
extern "C" __global__ void quantize_q8_1_kernel(const float* x, int n_in, int n_tokens, int8_t* q8, float2* ys) {
    const int nb = n_in / 32;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = gid / Q8_1_LANES;
    const int sub = gid % Q8_1_LANES;
    if (idx >= n_tokens * nb) return;
    const int t = idx / nb, b = idx % nb;
    const float* xb = x + (size_t)t * n_in + b * 32 + sub * 4;
    float4 xv;
    memcpy(&xv, xb, 16);
    float amax = fmaxf(fmaxf(fabsf(xv.x), fabsf(xv.y)), fmaxf(fabsf(xv.z), fabsf(xv.w)));
    #pragma unroll
    for (int mask = Q8_1_LANES / 2; mask > 0; mask >>= 1) amax = fmaxf(amax, __shfl_xor(amax, mask));
    // Store d rounded to f16, but compute q from the unrounded amax.
    const float d = __half2float(__float2half(amax / 127.0f));
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const int q0 = (int)roundf(xv.x * inv), q1 = (int)roundf(xv.y * inv);
    const int q2 = (int)roundf(xv.z * inv), q3 = (int)roundf(xv.w * inv);
    const unsigned packed = (unsigned)(q0 & 0xFF) | ((unsigned)(q1 & 0xFF) << 8) | ((unsigned)(q2 & 0xFF) << 16) | ((unsigned)(q3 & 0xFF) << 24);
    memcpy(q8 + (size_t)t * n_in + b * 32 + sub * 4, &packed, 4);
    int sum_q = q0 + q1 + q2 + q3;
    #pragma unroll
    for (int mask = Q8_1_LANES / 2; mask > 0; mask >>= 1) sum_q += __shfl_xor(sum_q, mask);
    if (sub == 0) ys[idx] = make_float2(d, (float)sum_q);
}

// Q4_K integer WMMA with Q8_1 activations and K=64 staging.
//
// Per 32-element block b of a row r and token t, with Q4_K sub-block s:
//   sum_i (d*sc_s*q4_i - dmin*m_s) * x_i
//     = coef[s][r] * d_x[t][b] * DOT(q4, q8)  -  coef[8+s][r] * sum_x[t][b]
// DOT is the exact int32 WMMA result (two K=16 iu8 WMMAs into a zeroed
// int32 accumulator: max |15*127*32| = 60960, no saturation possible,
// clamp=false). sum_x = d_x * sum(q8). Reset the integer accumulator
// every 32 K because the scale and min coefficients change per block.
// Epilogue coefficients use the C fragment's output row, not its A load row.
//
// LDS rows are padded to 72 bytes: 8-byte fragment loads stay aligned
// (72 % 8 == 0) and the 16-lane gather at 18-DWORD stride lands on 16
// distinct banks instead of the 4-way collision a 64-byte stride gives.
// Fragment bytes are read from LDS as two aligned, little-endian ints.
#define I8_KS 64
#define I8_PAD 72
// Split-K (blockIdx.z): each block covers K in [z*k_split, (z+1)*k_split)
// -- k_split is a multiple of 256 so superblock/coef bookkeeping is
// unchanged -- and writes its partial sum to y + z*n_tokens*n_out. With
// one split, y is the real output and z == 0. With more, y is a
// [splits][n_tokens][n_out] scratch that splitk_reduce_kernel sums in
// fixed order for deterministic floating-point accumulation.
extern "C" __global__ void qmatmul_gemm_wmma_i8(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    const int m0 = blockIdx.x * 64;
    const int n0 = blockIdx.y * n_tile;   // n_tile in {16,32,64}: WMMA N-fragments sharing one staged weight tile
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;
    const int row_base = (lane / 16) * 8;
    const int rb = (n_in / QK_K) * BLOCK_Q4K_BYTES;
    const int nb = n_in / 32;
    // Live-tile branches are block-uniform and contain no barriers.
    const int nt = min(n_tile / 16, (n_tokens - n0 + 15) / 16);

    __shared__ __attribute__((aligned(16))) int8_t As8[64][I8_PAD];  // [m][k], nibbles 0..15
    __shared__ __attribute__((aligned(16))) int8_t Bs8[32][I8_PAD];  // [n][k], Q8_1 values
    __shared__ float2 ysh[32][I8_KS / 32];                            // [n][32-block] {d_x, sum_x}
    __shared__ float coef[16][64];                                    // [0..8) d*sc, [8..16) dmin*m

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
    const int k_beg = blockIdx.z * k_split;
    const int k_end = k_beg + k_split;
    y += (size_t)blockIdx.z * n_tokens * n_out;

    for (int k0 = k_beg; k0 < k_end; k0 += I8_KS) {
        if ((k0 % QK_K) == 0) {
            if (threadIdx.x < 64) {
                const int rm = threadIdx.x;
                const int row = m0 + rm;
                if (row < n_out) {
                    const uint8_t* block = w + (size_t)row * rb + (size_t)(k0 / QK_K) * BLOCK_Q4K_BYTES;
                    __half d_h, dmin_h;
                    memcpy(&d_h, block, 2);
                    memcpy(&dmin_h, block + 2, 2);
                    const float d = __half2float(d_h);
                    const float dmin = __half2float(dmin_h);
                    const uint8_t* scales = block + 4;
                    #pragma unroll
                    for (int s = 0; s < 8; s++) {
                        uint8_t sc, m;
                        get_scale_min_k4(s, scales, &sc, &m);
                        coef[s][rm] = d * (float)sc;
                        coef[8 + s][rm] = dmin * (float)m;
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; j++) coef[j][rm] = 0.0f;
                }
            }
            __syncthreads();
        }

        // A K=64 chunk at k0 is one 64-element "outer" group of a
        // Q4_K superblock: its 32 qs bytes hold the low nibbles of
        // elements 0..31 and the high nibbles of 32..63. So word wi of
        // the chunk (bytes 4wi..4wi+3) yields, little-endian, kk =
        // 4wi..4wi+3 from `v & 0x0F0F0F0F` and kk = 32+4wi..32+4wi+3
        // from `(v >> 4) & 0x0F0F0F0F` -- two aligned 4-byte LDS stores
        // per one 4-byte global load. Alignment: the
        // superblock is 144 bytes, qs starts at +16, the group at +32*g,
        // all multiples of 4. k < n_in always holds (n_in % 256 == 0).
        {
            const int outer = (k0 % QK_K) / 64;
            for (int e = threadIdx.x; e < 64 * 8; e += 128) {
                const int rm = e / 8, wi = e % 8;
                const int row = m0 + rm;
                unsigned v = 0;
                if (row < n_out) {
                    const uint8_t* qs = w + (size_t)row * rb + (size_t)(k0 / QK_K) * BLOCK_Q4K_BYTES
                                      + 4 + K_SCALE_SIZE + outer * 32;
                    memcpy(&v, qs + wi * 4, 4);
                }
                const unsigned lo = v & 0x0F0F0F0Fu;
                const unsigned hi = (v >> 4) & 0x0F0F0F0Fu;
                memcpy(&As8[rm][wi * 4], &lo, 4);
                memcpy(&As8[rm][32 + wi * 4], &hi, 4);
            }
        }
        // Activation tile: 4-byte copies of the Q8_1 int8 row segment,
        // consecutive lanes take consecutive words of one token
        // (coalesced), only the nt*16 live tokens staged, tokens past
        // n_tokens inside the last live tile zero-filled. n_in % 256 == 0
        // and k0 % 64 == 0 keep every load 4-byte aligned.
        {
            const int ncols = nt * 16;
            for (int e = threadIdx.x; e < ncols * (I8_KS / 4); e += 128) {
                const int tt = e / (I8_KS / 4), wi = e % (I8_KS / 4);
                const int t = n0 + tt;
                unsigned v = 0;
                if (t < n_tokens) {
                    memcpy(&v, q8 + (size_t)t * n_in + k0 + wi * 4, 4);
                }
                memcpy(&Bs8[tt][wi * 4], &v, 4);
            }
            for (int e = threadIdx.x; e < ncols * (I8_KS / 32); e += 128) {
                const int tt = e / (I8_KS / 32), bb = e % (I8_KS / 32);
                const int t = n0 + tt;
                const float2 v = (t < n_tokens) ? ys[(size_t)t * nb + k0 / 32 + bb] : make_float2(0.0f, 0.0f);
                ysh[tt][bb] = make_float2(v.x, v.x * v.y);
            }
        }
        __syncthreads();

        // Per 32-K block: one int32 accumulator live at a time, N tiles
        // in sequence (A re-read from LDS per tile is cheap; four live
        // v8i32 accumulators would not be).
        #pragma unroll
        for (int kb = 0; kb < I8_KS / 32; kb++) {
            const int s = ((k0 + kb * 32) % QK_K) / 32;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                if (ni < nt) {
                    v8i32_t iacc = {0, 0, 0, 0, 0, 0, 0, 0};
                    #pragma unroll
                    for (int kh = 0; kh < 2; kh++) {
                        const int ki = kb * 32 + kh * 16 + row_base;
                        v2i32_t a, b;
                        memcpy(&a, &As8[warp * 16 + col][ki], 8);
                        memcpy(&b, &Bs8[ni * 16 + col][ki], 8);
                        iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, iacc, false);
                    }
                    const float2 yt = ysh[ni * 16 + col][kb];
                    float8_t* accp = (ni == 0) ? &acc0 : (ni == 1) ? &acc1 : (ni == 2) ? &acc2 : &acc3;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        const int rl = warp * 16 + row_base + j;
                        (*accp)[j] += coef[s][rl] * yt.x * (float)iacc[j] - coef[8 + s][rl] * yt.y;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store: masked on row and token, one N tile at a time.
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int row = m0 + warp * 16 + row_base + j;
        if (row >= n_out) continue;
        int t = n0 + col;
        if (t < n_tokens) y[(size_t)t * n_out + row] = acc0[j];
        t = n0 + 16 + col;
        if (nt > 1 && t < n_tokens) y[(size_t)t * n_out + row] = acc1[j];
        t = n0 + 32 + col;
        if (nt > 2 && t < n_tokens) y[(size_t)t * n_out + row] = acc2[j];
        t = n0 + 48 + col;
        if (nt > 3 && t < n_tokens) y[(size_t)t * n_out + row] = acc3[j];
    }
}

// Two-byte-aligned 32-bit load. A Q6_K superblock is 210 bytes, so odd
// superblocks sit 2 mod 4 -- a 4-byte load there is unaligned. Every
// offset used below is even, so two 16-bit loads are always aligned.
__device__ __forceinline__ unsigned load_u32_2aligned(const uint8_t* p) {
    unsigned short lo, hi;
    memcpy(&lo, p, 2);
    memcpy(&hi, p + 2, 2);
    return (unsigned)lo | ((unsigned)hi << 16);
}

// int8-WMMA prefill matmul, Q6_K. Same tile geometry, scratch layout and
// epilogue shape as qmatmul_gemm_wmma_i8, with the two things Q6_K
// changes:
//
// 1. Its scale changes every 16 elements (16 signed int8 scales per
//    256-element superblock), so the int32 accumulator is reset per
//    K=16 WMMA and scaled immediately: y += coef[ko/16][r] * d_x[t][b]
//    * DOT16 with ko the element offset inside the superblock -- ko/16
//    is exactly the 8*idx + 2*quadrant + is index dequant_q6k_elem
//    derives the long way. No min term. Max |31*127*16| = 62992, no
//    saturation.
// 2. Values are stored as u = q+32 in 0..63, low 4 bits in ql and the
//    top 2 in qh. Per K=64 chunk c of a superblock: idx = c/2 selects
//    the 128-element half (ql += 64*idx, qh += 32*idx), c%2 selects
//    quadrants {0,1} (ql low nibbles, qh bits 0-3) or {2,3} (ql high
//    nibbles, qh bits 4-7) -- the four (ql-nibble, qh-bit-pair) tuples
//    copied from dequant_q6k_elem. Word wi covers l = 4wi..4wi+3 for
//    both the chunk's first 32 (kk = l) and second 32 (kk = 32+l)
//    elements. The 6-bit u is re-centred per byte with no per-byte
//    loop: (u & 0x1F1F1F1F) | ((~u & 0x20202020) * 7) is u-32 in two's
//    complement for each byte (0x20*7 = 0xE0, no cross-byte carry).
extern "C" __global__ void qmatmul_gemm_wmma_i8_q6k(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    (void)n_tile;  // shared launcher signature; this kernel is 16-token-tile only (see the Q4_K kernel for the tunable)
    const int m0 = blockIdx.x * 64;
    const int n0 = blockIdx.y * 16;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;
    const int row_base = (lane / 16) * 8;
    const int rb = (n_in / QK_K) * BLOCK_Q6K_BYTES;
    const int nb = n_in / 32;

    __shared__ __attribute__((aligned(16))) int8_t As8[64][I8_PAD];
    __shared__ __attribute__((aligned(16))) int8_t Bs8[16][I8_PAD];
    __shared__ float2 ysh[16][I8_KS / 32];
    __shared__ float coef[16][64];  // coef[j][r] = d[r] * sc[r][j]

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const int k_beg = blockIdx.z * k_split;  // split-K, see qmatmul_gemm_wmma_i8
    const int k_end = k_beg + k_split;
    y += (size_t)blockIdx.z * n_tokens * n_out;

    for (int k0 = k_beg; k0 < k_end; k0 += I8_KS) {
        if ((k0 % QK_K) == 0) {
            if (threadIdx.x < 64) {
                const int rm = threadIdx.x;
                const int row = m0 + rm;
                if (row < n_out) {
                    const uint8_t* block = w + (size_t)row * rb + (size_t)(k0 / QK_K) * BLOCK_Q6K_BYTES;
                    const int8_t* sc = (const int8_t*)(block + 128 + 64);
                    __half d_h;
                    memcpy(&d_h, block + 128 + 64 + 16, 2);
                    const float d = __half2float(d_h);
                    #pragma unroll
                    for (int j = 0; j < 16; j++) coef[j][rm] = d * (float)sc[j];
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; j++) coef[j][rm] = 0.0f;
                }
            }
            __syncthreads();
        }

        {
            const int c = (k0 % QK_K) / 64;
            const int idx = c / 2;
            const int shift = (c % 2) * 4;  // 0: quadrants {0,1}; 4: quadrants {2,3}
            for (int e = threadIdx.x; e < 64 * 8; e += 128) {
                const int rm = e / 8, wi = e % 8;
                const int row = m0 + rm;
                unsigned lo = 0, hi = 0;
                if (row < n_out) {
                    const uint8_t* block = w + (size_t)row * rb + (size_t)(k0 / QK_K) * BLOCK_Q6K_BYTES;
                    const uint8_t* ql = block + 64 * idx;
                    const uint8_t* qh = block + 128 + 32 * idx;
                    const unsigned a = load_u32_2aligned(ql + wi * 4);        // ql[l]
                    const unsigned b = load_u32_2aligned(ql + 32 + wi * 4);   // ql[l+32]
                    const unsigned h = load_u32_2aligned(qh + wi * 4);        // qh[l]
                    const unsigned ulo = ((a >> shift) & 0x0F0F0F0Fu) | (((h >> shift) & 0x03030303u) << 4);
                    const unsigned uhi = ((b >> shift) & 0x0F0F0F0Fu) | (((h >> (shift + 2)) & 0x03030303u) << 4);
                    lo = (ulo & 0x1F1F1F1Fu) | ((~ulo & 0x20202020u) * 7u);
                    hi = (uhi & 0x1F1F1F1Fu) | ((~uhi & 0x20202020u) * 7u);
                }
                memcpy(&As8[rm][wi * 4], &lo, 4);
                memcpy(&As8[rm][32 + wi * 4], &hi, 4);
            }
        }
        for (int e = threadIdx.x; e < 16 * (I8_KS / 4); e += 128) {
            const int tt = e / (I8_KS / 4), wi = e % (I8_KS / 4);
            const int t = n0 + tt;
            unsigned v = 0;
            if (t < n_tokens) {
                memcpy(&v, q8 + (size_t)t * n_in + k0 + wi * 4, 4);
            }
            memcpy(&Bs8[tt][wi * 4], &v, 4);
        }
        if (threadIdx.x < 16 * (I8_KS / 32)) {
            const int tt = threadIdx.x / (I8_KS / 32), bb = threadIdx.x % (I8_KS / 32);
            const int t = n0 + tt;
            const float2 v = (t < n_tokens) ? ys[(size_t)t * nb + k0 / 32 + bb] : make_float2(0.0f, 0.0f);
            ysh[tt][bb] = make_float2(v.x, v.x * v.y);
        }
        __syncthreads();

        #pragma unroll
        for (int kb = 0; kb < I8_KS / 32; kb++) {
            const float dx = ysh[col][kb].x;
            #pragma unroll
            for (int kh = 0; kh < 2; kh++) {
                const int ki = kb * 32 + kh * 16 + row_base;
                v2i32_t a, b;
                memcpy(&a, &As8[warp * 16 + col][ki], 8);
                memcpy(&b, &Bs8[col][ki], 8);
                v8i32_t iacc = {0, 0, 0, 0, 0, 0, 0, 0};
                iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, iacc, false);
                const int cj = ((k0 + kb * 32 + kh * 16) % QK_K) / 16;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int rl = warp * 16 + row_base + j;
                    acc[j] += coef[cj][rl] * dx * (float)iacc[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int row = m0 + warp * 16 + row_base + j;
        const int t = n0 + col;
        if (row < n_out && t < n_tokens) {
            y[(size_t)t * n_out + row] = acc[j];
        }
    }
}

// Sums split-K partials [splits][n] into y[n] in fixed split order.
extern "C" __global__ void splitk_reduce_kernel(const float* partial, int splits, int n, float* y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int z = 0; z < splits; z++) {
        s += partial[(size_t)z * n + i];
    }
    y[i] = s;
}

// K=256 staging: weights and coefficients once per superblock, activations
// in two 128-K halves. A and B share a K permutation: lane half h feeds
// physical indices 16h+j and 16h+8+j to the two K=16 WMMAs.
// This preserves dot products while allowing 16-byte fragment loads.
// Row strides of 272 B (weights) and 144 B (activations) distribute
// sixteen 16-byte lane loads across all 64 LDS banks.
#define I9_AS_STRIDE 272   // 256 K + 16 pad bytes = 68 ints, 68 % 8 == 4
#define I9_BS_STRIDE 144   // 128 K + 16 pad bytes = 36 ints, 36 % 8 == 4
typedef int v4i32_t __attribute__((ext_vector_type(4)));

// Load before storing to overlap memory latency. Clamped indices keep
// loads valid; padded outputs are masked or padded activations zeroed.
// Each call stages one 16-token x 128-K tile, four words per thread.
__device__ __forceinline__ void i9_stage_y_tile(
    int8_t (*Bs8)[I9_BS_STRIDE], const int8_t* q8, int n_in, int n_tokens, int t0, int tt0, int kbase
) {
    // Four b32 words per thread, consecutive lanes covering one token's
    // contiguous 128-byte row: the LDS stores are conflict-free at the
    // 144-byte stride.
    unsigned v[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int e = threadIdx.x + 128 * i;
        const int tt = e / 32, wi = e % 32;
        const int t = min(t0 + tt, n_tokens - 1);
        memcpy(&v[i], q8 + (size_t)t * n_in + kbase + wi * 4, 4);
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int e = threadIdx.x + 128 * i;
        const int tt = e / 32, wi = e % 32;
        const unsigned val = (t0 + tt < n_tokens) ? v[i] : 0u;
        memcpy(&Bs8[tt0 + tt][wi * 4], &val, 4);
    }
}

// One 16-token tile's 8 per-superblock Q8_1 scale pairs: 128 entries,
// one per thread.
__device__ __forceinline__ void i9_stage_ys_tile(
    float2 (*ysh)[8], const float2* ys, int nb, int n_tokens, int t0, int tt0, int sb
) {
    const int tt = threadIdx.x / 8, bb = threadIdx.x % 8;
    const int t = min(t0 + tt, n_tokens - 1);
    const float2 v = ys[(size_t)t * nb + sb * 8 + bb];
    ysh[tt0 + tt][bb] = (t0 + tt < n_tokens) ? make_float2(v.x, v.x * v.y) : make_float2(0.0f, 0.0f);
}

// Load all fragments and scales before issuing independent WMMA accumulators.
// The epilogue consumes them only after all instructions have been issued.
// NT is selected by a uniform switch, excluding inactive token tiles.
template <int NT>
__device__ __forceinline__ void i9_kb_step(
    const int8_t (*As8)[I9_AS_STRIDE], const int8_t (*Bs8)[I9_BS_STRIDE], const float2 (*ysh)[8], const float (*coef)[64],
    int warp, int col, int row_base, int khalf, int hh, int kb4, int kb,
    float8_t& acc0, float8_t& acc1, float8_t& acc2, float8_t& acc3
) {
    v4i32_t a4;
    memcpy(&a4, &As8[warp * 16 + col][hh * 128 + kb4 * 32 + khalf], 16);
    v4i32_t b4[NT];
    float2 yt[NT];
    #pragma unroll
    for (int ni = 0; ni < NT; ni++) {
        memcpy(&b4[ni], &Bs8[ni * 16 + col][kb4 * 32 + khalf], 16);
        yt[ni] = ysh[ni * 16 + col][kb];
    }
    float cs[8], cm[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int rl = warp * 16 + row_base + j;
        cs[j] = coef[kb][rl];
        cm[j] = coef[8 + kb][rl];
    }
    const v2i32_t a_lo = {a4[0], a4[1]};
    const v2i32_t a_hi = {a4[2], a4[3]};
    v8i32_t iacc[NT];
    #pragma unroll
    for (int ni = 0; ni < NT; ni++) {
        const v2i32_t b_lo = {b4[ni][0], b4[ni][1]};
        iacc[ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_lo, true, b_lo, (v8i32_t){0, 0, 0, 0, 0, 0, 0, 0}, false);
    }
    #pragma unroll
    for (int ni = 0; ni < NT; ni++) {
        const v2i32_t b_hi = {b4[ni][2], b4[ni][3]};
        iacc[ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_hi, true, b_hi, iacc[ni], false);
    }
    #pragma unroll
    for (int ni = 0; ni < NT; ni++) {
        float8_t* accp = (ni == 0) ? &acc0 : (ni == 1) ? &acc1 : (ni == 2) ? &acc2 : &acc3;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            (*accp)[j] += cs[j] * yt[ni].x * (float)iacc[ni][j] - cm[j] * yt[ni].y;
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 2) qmatmul_gemm_wmma_i8_k256(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    const int m0 = blockIdx.x * 64;
    const int n0 = blockIdx.y * n_tile;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;
    const int row_base = (lane / 16) * 8;
    const int khalf = (lane / 16) * 16;   // physical K offset of this lane's 16-byte fragment within a 32-K block
    const int rb = (n_in / QK_K) * BLOCK_Q4K_BYTES;
    const int nb = n_in / 32;
    const int nt = min(n_tile / 16, (n_tokens - n0 + 15) / 16);   // block-uniform

    // Dynamic LDS, sized by the host to the live token tile: 64x272
    // nibbles + n_tile x 144 activations + n_tile x 8 scale pairs + coef.
    extern __shared__ __attribute__((aligned(16))) int8_t i9_smem[];
    int8_t (*As8)[I9_AS_STRIDE] = (int8_t (*)[I9_AS_STRIDE])i9_smem;                       // [m][k], one superblock of nibbles
    int8_t (*Bs8)[I9_BS_STRIDE] = (int8_t (*)[I9_BS_STRIDE])(i9_smem + 64 * I9_AS_STRIDE); // [n][k], one 128-K half of Q8_1 values
    float2 (*ysh)[8] = (float2 (*)[8])(i9_smem + 64 * I9_AS_STRIDE + n_tile * I9_BS_STRIDE); // [n][32-block] {d_x, s_x}
    float (*coef)[64] = (float (*)[64])(i9_smem + 64 * I9_AS_STRIDE + n_tile * I9_BS_STRIDE + n_tile * 8 * 8); // [0..8) d*sc, [8..16) dmin*m

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
    const int k_beg = blockIdx.z * k_split;
    const int k_end = k_beg + k_split;
    y += (size_t)blockIdx.z * n_tokens * n_out;
    const int ncols = nt * 16;

    for (int k0 = k_beg; k0 < k_end; k0 += QK_K) {
        const int sb = k0 / QK_K;
        // --- superblock staging: coefficients (64 producer threads),
        //     64 rows x 256 K of nibbles as 32 words per row (16 per
        //     thread), and this tile's 8 per-token Q8_1 scale pairs. ---
        if (threadIdx.x < 64) {
            // One 16-byte load of the superblock header (d, dmin, 12 scale
            // bytes), decoded from registers. Rows past n_out are clamped; their coef
            // rows are never consumed by a stored output.
            const int rm = threadIdx.x;
            const int row = min(m0 + rm, n_out - 1);
            const uint8_t* block = w + (size_t)row * rb + (size_t)sb * BLOCK_Q4K_BYTES;
            v4i32_t hdr;
            memcpy(&hdr, block, 16);   // superblocks are 144 = 9*16 bytes: always 16-aligned
            uint8_t hb[16];
            memcpy(hb, &hdr, 16);
            __half d_h, dmin_h;
            memcpy(&d_h, hb, 2);
            memcpy(&dmin_h, hb + 2, 2);
            const float d = __half2float(d_h);
            const float dmin = __half2float(dmin_h);
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                uint8_t sc, m;
                get_scale_min_k4(s, hb + 4, &sc, &m);
                coef[s][rm] = d * (float)sc;
                coef[8 + s][rm] = dmin * (float)m;
            }
        }
        {
            // 64 rows x 32 words of qs, 16 words per thread: all 16 loads
            // in flight before the first split/store. Rows past n_out are
            // clamped (valid memory), their outputs are masked at the
            // store; no branch anywhere in this loop.
            // Each thread takes 16 contiguous qs bytes (one 16-byte chunk c
            // of one row): 4 global_load_b128 per superblock instead of 16
            // b32 -- a b128 cannot be split by the scheduler, so batching
            // no longer depends on it. Chunk c holds words 4c..4c+3; word
            // wi = 4c+q maps to kk = (wi/8)*64 + (wi%8)*4 = (c/2)*64 +
            // (c%2)*16 + 4q, so the four low-nibble words are contiguous
            // in LDS (one 16-byte store) and so are the four high-nibble
            // words at +32. Alignment: superblock 144 = 9*16, qs at +16,
            // chunks at 16c -- every load and store is 16-byte aligned.
            v4i32_t v[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int e = threadIdx.x + 128 * i;
                const int rm = e / 8, c = e % 8;
                const int row = min(m0 + rm, n_out - 1);
                const uint8_t* qs = w + (size_t)row * rb + (size_t)sb * BLOCK_Q4K_BYTES + 4 + K_SCALE_SIZE;
                memcpy(&v[i], qs + c * 16, 16);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int e = threadIdx.x + 128 * i;
                const int rm = e / 8, c = e % 8;
                const int kk = (c / 2) * 64 + (c % 2) * 16;
                v4i32_t lo, hi;
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    lo[q] = (int)((unsigned)v[i][q] & 0x0F0F0F0Fu);
                    hi[q] = (int)(((unsigned)v[i][q] >> 4) & 0x0F0F0F0Fu);
                }
                memcpy(&As8[rm][kk], &lo, 16);
                memcpy(&As8[rm][kk + 32], &hi, 16);
            }
        }
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            if (ni < nt) i9_stage_ys_tile(ysh, ys, nb, n_tokens, n0 + ni * 16, ni * 16, sb);
        }

        // Keep this loop rolled to bound register pressure.
        #pragma unroll 1
        for (int hh = 0; hh < 2; hh++) {
            // --- activation half hh: 128 K x (nt*16) tokens, one tile at a time ---
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                if (ni < nt) i9_stage_y_tile(Bs8, q8, n_in, n_tokens, n0 + ni * 16, ni * 16, k0 + hh * 128);
            }
            __syncthreads();

            #pragma unroll 1
            for (int kb4 = 0; kb4 < 4; kb4++) {
                const int kb = hh * 4 + kb4;   // 32-K block index within the superblock, 0..7
                // Uniform dispatch on the live tile count (readfirstlane
                // makes it an SGPR: the per-tile `if (ni < nt)` compiled to
                // saveexec/execz masking, treated as divergent).
                switch (__builtin_amdgcn_readfirstlane(nt)) {
                    case 4: i9_kb_step<4>(As8, Bs8, ysh, coef, warp, col, row_base, khalf, hh, kb4, kb, acc0, acc1, acc2, acc3); break;
                    case 3: i9_kb_step<3>(As8, Bs8, ysh, coef, warp, col, row_base, khalf, hh, kb4, kb, acc0, acc1, acc2, acc3); break;
                    case 2: i9_kb_step<2>(As8, Bs8, ysh, coef, warp, col, row_base, khalf, hh, kb4, kb, acc0, acc1, acc2, acc3); break;
                    default: i9_kb_step<1>(As8, Bs8, ysh, coef, warp, col, row_base, khalf, hh, kb4, kb, acc0, acc1, acc2, acc3); break;
                }
            }
            __syncthreads();   // before the next half (or superblock) overwrites Bs8 / As8 / coef
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int row = m0 + warp * 16 + row_base + j;
        if (row >= n_out) continue;
        int t = n0 + col;
        if (t < n_tokens) y[(size_t)t * n_out + row] = acc0[j];
        t = n0 + 16 + col;
        if (nt > 1 && t < n_tokens) y[(size_t)t * n_out + row] = acc1[j];
        t = n0 + 32 + col;
        if (nt > 2 && t < n_tokens) y[(size_t)t * n_out + row] = acc2[j];
        t = n0 + 48 + col;
        if (nt > 3 && t < n_tokens) y[(size_t)t * n_out + row] = acc3[j];
    }
}

// Q6_K on the K=256 skeleton above. Two things Q6_K forces differently:
//
// 1. Its scale changes every 16 K, so the K-permuted 16-byte fragment
//    trick is *wrong* here: one WMMA would mix physical K 0..7 and
//    16..23, two scale groups. Fragments are natural-order 8-byte loads
//    (lane half h = physical K 16*kb + 8h .. +7), one WMMA per int32
//    reset, scaled by coef[ko/16][r] * d_x. At the 272-byte stride the
//    32-lane b64 gather is still conflict-free (int offsets 68*col + 2h
//    cover 64 banks exactly once).
// 2. Superblocks are 210 bytes, so odd ones are only 2-byte aligned:
//    staging reads 16-bit pairs, batched per group of four units with clamped rows and
//    no branches. A unit is (row, half idx, 4-element group g): three
//    words -- ql[4g], ql[32+4g], qh[4g] -- yield four stored words at
//    idx*128 + 4g + {0, 32, 64, 96}, the four (ql-nibble, qh-bit-pair)
//    tuples of dequant_q6k_elem, each recentred by
//    (u & 0x1F1F1F1F) | ((~u & 0x20202020) * 7) == u-32 per byte.
__device__ __forceinline__ void i9_q6k_unit(
    const uint8_t* block, int idx, int g, unsigned* a, unsigned* b, unsigned* h
) {
    const uint8_t* ql = block + 64 * idx;
    const uint8_t* qh = block + 128 + 32 * idx;
    *a = load_u32_2aligned(ql + 4 * g);
    *b = load_u32_2aligned(ql + 32 + 4 * g);
    *h = load_u32_2aligned(qh + 4 * g);
}
__device__ __forceinline__ unsigned i9_q6k_recentre(unsigned u) {
    return (u & 0x1F1F1F1Fu) | ((~u & 0x20202020u) * 7u);
}

extern "C" __global__ void __launch_bounds__(128, 2) qmatmul_gemm_wmma_i8_q6k_k256(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    const int m0 = blockIdx.x * 64;
    const int n0 = blockIdx.y * n_tile;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;
    const int row_base = (lane / 16) * 8;
    const int k8 = (lane / 16) * 8;   // natural-order 8-byte fragment offset within a 16-K block
    const int rb = (n_in / QK_K) * BLOCK_Q6K_BYTES;
    const int nb = n_in / 32;
    const int nt = min(n_tile / 16, (n_tokens - n0 + 15) / 16);

    extern __shared__ __attribute__((aligned(16))) int8_t i9_smem[];
    int8_t (*As8)[I9_AS_STRIDE] = (int8_t (*)[I9_AS_STRIDE])i9_smem;
    int8_t (*Bs8)[I9_BS_STRIDE] = (int8_t (*)[I9_BS_STRIDE])(i9_smem + 64 * I9_AS_STRIDE);
    float2 (*ysh)[8] = (float2 (*)[8])(i9_smem + 64 * I9_AS_STRIDE + n_tile * I9_BS_STRIDE);
    float (*coef)[64] = (float (*)[64])(i9_smem + 64 * I9_AS_STRIDE + n_tile * I9_BS_STRIDE + n_tile * 8 * 8);   // coef[j][r] = d[r]*sc[r][j], j in 0..16

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
    const int k_beg = blockIdx.z * k_split;
    const int k_end = k_beg + k_split;
    y += (size_t)blockIdx.z * n_tokens * n_out;

    for (int k0 = k_beg; k0 < k_end; k0 += QK_K) {
        const int sb = k0 / QK_K;
        if (threadIdx.x < 64) {
            // 16 int8 scales at +192 and d at +208, as four 2-aligned words + one half.
            const int rm = threadIdx.x;
            const int row = min(m0 + rm, n_out - 1);
            const uint8_t* block = w + (size_t)row * rb + (size_t)sb * BLOCK_Q6K_BYTES;
            unsigned sw[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) sw[i] = load_u32_2aligned(block + 128 + 64 + 4 * i);
            unsigned short d_bits;
            memcpy(&d_bits, block + 128 + 64 + 16, 2);
            __half d_h;
            memcpy(&d_h, &d_bits, 2);
            const float d = __half2float(d_h);
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                const int8_t sc = (int8_t)((sw[j / 4] >> (8 * (j % 4))) & 0xFF);
                coef[j][rm] = d * (float)sc;
            }
        }
        {
            // 64 rows x 2 halves x 8 groups = 1024 units, 8 per thread, in
            // two batches of four: 12 word-pairs in flight per batch.
            #pragma unroll
            for (int grp = 0; grp < 2; grp++) {
                unsigned a[4], b[4], h[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int e = threadIdx.x + 128 * (grp * 4 + i);
                    const int rm = e / 16, idx = (e / 8) % 2, g = e % 8;
                    const int row = min(m0 + rm, n_out - 1);
                    const uint8_t* block = w + (size_t)row * rb + (size_t)sb * BLOCK_Q6K_BYTES;
                    i9_q6k_unit(block, idx, g, &a[i], &b[i], &h[i]);
                }
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int e = threadIdx.x + 128 * (grp * 4 + i);
                    const int rm = e / 16, idx = (e / 8) % 2, g = e % 8;
                    const int base = idx * 128 + 4 * g;
                    const unsigned q0 = i9_q6k_recentre((a[i] & 0x0F0F0F0Fu) | ((h[i] & 0x03030303u) << 4));
                    const unsigned q1 = i9_q6k_recentre((b[i] & 0x0F0F0F0Fu) | (((h[i] >> 2) & 0x03030303u) << 4));
                    const unsigned q2 = i9_q6k_recentre(((a[i] >> 4) & 0x0F0F0F0Fu) | (((h[i] >> 4) & 0x03030303u) << 4));
                    const unsigned q3 = i9_q6k_recentre(((b[i] >> 4) & 0x0F0F0F0Fu) | (((h[i] >> 6) & 0x03030303u) << 4));
                    memcpy(&As8[rm][base], &q0, 4);
                    memcpy(&As8[rm][base + 32], &q1, 4);
                    memcpy(&As8[rm][base + 64], &q2, 4);
                    memcpy(&As8[rm][base + 96], &q3, 4);
                }
            }
        }
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            if (ni < nt) i9_stage_ys_tile(ysh, ys, nb, n_tokens, n0 + ni * 16, ni * 16, sb);
        }

        #pragma unroll 1
        for (int hh = 0; hh < 2; hh++) {
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                if (ni < nt) i9_stage_y_tile(Bs8, q8, n_in, n_tokens, n0 + ni * 16, ni * 16, k0 + hh * 128);
            }
            __syncthreads();

            #pragma unroll 1
            for (int kb16 = 0; kb16 < 8; kb16++) {   // 16-K blocks within this half
                const int ko = hh * 128 + kb16 * 16;   // element offset within the superblock
                v2i32_t a;
                memcpy(&a, &As8[warp * 16 + col][ko + k8], 8);
                float cs[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) cs[j] = coef[ko / 16][warp * 16 + row_base + j];
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    if (ni < nt) {
                        v2i32_t b;
                        memcpy(&b, &Bs8[ni * 16 + col][kb16 * 16 + k8], 8);
                        v8i32_t iacc = {0, 0, 0, 0, 0, 0, 0, 0};
                        iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, iacc, false);
                        const float dx = ysh[ni * 16 + col][ko / 32].x;
                        float8_t* accp = (ni == 0) ? &acc0 : (ni == 1) ? &acc1 : (ni == 2) ? &acc2 : &acc3;
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            (*accp)[j] += cs[j] * dx * (float)iacc[j];
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int row = m0 + warp * 16 + row_base + j;
        if (row >= n_out) continue;
        int t = n0 + col;
        if (t < n_tokens) y[(size_t)t * n_out + row] = acc0[j];
        t = n0 + 16 + col;
        if (nt > 1 && t < n_tokens) y[(size_t)t * n_out + row] = acc1[j];
        t = n0 + 32 + col;
        if (nt > 2 && t < n_tokens) y[(size_t)t * n_out + row] = acc2[j];
        t = n0 + 48 + col;
        if (nt > 3 && t < n_tokens) y[(size_t)t * n_out + row] = acc3[j];
    }
}

// One block per (KV head, query tile), one warp per query head in its GQA group.
// Heads share staged K/V tiles. Each lane owns up to four query/output
// dimensions; online softmax tracks a max and denominator per (query, head).
// Causal visibility and key bounds are warp-uniform.
#define ATT_QT 8
#define ATT_KT 16
extern "C" __global__ void attn_prefill_tiled_kernel(
    const float* q, const _Float16* k, const _Float16* v, float* o,
    int head_dim, int n_q_heads, int n_kv_heads, int n_q, int prefix, int qt
) {
    // More queries per block share K/V loads but reduce the grid size.
    const int group = n_q_heads / n_kv_heads;
    const int kvh = blockIdx.x;
    const int q0 = blockIdx.y * qt;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int qh = kvh * group + warp;
    const int dpl = (head_dim + 31) / 32;   // dims per lane, <= 4
    const int kv_row = n_kv_heads * head_dim;
    const float scale = rsqrtf((float)head_dim);

    extern __shared__ float att_smem[];   // Ks[ATT_KT][head_dim] then Vs[ATT_KT][head_dim]
    float* Ks = att_smem;
    float* Vs = att_smem + ATT_KT * head_dim;

    float qv[ATT_QT][4], ov[ATT_QT][4], mrun[ATT_QT], lrun[ATT_QT];
    #pragma unroll
    for (int qi = 0; qi < ATT_QT; qi++) {
        const int pos = min(q0 + qi, n_q - 1);   // clamped: slots past n_q compute garbage that is never stored
        const float* qp = q + ((size_t)pos * n_q_heads + qh) * head_dim;
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            const int dd = lane * dpl + d;
            qv[qi][d] = (d < dpl && dd < head_dim) ? qp[dd] : 0.0f;
            ov[qi][d] = 0.0f;
        }
        mrun[qi] = -INFINITY;
        lrun[qi] = 0.0f;
    }

    // Keys any query of this tile can see: t < prefix + last_pos + 1.
    const int n_keys = prefix + min(q0 + qt, n_q);

    const int n4 = ATT_KT * head_dim / 4;   // head_dim % 4 == 0 (4 on the small tests, 128 in production)
    for (int t0 = 0; t0 < n_keys; t0 += ATT_KT) {
        // Issue four float4 loads per thread before any store.
        // Keys past n_keys read the clamped last key and are zeroed by a
        // select; never a branch around the load.
        for (int base = threadIdx.x; base < n4; base += blockDim.x * 4) {
            float4 kr[4], vr[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int e4 = base + i * blockDim.x;
                const int tt = (e4 * 4) / head_dim, d = (e4 * 4) % head_dim;
                const int t = min(t0 + tt, n_keys - 1);
                const size_t off = (size_t)t * kv_row + (size_t)kvh * head_dim + d;
                const bool live = e4 < n4;
                half4_t kh, vh;
                memcpy(&kh, k + (live ? off : 0), 8);
                memcpy(&vh, v + (live ? off : 0), 8);
                kr[i] = make_float4((float)kh[0], (float)kh[1], (float)kh[2], (float)kh[3]);
                vr[i] = make_float4((float)vh[0], (float)vh[1], (float)vh[2], (float)vh[3]);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int e4 = base + i * blockDim.x;
                if (e4 < n4) {
                    const int tt = (e4 * 4) / head_dim;
                    const bool valid = t0 + tt < n_keys;
                    const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    ((float4*)Ks)[e4] = valid ? kr[i] : z;
                    ((float4*)Vs)[e4] = valid ? vr[i] : z;
                }
            }
        }
        __syncthreads();

        const int tiles_keys = min(ATT_KT, n_keys - t0);
        for (int tt = 0; tt < tiles_keys; tt++) {
            const int t = t0 + tt;
            const float* kt = Ks + tt * head_dim + lane * dpl;
            const float* vt = Vs + tt * head_dim + lane * dpl;
            #pragma unroll
            for (int qi = 0; qi < ATT_QT; qi++) {
                if (qi >= qt) break;   // block-uniform
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < 4; d++) {
                    if (d < dpl && lane * dpl + d < head_dim) s += qv[qi][d] * kt[d];
                }
                #pragma unroll
                for (int mask = 16; mask > 0; mask >>= 1) s += __shfl_xor(s, mask);
                s *= scale;
                // Warp-uniform: depends on the query slot and key only.
                if (q0 + qi < n_q && t <= prefix + q0 + qi) {
                    const float m_new = fmaxf(mrun[qi], s);
                    const float corr = __expf(mrun[qi] - m_new);   // 0 on the first key (mrun = -inf)
                    const float p = __expf(s - m_new);
                    lrun[qi] = lrun[qi] * corr + p;
                    #pragma unroll
                    for (int d = 0; d < 4; d++) {
                        if (d < dpl) ov[qi][d] = ov[qi][d] * corr + p * vt[d];
                    }
                    mrun[qi] = m_new;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int qi = 0; qi < ATT_QT; qi++) {
        const int pos = q0 + qi;
        if (qi >= qt || pos >= n_q) continue;
        float* op = o + ((size_t)pos * n_q_heads + qh) * head_dim;
        const float inv = 1.0f / lrun[qi];
        #pragma unroll
        for (int d = 0; d < 4; d++) {
            const int dd = lane * dpl + d;
            if (d < dpl && dd < head_dim) op[dd] = ov[qi][d] * inv;
        }
    }
}

// Flash attention using f16 WMMA with f32 accumulation:
//
//   S^T[key][query] = K[key][:] . Q^T[:][query]     (A = K, B = Q^T)
//
// leaves each lane holding 8 consecutive keys of ONE query (C fragment:
// rows kb..kb+7 of column lane%16), so the online softmax is per lane
// with a single xor-16 shuffle for the other 8 keys, and the resulting
// P^T[key][query] is already in B-fragment layout (8 consecutive k of
// column lane%16) for
//
//   O^T[d][query] += V^T[d][key] . P^T[key][query]  (A = V^T, B = P^T)
//
// -- no cross-lane repack anywhere. V^T's A fragment (8 consecutive keys
// at one d) is gathered from the natural Vs[key][d] with eight 16-bit LDS
// loads; the row stride (132 halves = 66 dwords) puts rows 8 apart on
// banks 16 apart so the two half-waves don't collide. Ks keeps the
// GEMM's 4-mod-8-dword stride for conflict-free 16-byte fragment loads.
// Q is pre-scaled by 1/sqrt(hd) * log2(e) so the softmax runs in exp2,
// which is the native v_exp_f32. Block per (KV head, 16 queries), one
// warp per q-head of the GQA group sharing the staged K/V tile.
// q, k, v and p are rounded to f16; accumulation and softmax statistics
// stay f32. Tiles a query cannot see are skipped
// by the loop bound; masked keys inside a tile get -inf before the max.
#define ATT_W_KS 136   // Ks row stride in halves (68 dwords = 4 mod 8): conflict-free b128 A loads
#define ATT_W_VS 132   // Vs row stride in halves (66 dwords): rows 8 apart -> banks 16 apart for the u16 gathers
template <int HD>
__device__ __forceinline__ void attn_prefill_wmma_body(
    const float* q, const _Float16* k, const _Float16* v, float* o, int n_q_heads, int n_kv_heads, int n_q, int prefix
) {
    constexpr int NDT = HD / 16;
    const int group = n_q_heads / n_kv_heads;
    const int kvh = blockIdx.x;
    const int q0 = blockIdx.y * 16;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;         // this lane's query slot: the S^T / O^T column
    const int kb = (lane / 16) * 8;    // this lane's first key (S^T row) / first d (O^T row) in a 16-tile
    const int qh = kvh * group + warp;
    const int kv_row = n_kv_heads * HD;
    const float qscale = rsqrtf((float)HD) * 1.4426950408889634f;

    __shared__ __attribute__((aligned(16))) _Float16 Ks[ATT_KT * ATT_W_KS];
    __shared__ __attribute__((aligned(16))) _Float16 Vs[ATT_KT * ATT_W_VS];

    // Q^T as B fragments: Q[query = col][d = dt*16 + kb + j], pre-scaled.
    // Slots past n_q read the clamped last query, compute, never store.
    const int qpos = q0 + col;
    half8_t qf[NDT];
    {
        const float* qp = q + ((size_t)min(qpos, n_q - 1) * n_q_heads + qh) * HD + kb;
        #pragma unroll
        for (int dt = 0; dt < NDT; dt++) {
            float4 a, b;
            memcpy(&a, qp + dt * 16, 16);
            memcpy(&b, qp + dt * 16 + 4, 16);
            qf[dt] = (half8_t){
                (_Float16)(a.x * qscale), (_Float16)(a.y * qscale), (_Float16)(a.z * qscale), (_Float16)(a.w * qscale),
                (_Float16)(b.x * qscale), (_Float16)(b.y * qscale), (_Float16)(b.z * qscale), (_Float16)(b.w * qscale),
            };
        }
    }
    float8_t oacc[NDT];
    #pragma unroll
    for (int dt = 0; dt < NDT; dt++) oacc[dt] = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};
    float m_run = -INFINITY, l_run = 0.0f;

    const int n_keys = prefix + min(q0 + 16, n_q);
    constexpr int CH = HD / 8;               // 8-dim chunks per key row
    for (int t0 = 0; t0 < n_keys; t0 += ATT_KT) {
        for (int e8 = threadIdx.x; e8 < ATT_KT * CH; e8 += blockDim.x) {
            const int tt = e8 / CH, d = (e8 % CH) * 8;
            const int t = min(t0 + tt, n_keys - 1);
            const size_t off = (size_t)t * kv_row + (size_t)kvh * HD + d;
            // f16 rows straight from the cache; keys past n_keys hold the
            // clamped last key (finite) and are masked before the max.
            half8_t kh, vh;
            memcpy(&kh, k + off, 16);
            memcpy(&vh, v + off, 16);
            memcpy(&Ks[tt * ATT_W_KS + d], &kh, 16);
            memcpy(&Vs[tt * ATT_W_VS + d], &vh, 8);        // 264-byte rows: 8-byte aligned only
            memcpy(&Vs[tt * ATT_W_VS + d + 4], ((const _Float16*)&vh) + 4, 8);
        }
        __syncthreads();

        // S^T = K . Q^T over the head dim.
        float8_t s = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int dt = 0; dt < NDT; dt++) {
            half8_t kf;
            memcpy(&kf, &Ks[col * ATT_W_KS + dt * 16 + kb], 16);
            s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf, qf[dt], s);
        }

        // Online softmax for query `col` over this tile's 16 keys: 8 here,
        // 8 in lane^16. Keys past n_keys or after the query are -inf.
        float mloc = -INFINITY;
        float sv[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int t = t0 + kb + j;
            sv[j] = (t < n_keys && t <= prefix + qpos) ? s[j] : -INFINITY;
            mloc = fmaxf(mloc, sv[j]);
        }
        mloc = fmaxf(mloc, __shfl_xor(mloc, 16));
        const float m_new = fmaxf(m_run, mloc);          // finite from tile 0 on: key 0 is visible to every query
        const float corr = exp2f(m_run - m_new);         // 0 on the first tile (m_run = -inf)
        float psum = 0.0f;
        half8_t pf;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const float p = exp2f(sv[j] - m_new);
            psum += p;
            pf[j] = (_Float16)p;
        }
        psum += __shfl_xor(psum, 16);
        l_run = l_run * corr + psum;
        m_run = m_new;

        // O^T += V^T . P^T: A = V^T[d = dt*16 + col][key = kb + j] gathered
        // from Vs[key][d]; B = P^T straight from the S^T fragment.
        #pragma unroll
        for (int dt = 0; dt < NDT; dt++) {
            #pragma unroll
            for (int j = 0; j < 8; j++) oacc[dt][j] *= corr;
            half8_t vf;
            #pragma unroll
            for (int j = 0; j < 8; j++) vf[j] = Vs[(kb + j) * ATT_W_VS + dt * 16 + col];
            oacc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(vf, pf, oacc[dt]);
        }
        __syncthreads();
    }

    if (qpos < n_q) {
        const float inv = 1.0f / l_run;
        float* op = o + ((size_t)qpos * n_q_heads + qh) * HD + kb;
        #pragma unroll
        for (int dt = 0; dt < NDT; dt++) {
            const float4 a = make_float4(oacc[dt][0] * inv, oacc[dt][1] * inv, oacc[dt][2] * inv, oacc[dt][3] * inv);
            const float4 b = make_float4(oacc[dt][4] * inv, oacc[dt][5] * inv, oacc[dt][6] * inv, oacc[dt][7] * inv);
            memcpy(op + dt * 16, &a, 16);
            memcpy(op + dt * 16 + 4, &b, 16);
        }
    }
}

extern "C" __global__ void attn_prefill_wmma_hd128(
    const float* q, const _Float16* k, const _Float16* v, float* o, int n_q_heads, int n_kv_heads, int n_q, int prefix
) {
    attn_prefill_wmma_body<128>(q, k, v, o, n_q_heads, n_kv_heads, n_q, prefix);
}

// Q4_K GEMM with a 64-row x 128-token tile on eight warps:
// four along the rows and two along the tokens. Each warp owns four N fragments.
template <int THREADS>
__device__ __forceinline__ void i9_stage_y_tile_t(
    int8_t (*Bs8)[I9_BS_STRIDE], const int8_t* q8, int n_in, int n_tokens, int t0, int tt0, int kbase
) {
    constexpr int N = 512 / THREADS;   // words per thread for one 16-token x 128-K tile
    unsigned v[N];
    #pragma unroll
    for (int i = 0; i < N; i++) {
        const int e = threadIdx.x + THREADS * i;
        const int tt = e / 32, wi = e % 32;
        const int t = min(t0 + tt, n_tokens - 1);
        memcpy(&v[i], q8 + (size_t)t * n_in + kbase + wi * 4, 4);
    }
    #pragma unroll
    for (int i = 0; i < N; i++) {
        const int e = threadIdx.x + THREADS * i;
        const int tt = e / 32, wi = e % 32;
        const unsigned val = (t0 + tt < n_tokens) ? v[i] : 0u;
        memcpy(&Bs8[tt0 + tt][wi * 4], &val, 4);
    }
}

template <int NWN>   // warps along the token axis: 2 -> 256 threads, up to 128 tokens
__device__ __forceinline__ void i9_gemm_q4k_body(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    constexpr int THREADS = 128 * NWN;
    constexpr int MAX_NT = 4 * NWN;
    const int m0 = blockIdx.x * 64;
    const int n0 = blockIdx.y * n_tile;
    const int warp = threadIdx.x / 32;
    const int warp_m = warp % 4;
    const int warp_n = warp / 4;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;
    const int row_base = (lane / 16) * 8;
    const int khalf = (lane / 16) * 16;
    const int rb = (n_in / QK_K) * BLOCK_Q4K_BYTES;
    const int nb = n_in / 32;
    const int nt = min(n_tile / 16, (n_tokens - n0 + 15) / 16);   // live 16-token tiles, <= MAX_NT

    extern __shared__ __attribute__((aligned(16))) int8_t i9_smem[];
    int8_t (*As8)[I9_AS_STRIDE] = (int8_t (*)[I9_AS_STRIDE])i9_smem;
    int8_t (*Bs8)[I9_BS_STRIDE] = (int8_t (*)[I9_BS_STRIDE])(i9_smem + 64 * I9_AS_STRIDE);
    float2 (*ysh)[8] = (float2 (*)[8])(i9_smem + 64 * I9_AS_STRIDE + n_tile * I9_BS_STRIDE);
    float (*coef)[64] = (float (*)[64])(i9_smem + 64 * I9_AS_STRIDE + n_tile * I9_BS_STRIDE + n_tile * 8 * 8);

    float8_t acc0 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc1 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc2 = {0, 0, 0, 0, 0, 0, 0, 0};
    float8_t acc3 = {0, 0, 0, 0, 0, 0, 0, 0};
    const int k_beg = blockIdx.z * k_split;
    const int k_end = k_beg + k_split;
    y += (size_t)blockIdx.z * n_tokens * n_out;

    for (int k0 = k_beg; k0 < k_end; k0 += QK_K) {
        const int sb = k0 / QK_K;
        if (threadIdx.x < 64) {
            const int rm = threadIdx.x;
            const int row = min(m0 + rm, n_out - 1);
            const uint8_t* block = w + (size_t)row * rb + (size_t)sb * BLOCK_Q4K_BYTES;
            v4i32_t hdr;
            memcpy(&hdr, block, 16);
            uint8_t hb[16];
            memcpy(hb, &hdr, 16);
            __half d_h, dmin_h;
            memcpy(&d_h, hb, 2);
            memcpy(&dmin_h, hb + 2, 2);
            const float d = __half2float(d_h);
            const float dmin = __half2float(dmin_h);
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                uint8_t sc, m;
                get_scale_min_k4(s, hb + 4, &sc, &m);
                coef[s][rm] = d * (float)sc;
                coef[8 + s][rm] = dmin * (float)m;
            }
        }
        {
            constexpr int NCH = 512 / THREADS;   // 16-byte weight chunks per thread
            v4i32_t v[NCH];
            #pragma unroll
            for (int i = 0; i < NCH; i++) {
                const int e = threadIdx.x + THREADS * i;
                const int rm = e / 8, c = e % 8;
                const int row = min(m0 + rm, n_out - 1);
                const uint8_t* qs = w + (size_t)row * rb + (size_t)sb * BLOCK_Q4K_BYTES + 4 + K_SCALE_SIZE;
                memcpy(&v[i], qs + c * 16, 16);
            }
            #pragma unroll
            for (int i = 0; i < NCH; i++) {
                const int e = threadIdx.x + THREADS * i;
                const int rm = e / 8, c = e % 8;
                const int kk = (c / 2) * 64 + (c % 2) * 16;
                v4i32_t lo, hi;
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    lo[q] = (int)((unsigned)v[i][q] & 0x0F0F0F0Fu);
                    hi[q] = (int)(((unsigned)v[i][q] >> 4) & 0x0F0F0F0Fu);
                }
                memcpy(&As8[rm][kk], &lo, 16);
                memcpy(&As8[rm][kk + 32], &hi, 16);
            }
        }
        #pragma unroll
        for (int ti = 0; ti < MAX_NT; ti++) {
            if (ti < nt && threadIdx.x < 128) i9_stage_ys_tile(ysh, ys, nb, n_tokens, n0 + ti * 16, ti * 16, sb);
        }

        #pragma unroll 1
        for (int hh = 0; hh < 2; hh++) {
            #pragma unroll
            for (int ti = 0; ti < MAX_NT; ti++) {
                if (ti < nt) i9_stage_y_tile_t<THREADS>(Bs8, q8, n_in, n_tokens, n0 + ti * 16, ti * 16, k0 + hh * 128);
            }
            __syncthreads();

            #pragma unroll 1
            for (int kb4 = 0; kb4 < 4; kb4++) {
                const int kb = hh * 4 + kb4;
                v4i32_t a4;
                memcpy(&a4, &As8[warp_m * 16 + col][hh * 128 + kb4 * 32 + khalf], 16);
                const v2i32_t a_lo = {a4[0], a4[1]};
                const v2i32_t a_hi = {a4[2], a4[3]};
                float cs[8], cm[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const int rl = warp_m * 16 + row_base + j;
                    cs[j] = coef[kb][rl];
                    cm[j] = coef[8 + kb][rl];
                }
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    const int gt = warp_n * 4 + ni;   // this warp's global 16-token tile
                    if (gt < nt) {
                        v4i32_t b4;
                        memcpy(&b4, &Bs8[gt * 16 + col][kb4 * 32 + khalf], 16);
                        const v2i32_t b_lo = {b4[0], b4[1]};
                        const v2i32_t b_hi = {b4[2], b4[3]};
                        v8i32_t iacc = {0, 0, 0, 0, 0, 0, 0, 0};
                        iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_lo, true, b_lo, iacc, false);
                        iacc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_hi, true, b_hi, iacc, false);
                        const float2 yt = ysh[gt * 16 + col][kb];
                        float8_t* accp = (ni == 0) ? &acc0 : (ni == 1) ? &acc1 : (ni == 2) ? &acc2 : &acc3;
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            (*accp)[j] += cs[j] * yt.x * (float)iacc[j] - cm[j] * yt.y;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int row = m0 + warp_m * 16 + row_base + j;
        if (row >= n_out) continue;
        const int tb = n0 + warp_n * 64 + col;
        int t = tb;
        if (warp_n * 4 + 0 < nt && t < n_tokens) y[(size_t)t * n_out + row] = acc0[j];
        t = tb + 16;
        if (warp_n * 4 + 1 < nt && t < n_tokens) y[(size_t)t * n_out + row] = acc1[j];
        t = tb + 32;
        if (warp_n * 4 + 2 < nt && t < n_tokens) y[(size_t)t * n_out + row] = acc2[j];
        t = tb + 48;
        if (warp_n * 4 + 3 < nt && t < n_tokens) y[(size_t)t * n_out + row] = acc3[j];
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) qmatmul_gemm_wmma_i8_k256_j128(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    i9_gemm_q4k_body<2>(w, n_out, n_in, n_tokens, q8, ys, y, k_split, n_tile);
}

// Compile-time NT permits all fragment and scale loads before the first multiply.
// Partial token tiles read a clamped live row and are masked at the store.
// LDS token metadata packs f16 d and int16 sum(q) into four bytes.
// Reconstruct s = d * sum(q) in f32; weight coefficients also remain f32.
struct q8_ts_t {
    __half d;
    short sumq;
};

// NT = 16-token tiles per warp (each warp owns 16 rows x 16*NT tokens,
// 8*NT outputs per lane); NWR = warps along the rows, so the block is
// 16*NWR rows x 16*NT tokens on 32*NWR threads.
// Token scales are staged per 128-K half to keep the 128x128 tile under
// the 64KB per-workgroup LDS cap.
template <int NT, int NWR, bool Packed = false>
__device__ __forceinline__ void i9_gemm_q4k_nt_body(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split
) {
    constexpr int n_tile = 16 * NT;
    constexpr int ROWS = 16 * NWR;
    constexpr int THREADS = 32 * NWR;
    constexpr int B_CPT = n_tile * 8 / THREADS;   // 16-byte chunks of the n_tile x 128-K activation tile per thread
    constexpr int AS_BYTES = ROWS * I9_AS_STRIDE;
    const int m0 = blockIdx.x * ROWS;
    const int n0 = blockIdx.y * n_tile;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;
    const int row_base = (lane / 16) * 8;
    const int khalf = (lane / 16) * 16;
    constexpr int block_bytes = Packed ? BLOCK_Q4K_PACKED_BYTES : BLOCK_Q4K_BYTES;
    const int rb = (n_in / QK_K) * block_bytes;
    const int nb = n_in / 32;

    extern __shared__ __attribute__((aligned(16))) int8_t i9_smem[];
    int8_t (*As8)[I9_AS_STRIDE] = (int8_t (*)[I9_AS_STRIDE])i9_smem;
    int8_t (*Bs8)[I9_BS_STRIDE] = (int8_t (*)[I9_BS_STRIDE])(i9_smem + AS_BYTES);
    // The min term sum_kb (dmin*m_kb) * s_kb is a K=8 dot per (row, token)
    // per superblock and leaves the kb loop: one f16 WMMA per token tile
    // with m (6-bit, exact in f16) against s split as s_hi = f16(s),
    // s_lo = f16(s - s_hi) packed along K=16 -- every product exact in
    // f32, retaining approximately 22 bits of s.
    _Float16 (*ysh)[4] = (_Float16 (*)[4])(i9_smem + AS_BYTES + n_tile * I9_BS_STRIDE);            // [token][block of this 128-K half] d_x
    _Float16 (*shl)[16] = (_Float16 (*)[16])(i9_smem + AS_BYTES + n_tile * I9_BS_STRIDE + n_tile * 4 * 2); // [token][s_hi x8, s_lo x8]
    float (*coef)[ROWS] = (float (*)[ROWS])(i9_smem + AS_BYTES + n_tile * I9_BS_STRIDE + n_tile * 4 * 2 + n_tile * 16 * 2); // [sub-block][row] d*sc
    float* dminr = (float*)(i9_smem + AS_BYTES + n_tile * I9_BS_STRIDE + n_tile * 4 * 2 + n_tile * 16 * 2 + 8 * ROWS * 4); // [row] dmin
    _Float16 (*mrow)[8] = (_Float16 (*)[8])(i9_smem + AS_BYTES + n_tile * I9_BS_STRIDE + n_tile * 4 * 2 + n_tile * 16 * 2 + 8 * ROWS * 4 + ROWS * 4); // [row][sub-block] m

    float8_t acc[NT];
    #pragma unroll
    for (int ni = 0; ni < NT; ni++) acc[ni] = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};
    const int k_beg = blockIdx.z * k_split;
    const int k_end = k_beg + k_split;
    y += (size_t)blockIdx.z * n_tokens * n_out;

    for (int k0 = k_beg; k0 < k_end; k0 += QK_K) {
        const int sb = k0 / QK_K;
        {
            // Two threads per row, four sub-blocks each: every thread busy.
            // Compile-time header indices keep the byte array in registers;
            // select the thread's half after decoding.
            const int rm = threadIdx.x / 2;
            const bool upper = threadIdx.x % 2;
            const int row = min(m0 + rm, n_out - 1);
            const uint8_t* block = w + (size_t)row * rb + (size_t)sb * block_bytes;
            v4i32_t hdr;
            memcpy(&hdr, block, 16);
            uint8_t hb[16];
            memcpy(hb, &hdr, 16);
            __half d_h, dmin_h;
            memcpy(&d_h, hb, 2);
            memcpy(&dmin_h, hb + 2, 2);
            const float d = __half2float(d_h);
            const float dmin = __half2float(dmin_h);
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                uint8_t sc_lo, m_lo, sc_hi, m_hi;
                q4k_scale_min<Packed>(s, hb, &sc_lo, &m_lo);
                q4k_scale_min<Packed>(s + 4, hb, &sc_hi, &m_hi);
                const float sc = upper ? (float)sc_hi : (float)sc_lo;
                const float m = upper ? (float)m_hi : (float)m_lo;
                coef[s + (upper ? 4 : 0)][rm] = d * sc;
                mrow[rm][s + (upper ? 4 : 0)] = (_Float16)m;
            }
            if (!upper) dminr[rm] = dmin;
        }
        {
            v4i32_t v[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int e = threadIdx.x + THREADS * i;
                const int rm = e / 8, c = e % 8;
                const int row = min(m0 + rm, n_out - 1);
                const uint8_t* qs = w + (size_t)row * rb + (size_t)sb * block_bytes + (Packed ? Q4K_PACKED_QUANTS_OFFSET : 16);
                memcpy(&v[i], qs + c * 16, 16);
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int e = threadIdx.x + THREADS * i;
                const int rm = e / 8, c = e % 8;
                const int kk = (c / 2) * 64 + (c % 2) * 16;
                v4i32_t lo, hi;
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    lo[q] = (int)((unsigned)v[i][q] & 0x0F0F0F0Fu);
                    hi[q] = (int)(((unsigned)v[i][q] >> 4) & 0x0F0F0F0Fu);
                }
                memcpy(&As8[rm][kk], &lo, 16);
                memcpy(&As8[rm][kk + 32], &hi, 16);
            }
        }

        #pragma unroll 1
        for (int hh = 0; hh < 2; hh++) {
            {
                // 16-byte moves: a token's 128 K of this half is 8 chunks.
                // Token slots past n_tokens read the clamped last token and
                // are never stored.
                v4i32_t v[B_CPT];
                #pragma unroll
                for (int i = 0; i < B_CPT; i++) {
                    const int e = threadIdx.x + THREADS * i;   // over all n_tile x 8 chunks
                    const int tt = e / 8, c = e % 8;
                    const int t = min(n0 + tt, n_tokens - 1);
                    memcpy(&v[i], q8 + (size_t)t * n_in + k0 + hh * 128 + c * 16, 16);
                }
                #pragma unroll
                for (int i = 0; i < B_CPT; i++) {
                    const int e = threadIdx.x + THREADS * i;
                    const int tt = e / 8, c = e % 8;
                    memcpy(&Bs8[tt][c * 16], &v[i], 16);
                }
            }
            // This half's four token scales per token (clamped slots
            // likewise carry the last token's scales and are never stored):
            // d for the kb epilogue, and s = d * sum(q) -- exact in f32 --
            // split hi/lo for the superblock's min-term WMMA. One read of
            // ys per (token, block) per superblock.
            #pragma unroll
            for (int i = 0; i < (n_tile * 4 + THREADS - 1) / THREADS; i++) {
                const int e = threadIdx.x + THREADS * i;
                if (e < n_tile * 4) {
                    const int tt = e / 4, bb = e % 4;
                    const int t = min(n0 + tt, n_tokens - 1);
                    const float2 v = ys[(size_t)t * nb + sb * 8 + hh * 4 + bb];
                    // The quantizer rounds d to f16, so this conversion is lossless.
                    ysh[tt][bb] = (_Float16)v.x;
                    const float sx = v.x * v.y;
                    const _Float16 hi = (_Float16)sx;
                    shl[tt][hh * 4 + bb] = hi;
                    shl[tt][8 + hh * 4 + bb] = (_Float16)(sx - (float)hi);
                }
            }
            __syncthreads();

            #pragma unroll 1
            for (int kb4 = 0; kb4 < 4; kb4++) {
                const int kb = hh * 4 + kb4;
                v4i32_t a4;
                memcpy(&a4, &As8[warp * 16 + col][hh * 128 + kb4 * 32 + khalf], 16);
                v4i32_t b4[NT];
                _Float16 yt[NT];
                #pragma unroll
                for (int ni = 0; ni < NT; ni++) {
                    memcpy(&b4[ni], &Bs8[ni * 16 + col][kb4 * 32 + khalf], 16);
                    yt[ni] = ysh[ni * 16 + col][kb4];
                }
                float cf[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) cf[j] = coef[kb][warp * 16 + row_base + j];
                const v2i32_t a_lo = {a4[0], a4[1]};
                const v2i32_t a_hi = {a4[2], a4[3]};
                // Consume groups of NG to bound live accumulators without
                // discarding the prefetched B fragments.
                constexpr int NG = NT > 4 ? 4 : NT;
                #pragma unroll
                for (int g = 0; g < NT; g += NG) {
                    v8i32_t iacc[NG];
                    #pragma unroll
                    for (int ni = 0; ni < NG; ni++) {
                        const v2i32_t b_lo = {b4[g + ni][0], b4[g + ni][1]};
                        iacc[ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_lo, true, b_lo, (v8i32_t){0, 0, 0, 0, 0, 0, 0, 0}, false);
                    }
                    #pragma unroll
                    for (int ni = 0; ni < NG; ni++) {
                        const v2i32_t b_hi = {b4[g + ni][2], b4[g + ni][3]};
                        iacc[ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_hi, true, b_hi, iacc[ni], false);
                    }
                    #pragma unroll
                    for (int ni = 0; ni < NG; ni++) {
                        const float dx = (float)yt[g + ni];
                        #pragma unroll
                        for (int j = 0; j < 8; j++) {
                            // mul, cvt, fma -- written as one fmaf into acc
                            // (`acc += a*b*c` compiled to an extra add).
                            acc[g + ni][j] = fmaf(cf[j] * dx, (float)iacc[ni][j], acc[g + ni][j]);
                        }
                    }
                    // Without this the scheduler hoists the next group's
                    // WMMAs above this epilogue (all NT int32 accumulators
                    // live) and pays for it by recycling two B-fragment
                    // register quads: load two, wait, multiply, reload.
                    if (NT > NG) __builtin_amdgcn_sched_barrier(0);
                }
            }
            if (hh == 1) {
                // Min term for this superblock: acc[row][tok] -= dmin[row] *
                // sum_kb m[row][kb] * s[tok][kb], the dot on the matrix core.
                // A = m (same 8 halves for both lane halves, k and k+8), B =
                // {s_hi, s_lo}; C lands in the accumulator's own layout.
                half8_t mf;
                memcpy(&mf, &mrow[warp * 16 + col][0], 16);
                float dm[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) dm[j] = dminr[warp * 16 + row_base + j];
                #pragma unroll
                for (int ni = 0; ni < NT; ni++) {
                    half8_t sf;
                    memcpy(&sf, &shl[ni * 16 + col][khalf / 2], 16);
                    const float8_t mt = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(mf, sf, (float8_t){0, 0, 0, 0, 0, 0, 0, 0});
                    #pragma unroll
                    for (int j = 0; j < 8; j++) acc[ni][j] = fmaf(-dm[j], mt[j], acc[ni][j]);
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int row = m0 + warp * 16 + row_base + j;
        if (row >= n_out) continue;
        #pragma unroll
        for (int ni = 0; ni < NT; ni++) {
            const int t = n0 + ni * 16 + col;
            if (t < n_tokens) y[(size_t)t * n_out + row] = acc[ni][j];
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 2) qmatmul_gemm_wmma_i8_k256_nt1(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    (void)n_tile;
    i9_gemm_q4k_nt_body<1, 4>(w, n_out, n_in, n_tokens, q8, ys, y, k_split);
}
extern "C" __global__ void __launch_bounds__(128, 2) qmatmul_gemm_wmma_i8_k256_nt2(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    (void)n_tile;
    i9_gemm_q4k_nt_body<2, 4>(w, n_out, n_in, n_tokens, q8, ys, y, k_split);
}
extern "C" __global__ void __launch_bounds__(128, 2) qmatmul_gemm_wmma_i8_k256_nt4(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    (void)n_tile;
    i9_gemm_q4k_nt_body<4, 4>(w, n_out, n_in, n_tokens, q8, ys, y, k_split);
}
// Large-batch geometry: 128 rows x 128 tokens, 256 threads,
// 64 outputs per lane. Occupancy 2 needs ~127KB of LDS per WGP.
extern "C" __global__ void __launch_bounds__(256, 2) qmatmul_gemm_wmma_i8_k256_nt8(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    (void)n_tile;
    i9_gemm_q4k_nt_body<8, 8>(w, n_out, n_in, n_tokens, q8, ys, y, k_split);
}

#define PREPARED_Q4K_GEMM(NT, NWR, THREADS) \
extern "C" __global__ void __launch_bounds__(THREADS, 2) qmatmul_gemm_wmma_i8_k256_packed_nt##NT( \
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile \
) { \
    (void)n_tile; \
    i9_gemm_q4k_nt_body<NT, NWR, true>(w, n_out, n_in, n_tokens, q8, ys, y, k_split); \
}
PREPARED_Q4K_GEMM(1, 4, 128)
PREPARED_Q4K_GEMM(2, 4, 128)
PREPARED_Q4K_GEMM(4, 4, 128)
PREPARED_Q4K_GEMM(8, 8, 256)
#undef PREPARED_Q4K_GEMM

// Q6_K on the compile-time-NT skeleton: same staging as
// qmatmul_gemm_wmma_i8_q6k_k256 (2-byte-pair loads, per-byte
// recentring), natural-order 8-byte fragments and one WMMA per int32
// accumulator per 16 K (the scale granularity), with every LDS load of a
// 16-K block -- A, B x NT, NT token scales, 8 coefficients -- issued
// before the first use and no branch in the loop.
// NT token tiles per warp, NWR warps along the rows (block = 16*NWR rows x
// 16*NT tokens on 32*NWR threads), as the Q4_K body. <8, 8> is the
// 128x128 / 256-thread geometry. No min term in Q6_K, so the kb epilogue
// is mul, cvt, fma and the token scale is d alone.
template <int NT, int NWR>
__device__ __forceinline__ void i9_gemm_q6k_nt_body(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split
) {
    constexpr int n_tile = 16 * NT;
    constexpr int ROWS = 16 * NWR;
    constexpr int THREADS = 32 * NWR;
    constexpr int B_CPT = n_tile * 8 / THREADS;   // 16-byte chunks of the n_tile x 128-K activation tile per thread
    const int m0 = blockIdx.x * ROWS;
    const int n0 = blockIdx.y * n_tile;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int col = lane % 16;
    const int row_base = (lane / 16) * 8;
    const int k8 = (lane / 16) * 8;
    const int rb = (n_in / QK_K) * BLOCK_Q6K_BYTES;
    const int nb = n_in / 32;

    extern __shared__ __attribute__((aligned(16))) int8_t i9_smem[];
    int8_t (*As8)[I9_AS_STRIDE] = (int8_t (*)[I9_AS_STRIDE])i9_smem;
    int8_t (*Bs8)[I9_BS_STRIDE] = (int8_t (*)[I9_BS_STRIDE])(i9_smem + ROWS * I9_AS_STRIDE);
    _Float16 (*ysh)[4] = (_Float16 (*)[4])(i9_smem + ROWS * I9_AS_STRIDE + n_tile * I9_BS_STRIDE);          // [token][block of this 128-K half] d_x
    float (*coef)[ROWS] = (float (*)[ROWS])(i9_smem + ROWS * I9_AS_STRIDE + n_tile * I9_BS_STRIDE + n_tile * 4 * 2); // [16-K block][row] d*sc

    float8_t acc[NT];
    #pragma unroll
    for (int ni = 0; ni < NT; ni++) acc[ni] = (float8_t){0, 0, 0, 0, 0, 0, 0, 0};
    const int k_beg = blockIdx.z * k_split;
    const int k_end = k_beg + k_split;
    y += (size_t)blockIdx.z * n_tokens * n_out;

    for (int k0 = k_beg; k0 < k_end; k0 += QK_K) {
        const int sb = k0 / QK_K;
        {
            // Two threads per row, eight of the sixteen scales each; the
            // thread's two scale words are picked with selects, never by
            // indexing an array at runtime (that goes to scratch).
            const int rm = threadIdx.x / 2;
            const bool upper = threadIdx.x % 2;
            const int row = min(m0 + rm, n_out - 1);
            const uint8_t* block = w + (size_t)row * rb + (size_t)sb * BLOCK_Q6K_BYTES;
            unsigned sw[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) sw[i] = load_u32_2aligned(block + 128 + 64 + 4 * i);
            const unsigned w0 = upper ? sw[2] : sw[0];
            const unsigned w1 = upper ? sw[3] : sw[1];
            unsigned short d_bits;
            memcpy(&d_bits, block + 128 + 64 + 16, 2);
            __half d_h;
            memcpy(&d_h, &d_bits, 2);
            const float d = __half2float(d_h);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned wj = j < 4 ? w0 : w1;
                const int8_t sc = (int8_t)((wj >> (8 * (j % 4))) & 0xFF);
                coef[(upper ? 8 : 0) + j][rm] = d * (float)sc;
            }
        }
        {
            #pragma unroll
            for (int grp = 0; grp < 2; grp++) {
                unsigned a[4], b[4], h[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int e = threadIdx.x + THREADS * (grp * 4 + i);
                    const int rm = e / 16, idx = (e / 8) % 2, g = e % 8;
                    const int row = min(m0 + rm, n_out - 1);
                    const uint8_t* block = w + (size_t)row * rb + (size_t)sb * BLOCK_Q6K_BYTES;
                    i9_q6k_unit(block, idx, g, &a[i], &b[i], &h[i]);
                }
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int e = threadIdx.x + THREADS * (grp * 4 + i);
                    const int rm = e / 16, idx = (e / 8) % 2, g = e % 8;
                    const int base = idx * 128 + 4 * g;
                    const unsigned q0 = i9_q6k_recentre((a[i] & 0x0F0F0F0Fu) | ((h[i] & 0x03030303u) << 4));
                    const unsigned q1 = i9_q6k_recentre((b[i] & 0x0F0F0F0Fu) | (((h[i] >> 2) & 0x03030303u) << 4));
                    const unsigned q2 = i9_q6k_recentre(((a[i] >> 4) & 0x0F0F0F0Fu) | (((h[i] >> 4) & 0x03030303u) << 4));
                    const unsigned q3 = i9_q6k_recentre(((b[i] >> 4) & 0x0F0F0F0Fu) | (((h[i] >> 6) & 0x03030303u) << 4));
                    memcpy(&As8[rm][base], &q0, 4);
                    memcpy(&As8[rm][base + 32], &q1, 4);
                    memcpy(&As8[rm][base + 64], &q2, 4);
                    memcpy(&As8[rm][base + 96], &q3, 4);
                }
            }
        }

        #pragma unroll 1
        for (int hh = 0; hh < 2; hh++) {
            {
                // 16-byte moves; clamped token slots carry the last token
                // and are never stored.
                v4i32_t v[B_CPT];
                #pragma unroll
                for (int i = 0; i < B_CPT; i++) {
                    const int e = threadIdx.x + THREADS * i;
                    const int tt = e / 8, c = e % 8;
                    const int t = min(n0 + tt, n_tokens - 1);
                    memcpy(&v[i], q8 + (size_t)t * n_in + k0 + hh * 128 + c * 16, 16);
                }
                #pragma unroll
                for (int i = 0; i < B_CPT; i++) {
                    const int e = threadIdx.x + THREADS * i;
                    const int tt = e / 8, c = e % 8;
                    memcpy(&Bs8[tt][c * 16], &v[i], 16);
                }
            }
            #pragma unroll
            for (int i = 0; i < (n_tile * 4 + THREADS - 1) / THREADS; i++) {
                const int e = threadIdx.x + THREADS * i;
                if (e < n_tile * 4) {
                    const int tt = e / 4, bb = e % 4;
                    const int t = min(n0 + tt, n_tokens - 1);
                    ysh[tt][bb] = (_Float16)ys[(size_t)t * nb + sb * 8 + hh * 4 + bb].x;
                }
            }
            __syncthreads();

            #pragma unroll 1
            for (int kb16 = 0; kb16 < 8; kb16++) {
                const int ko = hh * 128 + kb16 * 16;
                v2i32_t a;
                memcpy(&a, &As8[warp * 16 + col][ko + k8], 8);
                v2i32_t b[NT];
                float dx[NT];
                #pragma unroll
                for (int ni = 0; ni < NT; ni++) {
                    memcpy(&b[ni], &Bs8[ni * 16 + col][kb16 * 16 + k8], 8);
                    dx[ni] = (float)ysh[ni * 16 + col][kb16 / 2];
                }
                float cs[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) cs[j] = coef[ko / 16][warp * 16 + row_base + j];
                constexpr int NG = NT > 4 ? 4 : NT;
                #pragma unroll
                for (int g = 0; g < NT; g += NG) {
                    v8i32_t iacc[NG];
                    #pragma unroll
                    for (int ni = 0; ni < NG; ni++) {
                        iacc[ni] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b[g + ni], (v8i32_t){0, 0, 0, 0, 0, 0, 0, 0}, false);
                    }
                    #pragma unroll
                    for (int ni = 0; ni < NG; ni++) {
                        #pragma unroll
                        for (int j = 0; j < 8; j++) acc[g + ni][j] = fmaf(cs[j] * dx[g + ni], (float)iacc[ni][j], acc[g + ni][j]);
                    }
                    if (NT > NG) __builtin_amdgcn_sched_barrier(0);
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int row = m0 + warp * 16 + row_base + j;
        if (row >= n_out) continue;
        #pragma unroll
        for (int ni = 0; ni < NT; ni++) {
            const int t = n0 + ni * 16 + col;
            if (t < n_tokens) y[(size_t)t * n_out + row] = acc[ni][j];
        }
    }
}

extern "C" __global__ void __launch_bounds__(128, 2) qmatmul_gemm_wmma_i8_q6k_k256_nt1(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    (void)n_tile;
    i9_gemm_q6k_nt_body<1, 4>(w, n_out, n_in, n_tokens, q8, ys, y, k_split);
}
extern "C" __global__ void __launch_bounds__(128, 2) qmatmul_gemm_wmma_i8_q6k_k256_nt2(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    (void)n_tile;
    i9_gemm_q6k_nt_body<2, 4>(w, n_out, n_in, n_tokens, q8, ys, y, k_split);
}
extern "C" __global__ void __launch_bounds__(128, 2) qmatmul_gemm_wmma_i8_q6k_k256_nt4(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    (void)n_tile;
    i9_gemm_q6k_nt_body<4, 4>(w, n_out, n_in, n_tokens, q8, ys, y, k_split);
}
extern "C" __global__ void __launch_bounds__(256, 2) qmatmul_gemm_wmma_i8_q6k_k256_nt8(
    const uint8_t* w, int n_out, int n_in, int n_tokens, const int8_t* q8, const float2* ys, float* y, int k_split, int n_tile
) {
    (void)n_tile;
    i9_gemm_q6k_nt_body<8, 8>(w, n_out, n_in, n_tokens, q8, ys, y, k_split);
}

#endif

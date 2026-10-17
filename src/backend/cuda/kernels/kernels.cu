// CUDA kernels for NVIDIA (Ampere sm_86 by default). Compiled to PTX by
// clang (no CUDA toolkit: -nocudainc -nocudalib), so the qualifiers and
// math below come from NVVM builtins rather than cuda_runtime.h.

#include <__clang_cuda_builtin_vars.h>

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
#define KERNEL extern "C" __global__

typedef unsigned char u8;
typedef signed char i8;
struct __attribute__((aligned(16))) uint4 { unsigned x, y, z, w; };
struct __attribute__((aligned(16))) float4 { float x, y, z, w; };
// A register vector: a conditionally zeroed uint4 struct is copied
// through local memory, this is not.
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));

#define DT_F32 0
#define DT_F16 1
#define DT_Q4K 2
#define DT_Q6K 3
#define DT_BF16 4
#define DT_Q5K 5
#define DT_Q8_0 6
#define BLOCK_Q5K 176

// RoPE frequencies computed on the host with the CPU backend's powf, so
// the angle is the same f32 product. head_dim / 2 <= 256 (ROPE_PAIRS in
// cuda.rs).
struct Freqs {
    float f[256];
};

// clang takes a cast pointer's alignment from its byte-typed source
// (`*(unsigned*)&i8_array[k]` is an align-1 load), and NVPTX then splits
// the access into byte ops. These restate the alignment the layout
// guarantees.
#define AT32(p) (*(unsigned*)__builtin_assume_aligned((p), 4))
#define AT128(p) (*(uint4*)__builtin_assume_aligned((p), 16))

__device__ static inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, o, 31);
    return v;
}

__device__ static inline float fexp(float x) {
    return __nvvm_ex2_approx_ftz_f(x * 1.4426950408889634f);
}

__device__ static inline float half_at(const u8* p) {
    return (float)*(const _Float16*)p;
}

__device__ static inline float bf16_at(const u8* p) {
    return __builtin_bit_cast(float, (unsigned)*(const unsigned short*)p << 16);
}

// sin/cos to ~1 ulp: quadrant reduction in double (the angle is an exact
// f32, up to context length in radians), cephes polynomials on [-pi/4, pi/4].
__device__ static inline void sincos_acc(float a, float* s_out, float* c_out) {
    double ad = (double)a;
    long long k = __nvvm_d2ll_rn(ad * 0.63661977236758134308);
    float r = (float)(ad - (double)k * 1.57079632679489661923);
    float z = r * r;
    float s = r + r * z * (-1.6666654611e-1f + z * (8.3321608736e-3f + z * -1.9515295891e-4f));
    float c = 1.0f - 0.5f * z + z * z * (4.166664568298827e-2f + z * (-1.388731625493765e-3f + z * 2.443315711809948e-5f));
    switch ((int)(k & 3)) {
        case 0: *s_out = s; *c_out = c; break;
        case 1: *s_out = c; *c_out = -s; break;
        case 2: *s_out = -s; *c_out = -c; break;
        default: *s_out = -c; *c_out = s; break;
    }
}

KERNEL void add_f32(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

KERNEL void silu_mul_f32(const float* gate, const float* up, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        y[i] = (g / (1.0f + fexp(-g))) * up[i];
    }
}

// One 256-thread block per token.
KERNEL void rmsnorm_f32(const float* w, const float* x, float* y, int dim, float eps) {
    __shared__ float part[8];
    const float* xt = x + (long)blockIdx.x * dim;
    float* yt = y + (long)blockIdx.x * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xt[i] * xt[i];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) total += part[i];
    float scale = 1.0f / __nvvm_sqrt_rn_f(total / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) yt[i] = xt[i] * scale * w[i];
}

// sum = a + b, then y = rmsnorm(sum) * w: add_f32 and rmsnorm_f32 in
// one launch, same sums in the same order. Each thread re-reads only the
// sum elements it wrote, so sum may alias a or b.
KERNEL void add_rmsnorm_f32(const float* w, const float* a, const float* b, float* sum, float* y, int dim, float eps) {
    __shared__ float part[8];
    long base = (long)blockIdx.x * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float s = a[base + i] + b[base + i];
        sum[base + i] = s;
        ss += s * s;
    }
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    float total = 0.0f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) total += part[i];
    float scale = 1.0f / __nvvm_sqrt_rn_f(total / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) y[base + i] = sum[base + i] * scale * w[i];
}

// q/k are [n_tokens, n_heads, head_dim], interleaved pairs (2i, 2i+1).
// x = pair, y = head (q heads then kv heads), z = token.
KERNEL void rope_f32(float* q, float* k, int pos, int head_dim, int n_q_heads, int n_kv_heads, Freqs fr) {
    int pair = threadIdx.x;
    int head = blockIdx.y;
    int t = blockIdx.z;
    if (pair >= head_dim / 2) return;
    bool is_q = head < n_q_heads;
    int n_heads = is_q ? n_q_heads : n_kv_heads;
    int local = is_q ? head : head - n_q_heads;
    float* buf = is_q ? q : k;
    float ang = (float)(unsigned)(pos + t) * fr.f[pair];
    float s, c;
    sincos_acc(ang, &s, &c);
    long base = ((long)t * n_heads + local) * head_dim + 2 * pair;
    float x0 = buf[base];
    float x1 = buf[base + 1];
    buf[base] = x0 * c - x1 * s;
    buf[base + 1] = x0 * s + x1 * c;
}

// rope_f32 then kv_write_f16 in one launch: q rotates in place, k's
// rotated pair and v's pair go straight to the f16 caches at element
// `offset` (k itself is left unrotated). Same rounding as the two-step
// path: the rotated f32 is converted once. Under graph replay `pp` holds
// (token, pos) and overrides pos and offset (one token, pos rows in).
KERNEL void rope_kv_f16(float* q, const float* k, const float* v, _Float16* dk, _Float16* dv, int pos, int head_dim, int n_q_heads, int n_kv_heads, long long offset, Freqs fr, const unsigned* pp) {
    if (pp) {
        pos = (int)pp[1];
        offset = (long long)pos * n_kv_heads * head_dim;
    }
    int pair = threadIdx.x;
    int head = blockIdx.y;
    int t = blockIdx.z;
    if (pair >= head_dim / 2) return;
    bool is_q = head < n_q_heads;
    int n_heads = is_q ? n_q_heads : n_kv_heads;
    int local = is_q ? head : head - n_q_heads;
    float ang = (float)(unsigned)(pos + t) * fr.f[pair];
    float s, c;
    sincos_acc(ang, &s, &c);
    long base = ((long)t * n_heads + local) * head_dim + 2 * pair;
    const float* src = is_q ? q : k;
    float x0 = src[base];
    float x1 = src[base + 1];
    float r0 = x0 * c - x1 * s;
    float r1 = x0 * s + x1 * c;
    if (is_q) {
        q[base] = r0;
        q[base + 1] = r1;
    } else {
        dk[offset + base] = (_Float16)r0;
        dk[offset + base + 1] = (_Float16)r1;
        dv[offset + base] = (_Float16)v[base];
        dv[offset + base + 1] = (_Float16)v[base + 1];
    }
}

// f32 K/V rows into the f16 cache at element `offset`, rounded to
// nearest even once on the way in.
KERNEL void kv_write_f16(const float* k, const float* v, _Float16* dk, _Float16* dv, int n, long long offset, const unsigned* pp) {
    if (pp) offset = (long long)pp[1] * n;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dk[offset + i] = (_Float16)k[i];
        dv[offset + i] = (_Float16)v[i];
    }
}

__device__ static inline void f16x2(unsigned w, float* lo, float* hi) {
    *lo = (float)__builtin_bit_cast(_Float16, (unsigned short)w);
    *hi = (float)__builtin_bit_cast(_Float16, (unsigned short)(w >> 16));
}

// q/o: [n_q, n_q_heads, head_dim] f32; k/v rows: [kv_len, n_kv_heads,
// head_dim] f16, head_dim even and <= 256. Query i attends to keys
// [0, prefix + i]. One block per (G query heads, query, key chunk); the G
// heads share a kv head, so each K/V row is read once for all of them.
// ATTN_WARPS warps: warp w streams the chunk's keys w, w + ATTN_WARPS, ...
// through G online softmaxes (lane holds the dim pairs 2 lane + 64 c),
// then the warps' (max, sum, acc) partials merge in shared memory, one
// q head at a time. With one chunk the block writes o; with several it
// writes each head's (m, l, acc) to part[(query, head)][chunk] for
// attn_merge.
#define ATTN_WARPS 8
template <int G>
__device__ static inline void attn_group(
    const float* q, const _Float16* k, const _Float16* v, float* o, float* part,
    int head_dim, int n_q_heads, int n_kv_heads, int group, int prefix, int chunk
) {
    __shared__ float part_m[ATTN_WARPS];
    __shared__ float part_l[ATTN_WARPS];
    __shared__ float part_acc[ATTN_WARPS][256];
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    // Block x serves q heads x G .. x G + G - 1 of the same kv head
    // (group = n_q_heads / n_kv_heads, a multiple of G).
    int kvh = blockIdx.x * G / group;
    int qi = blockIdx.y;
    int split = blockIdx.z;
    int n_splits = gridDim.z;
    long kv_row = (long)n_kv_heads * head_dim;
    int visible = prefix + qi + 1;
    int begin = split * chunk;
    int end = begin + chunk < visible ? begin + chunk : visible;
    // Row of the group's first q head; head g is row0 + g.
    long row0 = (long)qi * n_q_heads + (long)blockIdx.x * G;
    int stride = head_dim + 2;
    if (begin >= end) {
        // Causally masked chunk: empty partials (exp(-inf) = 0 in the merge).
        int t = warp * 32 + lane;
        for (int g = 0; g < G; g++) {
            float* pp = part + ((row0 + g) * n_splits + split) * stride;
            if (t == 0) {
                pp[0] = -__builtin_inff();
                pp[1] = 0.0f;
            }
            for (int d = t; d < head_dim; d += ATTN_WARPS * 32) pp[2 + d] = 0.0f;
        }
        return;
    }
    float scale = 1.0f / __nvvm_sqrt_rn_f((float)head_dim);
    float qv[G][8];
    float acc[G][8];
    float m[G];
    float l[G];
#pragma unroll
    for (int g = 0; g < G; g++) {
        const float* qp = q + (row0 + g) * head_dim;
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int d = 2 * lane + 64 * c;
            qv[g][2 * c] = d < head_dim ? qp[d] : 0.0f;
            qv[g][2 * c + 1] = d < head_dim ? qp[d + 1] : 0.0f;
            acc[g][2 * c] = 0.0f;
            acc[g][2 * c + 1] = 0.0f;
        }
        m[g] = -__builtin_inff();
        l[g] = 0.0f;
    }
    for (int t = begin + warp; t < end; t += ATTN_WARPS) {
        const unsigned* kp = (const unsigned*)(k + t * kv_row + (long)kvh * head_dim);
        const unsigned* vp = (const unsigned*)(v + t * kv_row + (long)kvh * head_dim);
        float kf[8];
        float vf[8];
#pragma unroll
        for (int c = 0; c < 4; c++) {
            bool in = 2 * lane + 64 * c < head_dim;
            unsigned kw = in ? kp[lane + 32 * c] : 0u;
            unsigned vw = in ? vp[lane + 32 * c] : 0u;
            f16x2(kw, &kf[2 * c], &kf[2 * c + 1]);
            f16x2(vw, &vf[2 * c], &vf[2 * c + 1]);
        }
#pragma unroll
        for (int g = 0; g < G; g++) {
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) dot += qv[g][i] * kf[i];
            float s = warp_sum(dot) * scale;
            float m_new = s > m[g] ? s : m[g];
            float corr = fexp(m[g] - m_new);
            float p = fexp(s - m_new);
            l[g] = l[g] * corr + p;
#pragma unroll
            for (int i = 0; i < 8; i++) acc[g][i] = acc[g][i] * corr + p * vf[i];
            m[g] = m_new;
        }
    }
#pragma unroll
    for (int g = 0; g < G; g++) {
        if (g > 0) __syncthreads(); // the previous head's merge is done reading
        if (lane == 0) {
            part_m[warp] = m[g];
            part_l[warp] = l[g];
        }
#pragma unroll
        for (int c = 0; c < 4; c++) {
            int d = 2 * lane + 64 * c;
            if (d < head_dim) {
                part_acc[warp][d] = acc[g][2 * c];
                part_acc[warp][d + 1] = acc[g][2 * c + 1];
            }
        }
        __syncthreads();
        // Warp 0 always sees the chunk's first key, so the block max is
        // finite and warps that saw no keys (m = -inf, l = 0) contribute
        // exp(-inf) = 0.
        float gm = part_m[0];
        for (int w = 1; w < ATTN_WARPS; w++) gm = part_m[w] > gm ? part_m[w] : gm;
        float gl = 0.0f;
        for (int w = 0; w < ATTN_WARPS; w++) gl += part_l[w] * fexp(part_m[w] - gm);
        long row = row0 + g;
        if (n_splits == 1) {
            float* op = o + row * head_dim;
            for (int d = warp * 32 + lane; d < head_dim; d += ATTN_WARPS * 32) {
                float sum = 0.0f;
                for (int w = 0; w < ATTN_WARPS; w++) sum += part_acc[w][d] * fexp(part_m[w] - gm);
                op[d] = sum / gl;
            }
            continue;
        }
        float* pp = part + (row * n_splits + split) * stride;
        if (warp == 0 && lane == 0) {
            pp[0] = gm;
            pp[1] = gl;
        }
        for (int d = warp * 32 + lane; d < head_dim; d += ATTN_WARPS * 32) {
            float sum = 0.0f;
            for (int w = 0; w < ATTN_WARPS; w++) sum += part_acc[w][d] * fexp(part_m[w] - gm);
            pp[2 + d] = sum;
        }
    }
}

// One entry point per G; the host takes the largest that divides the
// GQA group, launching n_q_heads / G blocks along x. Under graph replay
// `pp` holds (token, pos): one query at pos, so the prefix is pos.
#define ATTN_ENTRY(G)                                                                   \
    KERNEL void attn_f16kv_g##G(                                                        \
        const float* q, const _Float16* k, const _Float16* v, float* o, float* part,    \
        int head_dim, int n_q_heads, int n_kv_heads, int group, int prefix, int chunk,  \
        const unsigned* pp                                                              \
    ) {                                                                                 \
        if (pp) prefix = (int)pp[1];                                                    \
        attn_group<G>(q, k, v, o, part, head_dim, n_q_heads, n_kv_heads, group, prefix, chunk); \
    }
ATTN_ENTRY(1)
ATTN_ENTRY(2)
ATTN_ENTRY(4)
ATTN_ENTRY(8)

// o[(query, head)] from attn_f16kv_g*'s per-chunk partials, merged in chunk
// order so the result does not depend on scheduling. Chunk 0 always holds
// key 0, so the max is finite. Block of 256 threads per (head, query).
KERNEL void attn_merge(const float* part, float* o, int head_dim, int n_q_heads, int n_splits) {
    long row = (long)blockIdx.y * n_q_heads + blockIdx.x;
    int stride = head_dim + 2;
    const float* pp = part + row * n_splits * stride;
    float gm = pp[0];
    for (int s = 1; s < n_splits; s++) gm = pp[s * stride] > gm ? pp[s * stride] : gm;
    float gl = 0.0f;
    for (int s = 0; s < n_splits; s++) gl += pp[s * stride + 1] * fexp(pp[s * stride] - gm);
    float* op = o + row * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int s = 0; s < n_splits; s++) sum += pp[s * stride + 2 + d] * fexp(pp[s * stride] - gm);
        op[d] = sum / gl;
    }
}

// y[t, row] = dot(W[row], x[t]); f32 or f16 weights. Block (32, 4): one
// warp per row, grid (rows / 4, n_tokens).
KERNEL void gemv_f(const u8* w, const float* x, float* y, int n_out, int n_in, int dtype) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int t = blockIdx.y;
    if (row >= n_out) return;
    const float* xt = x + (long)t * n_in;
    float acc = 0.0f;
    if (dtype == DT_F32) {
        const float* wr = (const float*)w + (long)row * n_in;
        for (int i = threadIdx.x; i < n_in; i += 32) acc += wr[i] * xt[i];
    } else if (dtype == DT_BF16) {
        const u8* wr = w + (long)row * n_in * 2;
        for (int i = threadIdx.x; i < n_in; i += 32) acc += bf16_at(wr + 2 * i) * xt[i];
    } else {
        const u8* wr = w + (long)row * n_in * 2;
        for (int i = threadIdx.x; i < n_in; i += 32) acc += half_at(wr + 2 * i) * xt[i];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[(long)t * n_out + row] = acc;
}

// Q4_K 6-bit scale and min j (0..7) from the 12 scale bytes as words
// w0..w2 (bytes 0-3, 4-7, 8-11). Shifts, not a byte array: a runtime
// index into a local array puts it in local memory.
__device__ static inline void q4k_scale_min_w(unsigned w0, unsigned w1, unsigned w2, int j, int* scale, int* mn) {
    int sh = 8 * (j & 3);
    unsigned b0 = (w0 >> sh) & 0xFF, b1 = (w1 >> sh) & 0xFF;
    if (j < 4) {
        *scale = b0 & 63;
        *mn = b1 & 63;
    } else {
        unsigned b2 = (w2 >> sh) & 0xFF;
        *scale = (b2 & 0xF) | ((b0 >> 6) << 4);
        *mn = (b2 >> 4) | ((b1 >> 6) << 4);
    }
}

__device__ static inline void q4k_scale_min(const u8* sc, int j, int* scale, int* mn) {
    q4k_scale_min_w(AT32(sc), AT32(sc + 4), AT32(sc + 8), j, scale, mn);
}

// Element e (0..255) of a Q5_K block: Q4_K's scales, plus a high bit per
// element in qh (bytes 16..47) ahead of the 4-bit qs (48..175).
__device__ static inline float q5k_at(const u8* blk, int e) {
    int j = e / 32, l = e % 32, scale, mn;
    q4k_scale_min(blk + 4, j, &scale, &mn);
    int lo = (blk[48 + (j >> 1) * 32 + l] >> ((j & 1) * 4)) & 0xF;
    int hi = (blk[16 + l] >> j) & 1;
    return half_at(blk) * (float)scale * (float)(lo | hi << 4) - half_at(blk + 2) * (float)mn;
}

// Matvec K-quant mapping: 8 lanes per 256-element block, so a warp
// covers 4 consecutive blocks (576 B of Q4_K) per iteration and each lane
// owns 32 elements of its block. A warp computes Q_ROWS rows, so the 32
// activations a lane holds in registers are read once for all of them.
#define Q_ROWS 4

// Q4_K block (144 B, 4-byte aligned): f16 d, f16 dmin, 12 scale bytes,
// 128 qs bytes. Lane c of the block's 8 reads qs[32p + 16h .. +16) for
// p = c/2, h = c&1: low nibbles are sub-block 2p (xv[0..16)), high
// nibbles 2p+1 (xv[16..32)); sl/sh are those activations' sums.
__device__ static inline float q4k_part_dot(const u8* blk, const float* xv, float sl, float sh, int c) {
    unsigned dd = *(const unsigned*)blk;
    float d = (float)*(const _Float16*)&dd;
    float dmin = (float)((const _Float16*)&dd)[1];
    unsigned w0 = AT32(blk + 4), w1 = AT32(blk + 8), w2 = AT32(blk + 12);
    int p = c >> 1;
    int s0, m0, s1, m1;
    q4k_scale_min_w(w0, w1, w2, 2 * p, &s0, &m0);
    q4k_scale_min_w(w0, w1, w2, 2 * p + 1, &s1, &m1);
    uint4 q = *(const uint4*)(blk + 16 + 32 * p + 16 * (c & 1));
    unsigned qw[4] = {q.x, q.y, q.z, q.w};
    float ql = 0.0f, qhs = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned w = qw[i];
        const float* a = xv + 4 * i;
        const float* b = xv + 16 + 4 * i;
        ql += (float)(w & 0xF) * a[0] + (float)((w >> 8) & 0xF) * a[1] + (float)((w >> 16) & 0xF) * a[2] + (float)((w >> 24) & 0xF) * a[3];
        qhs += (float)((w >> 4) & 0xF) * b[0] + (float)((w >> 12) & 0xF) * b[1] + (float)((w >> 20) & 0xF) * b[2] + (float)(w >> 28) * b[3];
    }
    return d * ((float)s0 * ql + (float)s1 * qhs) - dmin * ((float)m0 * sl + (float)m1 * sh);
}

// Q6_K block (210 B, 2-byte aligned): 128 ql, 64 qh, 16 i8 scales, f16 d.
// Lane c: half h = c/4, l in [8(c%4), +8). Each l yields elements
// 128h + l + 32g for g = 0..3 (ql[64h+l], ql[64h+l+32] nibbles, qh[32h+l]
// bit pairs), so a lane owns 4 runs of 8 elements (xv[8g..8g+8)), one
// scale each.
__device__ static inline float q6k_part_dot(const u8* blk, const float* xv, int c) {
    int h = c >> 2;
    int l0 = (c & 3) * 8;
    const unsigned short* a16 = (const unsigned short*)(blk + 64 * h + l0);
    const unsigned short* b16 = (const unsigned short*)(blk + 64 * h + l0 + 32);
    const unsigned short* h16 = (const unsigned short*)(blk + 128 + 32 * h + l0);
    u8 qa[8], qb[8], qhb[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned short va = a16[i], vb = b16[i], vh = h16[i];
        qa[2 * i] = (u8)va; qa[2 * i + 1] = (u8)(va >> 8);
        qb[2 * i] = (u8)vb; qb[2 * i + 1] = (u8)(vb >> 8);
        qhb[2 * i] = (u8)vh; qhb[2 * i + 1] = (u8)(vh >> 8);
    }
    const i8* scales = (const i8*)(blk + 192);
    float d = half_at(blk + 208);
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int hi = qhb[i];
        acc[0] += (float)((int)((qa[i] & 0xF) | ((hi & 3) << 4)) - 32) * xv[i];
        acc[1] += (float)((int)((qb[i] & 0xF) | (((hi >> 2) & 3) << 4)) - 32) * xv[8 + i];
        acc[2] += (float)((int)((qa[i] >> 4) | (((hi >> 4) & 3) << 4)) - 32) * xv[16 + i];
        acc[3] += (float)((int)((qb[i] >> 4) | (((hi >> 6) & 3) << 4)) - 32) * xv[24 + i];
    }
    int si = 8 * h + l0 / 16;
    return d * ((float)scales[si] * acc[0] + (float)scales[si + 2] * acc[1] + (float)scales[si + 4] * acc[2] + (float)scales[si + 6] * acc[3]);
}

// Rows row0 .. row0 + Q_ROWS - 1 of w (dtype 2 = Q4_K, 3 = Q6_K; n_in %
// 256 == 0) against token row xt, one warp: every lane ends with the
// Q_ROWS sums in out[]. Rows past n_out are 0.
__device__ static inline __attribute__((always_inline)) void gemv_q_warp(const u8* w, const float* xt, int row0, int n_out, int n_in, int dtype, int block_bytes, float* out) {
    int lane = threadIdx.x;
    int sub = lane >> 3;
    int c = lane & 7;
    int n_blocks = n_in / 256;
    long row_bytes = (long)n_blocks * block_bytes;
    float acc[Q_ROWS] = {};
    float xv[32];
    if (dtype == DT_Q4K) {
        int p = c >> 1;
        int h = c & 1;
        for (int b = sub; b < n_blocks; b += 4) {
            const float4* xl = (const float4*)(xt + b * 256 + 64 * p + 16 * h);
            const float4* xh = (const float4*)(xt + b * 256 + 64 * p + 32 + 16 * h);
            float sl = 0.0f, sh = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float4 a = xl[i];
                float4 bb = xh[i];
                xv[4 * i] = a.x; xv[4 * i + 1] = a.y; xv[4 * i + 2] = a.z; xv[4 * i + 3] = a.w;
                xv[16 + 4 * i] = bb.x; xv[16 + 4 * i + 1] = bb.y; xv[16 + 4 * i + 2] = bb.z; xv[16 + 4 * i + 3] = bb.w;
                sl += a.x + a.y + a.z + a.w;
                sh += bb.x + bb.y + bb.z + bb.w;
            }
#pragma unroll
            for (int r = 0; r < Q_ROWS; r++) {
                if (row0 + r < n_out) acc[r] += q4k_part_dot(w + (row0 + r) * row_bytes + (long)b * 144, xv, sl, sh, c);
            }
        }
    } else {
        int h = c >> 2;
        int l0 = (c & 3) * 8;
        for (int b = sub; b < n_blocks; b += 4) {
            const float* xs = xt + b * 256 + 128 * h + l0;
#pragma unroll
            for (int g = 0; g < 4; g++) {
                float4 a = *(const float4*)(xs + 32 * g);
                float4 bb = *(const float4*)(xs + 32 * g + 4);
                xv[8 * g] = a.x; xv[8 * g + 1] = a.y; xv[8 * g + 2] = a.z; xv[8 * g + 3] = a.w;
                xv[8 * g + 4] = bb.x; xv[8 * g + 5] = bb.y; xv[8 * g + 6] = bb.z; xv[8 * g + 7] = bb.w;
            }
#pragma unroll
            for (int r = 0; r < Q_ROWS; r++) {
                if (row0 + r < n_out) acc[r] += q6k_part_dot(w + (row0 + r) * row_bytes + (long)b * 210, xv, c);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < Q_ROWS; r++) out[r] = warp_sum(acc[r]);
}

// Block (32, 4), each warp Q_ROWS consecutive rows: grid (ceil(rows /
// (4 * Q_ROWS)), n_tokens).
KERNEL void gemv_q(const u8* w, const float* x, float* y, int n_out, int n_in, int dtype, int block_bytes) {
    int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * Q_ROWS;
    int t = blockIdx.y;
    if (row0 >= n_out) return;
    float v[Q_ROWS];
    gemv_q_warp(w, x + (long)t * n_in, row0, n_out, n_in, dtype, block_bytes, v);
#pragma unroll
    for (int r = 0; r < Q_ROWS; r++) {
        if (threadIdx.x == 0 && row0 + r < n_out) y[(long)t * n_out + row0 + r] = v[r];
    }
}

// Three K-quant matrices over the same x (the q, k and v projections) in
// one launch. Block x walks matrix 0's row blocks, then 1's, then 2's;
// each matrix keeps its own dtype and output. Selected with ternaries so
// the by-value arrays are never indexed at run time (that would copy them
// to local memory).
struct Gemv3 {
    const u8* w[3];
    float* y[3];
    int n_out[3];
    int dtype[3];
    int block_bytes[3];
    int blocks[3];
};

KERNEL void gemv_q3(Gemv3 m, const float* x, int n_in) {
    int bx = blockIdx.x;
    int i = bx < m.blocks[0] ? 0 : bx < m.blocks[0] + m.blocks[1] ? 1 : 2;
    int first = i == 0 ? 0 : i == 1 ? m.blocks[0] : m.blocks[0] + m.blocks[1];
    const u8* w = i == 0 ? m.w[0] : i == 1 ? m.w[1] : m.w[2];
    float* y = i == 0 ? m.y[0] : i == 1 ? m.y[1] : m.y[2];
    int n_out = i == 0 ? m.n_out[0] : i == 1 ? m.n_out[1] : m.n_out[2];
    int dtype = i == 0 ? m.dtype[0] : i == 1 ? m.dtype[1] : m.dtype[2];
    int bb = i == 0 ? m.block_bytes[0] : i == 1 ? m.block_bytes[1] : m.block_bytes[2];
    int row0 = ((bx - first) * blockDim.y + threadIdx.y) * Q_ROWS;
    int t = blockIdx.y;
    if (row0 >= n_out) return;
    float v[Q_ROWS];
    gemv_q_warp(w, x + (long)t * n_in, row0, n_out, n_in, dtype, bb, v);
#pragma unroll
    for (int r = 0; r < Q_ROWS; r++) {
        if (threadIdx.x == 0 && row0 + r < n_out) y[(long)t * n_out + row0 + r] = v[r];
    }
}

// y = SiLU(gate x) * (up x): both matvecs and silu_mul_f32 in one launch,
// a warp computing the same Q_ROWS rows of each. Same grid as gemv_q.
KERNEL void gemv_q_swiglu(const u8* wg, const u8* wu, const float* x, float* y, int n_out, int n_in, int dtype_g, int bb_g, int dtype_u, int bb_u) {
    int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * Q_ROWS;
    int t = blockIdx.y;
    if (row0 >= n_out) return;
    const float* xt = x + (long)t * n_in;
    float g[Q_ROWS], u[Q_ROWS];
    gemv_q_warp(wg, xt, row0, n_out, n_in, dtype_g, bb_g, g);
    gemv_q_warp(wu, xt, row0, n_out, n_in, dtype_u, bb_u, u);
#pragma unroll
    for (int r = 0; r < Q_ROWS; r++) {
        if (threadIdx.x == 0 && row0 + r < n_out) y[(long)t * n_out + row0 + r] = (g[r] / (1.0f + fexp(-g[r]))) * u[r];
    }
}

// Gather dequantized rows; one thread per output element, grid
// (n_embd / 256, n_tokens).
KERNEL void embed_rows(const u8* table, const int* ids, float* out, int n_embd, int dtype, int block_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (i >= n_embd) return;
    long id = ids[t];
    float v;
    switch (dtype) {
    case DT_F32:
        v = ((const float*)table)[id * n_embd + i];
        break;
    case DT_F16:
        v = half_at(table + (id * n_embd + i) * 2);
        break;
    case DT_BF16:
        v = bf16_at(table + (id * n_embd + i) * 2);
        break;
    case DT_Q5K:
        v = q5k_at(table + (id * (n_embd / 256) + i / 256) * BLOCK_Q5K, i % 256);
        break;
    case DT_Q8_0: {
        const u8* blk = table + (id * (n_embd / 32) + i / 32) * 34;
        v = half_at(blk) * (float)((const i8*)(blk + 2))[i % 32];
        break;
    }
    default: {
        const u8* blk = table + (id * (n_embd / 256) + i / 256) * block_bytes;
        int e = i % 256;
        if (dtype == DT_Q4K) {
            int j = e / 32;
            int l = e % 32;
            int scale, mn;
            q4k_scale_min(blk + 4, j, &scale, &mn);
            int q = (blk[16 + (j >> 1) * 32 + l] >> ((j & 1) * 4)) & 0xF;
            v = half_at(blk) * (float)scale * (float)q - half_at(blk + 2) * (float)mn;
        } else {
            int j = (e / 128) * 128;
            int quad = (e % 128) / 32;
            int l = e % 32;
            int lo = (blk[j / 2 + l + ((quad & 1) ? 32 : 0)] >> ((quad & 2) ? 4 : 0)) & 0xF;
            int qv = lo | (((blk[128 + j / 4 + l] >> (quad * 2)) & 3) << 4);
            int sc = ((const i8*)(blk + 192))[(j + quad * 32 + l) / 16];
            v = half_at(blk + 208) * (float)sc * (float)(qv - 32);
        }
        break;
    }
    }
    out[(long)t * n_embd + i] = v;
}

// ===========================================================================
// int8 prefill GEMM (the ROCm backend's round-8 design, on mma.sync).
// Activations are quantized to Q8_1 (int8 per 32 elements, f16-rounded
// scale d, integer block sum); weights are staged into shared memory as
// their raw quantized values with no scale arithmetic; mma.sync
// m16n8k16 s8 x s8 -> s32 gives exact integer dots per 16/32 elements,
// and a float epilogue applies the per-row and per-token factors.
// ===========================================================================

__device__ static inline int warp_isum_bfly(int v, int width) {
    for (int o = width / 2; o > 0; o >>= 1) v += __nvvm_shfl_sync_bfly_i32(0xffffffffu, v, o, 31);
    return v;
}

__device__ static inline float warp_fmax_bfly(float v, int width) {
    for (int o = width / 2; o > 0; o >>= 1) v = __builtin_fmaxf(v, __nvvm_shfl_sync_bfly_f32(0xffffffffu, v, o, 31));
    return v;
}

// Q8_1: 8 lanes per 32-float block, a float4 each. d = f16(amax / 127),
// q = roundf(x * 127 / amax); ys stores {d, sum(q)} for each block.
// The GEMM reconstructs s = d * sum(q). x is [n_tokens][n_in], n_in % 32 == 0.
// Every lane reaches the shuffles; lanes past the end load zeros, store nothing.
#define Q8_LANES 8
KERNEL void quantize_q8_1(const float* x, i8* q8, float* ys, int n_blocks) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = gid / Q8_LANES;
    int sub = gid % Q8_LANES;
    bool live = idx < n_blocks;
    float4 v = {0.0f, 0.0f, 0.0f, 0.0f};
    if (live) v = *(const float4*)(x + (long)idx * 32 + sub * 4);
    float amax = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(v.x), __builtin_fabsf(v.y)),
                                 __builtin_fmaxf(__builtin_fabsf(v.z), __builtin_fabsf(v.w)));
    amax = warp_fmax_bfly(amax, Q8_LANES);
    float d = (float)(_Float16)(amax / 127.0f);
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q0 = (int)__builtin_roundf(v.x * inv), q1 = (int)__builtin_roundf(v.y * inv);
    int q2 = (int)__builtin_roundf(v.z * inv), q3 = (int)__builtin_roundf(v.w * inv);
    int sum = warp_isum_bfly(q0 + q1 + q2 + q3, Q8_LANES);
    if (!live) return;
    unsigned packed = (unsigned)(q0 & 0xFF) | ((unsigned)(q1 & 0xFF) << 8) | ((unsigned)(q2 & 0xFF) << 16) | ((unsigned)(q3 & 0xFF) << 24);
    AT32(q8 + (long)idx * 32 + sub * 4) = packed;
    if (sub == 0) {
        ys[2 * (long)idx] = d;
        ys[2 * (long)idx + 1] = (float)sum;
    }
}

// Signed 4-way byte dot plus accumulator (sm_61+).
__device__ static inline int dp4a(unsigned a, unsigned b, int c) {
    int r;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// gemv_q on Q8_1 activations: same grid, lanes and Q_ROWS as gemv_q, but
// the 32 activations a lane owns are 8 int8 words instead of 32 floats,
// and the dots are dp4a. ys[2 * (t * n_in / 32 + blk)] = {d, sum(q)}.
//
// Q4_K lane c: nibble words qs[32p + 16h, +16) against q8 at 64p + 16h
// (low nibbles, sub-block 2p) and 64p + 32 + 16h (high, 2p + 1). The
// lane with h == 0 owns both sub-blocks' min terms, since sum(q) covers
// the whole 32-element block.
__device__ static inline float q4k_part_dot_q8(const u8* blk, const i8* x, const float* ys, int c) {
    unsigned dd = *(const unsigned*)blk;
    float d = (float)*(const _Float16*)&dd;
    float dmin = (float)((const _Float16*)&dd)[1];
    unsigned w0 = AT32(blk + 4), w1 = AT32(blk + 8), w2 = AT32(blk + 12);
    int p = c >> 1, h = c & 1;
    int s0, m0, s1, m1;
    q4k_scale_min_w(w0, w1, w2, 2 * p, &s0, &m0);
    q4k_scale_min_w(w0, w1, w2, 2 * p + 1, &s1, &m1);
    uint4 q = *(const uint4*)(blk + 16 + 32 * p + 16 * h);
    uint4 xl = AT128(x + 64 * p + 16 * h);
    uint4 xh = AT128(x + 64 * p + 32 + 16 * h);
    const unsigned m = 0x0F0F0F0Fu;
    int lo = dp4a(q.x & m, xl.x, dp4a(q.y & m, xl.y, dp4a(q.z & m, xl.z, dp4a(q.w & m, xl.w, 0))));
    int hi = dp4a((q.x >> 4) & m, xh.x, dp4a((q.y >> 4) & m, xh.y, dp4a((q.z >> 4) & m, xh.z, dp4a((q.w >> 4) & m, xh.w, 0))));
    float dx0 = ys[4 * p], sq0 = ys[4 * p + 1];
    float dx1 = ys[4 * p + 2], sq1 = ys[4 * p + 3];
    float r = d * ((float)s0 * dx0 * (float)lo + (float)s1 * dx1 * (float)hi);
    if (h == 0) r -= dmin * dx0 * (float)m0 * sq0 + dmin * dx1 * (float)m1 * sq1;
    return r;
}

// Q6_K lane c: the 4 runs of 8 elements q6k_part_dot takes, as byte
// words. q + 0x60 then ^ 0x80 recenters each byte from [0, 63] to
// [-32, 31] with no carry between bytes.
__device__ static inline float q6k_part_dot_q8(const u8* blk, const i8* x, const float* ys, int c) {
    int h = c >> 2;
    int l0 = (c & 3) * 8;
    const unsigned short* a16 = (const unsigned short*)(blk + 64 * h + l0);
    const unsigned short* b16 = (const unsigned short*)(blk + 64 * h + l0 + 32);
    const unsigned short* h16 = (const unsigned short*)(blk + 128 + 32 * h + l0);
    const i8* scales = (const i8*)(blk + 192);
    int si = 8 * h + l0 / 16;
    const unsigned m4 = 0x0F0F0F0Fu, m2 = 0x03030303u;
    int dot[4] = {0, 0, 0, 0};
#pragma unroll
    for (int k = 0; k < 2; k++) {
        unsigned A = a16[2 * k] | ((unsigned)a16[2 * k + 1] << 16);
        unsigned B = b16[2 * k] | ((unsigned)b16[2 * k + 1] << 16);
        unsigned H = h16[2 * k] | ((unsigned)h16[2 * k + 1] << 16);
        unsigned v[4] = {
            (A & m4) | ((H & m2) << 4),
            (B & m4) | (((H >> 2) & m2) << 4),
            ((A >> 4) & m4) | (((H >> 4) & m2) << 4),
            ((B >> 4) & m4) | (((H >> 6) & m2) << 4),
        };
#pragma unroll
        for (int g = 0; g < 4; g++) {
            unsigned xw = AT32(x + 128 * h + 32 * g + l0 + 4 * k);
            dot[g] = dp4a((v[g] + 0x60606060u) ^ 0x80808080u, xw, dot[g]);
        }
    }
    float acc = 0.0f;
#pragma unroll
    for (int g = 0; g < 4; g++) acc += (float)scales[si + 2 * g] * ys[2 * (4 * h + g)] * (float)dot[g];
    return half_at(blk + 208) * acc;
}

KERNEL void gemv_q8(const u8* w, const i8* q8, const float* ys, float* y, int n_out, int n_in, int dtype, int block_bytes) {
    int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * Q_ROWS;
    int t = blockIdx.y;
    if (row0 >= n_out) return;
    int lane = threadIdx.x;
    int sub = lane >> 3;
    int c = lane & 7;
    int n_blocks = n_in / 256;
    long row_bytes = (long)n_blocks * block_bytes;
    const i8* xt = q8 + (long)t * n_in;
    const float* yst = ys + 2 * ((long)t * n_in / 32);
    float acc[Q_ROWS] = {};
    for (int b = sub; b < n_blocks; b += 4) {
#pragma unroll
        for (int r = 0; r < Q_ROWS; r++) {
            if (row0 + r >= n_out) continue;
            const u8* blk = w + (row0 + r) * row_bytes + (long)b * block_bytes;
            acc[r] += dtype == DT_Q4K ? q4k_part_dot_q8(blk, xt + b * 256, yst + 16 * b, c)
                                      : q6k_part_dot_q8(blk, xt + b * 256, yst + 16 * b, c);
        }
    }
#pragma unroll
    for (int r = 0; r < Q_ROWS; r++) {
        float v = warp_sum(acc[r]);
        if (lane == 0 && row0 + r < n_out) y[(long)t * n_out + row0 + r] = v;
    }
}

// Block tile: GM rows (4 warps x 16) x GN tokens (4 n-fragments of 8),
// K one 256-element superblock per step. Shared rows are 272 bytes: a
// fragment load (8 rows x 4 words) then hits word 68g + tig, 32 distinct
// banks, where a 256-byte stride would put all 8 rows on one bank.
#define GM 64
#define GN 32
#define GK_PAD 272

// mma.sync m16n8k16 .s8 (PTX ISA "Matrix Fragments for mma.m16n8k16"):
// g = lane / 4, tig = lane % 4. A (row-major 16x16): a0 = row g, k
// [4tig, 4tig+4); a1 = row g+8, same k. B (col-major 16x8): b0 = column
// (token) g, k [4tig, 4tig+4). C: c0/c1 = row g, tokens 2tig, 2tig+1;
// c2/c3 = row g+8, the same tokens. Bytes are little-endian, lowest k first.
__device__ static inline void mma_s8_k16(int* c, unsigned a0, unsigned a1, unsigned b0) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#else
    __builtin_trap();
#endif
}

// y[t][row] = dot(dequant(w[row]), x_hat[t]) for Q6_K, with x
// pre-quantized by quantize_q8_1. 128 threads; grid (ceil(n_out / GM),
// ceil(n_tokens / GN)); n_in % 256 == 0. Q6_K has a scale per 16, so each
// k16 mma gets its own accumulator:
//   coef[r][j] = d sc_j, sum = dx (coef[r][2s] DOT_lo + coef[r][2s+1] DOT_hi).
// The 210-byte blocks are only 2-byte aligned, too narrow for cp.async,
// so this one stages synchronously (unlike gemm_q4k).
KERNEL void gemm_q6k(const u8* w, const i8* q8, const float* ys, float* y, int n_out, int n_in, int n_tokens) {
    __shared__ __attribute__((aligned(16))) i8 Ws[GM][GK_PAD];
    __shared__ __attribute__((aligned(16))) i8 Xs[GN][GK_PAD];
    __shared__ float coef[GM][16];
    __shared__ float xd[GN][8];
    __shared__ float xs[GN][8];

    int m0 = blockIdx.x * GM;
    int n0 = blockIdx.y * GN;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int g = lane >> 2;
    int tig = lane & 3;
    int n_sb = n_in / 256;
    long row_bytes = (long)n_sb * 210;
    int nb = n_in / 32;

    float acc[4][4] = {};

    for (int sb = 0; sb < n_sb; sb++) {
        __syncthreads();
        // Element e = 128h + 32quad + l: low bits from ql[64h + l +
        // 32(quad & 1)] (nibble quad >> 1), high bits qh[32h + l] >> 2quad.
        for (int i = tid; i < GM * 64; i += 128) {
            int r = i >> 6, e0 = (i & 63) * 4;
            unsigned packed = 0;
            if (m0 + r < n_out) {
                const u8* blk = w + (m0 + r) * row_bytes + (long)sb * 210;
                int h = e0 >> 7, quad = (e0 & 127) >> 5, l = e0 & 31;
                const u8* ql = blk + 64 * h + l + 32 * (quad & 1);
                const u8* qh = blk + 128 + 32 * h + l;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    int v = ((ql[k] >> (4 * (quad >> 1))) & 0xF) | (((qh[k] >> (2 * quad)) & 3) << 4);
                    packed |= (unsigned)((v - 32) & 0xFF) << (8 * k);
                }
            }
            AT32(&Ws[r][e0]) = packed;
        }
        for (int i = tid; i < GM * 16; i += 128) {
            int r = i >> 4, j = i & 15;
            float c = 0.0f;
            if (m0 + r < n_out) {
                const u8* blk = w + (m0 + r) * row_bytes + (long)sb * 210;
                c = half_at(blk + 208) * (float)((const i8*)(blk + 192))[j];
            }
            coef[r][j] = c;
        }
        for (int i = tid; i < GN * 16; i += 128) {
            int n = i >> 4, c = i & 15;
            u32x4 v = 0;
            if (n0 + n < n_tokens) v = *(const u32x4*)__builtin_assume_aligned(q8 + (long)(n0 + n) * n_in + sb * 256 + 16 * c, 16);
            *(u32x4*)__builtin_assume_aligned(&Xs[n][16 * c], 16) = v;
        }
        for (int i = tid; i < GN * 8; i += 128) {
            int n = i >> 3, j = i & 7;
            float d = 0.0f, s = 0.0f;
            if (n0 + n < n_tokens) {
                long b = (long)(n0 + n) * nb + sb * 8 + j;
                d = ys[2 * b];
                s = d * ys[2 * b + 1];
            }
            xd[n][j] = d;
            xs[n][j] = s;
        }
        __syncthreads();

        int wr = warp * 16;
#pragma unroll 2
        for (int s = 0; s < 8; s++) {
            int k0 = 32 * s + 4 * tig;
            unsigned a00 = AT32(&Ws[wr + g][k0]);
            unsigned a01 = AT32(&Ws[wr + g + 8][k0]);
            unsigned a10 = AT32(&Ws[wr + g][k0 + 16]);
            unsigned a11 = AT32(&Ws[wr + g + 8][k0 + 16]);
            float cr0 = coef[wr + g][2 * s];
            float cr1 = coef[wr + g + 8][2 * s];
            float mr0 = coef[wr + g][2 * s + 1];
            float mr1 = coef[wr + g + 8][2 * s + 1];
#pragma unroll
            for (int nf = 0; nf < 4; nf++) {
                int tn = 8 * nf;
                unsigned b0 = AT32(&Xs[tn + g][k0]);
                unsigned b1 = AT32(&Xs[tn + g][k0 + 16]);
                int t0 = tn + 2 * tig, t1 = t0 + 1;
                float d0 = xd[t0][s], d1 = xd[t1][s];
                int lo[4] = {0, 0, 0, 0}, hi[4] = {0, 0, 0, 0};
                mma_s8_k16(lo, a00, a01, b0);
                mma_s8_k16(hi, a10, a11, b1);
                acc[nf][0] += d0 * (cr0 * (float)lo[0] + mr0 * (float)hi[0]);
                acc[nf][1] += d1 * (cr0 * (float)lo[1] + mr0 * (float)hi[1]);
                acc[nf][2] += d0 * (cr1 * (float)lo[2] + mr1 * (float)hi[2]);
                acc[nf][3] += d1 * (cr1 * (float)lo[3] + mr1 * (float)hi[3]);
            }
        }
    }

    int r0 = m0 + warp * 16 + g;
#pragma unroll
    for (int nf = 0; nf < 4; nf++) {
        int t0 = n0 + 8 * nf + 2 * tig;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = r0 + (i >> 1) * 8;
            int t = t0 + (i & 1);
            if (r < n_out && t < n_tokens) y[(long)t * n_out + r] = acc[nf][i];
        }
    }
}

// 16-byte global -> shared copy that does not hold the issuing thread;
// src_bytes 0 zero-fills (rows and tokens past the edge). The shared
// operand is a 32-bit shared-window address.
__device__ static inline void cp_async16(void* dst, const void* src, int src_bytes) {
#if __CUDA_ARCH__ >= 800
    asm volatile("{ .reg .u64 a; .reg .u32 s; cvta.to.shared.u64 a, %0; cvt.u32.u64 s, a;"
                 " cp.async.cg.shared.global [s], [%1], 16, %2; }" ::"l"(dst), "l"(src), "r"(src_bytes)
                 : "memory");
#else
    __builtin_trap();
#endif
}

__device__ static inline void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// Waits until at most one committed group is still in flight.
__device__ static inline void cp_async_wait1() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

#define GQ4_ROW 144
#define GY_PAD 20

// Q4_K, per 32-element sub-block s of row r and token t:
//   sum_i (d sc_s q_i - dmin m_s) x_i = d sc_s dx DOT(q, q8) - dmin m_s dx sum(q8)
// DOT is two k16 mmas into one int32 accumulator (|15 * 127 * 32| fits).
// gemm_q6k's tiling, with the next superblock's tiles copied by cp.async
// while the current one is in the mma loop. Everything staged is a raw
// copy: the 144-byte weight blocks (unpacked to nibbles and coefficients
// in registers), the Q8_1 rows and their (d, sum) pairs, so no thread
// waits on global memory between the two barriers. Same grid as gemm_q6k. Shared rows of 144 bytes put a fragment load at word 36g + tig
// (+ const), banks 4g + tig: all 32 distinct. ys rows padded to 20 floats
// so the four token pairs a fragment reads land on banks 8 tig apart.
KERNEL void gemm_q4k(const u8* w, const i8* q8, const float* ys, float* y, int n_out, int n_in, int n_tokens) {
    __shared__ __attribute__((aligned(16))) u8 Wr[2][GM][GQ4_ROW];
    __shared__ __attribute__((aligned(16))) i8 Xs[2][GN][GK_PAD];
    __shared__ __attribute__((aligned(16))) float Ys[2][GN][GY_PAD];

    int m0 = blockIdx.x * GM;
    int n0 = blockIdx.y * GN;
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int g = lane >> 2;
    int tig = lane & 3;
    int n_sb = n_in / 256;
    long row_bytes = (long)n_sb * GQ4_ROW;
    int nb = n_in / 32;

    // Issues superblock sb into stage buffer b: 64 x 9 weight chunks,
    // 32 x 16 activation chunks, 32 x 4 (d, sum) chunks.
    auto stage = [&](int sb, int b) {
        for (int i = tid; i < GM * 9; i += 128) {
            int r = i / 9, c = i - 9 * r;
            bool ok = m0 + r < n_out;
            const u8* src = ok ? w + (m0 + r) * row_bytes + (long)sb * GQ4_ROW + 16 * c : w;
            cp_async16(&Wr[b][r][16 * c], src, ok ? 16 : 0);
        }
        for (int i = tid; i < GN * 16; i += 128) {
            int n = i >> 4, c = i & 15;
            bool ok = n0 + n < n_tokens;
            const i8* src = ok ? q8 + (long)(n0 + n) * n_in + sb * 256 + 16 * c : q8;
            cp_async16(&Xs[b][n][16 * c], src, ok ? 16 : 0);
        }
        {
            int n = tid >> 2, c = tid & 3;
            bool ok = n0 + n < n_tokens;
            const float* src = ok ? ys + 2 * ((long)(n0 + n) * nb + sb * 8) + 4 * c : ys;
            cp_async16(&Ys[b][n][4 * c], src, ok ? 16 : 0);
        }
        cp_async_commit();
    };

    float acc[4][4] = {};
    int wr = warp * 16;
    stage(0, 0);
    for (int sb = 0; sb < n_sb; sb++) {
        int b = sb & 1;
        // The barrier retires every read of buffer b ^ 1 (superblock
        // sb - 1) before it is refilled; the empty group on the last
        // step keeps the wait count uniform.
        __syncthreads();
        if (sb + 1 < n_sb) stage(sb + 1, b ^ 1);
        else cp_async_commit();
        cp_async_wait1();
        __syncthreads();

        const u8* ba = Wr[b][wr + g];
        const u8* bb = Wr[b][wr + g + 8];
        float da = half_at(ba), ma = half_at(ba + 2);
        float db = half_at(bb), mb = half_at(bb + 2);
        unsigned sa0 = AT32(ba + 4), sa1 = AT32(ba + 8), sa2 = AT32(ba + 12);
        unsigned sb0 = AT32(bb + 4), sb1 = AT32(bb + 8), sb2 = AT32(bb + 12);
#pragma unroll
        for (int p = 0; p < 4; p++) {
            // qs byte 32p + e holds element e of sub-block 2p (low
            // nibble) and 2p + 1 (high nibble).
            unsigned qa0 = AT32(ba + 16 + 32 * p + 4 * tig), qa1 = AT32(ba + 32 + 32 * p + 4 * tig);
            unsigned qb0 = AT32(bb + 16 + 32 * p + 4 * tig), qb1 = AT32(bb + 32 + 32 * p + 4 * tig);
#pragma unroll
            for (int h = 0; h < 2; h++) {
                int s = 2 * p + h;
                unsigned a00 = (qa0 >> (4 * h)) & 0x0F0F0F0Fu, a01 = (qb0 >> (4 * h)) & 0x0F0F0F0Fu;
                unsigned a10 = (qa1 >> (4 * h)) & 0x0F0F0F0Fu, a11 = (qb1 >> (4 * h)) & 0x0F0F0F0Fu;
                int sca, mna, scb, mnb;
                q4k_scale_min_w(sa0, sa1, sa2, s, &sca, &mna);
                q4k_scale_min_w(sb0, sb1, sb2, s, &scb, &mnb);
                float cr0 = da * (float)sca, mr0 = ma * (float)mna;
                float cr1 = db * (float)scb, mr1 = mb * (float)mnb;
                int k0 = 32 * s + 4 * tig;
#pragma unroll
                for (int nf = 0; nf < 4; nf++) {
                    int tn = 8 * nf;
                    unsigned b0 = AT32(&Xs[b][tn + g][k0]);
                    unsigned b1 = AT32(&Xs[b][tn + g][k0 + 16]);
                    int t0 = tn + 2 * tig, t1 = t0 + 1;
                    float d0 = Ys[b][t0][2 * s], d1 = Ys[b][t1][2 * s];
                    float s0 = d0 * Ys[b][t0][2 * s + 1], s1 = d1 * Ys[b][t1][2 * s + 1];
                    int c[4] = {0, 0, 0, 0};
                    mma_s8_k16(c, a00, a01, b0);
                    mma_s8_k16(c, a10, a11, b1);
                    acc[nf][0] += cr0 * d0 * (float)c[0] - mr0 * s0;
                    acc[nf][1] += cr0 * d1 * (float)c[1] - mr0 * s1;
                    acc[nf][2] += cr1 * d0 * (float)c[2] - mr1 * s0;
                    acc[nf][3] += cr1 * d1 * (float)c[3] - mr1 * s1;
                }
            }
        }
    }

    int r0 = m0 + wr + g;
#pragma unroll
    for (int nf = 0; nf < 4; nf++) {
        int t0 = n0 + 8 * nf + 2 * tig;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = r0 + (i >> 1) * 8;
            int t = t0 + (i & 1);
            if (r < n_out && t < n_tokens) y[(long)t * n_out + r] = acc[nf][i];
        }
    }
}

#include "decoder_ops.cu"

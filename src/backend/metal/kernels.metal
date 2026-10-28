// The host checks all shapes and bounds;
// guards below cover padded threadgroups. Scalar ABI arrays match metal.rs.
// GGUF Q4_K/Q6_K are unpacked in place, never expanded into resident f32 weights.
#include <metal_stdlib>
#if METAL_MATRIX_TILE
#include <metal_simdgroup_matrix>
#endif
using namespace metal;

kernel void layernorm_f32(
    device const float* w [[buffer(0)]], device const float* bias [[buffer(1)]],
    device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
    constant uint* p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]], uint groups [[simdgroups_per_threadgroup]]
) {
    ulong row = ulong(group) * groups + sg;
    if (row >= p[1]) return;
    ulong base = row * p[0];
    float sum = 0.0f;
    for (ulong i = lane; i < p[0]; i += lanes) sum += x[base + i];
    float mean = simd_sum(sum) / float(p[0]);
    float variance = 0.0f;
    for (ulong i = lane; i < p[0]; i += lanes) {
        float centered = x[base + i] - mean;
        variance += centered * centered;
    }
    float scale = 1.0f / sqrt(simd_sum(variance) / float(p[0]) + as_type<float>(p[2]));
    for (ulong i = lane; i < p[0]; i += lanes)
        y[base + i] = (x[base + i] - mean) * scale * w[i] + (p[3] ? bias[i] : 0.0f);
}

kernel void gelu_f32(
    device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (i >= p[0]) return;
    // Same erf approximation as the CPU oracle (not tanh GELU).
    float z = x[i] * 0.7071067811865475f;
    float t = 1.0f / (1.0f + 0.3275911f * abs(z));
    float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
        + t * (-1.453152027f + t * 1.061405429f))));
    float erf_z = copysign(1.0f - poly * exp(-z * z), z);
    y[i] = 0.5f * x[i] * (1.0f + erf_z);
}

kernel void add_bias_f32(
    device const float* bias [[buffer(0)]], device float* y [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (i < p[0]) y[i] += bias[i % p[1]];
}

kernel void argmax_f32(
    device const float* x [[buffer(0)]], device uint* index [[buffer(1)]],
    constant uint* p [[buffer(2)]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint lanes [[threads_per_simdgroup]]
) {
    if (sg != 0) return;
    float best = -INFINITY;
    uint at = 0xffffffffu;
    bool invalid = false;
    for (ulong i = lane; i < p[0]; i += lanes) {
        float value = x[i];
        invalid = invalid || !isfinite(value);
        if (value > best || (value == best && i < at)) { best = value; at = uint(i); }
    }
    float maximum = simd_max(best);
    uint winner = simd_min(best == maximum ? at : 0xffffffffu);
    bool bad = simd_any(invalid);
    if (lane == 0) index[0] = bad ? 0xffffffffu : winner;
}

// DTYPE_* and dispatch constants are injected by metal/abi.rs.
inline float weight_at(device const uchar* w, ulong index, uint dtype) {
    if (dtype == DTYPE_F32) return reinterpret_cast<device const float*>(w)[index];
    if (dtype == DTYPE_F16) return float(reinterpret_cast<device const half*>(w)[index]);
    if (dtype == DTYPE_BF16) return as_type<float>(uint(reinterpret_cast<device const ushort*>(w)[index]) << 16);
    uint e = uint(index % 256);
    if (dtype == DTYPE_Q4K || dtype == DTYPE_Q5K) {
        device const uchar* b = w + (index / 256) * (dtype == DTYPE_Q4K ? 144 : 176);
        float d = float(*reinterpret_cast<device const half*>(b));
        float dmin = float(*reinterpret_cast<device const half*>(b + 2));
        device const uchar* sc = b + 4;
        uint j = e / 32;
        uint scale, minimum;
        if (j < 4) {
            scale = sc[j] & 63;
            minimum = sc[j + 4] & 63;
        } else {
            scale = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4);
            minimum = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
        }
        uint quant = (b[(dtype == DTYPE_Q4K ? 16 : 48) + (j / 2) * 32 + e % 32] >> ((j & 1) * 4)) & 15;
        if (dtype == DTYPE_Q5K) quant |= ((b[16 + e % 32] >> j) & 1) << 4;
        return (d * float(scale)) * float(quant) - dmin * float(minimum);
    }
    device const uchar* b = w + (index / 256) * 210;
    uint half_block = e / 128;
    uint quadrant = (e % 128) / 32;
    uint lane = e % 32;
    uint lo = b[half_block * 64 + (quadrant & 1) * 32 + lane];
    uint hi = b[128 + half_block * 32 + lane];
    uint quant = ((lo >> ((quadrant / 2) * 4)) & 15)
        | (((hi >> (quadrant * 2)) & 3) << 4);
    int scale = int(reinterpret_cast<device const char*>(b + 192)[e / 16]);
    float d = float(*reinterpret_cast<device const half*>(b + 208));
    return (d * float(scale)) * float(int(quant) - 32);
}

kernel void copy_bytes(
    device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
    constant ulong* p [[buffer(2)]], uint i [[thread_position_in_grid]]
) {
    if (ulong(i) < p[2]) dst[p[1] + i] = src[p[0] + i];
}

kernel void scale_f32(device float* x [[buffer(0)]], constant uint* p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0]) x[i] *= as_type<float>(p[1]);
}

kernel void softcap_f32(device float* x [[buffer(0)]], constant uint* p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < p[0]) { float cap = as_type<float>(p[1]); x[i] = cap * tanh(x[i] / cap); }
}

kernel void gelu_tanh_mul_f32(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint* p [[buffer(3)]], uint i [[thread_position_in_grid]]
) {
    if (i < p[0]) {
        float x = a[i];
        y[i] = 0.5f * x * (1.0f + tanh(0.7978846f * x * (1.0f + 0.044715f * x * x))) * b[i];
    }
}

kernel void rope_split_half(
    device float* x [[buffer(0)]], device const float* factors [[buffer(1)]],
    device const float* frequencies [[buffer(2)]],
    constant uint* p [[buffer(3)]], uint2 id [[thread_position_in_grid]]
) {
    uint half_dim = p[0] / 2;
    if (id.x >= half_dim || id.y >= p[1]) return;
    ulong i = ulong(id.y) * p[0] + id.x, j = i + half_dim;
    float angle = float(p[2]) * frequencies[id.x] / factors[id.x];
    float s = sin(angle), c = cos(angle), a = x[i], b = x[j];
    x[i] = a * c - b * s;
    x[j] = a * s + b * c;
}

kernel void add_f32(
    device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p[0]) y[i] = a[i] + b[i];
}

kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint* p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i < p[0]) {
        float g = gate[i];
        y[i] = (g / (1.0f + exp(-g))) * up[i];
    }
}

kernel void rmsnorm_f32(
    device const float* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint* p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]], uint groups [[simdgroups_per_threadgroup]]
) {
    ulong t = ulong(group) * groups + sg;
    uint dim = p[0];
    if (t >= p[1]) return;
    ulong base = t * dim;
    float sum = 0.0f;
    for (ulong i = lane; i < dim; i += lanes) sum += x[base + i] * x[base + i];
    sum = simd_sum(sum);
    float scale = 1.0f / sqrt(sum / float(dim) + as_type<float>(p[2]));
    for (ulong i = lane; i < dim; i += lanes) y[base + i] = x[base + i] * scale * w[i];
}

// Preserve the residual sum as well as the normalized output. A lane
// rereads only the elements it wrote itself; no cross-lane memory exchange
// is needed beyond the sum-of-squares SIMD reduction.
kernel void add_rmsnorm_f32(
    device const float* w [[buffer(0)]], device const float* a [[buffer(1)]],
    device const float* b [[buffer(2)]], device float* residual [[buffer(3)]],
    device float* y [[buffer(4)]], constant uint* p [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]], uint groups [[simdgroups_per_threadgroup]]
) {
    ulong t = ulong(group) * groups + sg;
    uint dim = p[0];
    if (t >= p[1]) return;
    ulong base = t * dim;
    float sum = 0.0f;
    for (ulong i = lane; i < dim; i += lanes) {
        float value = a[base + i] + b[base + i];
        residual[base + i] = value;
        sum += value * value;
    }
    sum = simd_sum(sum);
    float scale = 1.0f / sqrt(sum / float(dim) + as_type<float>(p[2]));
    for (ulong i = lane; i < dim; i += lanes) y[base + i] = residual[base + i] * scale * w[i];
}

// Interleaved RoPE pairs, as in the CPU implementation (not NeoX).
kernel void rope_f32(
    device float* q [[buffer(0)]], device float* k [[buffer(1)]],
    device const float* frequencies [[buffer(2)]],
    constant uint* p [[buffer(3)]], uint3 id [[thread_position_in_grid]]
) {
    uint head_dim = p[1], q_heads = p[2], k_heads = p[3];
    if (id.x >= head_dim / 2 || id.y >= q_heads + k_heads || id.z >= p[4]) return;
    bool is_q = id.y < q_heads;
    uint heads = is_q ? q_heads : k_heads;
    uint head = is_q ? id.y : id.y - q_heads;
    device float* buf = is_q ? q : k;
    ulong base = (ulong(id.z) * heads + head) * head_dim + 2 * id.x;
    float freq = frequencies[id.x];
    float angle = float(p[0] + id.z) * freq;
    float s = sin(angle), c = cos(angle);
    float a = buf[base], b = buf[base + 1];
    buf[base] = a * c - b * s;
    buf[base + 1] = a * s + b * c;
}

// One SIMD group per output row/token, adjacent lanes load adjacent weights.
// Entire inactive groups return together; no partial-lane early return is
// allowed before simd_sum. Decode and small batches use this path.
kernel void qmatmul(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint* p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]], uint groups [[simdgroups_per_threadgroup]]
) {
    uint rows = p[0], cols = p[1];
    ulong row = ulong(group.x) * groups + sg;
    if (row >= rows || group.y >= p[2]) return;
    float sum = 0.0f;
    for (ulong i = lane; i < cols; i += lanes) {
        sum += weight_at(w, row * cols + i, p[3]) * x[ulong(group.y) * cols + i];
    }
    sum = simd_sum(sum);
    if (lane == 0) y[ulong(group.y) * rows + row] = sum;
}

// 32 output rows x 8 tokens, four SIMD groups. Packed weights are decoded
// only into a bounded threadgroup tile, reused across tokens. Both matrix
// operands and accumulation remain f32 (no half staging).
#if METAL_MATRIX_TILE
kernel void qmatmul_tile(
    device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]], constant uint* p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float wt[TILE_ROWS * TILE_K];
    threadgroup float xt[TILE_K * TILE_TOKENS];
    threadgroup float out[TILE_ROWS * TILE_TOKENS];
    ulong row0 = ulong(group.x) * TILE_ROWS;
    ulong token0 = ulong(group.y) * TILE_TOKENS;
    simdgroup_float8x8 sum = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (ulong k0 = 0; k0 < p[1]; k0 += TILE_K) {
        for (uint i = tid; i < TILE_ROWS * TILE_K; i += TILE_THREADS) {
            ulong r = row0 + i / TILE_K, c = k0 + i % TILE_K;
            wt[i] = r < p[0] && c < p[1] ? weight_at(w, r * p[1] + c, p[3]) : 0.0f;
        }
        for (uint i = tid; i < TILE_K * TILE_TOKENS; i += TILE_THREADS) {
            ulong t = token0 + i % TILE_TOKENS, c = k0 + i / TILE_TOKENS;
            xt[i] = t < p[2] && c < p[1] ? x[t * p[1] + c] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_K; k += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, wt + sg * 8 * TILE_K + k, TILE_K);
            simdgroup_load(b, xt + k * TILE_TOKENS, TILE_TOKENS);
            simdgroup_multiply_accumulate(sum, a, b, sum);
        }
        // All groups must finish reading before the next tile overwrites staging.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(sum, out + sg * 8 * TILE_TOKENS, TILE_TOKENS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < TILE_ROWS * TILE_TOKENS; i += TILE_THREADS) {
        ulong r = row0 + i / TILE_TOKENS, t = token0 + i % TILE_TOKENS;
        if (r < p[0] && t < p[2]) y[t * p[0] + r] = out[i];
    }
}
#endif

// Gate/up matrices may have different packed formats but share a shape.
// Reuse x loads, reduce both dot products, and apply SiLU without storing
// intermediate vectors or submitting two additional command buffers.
kernel void ffn_gate_up(
    device const uchar* gate [[buffer(0)]], device const uchar* up [[buffer(1)]],
    device const float* x [[buffer(2)]], device float* y [[buffer(3)]],
    constant uint* p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]], uint groups [[simdgroups_per_threadgroup]]
) {
    uint rows = p[0], cols = p[1];
    ulong row = ulong(group.x) * groups + sg;
    if (row >= rows || group.y >= p[2]) return;
    float g = 0.0f, u = 0.0f;
    for (ulong i = lane; i < cols; i += lanes) {
        float value = x[ulong(group.y) * cols + i];
        ulong index = row * cols + i;
        g += weight_at(gate, index, p[3]) * value;
        u += weight_at(up, index, p[4]) * value;
    }
    g = simd_sum(g);
    u = simd_sum(u);
    if (lane == 0) y[ulong(group.y) * rows + row] = (g / (1.0f + exp(-g))) * u;
}

kernel void embed_rows(
    device const uchar* table [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint* p [[buffer(3)]],
    uint2 id [[thread_position_in_grid]]
) {
    if (id.x >= p[0] || id.y >= p[1]) return;
    out[ulong(id.y) * p[0] + id.x] = weight_at(table, ulong(ids[id.y]) * p[0] + id.x, p[2]);
}

inline float attn_score(
    device const float* q, device const float* k, ulong q_base, ulong k_base,
    uint dim, float scale
) {
    float dot = 0.0f;
    for (uint d = 0; d < dim; d++) dot += q[q_base + d] * k[k_base + d];
    return dot * scale;
}

// Causal GQA, f32 KV. Three score passes avoid a fixed context limit or
// per-dispatch scratch allocation. One thread owns one entire query head.
kernel void attn_f32(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device float* o [[buffer(3)]],
    constant uint* p [[buffer(4)]], uint2 id [[thread_position_in_grid]]
) {
    uint dim = p[0], q_heads = p[1], kv_heads = p[2];
    if (id.x >= q_heads || id.y >= p[3]) return;
    uint kv_head = id.x / (q_heads / kv_heads);
    ulong kv_row = ulong(kv_heads) * dim;
    ulong kv_base = ulong(kv_head) * dim;
    ulong q_base = (ulong(id.y) * q_heads + id.x) * dim;
    uint visible = p[4] + id.y + 1;
    float scale = as_type<float>(p[5]);
    uint first = p[6] == 0 || p[6] >= visible ? 0 : visible - p[6];
    float maximum = -INFINITY;
    for (uint t = first; t < visible; t++) {
        maximum = max(maximum, attn_score(q, k, q_base, ulong(t) * kv_row + kv_base, dim, scale));
    }
    float sum = 0.0f;
    for (uint t = first; t < visible; t++) {
        sum += exp(attn_score(q, k, q_base, ulong(t) * kv_row + kv_base, dim, scale) - maximum);
    }
    for (uint d = 0; d < dim; d++) o[q_base + d] = 0.0f;
    for (uint t = first; t < visible; t++) {
        ulong base = ulong(t) * kv_row + kv_base;
        float weight = exp(attn_score(q, k, q_base, base, dim, scale) - maximum) / sum;
        for (uint d = 0; d < dim; d++) o[q_base + d] += weight * v[base + d];
    }
}

// One SIMD group per query head. Running max/sum implements stable online
// softmax in one KV pass, without context-sized scratch or global output
// updates per key. Host and shader share the register-capacity limit.
kernel void attn_simd(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device float* o [[buffer(3)]],
    constant uint* p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint lanes [[threads_per_simdgroup]], uint groups [[simdgroups_per_threadgroup]]
) {
    uint dim = p[0], q_heads = p[1], kv_heads = p[2];
    ulong qh = ulong(group.x) * groups + sg;
    if (qh >= q_heads || group.y >= p[3]) return;
    ulong kv_head = qh / (q_heads / kv_heads);
    ulong kv_row = ulong(kv_heads) * dim;
    ulong kv_base = kv_head * dim;
    ulong q_base = (ulong(group.y) * q_heads + qh) * dim;
    uint visible = p[4] + group.y + 1;
    float scale = as_type<float>(p[5]);
    uint first = p[6] == 0 || p[6] >= visible ? 0 : visible - p[6];
    float query[ATTN_ELEMENTS_PER_LANE], accum[ATTN_ELEMENTS_PER_LANE];
    #pragma unroll
    for (uint j = 0; j < ATTN_ELEMENTS_PER_LANE; j++) {
        uint d = lane + j * lanes;
        query[j] = d < dim ? q[q_base + d] : 0.0f;
        accum[j] = 0.0f;
    }
    float maximum = -INFINITY, sum = 0.0f;
    for (uint t = first; t < visible; t++) {
        ulong base = ulong(t) * kv_row + kv_base;
        float dot = 0.0f;
        #pragma unroll
        for (uint j = 0; j < ATTN_ELEMENTS_PER_LANE; j++) {
            uint d = lane + j * lanes;
            if (d < dim) dot += query[j] * k[base + d];
        }
        float score = simd_sum(dot) * scale;
        float next_max = max(maximum, score);
        float alpha = exp(maximum - next_max);
        float beta = exp(score - next_max);
        sum = sum * alpha + beta;
        maximum = next_max;
        #pragma unroll
        for (uint j = 0; j < ATTN_ELEMENTS_PER_LANE; j++) {
            uint d = lane + j * lanes;
            if (d < dim) accum[j] = accum[j] * alpha + beta * v[base + d];
        }
    }
    #pragma unroll
    for (uint j = 0; j < ATTN_ELEMENTS_PER_LANE; j++) {
        uint d = lane + j * lanes;
        if (d < dim) o[q_base + d] = accum[j] / sum;
    }
}

// Gemma primitives (included at the end of kernels.cu): softcap/scale,
// tanh-GELU, split-half RoPE, windowed attention over the f16 KV cache up
// to 512-wide heads, and Q5_K matvec. Eager-only on the Rust side except
// the elementwise ones.

// tanh to a few f32 ulp without libdevice: odd series near 0 (where
// 1 - e^-2|x| cancels), the exp form elsewhere. NaN propagates.
__device__ static inline float tanh_acc(float x) {
    float a = x < 0.0f ? -x : x;
    if (a < 0.125f) {
        float z = x * x;
        return x * (1.0f + z * (-0.33333333f + z * (0.13333333f + z * -0.053968254f)));
    }
    float e = fexp(-2.0f * a);
    float t = (1.0f - e) / (1.0f + e);
    return x < 0.0f ? -t : t;
}

// op 0: x * value (scale); op 1: tanh(x / value) * value (softcap).
KERNEL void unary_f32(float* x, int n, float value, int op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = op == 0 ? x[i] * value : tanh_acc(x[i] / value) * value;
}

KERNEL void gelu_tanh_mul_f32(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = a[i];
        y[i] = 0.5f * x * (1.0f + tanh_acc(0.7978846f * x * (1.0f + 0.044715f * x * x))) * b[i];
    }
}

// x is whole heads of hd; pair j rotates (j, j + hd/2). fr holds the host's
// theta^(-2j/hd), so the angle is the CPU backend's f32 expression.
KERNEL void rope_split_half_f32(float* x, const float* factors, int n, int hd, unsigned pos, Freqs fr) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= n / 2) return;
    int half = hd / 2, j = pair % half;
    long i = (long)(pair / half) * hd + j;
    float angle = (float)pos * fr.f[j] / factors[j];
    float s, c;
    sincos_acc(angle, &s, &c);
    float a = x[i], b = x[i + half];
    x[i] = a * c - b * s;
    x[i + half] = a * s + b * c;
}

// Split-half RoPE over the first rotary_dim values of each head. Under CUDA
// graph replay pp holds (token, position); the host-filled Freqs table uses
// rotary_dim, matching the CPU operation's exponent denominator.
KERNEL void rope_partial_split_half_f32(float* x, int n, int hd, int rotary_dim, unsigned pos, Freqs fr, const unsigned* pp) {
    if (pp) pos = pp[1];
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs_per_head = rotary_dim / 2;
    int total = (n / hd) * pairs_per_head;
    if (pair >= total) return;
    int half = pairs_per_head;
    int j = pair % half;
    long i = (long)(pair / half) * hd + j;
    float angle = (float)pos * fr.f[j];
    float s, c;
    sincos_acc(angle, &s, &c);
    float a = x[i], b = x[i + half];
    x[i] = a * c - b * s;
    x[i + half] = a * s + b * c;
}

KERNEL void sigmoid_mul_f32(const float* gate, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / (1.0f + fexp(-gate[i]));
}

// Q8_0 matvec, one warp per output row. Each 32-element block stores an f16
// scale followed by 32 signed bytes. This path is used by Qwen3.5's alpha and
// beta projections, whose weights are commonly Q8_0 rather than K-quantized.
KERNEL void gemv_q8_0(const u8* w, const float* x, float* y, int n_out, int n_in) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int t = blockIdx.y;
    if (row >= n_out) return;
    const float* xt = x + (long)t * n_in;
    const u8* wr = w + (long)row * (n_in / 32) * 34;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += 32) {
        const u8* block = wr + (i / 32) * 34;
        acc += half_at(block) * (float)((const i8*)(block + 2))[i % 32] * xt[i];
    }
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[(long)t * n_out + row] = acc;
}

// One token of the causal depthwise convolution used by Qwen3.5 DeltaNet.
// Each channel has an independent filter and state window, so one thread can
// update one channel without a block-wide barrier.
KERNEL void ssm_conv_silu_f32(const float* w, float* state, const float* x, float* y, int channels, int taps) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const float* wc = w + (long)channel * taps;
    float* past = state + (long)channel * (taps - 1);
    float sum = x[channel] * wc[taps - 1];
    for (int i = 0; i < taps - 1; i++) {
        sum += past[i] * wc[i];
        if (i + 1 < taps - 1) past[i] = past[i + 1];
    }
    past[taps - 2] = x[channel];
    y[channel] = sum / (1.0f + fexp(-sum));
}

#define DELTA_MAX_HEAD_DIM 256

// Approximate softplus without libdevice. The log2 intrinsic is available in
// the CUDA device-only clang build; fexp is the same fast exponential used by
// the rest of the backend. The branch keeps the positive large-x path stable.
__device__ static inline float softplus_acc(float x) {
    if (x > 20.0f) return x;
    float e = x > 0.0f ? fexp(-x) : fexp(x);
    float log1p = __nvvm_lg2_approx_ftz_f(1.0f + e) * 0.6931471805599453f;
    return x > 0.0f ? x + log1p : log1p;
}

// One block per value head for one token of the gated DeltaNet recurrence.
// q/k normalization is shared by the value heads that map to the same key
// head only conceptually; each block keeps its normalized vectors in shared
// memory so the d x d state update can reuse them for every row.
KERNEL void gated_delta_f32(
    const float* qkv,
    const float* alpha,
    const float* beta,
    const float* dt_bias,
    const float* a,
    float* state,
    float* o,
    int n_k_heads,
    int n_v_heads,
    int head_dim,
    float eps
) {
    int h = blockIdx.x;
    int tid = threadIdx.x;
    if (h >= n_v_heads || head_dim > DELTA_MAX_HEAD_DIM) return;
    int kh = h % n_k_heads;
    __shared__ float qn[DELTA_MAX_HEAD_DIM];
    __shared__ float kn[DELTA_MAX_HEAD_DIM];
    __shared__ float red[DELTA_MAX_HEAD_DIM];

    if (tid < head_dim) {
        qn[tid] = qkv[(long)kh * head_dim + tid];
        kn[tid] = qkv[(long)(n_k_heads + kh) * head_dim + tid];
    } else {
        qn[tid] = 0.0f;
        kn[tid] = 0.0f;
    }
    red[tid] = qn[tid] * qn[tid];
    __syncthreads();
    for (int step = DELTA_MAX_HEAD_DIM / 2; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    float qscale = 1.0f / __nvvm_sqrt_rn_f(red[0] + eps);
    if (tid < head_dim) qn[tid] *= qscale;
    red[tid] = kn[tid] * kn[tid];
    __syncthreads();
    for (int step = DELTA_MAX_HEAD_DIM / 2; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    float kscale = 1.0f / __nvvm_sqrt_rn_f(red[0] + eps);
    if (tid < head_dim) kn[tid] *= kscale;
    __syncthreads();

    float gate = alpha[h] + dt_bias[h];
    float decay = fexp(softplus_acc(gate) * a[h]);
    float rate = 1.0f / (1.0f + fexp(-beta[h]));
    float out_scale = 1.0f / __nvvm_sqrt_rn_f((float)head_dim);
    const float* v = qkv + (long)(2 * n_k_heads + h) * head_dim;
    float* sh = state + (long)h * head_dim * head_dim;
    float* oh = o + (long)h * head_dim;

    for (int row = 0; row < head_dim; row++) {
        float dot = 0.0f;
        for (int col = tid; col < head_dim; col += blockDim.x) {
            long index = (long)row * head_dim + col;
            sh[index] *= decay;
            dot += sh[index] * kn[col];
        }
        red[tid] = dot;
        __syncthreads();
        for (int step = DELTA_MAX_HEAD_DIM / 2; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            __syncthreads();
        }
        float delta = (v[row] - red[0]) * rate;
        for (int col = tid; col < head_dim; col += blockDim.x) {
            long index = (long)row * head_dim + col;
            sh[index] += kn[col] * delta;
        }
        __syncthreads();
        float out_dot = 0.0f;
        for (int col = tid; col < head_dim; col += blockDim.x) {
            out_dot += sh[(long)row * head_dim + col] * qn[col];
        }
        red[tid] = out_dot;
        __syncthreads();
        for (int step = DELTA_MAX_HEAD_DIM / 2; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            __syncthreads();
        }
        if (tid == 0) oh[row] = red[0] * out_scale;
        __syncthreads();
    }
}

#define DW_THREADS 128
#define DW_CHUNK 256
#define DW_MAX_HEAD_DIM 512

// One 128-thread block per (query head, query). Keys stream in chunks of
// DW_CHUNK under an online softmax, so shared memory is fixed whatever the
// context. A warp per key for q.k (coalesced f16 row reads); thread d owns
// output dims d, d+128, d+256, d+384 for the V sum.
KERNEL void attn_windowed_f16kv(const float* q, const _Float16* k, const _Float16* v, float* o, int hd, int qheads, int kvheads, int prefix, float scale, int window) {
    __shared__ float qs[DW_MAX_HEAD_DIM], sc[DW_CHUNK], red[DW_THREADS];
    int lid = threadIdx.x, lane = lid & 31, warp = lid >> 5;
    int qh = blockIdx.x, qi = blockIdx.y;
    int kvh = qh / (qheads / kvheads);
    long qbase = ((long)qi * qheads + qh) * hd;
    long kvrow = (long)kvheads * hd;
    int visible = prefix + qi + 1;
    int start = window == 0 || visible <= window ? 0 : visible - window;
    for (int d = lid; d < hd; d += DW_THREADS) qs[d] = q[qbase + d];
    __syncthreads();
    float m = -__builtin_inff(), sum = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int c0 = start; c0 < visible; c0 += DW_CHUNK) {
        int n = visible - c0 < DW_CHUNK ? visible - c0 : DW_CHUNK;
        for (int t = warp; t < n; t += DW_THREADS / 32) {
            const _Float16* kr = k + (long)(c0 + t) * kvrow + (long)kvh * hd;
            float dot = 0.0f;
            for (int d = lane; d < hd; d += 32) dot += qs[d] * (float)kr[d];
            dot = warp_sum(dot);
            if (lane == 0) sc[t] = dot * scale;
        }
        __syncthreads();
        float local = -__builtin_inff();
        for (int t = lid; t < n; t += DW_THREADS) local = sc[t] > local ? sc[t] : local;
        red[lid] = local;
        __syncthreads();
        for (int s = DW_THREADS / 2; s > 0; s >>= 1) {
            if (lid < s && red[lid + s] > red[lid]) red[lid] = red[lid + s];
            __syncthreads();
        }
        float next = red[0] > m ? red[0] : m;
        float corr = fexp(m - next);
        __syncthreads();
        sum *= corr;
        for (int j = 0; j < 4; j++) acc[j] *= corr;
        local = 0.0f;
        for (int t = lid; t < n; t += DW_THREADS) {
            sc[t] = fexp(sc[t] - next);
            local += sc[t];
        }
        red[lid] = local;
        __syncthreads();
        for (int s = DW_THREADS / 2; s > 0; s >>= 1) {
            if (lid < s) red[lid] += red[lid + s];
            __syncthreads();
        }
        sum += red[0];
        for (int t = 0; t < n; t++) {
            const _Float16* vr = v + (long)(c0 + t) * kvrow + (long)kvh * hd;
            float p = sc[t];
            for (int j = 0; j < 4; j++) {
                int d = lid + DW_THREADS * j;
                if (d < hd) acc[j] += p * (float)vr[d];
            }
        }
        __syncthreads();
        m = next;
    }
    for (int j = 0; j < 4; j++) {
        int d = lid + DW_THREADS * j;
        if (d < hd) o[qbase + d] = acc[j] / sum;
    }
}

// Q5_K matvec, one warp per row, element-wise dequant: Gemma's only Q5_K
// tensor is an embedding table, so this is for completeness, not speed.
KERNEL void gemv_q5k(const u8* w, const float* x, float* y, int n_out, int n_in) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int t = blockIdx.y;
    if (row >= n_out) return;
    const float* xt = x + (long)t * n_in;
    const u8* wr = w + (long)row * (n_in / 256) * BLOCK_Q5K;
    float acc = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += 32) acc += q5k_at(wr + (i / 256) * BLOCK_Q5K, i % 256) * xt[i];
    acc = warp_sum(acc);
    if (threadIdx.x == 0) y[(long)t * n_out + row] = acc;
}

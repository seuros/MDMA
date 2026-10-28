extern "C" __global__ void hybrid_layernorm_kernel(
    const float* w, const float* bias, const float* x, float* y,
    unsigned int width, float eps, unsigned int has_bias
) {
    __shared__ float reduction[256];
    unsigned int tid = threadIdx.x;
    size_t base = (size_t)blockIdx.x * width;
    float sum = 0.0f;
    for (size_t i = tid; i < width; i += 256) sum += x[base + i];
    reduction[tid] = sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    float mean = reduction[0] / (float)width;
    __syncthreads();
    float variance = 0.0f;
    for (size_t i = tid; i < width; i += 256) {
        float centered = x[base + i] - mean;
        variance += centered * centered;
    }
    reduction[tid] = variance;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    float scale = 1.0f / sqrtf(reduction[0] / (float)width + eps);
    for (size_t i = tid; i < width; i += 256)
        y[base + i] = (x[base + i] - mean) * scale * w[i] + (has_bias ? bias[i] : 0.0f);
}

extern "C" __global__ void hybrid_gelu_kernel(const float* x, float* y, unsigned int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.7071067811865475f));
}

extern "C" __global__ void hybrid_argmax_kernel(const float* x, unsigned int* index, unsigned int n) {
    __shared__ float values[256];
    __shared__ unsigned int indices[256], invalid[256];
    unsigned int tid = threadIdx.x;
    float best = -INFINITY;
    unsigned int at = 0xffffffffu, bad = 0;
    for (size_t i = tid; i < n; i += 256) {
        float v = x[i];
        bad |= !isfinite(v);
        if (v > best || (v == best && i < at)) { best = v; at = (unsigned int)i; }
    }
    values[tid] = best; indices[tid] = at; invalid[tid] = bad;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            unsigned int other = tid + stride;
            invalid[tid] |= invalid[other];
            if (values[other] > values[tid]
                || (values[other] == values[tid] && indices[other] < indices[tid])) {
                values[tid] = values[other]; indices[tid] = indices[other];
            }
        }
        __syncthreads();
    }
    if (tid == 0) index[0] = invalid[0] ? 0xffffffffu : indices[0];
}

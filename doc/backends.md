# Backend build details and execution modes

## Build details

The HIP kernels (`src/backend/rocm/kernels/kernels.hip`) are compiled by
`build.rs` with `hipcc` and embedded into the binary.

Vulkan's elementwise, embedding and RoPE launch sizes and unary-operation tags
live in `src/backend/vulkan/abi.rs`; dtype tags live in
`src/backend/dtype_code/tags.rs`. `build.rs` passes them to GLSL as compiler
defines, so host dispatch and shader compilation use the same values.

Metal uses `objc2-metal` and compiles embedded `src/backend/metal/kernels.metal`
source at backend startup, caching the compute pipelines. It does not need
MLX, Python, or the standalone `xcrun metal` compiler. Building Rust's native
dependencies still requires the macOS Command Line Tools and CMake.
Metal's host/shader constants live in `src/backend/metal/abi.rs`; shader
dtype defines come from the shared `dtype_code/tags.rs`.
Public backend operations and `Model::forward` are synchronous.
`Backend::decoder` creates a private queue/batch for each request, retaining
temporary buffers and joining GPU work before logits downloads or errors.
It is command batching, not graph replay. Set `MDMA_METAL_BATCH=0` to use
the eager decoder instead. Serving and `mdma-run` use the decoder interface.

On Apple-family GPUs supporting the matrix path, batches of at least eight
tokens use 32-row × 8-token tiles with f32 staging and accumulation.
Decode and small matrices retain the SIMD-group matvec path.
Set `MDMA_METAL_TILE=0` before backend initialization to disable tiling.
Llama prefill is batched; Gemma prefill remains sequential.
Unsupported weight types have no CPU fallback.
RoPE frequencies are cached, host-rounded model constants; rotations and
all activation computation stay on Metal.
Binary elementwise kernels accept extra output capacity without modifying
the unused tail, matching the CPU buffer contract.
The feature has no Apple dependencies on non-macOS targets, where selecting
`MDMA_BACKEND=metal` reports an explicit platform error.

## CPU matmul

The CPU dequantizes each weight row to f32 and sums its products in 16 fixed
lanes: deterministic on every machine and thread count, and vectorized by the
compiler. `MDMA_CPU_MATMUL=packed` (default `exact`) instead quantizes each
activation once per matmul to the weight type's dot type and multiplies rows
in their blocks with Candle's kernels, several times faster on Q4_K_M models
and not the same logits. `mdma-serve`, `mdma-run`, `mdma-bench` and `mdma`
read it at startup, refuse other values, and fix it for the process; the CPU
execution profile names it, so federation peers see which one runs. Build with
`RUSTFLAGS="-C target-cpu=native"` (as `just install` does) for the AVX2/NEON
kernels.

A binary built for a target's baseline still runs wider vectors where the
machine has them: at startup, `backend::cpu::init` detects the instruction
sets once and picks the exact path's dot and fused Q4_K kernels for them, the
`x86-64-v4` (AVX-512), `x86-64-v3` (AVX2, FMA, F16C) or baseline variant.
Generation only follows that choice; nothing is detected per request. All variants give the same
bits, so the choice is not part of the execution profile; `mdma info`,
`mdma-bench` and `mdma-serve`'s startup log show it. On aarch64 NEON is the
baseline; dotprod, i8mm, fp16, bf16, SVE and SVE2 are reported for kernels
that will use them.

## Execution modes

`Session` uses graph replay on supported HIP decode configurations.
Set `MDMA_HIP_GRAPHS=0` to use the eager fallback. Library callers can use
`Backend::decoder`; direct `Model::forward` remains eager.
`HipBackend::new_f32()` uses f32 activations and KV with eager GPU kernels.
It trades throughput and twice the KV storage for reference accuracy;
`HipBackend::new()` keeps the existing fast path.

On CUDA, `MDMA_CUDA_INT8` picks the K-quant int8 paths: `prefill`
(default, the int8 GEMM from 2 tokens up on sm_80+), `all` (plus the
dp4a decode matvec) or `off` (f32 activations only). CUDA decode replays
a captured graph per token; `MDMA_CUDA_GRAPHS=0` uses the eager path.
Qwen3.5's recurrent state is included in the captured graph and its Q8_0
alpha/beta projections use a dedicated matvec kernel.

For Vulkan, `MDMA_VK_F16_WEIGHTS=1` stages prefill weights as f16; anything
else keeps f32. `VulkanBackend::calibrate_f16` measures speed and logit drift
for a model when you want to decide. FP16 arithmetic and its staging
pipeline are optional: unsupported devices retain f32 staging, even with `1`.
Packed f16 model weights remain supported without FP16 arithmetic.
`MDMA_VK_PROFILE=1` flushes each operation
separately for wall-time profiling.

Vulkan fuses residual addition with RMSNorm by default. Eligible prefill
attention uses f32 K/V tiles shared across eight queries, without matrix
instructions. `MDMA_VK_FUSED_NORM=0` and `MDMA_VK_TILED_ATTN=0` restore the
separate/original kernels. Decode attention and heads wider than 128 keep
the original path.
See [selection, correctness gates and benchmarking](vulkan-fusion.md).

`MDMA_VK_INT8=1` opts into experimental Q4_K/Q6_K int8 prefill: cooperative
matrices on RADV RDNA3/4, otherwise accelerated packed dot products
(RDNA2-compatible). `MDMA_VK_INT8=dot` forces the packed-dot path.
It requantizes activations to Q8_1; decode and other weight types keep
their existing paths. It overrides f16 staging and skips its load-time
calibration; unsupported devices fall back. RDNA2 hardware validation
is pending.
Int8 remains off by default.
See [validation, limits and attribution](vulkan-int8.md).

Library users can opt into `Session::with_prefix_reuse()`. It is off by
default and not enabled by the gRPC server. Reuse a session only within a
single trust boundary: cache-hit timing can reveal prompt reuse, and
cached prompt KV remains resident. Each session retains at most one idle
cache, bounded by `n_ctx`; concurrent requests use separate caches.
`Session::clear_prefix_cache()` evicts the idle cache and prevents
in-flight requests from restoring it. This is session-local reuse, not
shared KV pages or cross-request batching.

K-quant decode and prefill use Q8_1 activations: f16-defined scales and
exact integer block sums, with FP32 accumulation. Activation rounding
differs from the FP32 reference; requantized-reference tests and the stored
MiniCheck greedy-continuation gates pass, without claiming parity on every prompt.

## Coral Edge TPU

The `coral` feature drives a Coral PCIe or M.2 card through the `apex` and
`gasket` drivers (FreeBSD 16 and Linux); `MDMA_CORAL_DEVICE` names the node
(`/dev/apex0` on FreeBSD, `/dev/apex_0` on Linux by default). Every program is
generated by mdma from the decoded instruction set; nothing comes from
edgetpu_compiler or libedgetpu at run time. Prefill runs on the CPU; a decoder
puts the model on the chip:

- **Matmuls** (`MDMA_CORAL_MODE=matmul`, the default, for any Llama model whose
  decode matmuls fit in the tiles' memory): q|k|v, o, gate|up and down run on the
  chip with one weight and output scale per compute tile and the input quantized
  over its own range at each call; an output a tile clamped is recomputed on the
  CPU. Attention and norms stay on the CPU, the classifier runs on the CPU in int8.
  The output ranges are calibrated on 8 stories the float model samples and
  cached in `$MDMA_CACHE_DIR/coral/` (default `~/.cache/mdma`).
- **Whole model** (`MDMA_CORAL_MODE=model`) for TinyStories-15M's geometry (dim
  288, FFN 768, 6 heads of 48, RoPE base 10000, at most 256 positions): all layers
  in one program per position count, the RMSNorm weights folded into the next
  matmuls, attention with the softmax on the scalar core, every activation and
  the KV rows in uint8 with one scale per tensor. The host keeps the embedding
  lookup, the final RMSNorm and the int8 classifier. The quantization is
  calibrated the same way; it and the generated programs are cached there too.

On stories15M, teacher-forced on stories the float model wrote (perplexity
3.18 in float): the matmul mode reaches 3.2 and 96% argmax agreement, the
whole model 3.60 and 85%.

## Strict ROCm reference mode

For TinyLlama and Gemma, `mdma-run` also accepts `MDMA_BACKEND=rocm` or `vulkan`
when built with the matching feature. For strict reference accuracy on ROCm:

```sh
MDMA_BACKEND=rocm MDMA_HIP_F32=1 cargo run --release --features rocm --bin mdma-run -- \
  /path/to/gemma-4-E2B-it-Q4_K_M.gguf "The capital of France is" 16
```

`mdma-serve` accepts the same `MDMA_HIP_F32=0|1` switch and freezes the selection at
startup; see the [federation runtime](federation/routing-runtime.md#deployment-notes).

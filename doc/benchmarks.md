# CPU and Metal decoder benchmarking

Use `mdma-bench` to measure model loading, logical native memory, prefill,
and fixed-length greedy decode in isolation:

```sh
cargo build --release --features metal --bin mdma-bench
MDMA_BACKEND=metal /usr/bin/time -l target/release/mdma-bench \
  /path/to/model.gguf 128 16 3
```

Use `MDMA_BACKEND=cpu` for the same CPU workload. For a CPU-only build, omit
`--features metal`. The arguments select 128 prompt tokens, 16 decode steps,
and three timed trials after one complete untimed warmup.

## Method and limits

- The input cycles the tokenization of `"The capital of France is Paris.\n"`.
  Each trial uses fresh KV. KV/workspace allocation is outside throughput
  timers; prefill workspace is released before decode.
- Decode includes greedy selection and logits downloads. EOS does not shorten
  the fixed workload. Gemma prefill is sequential.
- Load includes GGUF reading, tokenizer construction, and weight loading,
  but excludes backend initialization. Filesystem and driver/compiler caches
  are not forced cold.
- Logical native peak is weights + KV + max(prefill workspace, decode
  workspace). It excludes backend-private scratch, driver and host overhead.
  macOS `/usr/bin/time -l` reports process peak RSS separately. Neither is
  total RAM/VRAM usage; do not add the two metrics together.
- These are library measurements, not streaming latency, P95, sampling,
  network latency, or a comparison with another inference engine.

`MDMA_CPU_MATMUL=packed` times the CPU's packed path: each activation is
quantized once per matmul to the weight type's dot type (Q8_K for k-quants,
Q8_0 or Q8_1 otherwise) and Candle's block kernels take the rows, for one
token or a batch. The default `exact` path dequantizes rows to f32, which
matches the pinned llama.cpp checkpoints; requantizing changes logits and can
change greedy tokens within a few dozen steps.

In `exact`, a one-token Q4_K matmul never writes a row out: each weight is
dequantized in registers exactly as Candle does and added to the same 16
lanes, in the same order, so the result is bit-identical to dequantizing the
row, without storing and rereading a row of f32. The direct f32 SIMD
experiment measured below started this; its AVX2-only, differently ordered sum
is no longer a mode. Both
modes can be run in fresh processes with the checked-in runner (the matmul is
fixed once per process):

```sh
RAYON_NUM_THREADS=24 MDMA_CPU_LIST=0-23 \
  tools/bench-cpu-q4k.sh /path/to/Qwen3.5-9B-Q4_K_M.gguf results.txt
```

The runner records the revision, model SHA-256, CPU, thread pinning, compiler
flags, workload, and one JSON result for each of `exact` and `packed`
(`MDMA_CPU_MATMUL=packed`). The fused kernel's gate requires the same bits as
the dequantized row; the packed path uses a `0.5` fixture bound because
activation requantization is approximate:

```sh
cargo test --release --lib \
  backend::cpu::tests::q4k_fused_dot_is_the_exact_path_bit_for_bit_and_packed_stays_close
```

Run one model/backend per fresh process, serially, without competing inference
or builds. Record model checksum, release revision, Rust/driver versions,
hardware, power/thermal conditions, workload, and per-trial durations alongside
results. Run the relevant [correctness gates](testing.md) first.
Model weights are not bundled.

## Intel i7-14700F / RTX 4070 report

The following measurements were collected on 2026-10-16 at revision
`d67aae1`, with Rust `1.99.0`, 24 Rayon threads for CPU runs, and the standard
128-token prompt plus 16 decode steps. CPU binaries used
`RUSTFLAGS="-C target-cpu=native"`; the CUDA run used NVIDIA driver `595.91.07`.
Values are tokens per second; each run had one untimed warmup.

| Model and backend | Prefill | Decode | Timed trials |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B Q4_K_M, CPU | 1.16 | 1.51 | 1 exploratory trial |
| Gemma 4 E2B Q4_K_M, CPU | 6.75 | 5.55 | 3 |
| Gemma 4 E2B Q4_K_M, CUDA | 168.01 | 132.46 | 3 |

Hardware was an Intel i7-14700F (28 logical CPUs) with an NVIDIA RTX 4070.
The Qwen run used the Qwen3.5-9B Q4_K_M file with SHA-256
`03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`; the
Gemma file's SHA-256 was
`740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8`.

## Q4_K CPU path comparison

On the same i7-14700F, using 24 pinned Rayon threads, native release code,
Qwen3.5-9B Q4_K_M (SHA-256
`03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`),
128 prompt tokens, 16 decode steps, one warmup, and three timed trials at
revision `5ca4d44`, the modes measured:

| CPU mode | Prefill tok/s | Decode tok/s | Accuracy status |
| --- | ---: | ---: | --- |
| exact f32 dequantization | 1.741 ± 0.001 | 1.555 ± 0.004 | reference |
| packed Q8_K (`MDMA_CPU_Q4K_VEC_DOT=1`) | 3.568 ± 0.006 | 2.867 ± 0.003 | opt-in; can change logits |
| direct f32 SIMD (`MDMA_CPU_Q4K_F32_DOT=1`) | 2.992 ± 0.004 | 2.483 ± 0.005 | opt-in; max dot drift `2.8e-6` in the fixture |

After the fused Q4_K kernel and runtime ISA selection landed, the same host
measured the current paths at revision `b3e6fa8`:

| CPU mode | Prefill tok/s | Decode tok/s | Accuracy status |
| --- | ---: | ---: | --- |
| exact, fused Q4_K (`x86-64-v3`) | 3.051 ± 0.167 | 2.585 ± 0.175 | bit-identical to the dequantized reference |
| packed (`MDMA_CPU_MATMUL=packed`) | 10.173 ± 1.922 | 8.894 ± 1.411 | opt-in; can change logits |

These runs used the same Qwen3.5-9B Q4_K_M file, 24 pinned Rayon threads,
native release code, a 128-token prompt, 16 decode steps, one warmup, and
three timed trials. The exact path now provides the accuracy-preserving
baseline; the packed path remains the throughput option.

## x86_64 16-thread CPU report

Collected on 2026-10-16 on a 16-thread x86_64 FreeBSD 16-CURRENT host with
Rust `1.99.0`, `RUSTFLAGS="-C target-cpu=native"`, the standard 128-token
prompt plus 16 decode steps, one untimed warmup and three timed trials per
process, two processes per row, alternated. Gemma 4 E4B Q4_K_M, SHA-256
`85a896a047553e842f25297ee5b031d64ff30147d9c4af17b1e4b394cd1fab87`. Values are
tokens per second.

| CPU matmul | Prefill | Decode |
| --- | ---: | ---: |
| Sequential f32 dot (before) | 1.85 | 1.59 |
| `exact`: 16-lane f32 dot | 2.99 | 2.56 |
| `exact`: plus the fused Q4_K kernel | 6.12 | 5.40 |
| Packed Q4_K only (first experiment) | 6.26 | 5.64 |
| `packed`: every quantized type | 10.70 | 8.92 |

The host is an AMD Ryzen 7 5800H (AVX2). The same `exact` path from a default
build without `target-cpu=native`, as `cargo install` makes it:

| Default build | Prefill | Decode |
| --- | ---: | ---: |
| Kernels compiled for SSE2 only | 4.54 | 3.98 |
| `x86-64-v3` kernels picked at startup | 5.78 | 5.08 |

Greedy output of all three builds was identical on every prompt tried.

Greedy output of `exact` matched the sequential dot on every prompt tried, and
the pinned Gemma 4 E2B and TinyLlama CPU checkpoints pass with it. `packed`
matched `exact` on the same four generations (24 to 64 tokens) where Q4_K-only
packing changed tokens within 15.

## Metal batching and tiles

Use the same binary and `MDMA_BENCH_DECODER=executor` for both modes:

```sh
MDMA_BACKEND=metal MDMA_BENCH_DECODER=executor \
  MDMA_METAL_BATCH=1 MDMA_METAL_TILE=1 \
  /usr/bin/time -l target/release/mdma-bench \
  /path/to/model.gguf 128 16 3
```

Set both switches to `0` for the control. Executor setup remains outside the
decode timer; prefill always calls `Model::prefill`. Compare modes with identical
models, prompts, trial counts, and power conditions. A measured kernel or
workload gain does not establish a universal speedup.

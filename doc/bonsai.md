# Bonsai Q1_0 and Gemma CPU validation

MDMA runs both models natively on CPU. The reference implementation is Prism's
llama.cpp fork.

## Artifacts

| Checkpoint | SHA-256 |
|---|---|
| `Bonsai-1.7B-Q1_0.gguf` (248302272 bytes) | `3d7c6c90dd98717a203adb22d5eacd2581850e40aa5327e144b97766cae5f7e3` |
| `gemma-4-E2B-it-Q4_K_M.gguf` | `740185b21d22ceb83a11c3aa62ad5842ef32c70f6096d756bbee85a1e4ec34b8` |

Bonsai source: `prism-ml/Bonsai-1.7B-gguf`, revision
`210a9e99f79cb184909d49595906526eb2b3dd9a`.

## Native implementation

- GGUF v1/v2/v3 header parsing is native, retaining Candle's metadata `Value`
  representation. Counts, lengths, nesting, dimensions, alignment and byte
  arithmetic are checked; duplicate keys/names and malformed UTF-8 are rejected.
- `gguf::GgmlDType` is now MDMA-owned. Existing formats convert from Candle via
  `.into()`, and back via `.candle() -> Option<_>`. Q1_0 has no Candle equivalent.
  Callers previously passing MDMA's re-export directly to Candle quantizers
  must use the explicit conversion.
- Q1_0 (wire ID 41) stays packed: an fp16 scale and 128 LSB-first sign bits,
  18 bytes per block. CPU matmul/embedding decode rows, not the entire model.
  `MDMA_CPU_MATMUL=packed` falls back to the exact row path for this dtype.
  GPU kernel admission rejects Q1_0.
- Bonsai declares `general.architecture=qwen3` with static YaRN, factor 4,
  original context 8192, context 32768 and RoPE base 1000000. The decoder
  implements the correction ramp (default beta-fast 32/beta-slow 1),
  frequency interpolation/extrapolation and attention magnitude.
  Only `rope.scaling.type`, `factor` and `original_context_length` are admitted.
  Other scaling modes/overrides fail preflight rather than being ignored.
- Unscaled GGUF and HF Qwen paths use identity RoPE magnitude.

## Reproduce generation

```sh
export RUSTUP_TOOLCHAIN=1.99.0
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR="$PWD/.tmp/bonsai-validation/target"
cargo build --release --locked --no-default-features --features cli --bin mdma-run

MDMA_BACKEND=cpu MDMA_CACHE=off MDMA_JSON=1 RAYON_NUM_THREADS=2 \
  "$CARGO_TARGET_DIR/release/mdma-run" \
  "$HOME/models/bonsai-1.7b/Bonsai-1.7B-Q1_0.gguf" \
  'The capital of France is' 8

MDMA_BACKEND=cpu MDMA_CACHE=off MDMA_JSON=1 RAYON_NUM_THREADS=2 \
  "$CARGO_TARGET_DIR/release/mdma-run" \
  "$HOME/models/gemma-4-E2B-it-Q4_K_M.gguf" \
  'The capital of France is' 8
```

Observed results, including exact prompt and generated token equality with
the independent reference:

| Model | Prompt tokens | Generated tokens (EOS excluded) | Text |
|---|---|---|---|
| Bonsai | `[785,6722,315,9625,374]` | `[12095,11,323,279,6722,315,17689,374]` | ` Paris, and the capital of Spain is` |
| Gemma | `[2,818,5279,529,7001,563]` | `[9079,236761]` | ` Paris.` |

## Reference generation

Reference: `PrismML-Eng/llama.cpp`, `prism` branch pinned at
`c96baf4abe24d11a50f326e40b6223b687449d63`.

```sh
ref=/path/to/prism-llama.cpp
cmake -S "$ref" -B "$ref/build-mdma-reference" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_CUDA=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$ref/build-mdma-reference" --target llama-completion -j2
mkdir -p .tmp/bonsai-validation
clang++ -std=c++17 -O2 -I"$ref/include" -I"$ref/ggml/include" \
  tools/model-reference.cpp -L"$ref/build-mdma-reference/bin" \
  -Wl,-rpath,"$ref/build-mdma-reference/bin" -lllama \
  -o .tmp/bonsai-validation/model-reference
.tmp/bonsai-validation/model-reference \
  "$HOME/models/bonsai-1.7b/Bonsai-1.7B-Q1_0.gguf" \
  'The capital of France is' 8
```

Repeat with the Gemma path. The helper uses the reference's public C API,
CPU threads 2, context 256, microbatch 1, f32 KV, no flash attention, greedy
argmax with lowest-index tie breaking and special-token parsing.
It prints token JSON only and refuses generation limits outside 1..64.

## Regression checks

The fixture-free CPU suite checks Q1 sign order/scales, malformed rows,
matmul/embedding against independently expanded F32, all existing dtype sizes,
GGUF metadata parity with Candle, malformed headers and YaRN admission/math.
Embedding API and crate-package-content integration tests are included.

```sh
cargo test --locked --no-default-features
MDMA_TEST_BONSAI_GGUF="$HOME/models/bonsai-1.7b/Bonsai-1.7B-Q1_0.gguf" \
RAYON_NUM_THREADS=2 MDMA_CACHE=off \
cargo test --release --locked --no-default-features --lib \
  model::bonsai_tests::bonsai_cpu_checkpoint -- --ignored --exact --test-threads=1
MDMA_TEST_GEMMA4_GGUF="$HOME/models/gemma-4-E2B-it-Q4_K_M.gguf" \
RAYON_NUM_THREADS=2 MDMA_CACHE=off \
cargo test --release --locked --no-default-features --lib \
  model::checkpoint_tests::gemma4_cpu_checkpoint -- --ignored --exact --test-threads=1
```

# Tests

```sh
cargo test --release                       # CPU: unit + kernel tests, no model needed
cargo test --release --features rocm       # + fixture-free HIP kernel tests; requires an AMD GPU
cargo test --features metal --lib backend::metal  # macOS GPU tests; no model fixture needed
```

Coral Edge TPU tests (`tests/coral.rs`) check the generated programs against the
oracle fixtures in `data/fixtures/coral` on any host; on a card they also run
the programs and decode stories15M (the F32 `stories15M.gguf` from llama.cpp's
test models):

```sh
MDMA_CORAL_DEVICE=/dev/apex_0 MDMA_TEST_STORIES15M_GGUF="$HOME/models/stories15M.gguf" \
cargo test --release --features coral --test coral
```

All tests and benchmarks that load external GGUF checkpoints are ignored by default.
The separate native CPU [Bonsai Q1_0 gate](bonsai.md) pins the artifact SHA-256
and compares prompt/greedy token IDs with the Prism reference:

```sh
MDMA_TEST_BONSAI_GGUF="$HOME/models/bonsai-1.7b/Bonsai-1.7B-Q1_0.gguf" \
RAYON_NUM_THREADS=2 MDMA_CACHE=off \
cargo test --release --locked --no-default-features --lib \
  model::bonsai_tests::bonsai_cpu_checkpoint -- --ignored --exact --test-threads=1
```

Synthetic Llama, Gemma, ModernBERT, and Laya-head compatibility tests always run with `cargo test`;
`--features metal`, `rocm`, or `vulkan` adds GPU compatibility tests. Separate optional
TinyLlama/Gemma checkpoint gates use `MDMA_TEST_TINYLLAMA_GGUF` and
`MDMA_TEST_GEMMA4_GGUF`:

```sh
MDMA_TEST_TINYLLAMA_GGUF="$HOME/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" \
MDMA_TEST_GEMMA4_GGUF="$HOME/models/gemma-4-E2B-it-Q4_K_M.gguf" \
cargo test --release --features metal --lib model::checkpoint_tests -- \
  --ignored --test-threads=1
```

Omit `--features metal` for CPU-only tests. The reference is llama.cpp
`552f18f912a32ea86edf82e2b76431cb7131538d`, Metal, greedy, f32 KV, context
2048, microbatch 1, flash attention off. The gates require exact token IDs,
first-step log-probability error below `0.01`, and prefill/decode logit
drift below `0.005`. Fixtures and prompt bytes are in
`src/model/checkpoint_tests.rs`.
Metal checkpoint tests cover both eager and request-owned executors,
including the real 512-token attention window and near-2048 context.
Generated Metal batching/concurrency and mTLS compatibility tests are
always-on with `--features metal`; external checkpoints remain opt-in.

TinyLlama/Gemma GPU checks (run serially with enough free VRAM):

```sh
export MDMA_TEST_TINYLLAMA_GGUF="$HOME/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
export MDMA_TEST_GEMMA4_GGUF="$HOME/models/gemma-4-E2B-it-Q4_K_M.gguf"
cargo test --features vulkan --lib vulkan_checkpoint -- --ignored --test-threads=1
cargo test --features rocm --lib rocm_f32_checkpoint -- --ignored --test-threads=1
cargo test --features rocm --lib gemma4_rocm_greedy_checkpoint -- --ignored --nocapture --test-threads=1
# Known-failing diagnostic, not a passing ROCm correctness gate:
cargo test --features rocm --lib tinyllama_rocm_prefill_decode_diagnostic -- --ignored --nocapture --test-threads=1
cargo test --release --features cuda --lib _cuda_ -- --ignored --test-threads=1
```

The default HIP fast path is not an f32 log-probability parity mode.
TinyLlama's fast-path prefill/decode diagnostic remains ignored and
known-failing; no tolerance is relaxed to make it pass.
Fixture-free tests additionally cover BF16/Q5_K matmul and embedding layouts,
output canaries, grouped-query attention, cached prefixes, 512-token windows,
chunked softmax, and 512-wide heads against the CPU oracle.

CUDA checkpoint gates cover eager execution and graph replay. Its
`tinyllama_cuda_int8_prefill_diagnostic` stays ignored and known-failing:
int8 prefill is not the strict f32 parity path. Gemma decodes eagerly: graph capture
is limited to heads of at most 256, and its 512-wide global heads use the
windowed attention kernel.

Real long-context and two-model mTLS lifecycle gates use the same variables:

```sh
export MDMA_TEST_TINYLLAMA_GGUF="$HOME/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
export MDMA_TEST_GEMMA4_GGUF="$HOME/models/gemma-4-E2B-it-Q4_K_M.gguf"
cargo test --release --features metal --lib model::long_context_tests -- --ignored --test-threads=1
cargo test --release --features metal --lib grpc::tests::native -- --ignored --test-threads=1
cargo test --release --features metal --test checkpoint_cli -- --ignored --test-threads=1
```

The long fixtures check Gemma prefix lengths 511/512/513/528 and TinyLlama
length 2036, continued greedy generation, batched/incremental/sequential
prefill, invalid IDs, and exact KV capacity. Tolerances are unchanged.
`data/fixtures/model/` holds the token inputs and reference samples;
the samples were recorded from the pinned llama-server identified in each fixture.
The metadata gate also checks the Gemma text configuration pinned at
Google revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, including the
GGUF shared-layer FFN and proportional-RoPE transformations.
Fixture-free 512-window and two-architecture mTLS tests always run.

The MiniCheck gates below require **Bespoke-MiniCheck-7B-Q4_K_M.gguf**, not
an arbitrary GGUF: their tokenizer and continuation expectations are model-specific.
Pass its path through `MDMA_TEST_GGUF` and explicitly select ignored tests:

```sh
MDMA_TEST_GGUF=/models/Bespoke-MiniCheck-7B-Q4_K_M.gguf \
  cargo test --release --lib tokenizer::tests -- --ignored

MDMA_TEST_GGUF=/models/Bespoke-MiniCheck-7B-Q4_K_M.gguf \
  cargo test --release --lib model::tests -- --ignored

MDMA_TEST_GGUF="$HOME/models/Bespoke-MiniCheck-7B-Q4_K_M.gguf" \
  cargo test --release --lib session::tests -- --ignored --test-threads=1

MDMA_TEST_GGUF=/models/Bespoke-MiniCheck-7B-Q4_K_M.gguf \
  cargo test --release --features rocm --lib backend::rocm \
  -- --ignored --skip bench_ --test-threads=1
```

Setting the variable alone does not enable ignored tests. Explicit runs
fail if it is unset, empty, or points to a missing file; there is no local
fallback path or automatic download. GPU tests also require the matching
feature, toolchain and hardware. The mTLS transport tests need loopback
network access. Laya parity tests use the separate `MDMA_LAYA_PATH` checkpoint;
see [Laya checkpoint tests](laya.md#checkpoint-tests).

Run benchmarks individually with `--ignored --nocapture --test-threads=1`.
`MDMA_BENCH_T=256` scopes the HIP prefill benchmark to one prompt length.
Run tests from a repository checkout. The published crate excludes test
modules, test helpers, reference fixtures, `doc/`, `deploy/` service templates
and development tools. Compatibility tests remain always-on in the repository;
model weights are not bundled.

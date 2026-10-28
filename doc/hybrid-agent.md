# HybridAgent on CPU and ROCm

The native library and local CLI run the trained 8B checkpoint on CPU and AMD
ROCm. `model::hybrid` provides CPU/GPU layer offloading separately.

For direct use by another Rust application, see the
[owned Engine/Session embedding API](embedding.md). Tool interpretation and
execution belong to the application (`lsd`), not mdma.

## Run locally

Place the complete HF bundle at `$HOME/models/HybridAgent-8B-Agentic`:

```sh
CARGO_BUILD_JOBS=1 cargo build --release --features cli,rocm --bin mdma-ha

RAYON_NUM_THREADS=2 target/release/mdma-ha \
  --model "$HOME/models/HybridAgent-8B-Agentic" --backend rocm \
  --prompt $'<|im_start|>system\nUse <think> for reasoning, <latent> for internal steps, and <answer> for the final answer. No tools are available.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n' \
  --min-latent-steps 4 --max-latent-steps 24 --max-steps 80 --context 256
```

Use `--backend cpu` for CPU, with `RAYON_NUM_THREADS=2` to bound CPU workers.
CPU-only runner builds need `--features cli`, without `rocm`. Run heavy checks serially.
Prompts are raw, already-formatted text; chat templates are not rendered.
The CLI uses greedy text sampling with think/latent masks, handles EOS, and
counts text, continuous steps and exits against the same generation budget.
JSON output reports token IDs, visible text, latent/exit counts and timings.
Prefill/decode timing joins pending backend work.

The runner never executes generated Python or search calls. It stops on a
completed tool request or a fabricated `<result>`. A latent span may remain open
when the context or generation budget runs out.
`--force-latent` is a diagnostic, not normal generation, and requires a prompt
ending in `<think>`.

## Reference

- Paper: [Hybrid Reasoning Agents, arXiv 2610.18861](https://arxiv.org/abs/2610.18861).
- Official implementation: `togethercomputer/HybridAgent`, commit
  `1b41cebbd2388d083ee1820af421b7f9e24affb5`.
- Checkpoint: `togethercomputer/HybridAgent-8B-Agentic`, revision
  `13a5c61b1880d7bcc2fac2e28d478809722beb8f`.

The reference's `hybrid_agent/hybridlink.py` uses:

```
z = LayerNorm_pre(h)
z = GELU_erf(W1 z + b1)
e = LayerNorm_post(h + W2 z + b2)
```

Both LayerNorms have learned scales/biases and epsilon `1e-5`. `h` is the
backbone's last hidden state **after final RMSNorm**, not the raw residual.
Neither RMSNorm in place of LayerNorm nor tanh-GELU is interchangeable here.

In `hybrid_agent/engine.py`, exit is determined by the **unmasked LM logits'
argmax** on the current state, after the minimum latent steps. It is not a cosine
threshold. Cosine nearest-embedding lookup is an optional pseudo-text inspector.
At exit, feed the real `</latent>` embedding before resuming text generation.
The max-step check takes precedence over an anchor exit.

## Native implementation

`model::hybrid_agent::HybridAgent::load_bundle(backend, bundle_directory)` loads:

- The trained dense Qwen3 `model.safetensors` directly. BF16/F16/F32 matrices
  retain their source dtype; norms are converted to F32.
- The original bundle's `config.json`, `tokenizer.json`, `adapter_config.json`
  and `adapter.pt`. Candle's data-only PyTorch reader loads the eight adapter
  tensors; no Python interpreter or remote model code is executed.

The loader checks supported configuration, backbone geometry, vocabulary, RMS
epsilon, RoPE, tensor count/layout and dtypes before uploading. It requires
distinct in-vocabulary anchors and validates adapter names, shapes, contiguous
layouts, supported floating dtypes and finite values.
The backbone must use HybridAgent-trained weights.

The HF loader temporarily reads the safetensors file into host RAM (~16.4 GB for
this checkpoint). GPU loading checks available memory with headroom; KV/scratch
allocation can still fail. Prepared GGUF caches are not used: `MDMA_CACHE=auto`
or `off` is accepted, while `required` is rejected.
The alternative `HybridAgent::load(backend, gguf, bundle_directory)` remains
available for a compatible trained-backbone GGUF.

CPU provides the reference path. ROCm adds native device LayerNorm, erf-form
GELU and stable argmax, reusing device bias addition. ROCm uses
`HipBackend::new_f32()`; the default HIP fast mode is not the HA accuracy path.
Hidden states, HybridLink intermediates,
continuous inputs and KV stay in backend buffers. A latent step downloads
one `u32` exit readout, not a hidden vector or vocabulary-sized logit array.
Nonfinite logits invalidate the trajectory instead of choosing a token.
Text-mode greedy sampling explicitly downloads logits to the host.

### Trajectory API

`agent.decoder(backend, context, LatentLimits { min_steps, max_steps })` owns
one reusable backbone workspace, adapter scratch and a persistent KV cache.
Limits are bounded: `max_steps > 0` and `min_steps <= max_steps`.

- `feed_text(ids)`: append a prompt or externally supplied tool observation.
  All IDs and context capacity are checked before any append. Anchors inside
  external text do not switch generation mode.
- `logits()`: explicitly download current logits for a text-mode sampler.
- `token(id)`: consume one sampled text token; `<latent>` opens a span.
  A sampled `</latent>` is rejected.
- `latent_step()`: continue with HybridLink's continuous embedding, or consume
  the real exit anchor and return `LatentStep::Exit`.
- `position()`: KV positions consumed, including continuous latent positions.
- `finish()`: join pending work without downloading a hidden state or full logits.

Every continuous step and every exit consumes exactly one context position.
External text cannot be inserted into an open latent span. A backend failure
poisons the decoder: a partially updated layer cache must not be reused.
Different trajectories own different workspaces and caches.

## Validation

Small Linux regression checks:

```sh
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 \
  cargo test --lib model::hybrid_agent -- --test-threads=1
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 \
  cargo test --lib model::arch::qwen::hf -- --test-threads=1
CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test --release --features rocm --lib \
  rocm_hybridlink_and_trajectory_match_cpu -- --test-threads=1
```

Checkpoint checks (larger CPU/RAM/VRAM workload; run one at a time):

```sh
MDMA_TEST_HYBRID_AGENT_DIR="$HOME/models/HybridAgent-8B-Agentic" \
  CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test --release --lib \
  released_hybridlink_checkpoint_matches_reference -- --ignored --test-threads=1
MDMA_TEST_HYBRID_AGENT_DIR="$HOME/models/HybridAgent-8B-Agentic" \
  CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=2 cargo test --release --features rocm --lib \
  released_backbone_cpu_rocm_hybrid_trajectory -- --ignored --test-threads=1
```

Tests cover an independent f64/Candle-erf adapter oracle, explicit embedding
versus token input, persistent multi-turn text/latent/observation KV, min/max
exit ordering, context boundaries, nonfinite logits, malformed adapters and
failure invalidation. An instrumented backend checks that continuous steps
allocate/upload nothing and download only four bytes.

The trained 8B backbone's 13-position CPU/ROCm trajectory was checked on an
i9-14900K and Radeon AI PRO R9700/gfx1201. Across two latent spans and subsequent
text/observation continuation, maximum absolute logit error was `0.00029182434`
(tolerance: `5e-3 + 5e-4 * abs(cpu)`).
MDMA uses f32 activations/KV; the upstream engine uses BF16.

Metal validation requires macOS:

```sh
cargo test --features metal --lib backend::metal::tests::hybrid -- --test-threads=1
```

The tests compile the shaders and exercise primitive boundaries,
nonfinite/tie argmax behavior, adapter reference parity and a synthetic hybrid
trajectory against CPU. `--features metal` on Linux does **not** compile or
validate the macOS-gated Metal implementation.

## Remaining work

- Upstream-engine trajectory comparisons, naturally selected latent-span fixtures,
  and broader task evaluation.
- Native Mac shader/runtime validation and request-owned Metal batching.
- Additional sampling and cooperative prefill cancellation.
- Pseudo-text readout, multi-row batching and complete memory admission.

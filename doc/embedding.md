# Embedding mdma in a Rust application

`mdma` is the inference crate. Applications such as `lsd` own tools, routing,
approvals, prompt formatting, persistence and orchestration. No daemon, gRPC
client, sockets or async runtime is required for in-process inference.

## Dependency and features

```toml
[dependencies]
mdma = { version = "0.6", default-features = false, features = ["rocm"] }
```

Omit `rocm` for CPU only. The `cpu` default feature is currently a marker;
the reference CPU backend is available with `default-features = false` too.

| Feature | Surface |
|---|---|
| default (`cpu`) | local inference, loaders, tokenizers, backends, memory planning |
| `rocm`, `metal`, `cuda`, `vulkan`, `level0` | opt-in device backend; native SDK/compiler requirements still apply |
| `cli` | command-line frontends, including `mdma-ha`; does not enable the gRPC service |
| `service` | existing gRPC/TLS, async session, registry and forwarding modules |

`mdma`, `mdma-serve`, `mdma-client`, `mdma-fed` and `mdma-web` binaries require both `cli` and `service`.
The default library dependency does not build Rama/TLS, Tokio, CLI parsing,
service discovery or the protobuf generator. Cargo features are additive: if
another dependency enables `service`, Cargo will include it for that build.

## HybridAgent

```rust,no_run
use mdma::{
    backend::cpu::CpuBackend,
    hybrid_agent::{Engine, LatentLimits, LatentStep, Step},
};

fn main() -> mdma::Result<()> {
    let engine = Engine::load(CpuBackend, "/models/HybridAgent-8B-Agentic")?;
    let mut session = engine.session(512, LatentLimits::default())?;
    session.append("<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n")?;

    let mut tokens = Vec::new();
    for _ in 0..80 {
        match session.step()? {
            Some(Step::Token { token_id, end_of_turn }) => {
                tokens.push(token_id);
                if end_of_turn { break; }
            }
            Some(Step::Latent(LatentStep::Exit { token, .. })) => tokens.push(token),
            Some(Step::Latent(LatentStep::Thought { .. })) => {}
            None => break,
        }
    }
    session.finish()?;
    println!("{}", session.decode(&tokens)?);
    Ok(())
}
```

For ROCm f32 inference, load with:

```rust,no_run
# fn main() -> mdma::Result<()> {
let backend = mdma::backend::rocm::HipBackend::new_f32()?;
let engine = mdma::hybrid_agent::Engine::load(backend, "/models/HybridAgent-8B-Agentic")?;
# Ok(())
# }
```

### Ownership and continuation

- Load once; `engine.session(context, limits)` shares immutable weights and the
  tokenizer while allocating private KV/scratch. Context includes every prompt,
  generated token, continuous thought and exit position.
- Sessions own their handles and can outlive the `Engine`. Dropping a session
  frees its KV; weights/backend stay alive until their last owner is dropped.
- Each `step()` returns at most one generated position. The caller controls
  budgets, cancellation between steps, backpressure and stop boundaries. A
  budget stop can leave a latent span open; continue stepping to close it before
  appending an observation.
- After the application executes a requested tool, call
  `session.append(raw_observation)` (or `append_tokens`) and continue stepping.
  The same complete text-plus-latent KV stays resident. No model reload or replay
  of visible text is needed. mdma does not parse or execute the tool request.
- `Step::Token { end_of_turn: true, .. }` has consumed EOS. Later `step()` calls
  return `None` until a successful append starts another turn.
- `step()` provides deterministic, control-masked greedy sampling. A caller
  implementing another sampler can read `logits()` and consume `token(id)` in
  text mode. Continuous reasoning must still use `step()` in latent mode.
- `finish()` joins queued work without downloading the full vocabulary. Keep
  accumulated token IDs when decoding: one token can be only part of UTF-8.

Calls are blocking/device-enqueuing, not async. Use an application-owned worker
or blocking pool rather than blocking an async executor. `Send`/`Sync` follow the
backend and buffers; the CPU session is checked for both. No worker or global
session table is created by mdma. Pausing is in-memory only: dropping a session
or restarting the process loses its continuous KV.

Context preflight and backend allocation errors are returned as `mdma::Error`.
The application must bound simultaneous resident sessions.
See [checkpoint format, validation and limitations](hybrid-agent.md).

## Other native models

Existing public modules remain available:

- `gguf::GgufModel`, `model::Model` and `backend::Backend` for explicit
  load/prefill/forward and caller-owned KV.
- `inference::generate` for bounded local greedy GGUF generation, including
  optional CPU/GPU layer placement.
- `laya::Predictor` for native typed decisions.

HA's `Engine` uses the HybridAgent continuous-input decoder.

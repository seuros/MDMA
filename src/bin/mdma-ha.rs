//! Bounded, greedy local HybridAgent runner. Generated tools are never executed.
use mdma::backend::{Backend, cpu::CpuBackend};
use mdma::hybrid_agent::{Engine, LatentLimits, LatentStep, ExitReason, Step};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[derive(Debug, usage_rs::Cli)]
#[usage(bin = "mdma-ha")]
struct Args {
    /// Directory containing the trained HF backbone, tokenizer and adapter.
    #[usage(long)]
    model: PathBuf,
    /// Raw, already-formatted checkpoint prompt.
    #[usage(long)]
    prompt: String,
    /// Backend: cpu or rocm (ROCm uses f32 compute/KV).
    #[usage(long, default = "cpu")]
    backend: String,
    /// Maximum generated positions, counting text, latent steps and exits.
    #[usage(long, default = "64")]
    max_steps: usize,
    /// Total KV capacity including the prompt.
    #[usage(long, default = "512")]
    context: usize,
    #[usage(long, default = "4")]
    min_latent_steps: usize,
    #[usage(long, default = "200")]
    max_latent_steps: usize,
    /// Diagnostic: enter latent mode immediately after the raw prompt.
    #[usage(long)]
    force_latent: bool,
}

fn run<B: Backend + 'static>(b: B, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.max_steps == 0 || args.prompt.is_empty() || args.context == 0
        || args.min_latent_steps > args.max_latent_steps || args.max_latent_steps == 0
    { return Err("invalid generation, context or latent limits".into()); }
    if args.force_latent && !args.prompt.trim_end().ends_with("<think>") {
        return Err("--force-latent requires a prompt ending in <think>".into());
    }
    let started = Instant::now();
    eprintln!("loading {} on {}", args.model.display(), b.device_label().unwrap_or_default());
    let agent = Engine::load(b, &args.model)?;
    let loaded_ms = started.elapsed().as_millis();
    let prompt = agent.encode(&args.prompt)?;
    if prompt.is_empty() || prompt.len() >= args.context {
        return Err("prompt must fit the context with room for output".into());
    }
    let mut decoder = agent.session(args.context, LatentLimits {
        min_steps: args.min_latent_steps, max_steps: args.max_latent_steps,
    })?;
    let started = Instant::now();
    decoder.append_tokens(&prompt)?;
    decoder.finish()?;
    let prefill_ms = started.elapsed().as_millis();
    eprintln!("loaded in {loaded_ms} ms; {} prompt positions in {prefill_ms} ms", prompt.len());
    let started = Instant::now();
    let mut tokens = Vec::new();
    let (mut latent, mut anchor_exits, mut forced_exits) = (0, 0, 0);
    let mut reason = "budget";
    for step in 0..args.max_steps {
        if decoder.position() >= args.context { reason = "context"; break; }
        let event = if step == 0 && args.force_latent {
            decoder.token(agent.entry_token())?
        } else {
            let Some(event) = decoder.step()? else { reason = "eos"; break; };
            event
        };
        match event {
            Step::Latent(LatentStep::Thought { .. }) => {
                latent += 1;
                continue;
            }
            Step::Latent(LatentStep::Exit { token, reason, .. }) => {
                tokens.push(token);
                match reason { ExitReason::Anchor => anchor_exits += 1, ExitReason::MaxSteps => forced_exits += 1 }
                continue;
            }
            Step::Token { token_id, end_of_turn } => {
                tokens.push(token_id);
                if end_of_turn { reason = "eos"; break; }
            }
        }
        let text = agent.decode(&tokens)?;
        if text.contains("<result>") { reason = "fabricated_tool_result"; break; }
        if text.contains("</python>") || text.contains("</search>") { reason = "tool_requested"; break; }
    }
    decoder.finish()?;
    let elapsed_ms = started.elapsed().as_millis();
    let text = agent.decode(&tokens)?;
    println!("{}", json!({
        "backend":args.backend, "device":agent.device_label(), "text":text,
        "token_ids":tokens, "latent_steps":latent, "anchor_exits":anchor_exits,
        "forced_exits":forced_exits, "positions":decoder.position(), "prompt_positions":prompt.len(),
        "end_reason":reason, "load_ms":loaded_ms, "prefill_ms":prefill_ms, "decode_ms":elapsed_ms,
    }));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.backend.as_str() {
        "cpu" => run(CpuBackend, &args),
        #[cfg(feature = "rocm")]
        "rocm" => run(mdma::backend::rocm::HipBackend::new_f32()?, &args),
        _ => Err("backend must be cpu, or rocm with --features rocm".into()),
    }
}

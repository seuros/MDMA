// Minimal mTLS gRPC client for a running `mdma-serve`: streams a completion to
// stdout, or prints Introspect when given no prompt. Env, like the server:
//   MDMA_ADDR=host:port            (default localhost:6362; must match the server cert's SAN)
//   MDMA_TLS_CERT / MDMA_TLS_KEY   the client cert/key mdma-certs wrote (client.pem, client-key.pem)
//   MDMA_TLS_CA                    ca.pem
//   MDMA_MAX_TOKENS                default 64
//   MDMA_MODEL                     registry name; empty means the server's default
//   MDMA_READY_TIMEOUT_SECS        default 120; 0 makes one bounded Introspect check
//   MDMA_GREEDY                    1 requests greedy decoding without repetition penalties; default 0
//   MDMA_CHAT                      1 sends the prompt as one user turn in the model's chat format; default 0
//   MDMA_SYSTEM                    with MDMA_CHAT=1, a system message before that turn
//   MDMA_RPC_TIMEOUT_MS            optional 1..300000; a routed model defaults to 300000, the most a route runs
// Usage: mdma-client [prompt words...]
//        mdma-client --decide request.json   ({"state": ..., "questions": {...}} to a Laya model)
//        mdma-client --push PEER FILE [MODEL] (FILE.gguf from the server's models folder to a
//                                              peer, as MODEL; needs a MDMA_ADMIN_SPKI pin)
//        mdma-client --routes PEER [PREFIX]   (route the peer's ready models as PREFIX-MODEL
//                                              for this client, live; needs a MDMA_ADMIN_SPKI pin)
use mdma::grpc::client::mtls_client;
use mdma::grpc::{
    DecideRequest, ImportRoutesRequest, IntrospectRequest, PushRequest, SynthesizeRequest,
};
use mdma::registry::parameters_label;
use rama::error::BoxError;
use std::io::Write;
use tokio_stream::StreamExt;

#[path = "mdma-client/deadline.rs"]
mod deadline;
#[path = "mdma-client/readiness.rs"]
mod readiness;
#[path = "mdma-client/sampling.rs"]
mod sampling;

/// Has the server send FILE.gguf to `peer`, reporting progress on stderr.
async fn push(
    client: &mdma::grpc::client::Client,
    peer: &str,
    file: &str,
    model: String,
) -> Result<(), BoxError> {
    let mut progress = client
        .push(PushRequest {
            peer: peer.into(),
            file: file.into(),
            model,
        })
        .await?
        .into_inner();
    while let Some(report) = progress.next().await {
        let report = report?;
        if !report.installed.is_empty() {
            if report.present {
                println!("{peer} already holds these bytes as {}", report.installed);
            } else {
                println!("pushed {} to {peer}", report.installed);
            }
            return Ok(());
        }
        let gib = |bytes: u64| bytes as f64 / f64::from(1u32 << 30);
        eprintln!(
            "mdma-client: sent {:.2} of {:.2} GiB",
            gib(report.sent),
            gib(report.size)
        );
    }
    Err("the server ended the push without an outcome".into())
}

fn env(name: &str) -> Result<String, BoxError> {
    std::env::var(name).map_err(|_| format!("missing required env var {name}").into())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), BoxError> {
    let addr = std::env::var("MDMA_ADDR").unwrap_or_else(|_| "localhost:6362".to_string());
    let max_tokens: u32 = std::env::var("MDMA_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let ready_timeout = readiness::timeout(|name| std::env::var(name))?;
    let sampler = sampling::config(|name| std::env::var(name))?;
    let rpc_timeout = deadline::timeout(|name| std::env::var(name))?;
    let chat = match std::env::var("MDMA_CHAT").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(std::env::VarError::NotPresent) => false,
        _ => return Err("MDMA_CHAT must be 0 or 1".into()),
    };
    let system = std::env::var("MDMA_SYSTEM").unwrap_or_default();
    if !chat && !system.is_empty() {
        return Err("MDMA_SYSTEM requires MDMA_CHAT=1".into());
    }
    let client = mtls_client(
        &format!("https://{addr}"),
        env("MDMA_TLS_CERT")?.as_ref(),
        env("MDMA_TLS_KEY")?.as_ref(),
        env("MDMA_TLS_CA")?.as_ref(),
    )?;
    let model = std::env::var("MDMA_MODEL").unwrap_or_default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|flag| flag == "--push") {
        let (peer, file, model) = match &args[1..] {
            [peer, file] => (peer, file, String::new()),
            [peer, file, model] => (peer, file, model.clone()),
            _ => return Err("usage: mdma-client --push PEER FILE [MODEL]".into()),
        };
        return push(&client, peer, file, model).await;
    }
    if args.first().is_some_and(|flag| flag == "--routes") {
        let (peer, prefix) = match &args[1..] {
            [peer] => (peer.clone(), String::new()),
            [peer, prefix] => (peer.clone(), prefix.clone()),
            _ => return Err("usage: mdma-client --routes PEER [PREFIX]".into()),
        };
        let routes = client
            .import_routes(ImportRoutesRequest {
                peer,
                prefix,
                client_spki: String::new(),
            })
            .await?
            .into_inner()
            .routes;
        for route in routes {
            println!("{} -> {} {}", route.alias, route.peer_id, route.model);
        }
        return Ok(());
    }
    let decision = match args.as_slice() {
        [flag, path] if flag == "--decide" => {
            let request: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            let state_json = request
                .get("state")
                .ok_or("request needs a \"state\"")?
                .to_string();
            let questions_json = request
                .get("questions")
                .ok_or("request needs a \"questions\" object")?
                .to_string();
            Some(DecideRequest {
                model: model.clone(),
                state_json,
                questions_json,
            })
        }
        _ => None,
    };
    let prompt: String = args.join(" ");
    let selected = if prompt.is_empty() && model.is_empty() {
        None
    } else {
        Some(model.as_str())
    };
    let h = readiness::wait(
        || async { Ok(client.introspect(IntrospectRequest {}).await?.into_inner()) },
        selected,
        ready_timeout,
    )
    .await?;
    let target = if model.is_empty() {
        &h.default_model
    } else {
        &model
    };
    let routed = h
        .models
        .iter()
        .any(|m| &m.name == target && m.host.is_some());
    let rpc_timeout = rpc_timeout.or_else(|| routed.then_some(deadline::MAX));

    if let Some(request) = decision {
        let reply = client
            .decide(deadline::request(request, rpc_timeout)?)
            .await?
            .into_inner();
        let result: serde_json::Value = serde_json::from_str(&reply.result_json)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    if prompt.is_empty() {
        println!(
            "ready={} state={} backend={} default={}",
            h.ready, h.server_state, h.backend, h.default_model
        );
        for m in h.models {
            let size = if m.weight_bytes > 0 {
                format!(" {} MB", m.weight_bytes >> 20)
            } else {
                String::new()
            };
            // From the file header; unknown ones are left out.
            let facts: String = [
                (m.parameters > 0).then(|| format!(" params={}", parameters_label(m.parameters))),
                (!m.quantization.is_empty()).then(|| format!(" q={}", m.quantization)),
                (m.trained_ctx > 0).then(|| format!(" trained={}", m.trained_ctx)),
            ]
            .into_iter()
            .flatten()
            .collect();
            let target = m.host.map_or(m.backend, |host| host.name);
            println!(
                "  {:<20} {:<9} {:<10} ready={} {} leases={} arch={} n_ctx={}{facts}{size}",
                m.name, m.kind, target, m.ready, m.residency, m.leases, m.architecture, m.n_ctx,
            );
        }
        return Ok(());
    }
    let mut stream = client
        .synthesize(deadline::request(
            SynthesizeRequest {
                prompt,
                max_tokens,
                sampler,
                model,
                emit_events: false,
                chat,
                system,
            },
            rpc_timeout,
        )?)
        .await?
        .into_inner();
    let mut out = std::io::stdout();
    let started = std::time::Instant::now();
    let mut n = 0u32;
    while let Some(item) = stream.next().await {
        let token = item?;
        out.write_all(token.text.as_bytes())?;
        out.flush()?;
        n += 1;
    }
    let secs = started.elapsed().as_secs_f64();
    eprintln!(
        "\n[{n} tokens in {secs:.2}s, {:.1} tok/s]",
        f64::from(n) / secs
    );
    Ok(())
}

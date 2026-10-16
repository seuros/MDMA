// Browser UI for a running `mdma-serve`, on loopback. Env:
//   MDMA_WEB_LISTEN                 loopback ip:port (default 127.0.0.1:6364)
//   MDMA_ADDR                       mdma-serve host:port (default localhost:6362; must match its SAN)
//   MDMA_TLS_CERT / MDMA_TLS_KEY    client cert/key, as for mdma-client
//   MDMA_TLS_CA                     ca.pem
//   MDMA_WEB_MAX_STREAMS            concurrent playground generations, 1..8 (default 2)
//   MDMA_WEB_WATCH_MS               status snapshot interval, 250..60000 (default 2000)
//   MDMA_RPC_TIMEOUT_MS             request deadline, 1..300000 (default 300000)
#[path = "mdma-web/auth.rs"]
mod auth;
#[path = "mdma-web/config.rs"]
mod config;
#[path = "mdma-client/deadline.rs"]
mod deadline;
#[path = "mdma-web/leases.rs"]
mod leases;
#[path = "mdma-web/pages.rs"]
mod pages;
#[path = "mdma-web/playground.rs"]
mod playground;
#[path = "mdma-web/sse.rs"]
mod sse;
#[path = "mdma-web/upstream.rs"]
mod upstream;
#[path = "mdma-serve/vars.rs"]
mod vars;

use config::Config;
use pages::View;
use rama::Layer;
use rama::error::BoxError;
use rama::graceful::{Shutdown, ShutdownGuard};
use rama::http::layer::compression::CompressionLayer;
use rama::http::layer::error_handling::ErrorHandler;
use rama::http::server::HttpServer;
use rama::http::service::web::Router;
use rama::http::service::web::extract::{Path, State};
use rama::http::service::web::response::{
    Css, DatastarScript, DatastarSourceMap, Html, IntoResponse, Script,
};
use rama::rt::Executor;
use rama::tcp::server::TcpListener;
use std::sync::Arc;
use std::time::Duration;
use upstream::Upstream;

const SCENE_JS: &str = include_str!("mdma-web/assets/scene.js");
const STYLE_CSS: &str = include_str!("mdma-web/assets/style.css");
const SHUTDOWN_LIMIT: Duration = Duration::from_secs(10);

#[derive(serde::Deserialize)]
struct ViewPath {
    view: View,
}

fn render(view: View, upstream: &Upstream) -> Html<String> {
    Html(pages::page(view, &upstream.snapshot(), &upstream.pin))
}

async fn index(State(upstream): State<Arc<Upstream>>) -> impl IntoResponse {
    render(View::Memory, &upstream)
}

async fn view(
    State(upstream): State<Arc<Upstream>>,
    Path(path): Path<ViewPath>,
) -> impl IntoResponse {
    render(path.view, &upstream)
}

async fn serve(config: Config, guard: ShutdownGuard) -> Result<(), BoxError> {
    let client = mdma::grpc::client::mtls_client(
        &config.server,
        config.tls_cert.as_ref(),
        config.tls_key.as_ref(),
        config.tls_ca.as_ref(),
    )?;
    let rpc_timeout = deadline::timeout(|name| std::env::var(name))?.or(Some(deadline::MAX));
    let executor = Executor::graceful(guard.clone());
    let listener = TcpListener::bind_address(config.listen.to_string(), executor.clone()).await?;
    let bound = listener.local_addr()?;
    let token = auth::generate_token()?;
    let capability = auth::Capability::new(token.clone(), bound.port());
    let pin = mdma::grpc::client::spki_pin(config.tls_cert.as_ref())
        .map_err(|e| format!("MDMA_TLS_CERT {}: {e}", config.tls_cert))?;
    let upstream = Upstream::start(
        client,
        pin,
        rpc_timeout,
        config.max_streams,
        config.watch_interval,
        &guard,
    );

    let router = Router::new_with_state(upstream)
        .with_get("/", index)
        .with_get("/{view}", view)
        .with_get("/sse/status", sse::status)
        .with_post("/playground/run", playground::run)
        .with_post("/playground/stop", playground::stop)
        .with_post("/playground/model", playground::select)
        .with_post("/models/hold", leases::hold)
        .with_post("/models/release", leases::release)
        .with_get("/assets/datastar.js", DatastarScript::default())
        .with_get("/assets/datastar.js.map", DatastarSourceMap::default())
        .with_get("/assets/scene.js", Script(SCENE_JS))
        .with_get("/assets/style.css", Css(STYLE_CSS));
    let app =
        CompressionLayer::new().into_layer(auth::Guard::new(ErrorHandler::new(router), capability));

    eprintln!("mdma-web listening on http://{bound}/?token={token}");
    eprintln!(
        "mdma-web: forwarding to {} (max {} concurrent generations)",
        config.server, config.max_streams
    );
    listener
        .serve(HttpServer::auto(executor).service(Arc::new(app)))
        .await;
    Ok(())
}

fn main() -> Result<(), BoxError> {
    let config = Config::load(|name| std::env::var(name))?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(async {
        let shutdown = Shutdown::new(async {
            rama::graceful::default_signal().await;
            eprintln!("mdma-web stopping");
        });
        let server = serve(config, shutdown.guard());
        let drain = shutdown.shutdown_with_limit(SHUTDOWN_LIMIT);
        tokio::pin!(drain);
        let drained = tokio::select! {
            result = server => {
                result?;
                drain.await
            }
            result = &mut drain => result,
        };
        drained.map_err(|e| format!("mdma-web shutdown deadline exceeded: {e}"))?;
        Ok::<_, BoxError>(())
    });
    runtime.shutdown_timeout(Duration::ZERO);
    result
}

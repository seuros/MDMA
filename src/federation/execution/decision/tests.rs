use super::super::{deadline, ingress::Receipt};
use super::*;
use crate::{
    backend::cpu::CpuBackend,
    federation::{config::ExecuteExport, control::ExecutionEnvelope, tls::tests::Fixture},
    registry::{Options, Registry, Source},
};
use rama::{
    extensions::{Extensions, ExtensionsRef, Ingress},
    http::{HeaderMap, grpc::Code},
};
use std::assert_matches;

fn setup() -> (
    Fixture,
    Service,
    crate::federation::tls::Binding,
    DecideRequest,
) {
    setup_with_budget(NativeMemoryBudget::new(128 * 1024 * 1024))
}

fn setup_with_budget(
    budget: NativeMemoryBudget,
) -> (
    Fixture,
    Service,
    crate::federation::tls::Binding,
    DecideRequest,
) {
    let f = Fixture::new();
    let (binding, _) = f.bindings();
    let dir = crate::laya::compat_tests::checkpoint();
    let options = Options {
        frozen: true,
        ..Options::new(32)
    };
    let mut registry = Registry::new(
        Arc::new(CpuBackend),
        vec![("laya".into(), Source::Laya(dir.path().into()))],
        None,
        options,
    )
    .unwrap();
    assert_eq!(registry.preload(), Vec::new());
    registry.capture_execution_profiles().unwrap();
    let executor: Arc<dyn super::super::admission::Executor> =
        Arc::new(super::super::admission::Native {
            backend: Arc::new(CpuBackend),
            inventory: inventory::Native {
                registry: Arc::new(registry),
                shutdown: None,
                power: None,
                host: Arc::default(),
                max_output_tokens: usize::MAX,
            },
            decision_budget: budget,
        });
    let descriptor = executor.descriptor("laya").unwrap();
    let service = Service::new(f.store.clone(), executor);
    let message = DecideRequest {
        envelope: Some(ExecutionEnvelope {
            origin: "remote.mdma.test".into(),
            destination: "local.mdma.test".into(),
            request_id: vec![51; 16],
            model: "laya".into(),
            artifact_id: descriptor.artifact_id,
            profile_id: descriptor.profile_id,
            timeout_ms: 3000,
        }),
        state_json: r#"{"a": 1}"#.into(),
        questions_json: r#"{"q":{"type":"choice","instructions":"choose","criteria":["a","b"]}}"#
            .into(),
    };
    (f, service, binding, message)
}

fn grant(f: &Fixture, operation: Operation) {
    let mut policy = f
        .store
        .snapshot()
        .unwrap()
        .peer("remote.mdma.test")
        .unwrap()
        .policy
        .clone();
    policy.execute_exports = vec![ExecuteExport {
        model: "laya".into(),
        operation,
    }];
    f.store.approve_policy(policy).unwrap();
}

fn request(
    message: DecideRequest,
    binding: &crate::federation::tls::Binding,
) -> Request<DecideRequest> {
    let request = Request::new(message);
    let ingress = Extensions::new();
    ingress.insert(binding.clone());
    request.extensions().insert(Ingress(ingress));
    let mut headers = HeaderMap::new();
    headers.insert("grpc-timeout", "3000m".parse().unwrap());
    request
        .extensions()
        .insert(Receipt(Ok(deadline::Ingress::capture(&headers).unwrap())));
    request
}

#[tokio::test]
async fn decide_has_no_implicit_permission_from_synthesis_or_catalog() {
    let (f, service, binding, message) = setup();
    assert_eq!(
        service
            .decide(request(message.clone(), &binding))
            .await
            .err()
            .unwrap()
            .code(),
        Code::PermissionDenied
    );
    grant(&f, Operation::Synthesize);
    assert_eq!(
        service
            .decide(request(message.clone(), &binding))
            .await
            .err()
            .unwrap()
            .code(),
        Code::PermissionDenied
    );
    grant(&f, Operation::Decide);
    let done = service
        .decide(request(message.clone(), &binding))
        .await
        .unwrap();
    assert_eq!(done.response.request_id, vec![51; 16]);
    assert_eq!(done.response.executor, "local.mdma.test");
    assert_eq!(done.response.model, "laya");
    assert_eq!(
        done.response.artifact_id,
        message.envelope.as_ref().unwrap().artifact_id
    );
    assert_eq!(
        done.response.profile_id,
        message.envelope.as_ref().unwrap().profile_id
    );
    let result: serde_json::Value = serde_json::from_str(&done.response.result_json).unwrap();
    assert_eq!(result["model"], "laya-rl-agent");
    assert!(result["answers"].get("q").is_some());
    assert_eq!(result["usage"]["output_tokens"], 0);
    // No decoder/default/upstream operation may be substituted.
    assert_eq!(
        service
            .decide(request(message, &binding))
            .await
            .err()
            .unwrap()
            .code(),
        Code::AlreadyExists
    );
}

#[tokio::test]
async fn decide_combined_input_question_count_and_schema_are_sanitized() {
    let (f, service, binding, message) = setup();
    grant(&f, Operation::Decide);
    let cases = [
        ("not JSON sensitive!".into(), message.questions_json.clone()),
        ("{}".into(), "[]".into()),
        (
            "{}".into(),
            r#"{"secret-question-id":{"type":"bad"}}"#.into(),
        ),
        (" ".repeat(65_536), message.questions_json.clone()),
        (
            "{}".into(),
            serde_json::to_string(
                &(0..129)
                    .map(|i| (i.to_string(), serde_json::json!({})))
                    .collect::<serde_json::Map<_, _>>(),
            )
            .unwrap(),
        ),
    ];
    for (index, (state, questions)) in cases.into_iter().enumerate() {
        let mut message = message.clone();
        message.envelope.as_mut().unwrap().request_id = vec![index as u8 + 60; 16];
        message.state_json = state;
        message.questions_json = questions;
        let error = service
            .decide(request(message, &binding))
            .await
            .err()
            .unwrap();
        assert_eq!(error.code(), Code::InvalidArgument);
        assert_eq!(error.message(), "invalid_request");
    }
}

#[tokio::test]
async fn decide_preserves_empty_question_semantics_and_response_lifetime_budget() {
    let (f, service, binding, mut message) = setup();
    grant(&f, Operation::Decide);
    message.questions_json = "{}".into();
    let done = service
        .decide(request(message.clone(), &binding))
        .await
        .unwrap();
    let result: serde_json::Value = serde_json::from_str(&done.response.result_json).unwrap();
    assert_eq!(result["answers"], serde_json::json!({}));
    message.envelope.as_mut().unwrap().request_id = vec![52; 16];
    assert_eq!(
        service
            .decide(request(message.clone(), &binding))
            .await
            .err()
            .unwrap()
            .code(),
        Code::ResourceExhausted
    );
    drop(done);
    message.envelope.as_mut().unwrap().request_id = vec![53; 16];
    assert!(service.decide(request(message, &binding)).await.is_ok());
}

#[tokio::test]
async fn request_ids_are_shared_across_decide_and_synthesize() {
    let (f, service, binding, message) = setup();
    grant(&f, Operation::Decide);
    let done = service
        .decide(request(message.clone(), &binding))
        .await
        .unwrap();
    drop(done);
    grant(&f, Operation::Synthesize);
    let synth = crate::federation::control::SynthesizeRequest {
        envelope: message.envelope,
        prompt: "abc".into(),
        max_tokens: 1,
        sampler: None,
    };
    let request = {
        let req = Request::new(synth);
        let ingress = Extensions::new();
        ingress.insert(binding);
        req.extensions().insert(Ingress(ingress));
        let mut headers = HeaderMap::new();
        headers.insert("grpc-timeout", "3000m".parse().unwrap());
        req.extensions()
            .insert(Receipt(Ok(deadline::Ingress::capture(&headers).unwrap())));
        req
    };
    assert_matches!(service.synthesize(request).map(|_| ()), Err(s) if s.code() == Code::AlreadyExists);
}

#[test]
fn result_serializer_never_exceeds_byte_cap() {
    let value = serde_json::json!({"secret-output": "x".repeat(1024)});
    assert_eq!(
        encode(&value, 12).unwrap_err().code(),
        Code::ResourceExhausted
    );
    let rendered = encode(&value, 2048).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&rendered).unwrap(),
        value
    );
}

#[tokio::test]
async fn actual_laya_workspace_is_reserved_before_launch_and_refunded_on_preparation_drop() {
    let budget = NativeMemoryBudget::new(1024 * 1024);
    let (f, service, _, message) = setup_with_budget(budget.clone());
    grant(&f, Operation::Decide);
    let descriptor = service.executor.descriptor("laya").unwrap();
    let boundary: Boundary = Arc::new(|| true);
    let prepared = service
        .executor
        .prepare_decide(&message, &descriptor, &boundary)
        .unwrap();
    assert!(budget.used() > 0);
    drop(prepared);
    assert_eq!(budget.used(), 0);
    let (f, service, binding, message) = setup_with_budget(NativeMemoryBudget::new(1));
    grant(&f, Operation::Decide);
    assert_eq!(
        service
            .decide(request(message, &binding))
            .await
            .err()
            .unwrap()
            .code(),
        Code::ResourceExhausted
    );
}

struct Held {
    inner: Arc<dyn super::super::admission::Executor>,
    hooks: std::sync::Mutex<
        Option<(
            tokio::sync::oneshot::Sender<()>,
            std::sync::mpsc::Receiver<()>,
        )>,
    >,
    cancel: Arc<std::sync::Mutex<Option<CancellationToken>>>,
}

impl super::super::admission::Executor for Held {
    fn descriptor(&self, model: &str) -> Result<ModelDescriptor, Status> {
        self.inner.descriptor(model)
    }
    fn prepare(
        &self,
        message: &crate::federation::control::SynthesizeRequest,
        descriptor: &ModelDescriptor,
    ) -> Result<super::super::admission::Prepared, Status> {
        self.inner.prepare(message, descriptor)
    }
    fn prepare_decide(
        &self,
        message: &DecideRequest,
        descriptor: &ModelDescriptor,
        boundary: &Boundary,
    ) -> Result<Prepared, Status> {
        let prepared = self.inner.prepare_decide(message, descriptor, boundary)?;
        let Some((entered, release)) = self.hooks.lock().unwrap().take() else {
            return Ok(prepared);
        };
        let observed = self.cancel.clone();
        Ok(Prepared {
            job: Box::new(move |cancel, boundary| {
                *observed.lock().unwrap() = Some(cancel.clone());
                let _ = entered.send(());
                release.recv_timeout(Duration::from_secs(3)).unwrap();
                (prepared.job)(cancel, boundary)
            }),
        })
    }
}

#[tokio::test]
async fn unary_deadline_revocation_and_handler_drop_retain_actual_worker_memory_until_cleanup() {
    for cause in ["deadline", "revoke", "drop"] {
        let budget = NativeMemoryBudget::new(1024 * 1024);
        let (f, service, binding, mut message) = setup_with_budget(budget.clone());
        grant(&f, Operation::Decide);
        if cause == "deadline" {
            message.envelope.as_mut().unwrap().timeout_ms = 100;
        }
        let (entered, receive) = tokio::sync::oneshot::channel();
        let (release, held) = std::sync::mpsc::sync_channel(1);
        let cancel = Arc::new(std::sync::Mutex::new(None));
        let executor = Arc::new(Held {
            inner: service.executor.clone(),
            hooks: std::sync::Mutex::new(Some((entered, held))),
            cancel: cancel.clone(),
        });
        let service = Service::new(f.store.clone(), executor);
        let running = service.clone();
        let req = request(message.clone(), &binding);
        let task = tokio::spawn(async move { running.decide(req).await });
        tokio::time::timeout(Duration::from_secs(2), receive)
            .await
            .unwrap()
            .unwrap();
        assert!(budget.used() > 0);
        match cause {
            "deadline" => assert_eq!(
                task.await.unwrap().err().unwrap().code(),
                Code::DeadlineExceeded
            ),
            "revoke" => {
                f.store.suspend("remote.mdma.test").unwrap();
                assert_eq!(
                    task.await.unwrap().err().unwrap().code(),
                    Code::PermissionDenied
                );
            }
            _ => {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            }
        }
        assert!(cancel.lock().unwrap().as_ref().unwrap().is_cancelled());
        assert!(
            budget.used() > 0,
            "response release cannot free an unfinished job"
        );
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while budget.used() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}

struct WireTransport(
    rama::service::BoxService<
        rama::http::Request,
        rama::http::Response,
        rama::error::extra::OpaqueError,
    >,
);
impl rama::Service<rama::http::Request> for WireTransport {
    type Output = rama::http::Response;
    type Error = Status;
    async fn serve(&self, request: rama::http::Request) -> Result<Self::Output, Self::Error> {
        self.0
            .serve(request)
            .await
            .map_err(|_| Status::unavailable("transport_failed"))
    }
}

async fn wire(
    f: &Fixture,
    executor: Arc<dyn super::super::admission::Executor>,
) -> (
    crate::federation::control::FederationClient<WireTransport>,
    tokio::task::JoinHandle<()>,
) {
    use crate::federation::{
        config::{SigningKey, Window},
        control::{ControlService, FederationClient, Signer},
        signing::KeyPairHandle,
    };
    use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
    use rama::tls::{
        client::TlsClientConfig,
        rustls::{
            client::RustlsClientConfigExt,
            server::{RustlsServerConfigExt, TlsAcceptorLayer},
        },
        server::TlsServerConfig,
    };
    use rama::{
        Layer,
        http::{Body, client::EasyHttpConnectorBuilder, server::HttpServer},
        rt::Executor,
        tcp::server::TcpListener,
    };
    let key = KeyPairHandle::generate().unwrap();
    f.store
        .approve_local_signer(SigningKey {
            key_id: "ed25519:decision-test".into(),
            public_key: URL_SAFE_NO_PAD.encode(key.public_key_bytes()),
            window: Window {
                valid_from_ms: f.now_ms - 60_000,
                valid_until_ms: f.now_ms + 300_000,
            },
        })
        .unwrap();
    let signer = Signer::new(&f.store, "ed25519:decision-test".into(), key).unwrap();
    let service = ControlService::new(f.store.clone(), signer).with_execution(executor);
    let (client_config, server_config) = f.configs();
    let execution = Executor::default();
    let listener = TcpListener::bind_address("127.0.0.1:0", execution.clone())
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    let mut http = HttpServer::new_h2(execution);
    http.h2_mut().set_max_send_buf_size(1);
    http.h2_mut().set_max_concurrent_streams(2);
    let server = super::super::ingress::Connections::new(
        TlsAcceptorLayer::new(
            TlsServerConfig::new().with_modify_rustls_config(move |_| Ok(server_config.clone())),
        )
        .into_layer(crate::federation::tls::BindPeerService::new(
            f.store.clone(),
            f.now_ms + 300_000,
            http.service(super::super::ingress::Gate::new(service.into_server())),
        )),
    );
    let task = tokio::spawn(async move { listener.serve(server).await });
    let tls = TlsClientConfig::new()
        .with_server_name(rama::net::address::Host::from_static("local.mdma.test"))
        .with_modify_rustls_config(move |_| Ok(client_config.clone()));
    let connector = EasyHttpConnectorBuilder::new()
        .with_default_transport_connector()
        .with_default_dns_connector()
        .without_tls_proxy_support()
        .without_proxy_support()
        .with_tls_support_using_rustls(tls)
        .with_default_http_connector::<Body>(Executor::default())
        .without_connection_pool()
        .build_client::<Body, Body, _>();
    (
        FederationClient::new(
            WireTransport(rama::service::BoxService::new(connector)),
            format!("https://{address}").parse().unwrap(),
        ),
        task,
    )
}

#[tokio::test]
async fn actual_mtls_decide_returns_exact_joined_identity_and_does_not_accept_missing_deadline() {
    let (f, service, _, message) = setup();
    grant(&f, Operation::Decide);
    let (client, server) = wire(&f, service.executor.clone()).await;
    assert!(
        matches!(client.decide(message.clone()).await, Err(s) if s.code() == Code::InvalidArgument)
    );
    let mut request = Request::new(message.clone());
    request.try_set_timeout(Duration::from_secs(2)).unwrap();
    let response = client.decide(request).await.unwrap().into_inner();
    assert_eq!(response.request_id, vec![51; 16]);
    assert_eq!(response.executor, "local.mdma.test");
    assert_eq!(response.model, "laya");
    assert_eq!(
        response.profile_id,
        message.envelope.as_ref().unwrap().profile_id
    );
    assert_eq!(
        response.artifact_id,
        message.envelope.as_ref().unwrap().artifact_id
    );
    let result: serde_json::Value = serde_json::from_str(&response.result_json).unwrap();
    assert!(result["answers"].get("q").is_some());
    server.abort();
    let _ = server.await;
}

#[tokio::test]
async fn actual_http2_unary_client_reset_cancels_but_cannot_free_paused_native_worker_memory() {
    let budget = NativeMemoryBudget::new(1024 * 1024);
    let (f, service, _, message) = setup_with_budget(budget.clone());
    grant(&f, Operation::Decide);
    let (entered, receive) = tokio::sync::oneshot::channel();
    let (release, held) = std::sync::mpsc::sync_channel(1);
    let cancel = Arc::new(std::sync::Mutex::new(None));
    let executor = Arc::new(Held {
        inner: service.executor.clone(),
        hooks: std::sync::Mutex::new(Some((entered, held))),
        cancel: cancel.clone(),
    });
    let (client, server) = wire(&f, executor).await;
    let mut request = Request::new(message);
    request.try_set_timeout(Duration::from_secs(2)).unwrap();
    let call = tokio::spawn(async move { client.decide(request).await });
    tokio::time::timeout(Duration::from_secs(2), receive)
        .await
        .unwrap()
        .unwrap();
    assert!(budget.used() > 0);
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(1), async {
        while !cancel.lock().unwrap().as_ref().unwrap().is_cancelled() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert!(budget.used() > 0);
    release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while budget.used() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    server.abort();
    let _ = server.await;
}

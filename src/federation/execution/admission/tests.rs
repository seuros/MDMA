use super::*;
use crate::federation::{
    config::ExecuteExport, control::synthesize_event::Event, tls::tests::Fixture,
};
use rama::http::{HeaderMap, grpc::Code};
use std::time::Duration;
use tokio_stream::StreamExt;

fn setup() -> (Fixture, Service, Binding, SynthesizeRequest) {
    let fixture = Fixture::new();
    let (binding, _) = fixture.bindings();
    let native = crate::federation::inventory::tests::native_service();
    let executor = native.federation_executor();
    let descriptor = executor.descriptor("native").unwrap();
    let service = Service::new(fixture.store.clone(), executor);
    let message = SynthesizeRequest {
        envelope: Some(ExecutionEnvelope {
            origin: "remote.mdma.test".into(),
            destination: "local.mdma.test".into(),
            request_id: vec![1; 16],
            model: "native".into(),
            artifact_id: descriptor.artifact_id,
            profile_id: descriptor.profile_id,
            timeout_ms: 3000,
        }),
        prompt: "abc".into(),
        max_tokens: 3,
        sampler: Some(SamplerConfig {
            temperature: Some(0.),
            ..Default::default()
        }),
    };
    (fixture, service, binding, message)
}

fn grant(fixture: &Fixture, model: Option<&str>) {
    let mut policy = fixture
        .store
        .snapshot()
        .unwrap()
        .peer("remote.mdma.test")
        .unwrap()
        .policy
        .clone();
    policy.execute_exports = model
        .map(|model| ExecuteExport {
            model: model.into(),
            operation: Operation::Synthesize,
        })
        .into_iter()
        .collect();
    fixture.store.approve_policy(policy).unwrap();
}

fn request(message: SynthesizeRequest, binding: &Binding) -> Request<SynthesizeRequest> {
    let request = Request::new(message);
    let extensions = Extensions::new();
    extensions.insert(binding.clone());
    request.extensions().insert(Ingress(extensions));
    let mut headers = HeaderMap::new();
    headers.insert("grpc-timeout", "3000m".parse().unwrap());
    request
        .extensions()
        .insert(Receipt(Ok(super::super::deadline::Ingress::capture(
            &headers,
        )
        .unwrap())));
    request
}

fn failure(service: &Service, message: SynthesizeRequest, binding: &Binding) -> Code {
    match service.synthesize(request(message, binding)) {
        Err(status) => status.code(),
        Ok(_) => panic!("unexpected admission"),
    }
}

#[tokio::test]
async fn no_default_permission_and_permission_precedes_model_lookup() {
    let (f, service, binding, mut message) = setup();
    assert_eq!(
        failure(&service, message.clone(), &binding),
        Code::PermissionDenied
    );
    message.envelope.as_mut().unwrap().model = "secret".into();
    assert_eq!(
        failure(&service, message.clone(), &binding),
        Code::PermissionDenied
    );
    grant(&f, Some("secret"));
    assert_eq!(failure(&service, message, &binding), Code::NotFound);
}

#[tokio::test]
async fn exact_selection_identity_and_required_ingress_are_not_headers() {
    let (f, service, binding, message) = setup();
    grant(&f, Some("native"));
    for (index, field) in [
        "origin",
        "destination",
        "artifact",
        "profile",
        "id",
        "timeout",
    ]
    .into_iter()
    .enumerate()
    {
        let mut changed = message.clone();
        let envelope = changed.envelope.as_mut().unwrap();
        envelope.request_id = vec![index as u8 + 10; 16];
        let expected = match field {
            "origin" => {
                envelope.origin = "other.mdma.test".into();
                Code::Unauthenticated
            }
            "destination" => {
                envelope.destination = "other.mdma.test".into();
                Code::InvalidArgument
            }
            "artifact" => {
                envelope.artifact_id = format!("sha256:{}", "0".repeat(64));
                Code::FailedPrecondition
            }
            "profile" => {
                envelope.profile_id = format!("sha256:{}", "0".repeat(64));
                Code::FailedPrecondition
            }
            "id" => {
                envelope.request_id.clear();
                Code::InvalidArgument
            }
            _ => {
                envelope.timeout_ms = 0;
                Code::InvalidArgument
            }
        };
        assert_eq!(failure(&service, changed, &binding), expected, "{field}");
    }
    let req = Request::new(message);
    // Metadata cannot confer a private transport binding.
    assert!(matches!(service.synthesize(req), Err(s) if s.code() == Code::Unauthenticated));
}

#[tokio::test]
async fn accepted_tokens_completed_then_eof_and_resolved_presence() {
    let (f, service, binding, message) = setup();
    grant(&f, Some("native"));
    let mut stream = service
        .synthesize(request(message.clone(), &binding))
        .unwrap();
    let first = stream.next().await.unwrap().unwrap();
    let Some(Event::Accepted(accepted)) = first.event else {
        panic!("Accepted first");
    };
    assert_eq!(accepted.request_id, vec![1; 16]);
    assert_eq!(accepted.executor, "local.mdma.test");
    assert_eq!(accepted.model, "native");
    assert_eq!(accepted.prompt_tokens, 3);
    assert_eq!(accepted.max_tokens, 3);
    assert_eq!(accepted.n_ctx, 32);
    let resolved = accepted.sampler.unwrap();
    assert_eq!(resolved.temperature, Some(0.));
    assert_eq!(
        resolved.top_p,
        Some(crate::sampler::SamplerConfig::default().top_p)
    );
    let mut count = 0;
    loop {
        match stream.next().await.unwrap().unwrap().event.unwrap() {
            Event::Token(token) => {
                assert_eq!(token.index, count);
                count += 1;
            }
            Event::Completed(completed) => {
                assert_eq!(completed.emitted_tokens, count);
                assert_eq!(completed.request_id, vec![1; 16]);
                assert!(
                    ["eos", "max_tokens", "context_limit"]
                        .contains(&completed.finish_reason.as_str())
                );
                break;
            }
            Event::Accepted(_) => panic!("duplicate admission"),
        }
    }
    assert!(stream.next().await.is_none());
    assert_eq!(
        failure(&service, message.clone(), &binding),
        Code::AlreadyExists
    );
    let mut other = message.clone();
    other.envelope.as_mut().unwrap().request_id = vec![2; 16];
    // Compute has finished, but keeping this response still charges capacity.
    assert_eq!(
        failure(&service, other.clone(), &binding),
        Code::ResourceExhausted
    );
    drop(stream);
    other.envelope.as_mut().unwrap().request_id = vec![3; 16];
    assert!(service.synthesize(request(other, &binding)).is_ok());
}

#[tokio::test]
async fn never_polled_admission_has_fixed_deadline_and_no_success() {
    let (f, service, binding, mut message) = setup();
    grant(&f, Some("native"));
    message.envelope.as_mut().unwrap().timeout_ms = 10;
    let mut stream = service
        .synthesize(request(message.clone(), &binding))
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;
    // Independent expiry releases validating/native reservations, even before
    // this response is polled. Its response-lifetime permit remains charged.
    let claim = service
        .ledger
        .claim(
            "remote.mdma.test",
            RequestId::parse(&[9; 16]).unwrap(),
            &config::Limits::default(),
        )
        .unwrap();
    drop(claim);
    assert_eq!(
        stream.next().await.unwrap().unwrap_err().code(),
        Code::DeadlineExceeded
    );
    assert!(stream.next().await.is_none());
    assert_eq!(failure(&service, message, &binding), Code::AlreadyExists);
}

#[tokio::test]
async fn revocation_between_admission_and_launch_cannot_start_or_complete() {
    let (f, service, binding, message) = setup();
    grant(&f, Some("native"));
    let mut stream = service
        .synthesize(request(message.clone(), &binding))
        .unwrap();
    assert!(matches!(
        stream.next().await.unwrap().unwrap().event,
        Some(Event::Accepted(_))
    ));
    grant(&f, None);
    assert_eq!(
        stream.next().await.unwrap().unwrap_err().code(),
        Code::PermissionDenied
    );
    assert!(stream.next().await.is_none());
    grant(&f, Some("native"));
    assert_eq!(failure(&service, message, &binding), Code::AlreadyExists);
}

#[tokio::test]
async fn unpolled_response_drop_is_terminal_and_frees_prelaunch_charge() {
    let (f, service, binding, mut message) = setup();
    grant(&f, Some("native"));
    let stream = service
        .synthesize(request(message.clone(), &binding))
        .unwrap();
    drop(stream);
    assert_eq!(
        failure(&service, message.clone(), &binding),
        Code::AlreadyExists
    );
    message.envelope.as_mut().unwrap().request_id = vec![2; 16];
    assert!(service.synthesize(request(message, &binding)).is_ok());
}

#[tokio::test]
async fn output_cap_context_validation_and_invalid_sampler_are_sanitized() {
    let (f, service, binding, mut message) = setup();
    grant(&f, Some("native"));
    message.prompt = "sensitive prompt".repeat(20);
    let status = service
        .synthesize(request(message.clone(), &binding))
        .err()
        .unwrap();
    assert_eq!(status.code(), Code::ResourceExhausted);
    assert_eq!(status.message(), "limit_exceeded");
    message.envelope.as_mut().unwrap().request_id = vec![2; 16];
    message.prompt = "abc".into();
    message.sampler.as_mut().unwrap().top_p = Some(0.);
    assert_eq!(failure(&service, message, &binding), Code::InvalidArgument);
}

#[tokio::test]
async fn deadline_cancels_a_running_unpolled_backpressured_worker_but_keeps_response_charge() {
    let (f, service, binding, mut message) = setup();
    grant(&f, Some("native"));
    message.max_tokens = 29;
    message.envelope.as_mut().unwrap().timeout_ms = 100;
    let mut stream = service
        .synthesize(request(message.clone(), &binding))
        .unwrap();
    assert!(matches!(
        stream.next().await.unwrap().unwrap().event,
        Some(Event::Accepted(_))
    ));
    // The second poll launches the dedicated job.
    assert!(matches!(
        stream.next().await.unwrap().unwrap().event,
        Some(Event::Token(_))
    ));
    tokio::time::sleep(Duration::from_millis(130)).await;
    assert_eq!(
        stream.next().await.unwrap().unwrap_err().code(),
        Code::DeadlineExceeded
    );
    tokio::time::timeout(Duration::from_secs(2), async {
        for id in 10..30 {
            match service.ledger.claim(
                "remote.mdma.test",
                RequestId::parse(&[id; 16]).unwrap(),
                &config::Limits::default(),
            ) {
                Ok(claim) => {
                    drop(claim);
                    return;
                }
                Err(super::super::ledger::Error::Concurrency) => {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                }
                Err(other) => panic!("unexpected accounting error {other:?}"),
            }
        }
        panic!("worker did not release charge");
    })
    .await
    .unwrap();
    assert_eq!(service.responses.available_permits(), 0);
    drop(stream);
    assert_eq!(service.responses.available_permits(), 1);
}

#[tokio::test]
async fn a_live_stream_cannot_outlive_its_current_tls_pin_window() {
    let (f, service, binding, message) = setup();
    grant(&f, Some("native"));
    let mut stream = service.synthesize(request(message, &binding)).unwrap();
    assert!(matches!(
        stream.next().await.unwrap().unwrap().event,
        Some(Event::Accepted(_))
    ));
    // Shortened while the stream runs. The trust write may take longer than
    // the window on a loaded disk; the stream ends either way.
    let mut policy = f
        .store
        .snapshot()
        .unwrap()
        .peer("remote.mdma.test")
        .unwrap()
        .policy
        .clone();
    let until = utc_ms().unwrap() + 100;
    for pin in &mut policy.tls_pins {
        pin.window.valid_until_ms = until;
    }
    f.store.approve_policy(policy).unwrap();
    tokio::time::sleep(Duration::from_millis(
        until.saturating_sub(utc_ms().unwrap()) + 30,
    ))
    .await;
    assert_eq!(
        stream.next().await.unwrap().unwrap_err().code(),
        Code::PermissionDenied
    );
    assert!(stream.next().await.is_none());
}

struct ChangeDuringPrepare {
    inner: Arc<dyn Executor>,
    store: Arc<Store>,
    suspend: bool,
}

impl Executor for ChangeDuringPrepare {
    fn descriptor(&self, model: &str) -> Result<ModelDescriptor, Status> {
        self.inner.descriptor(model)
    }
    fn prepare(
        &self,
        request: &SynthesizeRequest,
        descriptor: &ModelDescriptor,
    ) -> Result<Prepared, Status> {
        let prepared = self.inner.prepare(request, descriptor)?;
        if self.suspend {
            self.store.suspend("remote.mdma.test").unwrap();
        } else {
            let mut policy = self
                .store
                .snapshot()
                .unwrap()
                .peer("remote.mdma.test")
                .unwrap()
                .policy
                .clone();
            policy.limits.requests_per_minute += 1;
            self.store.approve_policy(policy).unwrap();
        }
        Ok(prepared)
    }
}

#[tokio::test]
async fn policy_changes_during_validation_cannot_emit_accepted() {
    for suspend in [false, true] {
        let (f, service, binding, message) = setup();
        grant(&f, Some("native"));
        let executor = Arc::new(ChangeDuringPrepare {
            inner: service.executor.clone(),
            store: f.store.clone(),
            suspend,
        });
        let service = Service::new(f.store.clone(), executor);
        assert_eq!(
            failure(&service, message, &binding),
            if suspend {
                Code::PermissionDenied
            } else {
                Code::Aborted
            }
        );
        assert_eq!(service.responses.available_permits(), 1);
    }
}

#[tokio::test]
async fn power_caps_apply_at_admission_and_only_critical_stops_a_flipped_request() {
    use crate::power::{DegradedLimits, PowerStatus, ServerState as PowerState};
    let (f, _, binding, mut message) = setup();
    grant(&f, Some("native"));
    let status = |state| PowerStatus {
        state,
        reading: None,
    };
    let (tx, rx) = tokio::sync::watch::channel(PowerStatus::default());
    let native = crate::federation::inventory::tests::native_service().with_power(
        rx,
        DegradedLimits {
            n_ctx: 12,
            max_tokens: 2,
        },
    );
    let service = Service::new(f.store.clone(), native.federation_executor());
    let descriptor = service.executor.descriptor("native").unwrap();
    // Power at admission, a flip before Accepted, and the (n_ctx, max_tokens)
    // Accepted then reports; the flip neither re-caps nor aborts the request.
    for (id, admitted, flipped, caps) in [
        (1, PowerState::Degraded, PowerState::Serving, Some((12, 2))),
        (2, PowerState::Serving, PowerState::Degraded, Some((32, 3))),
        (3, PowerState::Serving, PowerState::Critical, None),
    ] {
        message.envelope.as_mut().unwrap().request_id = vec![id; 16];
        tx.send(status(admitted)).unwrap();
        let mut stream = service
            .synthesize(request(message.clone(), &binding))
            .unwrap();
        tx.send(status(flipped)).unwrap();
        let Some((n_ctx, max_tokens)) = caps else {
            assert_eq!(
                stream.next().await.unwrap().unwrap_err().code(),
                Code::Unavailable
            );
            assert!(stream.next().await.is_none());
            continue;
        };
        assert_eq!(service.executor.descriptor("native").unwrap(), descriptor);
        let Some(Event::Accepted(accepted)) = stream.next().await.unwrap().unwrap().event else {
            panic!("Accepted first");
        };
        assert_eq!(
            (accepted.prompt_tokens, accepted.n_ctx, accepted.max_tokens),
            (3, n_ctx, max_tokens)
        );
        let mut emitted = 0;
        let completed = loop {
            match stream.next().await.unwrap().unwrap().event.unwrap() {
                Event::Token(_) => emitted += 1,
                Event::Completed(completed) => break completed,
                Event::Accepted(_) => panic!("duplicate admission"),
            }
        };
        assert_eq!(completed.emitted_tokens, emitted);
        assert!(emitted <= max_tokens);
        assert!(stream.next().await.is_none());
    }
    // The degraded context refuses a prompt the configured one would take.
    message.envelope.as_mut().unwrap().request_id = vec![4; 16];
    message.prompt = "x".repeat(13);
    tx.send(status(PowerState::Degraded)).unwrap();
    assert_eq!(
        failure(&service, message, &binding),
        Code::ResourceExhausted
    );
}

#[tokio::test]
async fn sampler_absence_and_explicit_zero_are_not_conflated() {
    let (f, service, binding, mut message) = setup();
    grant(&f, Some("native"));
    message.sampler = None;
    let mut stream = service
        .synthesize(request(message.clone(), &binding))
        .unwrap();
    let Some(Event::Accepted(accepted)) = stream.next().await.unwrap().unwrap().event else {
        panic!();
    };
    assert_eq!(
        accepted.sampler.unwrap().temperature,
        Some(crate::sampler::SamplerConfig::default().temperature)
    );
    drop(stream);
    message.envelope.as_mut().unwrap().request_id = vec![2; 16];
    message.sampler = Some(SamplerConfig {
        temperature: Some(0.),
        min_p: Some(0.),
        ..Default::default()
    });
    let mut stream = service.synthesize(request(message, &binding)).unwrap();
    let Some(Event::Accepted(accepted)) = stream.next().await.unwrap().unwrap().event else {
        panic!();
    };
    let sampler = accepted.sampler.unwrap();
    assert_eq!(sampler.temperature, Some(0.));
    assert_eq!(sampler.min_p, Some(0.));
}

impl Service {
    pub(crate) fn is_idle(&self) -> bool {
        self.ledger.is_idle() && self.responses.available_permits() == 1
    }
}

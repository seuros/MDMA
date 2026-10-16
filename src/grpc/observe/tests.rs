use super::*;
use crate::backend::cpu::CpuBackend;
use crate::gguf::GgufModel;
use crate::grpc::proto::{self, mdma_server::Mdma};
use crate::memory::NativeMemoryBudget;
use crate::registry::{Options, Registry, Source};
use rama::graceful::Shutdown;
use std::sync::Arc;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

/// Two names over one GGUF, loaded under an accounting-only budget.
fn aliased() -> (
    MdmaService<CpuBackend>,
    NativeMemoryBudget,
    usize,
    tempfile::NamedTempFile,
) {
    let file = crate::gguf::tests::Fixture::new(1, 8)
        .with_tokenizer()
        .write();
    let models = ["a", "b"]
        .map(|name| (name.to_owned(), Source::Gguf(file.path().into())))
        .into();
    let weights =
        crate::model::memory::weight_bytes(&GgufModel::open(file.path()).unwrap()).unwrap();
    let budget = NativeMemoryBudget::new(usize::MAX);
    let registry = Registry::new(
        Arc::new(CpuBackend),
        models,
        Some("a"),
        Options {
            budget: Some(budget.clone()),
            ..Options::new(64)
        },
    )
    .unwrap();
    assert_eq!(registry.preload(), Vec::new());
    (
        MdmaService::new(registry, "cpu").with_memory_budget(budget.clone()),
        budget,
        weights,
        file,
    )
}

#[test]
fn interval_is_clamped_with_a_default() {
    for (requested, ms) in [
        (0, 1000),
        (1, 250),
        (250, 250),
        (5000, 5000),
        (u32::MAX, 60_000),
    ] {
        assert_eq!(
            interval(requested),
            Duration::from_millis(ms),
            "{requested}"
        );
    }
}

#[tokio::test]
async fn snapshot_reports_shared_uploads_geometry_and_live_requests() {
    let (service, budget, weights, _file) = aliased();
    let memory = service.memory_snapshot();
    assert_eq!(
        memory.limit_bytes, 0,
        "unset limit reads as accounting-only"
    );
    assert_eq!(
        (memory.used_bytes, memory.weights_bytes, memory.kv_bytes),
        (weights as u64, weights as u64, 0)
    );
    assert_eq!(
        memory.weights,
        [WeightUpload {
            upload_id: 0,
            bytes: weights as u64,
            reserved: true,
            models: vec!["a".into(), "b".into()]
        }]
    );
    assert!(
        memory.device.as_ref().is_some_and(|d| !d.name.is_empty()),
        "the backend labels its device"
    );
    assert_eq!(memory.active_requests, 0);

    let status = service
        .introspect(Request::new(proto::IntrospectRequest {}))
        .await
        .unwrap()
        .into_inner();
    let geometry = status.models[0]
        .geometry
        .clone()
        .expect("native decoders report geometry");
    assert_eq!(geometry.n_layers, 1);
    assert_eq!(geometry.layer_kv_widths.len(), 1);
    assert!(geometry.kv_bytes_per_token > 0);

    // A held generation keeps its request open; dropping it releases everything.
    let stream = service
        .synthesize(Request::new(proto::SynthesizeRequest {
            prompt: "hello".into(),
            max_tokens: 40,
            sampler: None,
            model: String::new(),
            emit_events: false,
            chat: false,
            system: String::new(),
        }))
        .await
        .unwrap()
        .into_inner();
    let during = service.memory_snapshot();
    assert_eq!(during.active_requests, 1);
    assert!(
        during.kv_bytes > 0 && during.workspace_bytes > 0,
        "{during:?}"
    );
    assert_eq!(
        during.used_bytes,
        during.weights_bytes + during.kv_bytes + during.workspace_bytes
    );
    drop(stream);
    assert_eq!(service.memory_snapshot().active_requests, 0);
    let deadline = Instant::now() + Duration::from_secs(5);
    while budget.snapshot().used != weights {
        assert!(
            Instant::now() < deadline,
            "request memory was not released: {:?}",
            budget.snapshot()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn watch_streams_a_snapshot_and_ends_on_shutdown() {
    let (service, _budget, weights, _file) = aliased();
    let trigger = CancellationToken::new();
    let shutdown = Shutdown::new(trigger.clone().cancelled_owned());
    let service = service.with_shutdown_guard(shutdown.guard_weak());
    let mut stream = service
        .watch(Request::new(proto::WatchRequest { interval_ms: 250 }))
        .await
        .unwrap()
        .into_inner();
    let first = stream.next().await.unwrap().unwrap();
    let status = first.status.unwrap();
    assert!(status.ready);
    assert_eq!(status.models.len(), 2);
    assert_eq!(first.memory.unwrap().weights_bytes, weights as u64);

    trigger.cancel();
    let end = tokio::time::timeout(Duration::from_secs(5), async {
        // A draining snapshot may still arrive before the stream ends.
        while let Some(item) = stream.next().await {
            item.unwrap();
        }
    })
    .await;
    assert!(end.is_ok(), "Watch must end when the server shuts down");
    shutdown
        .shutdown_with_limit(Duration::from_secs(2))
        .await
        .unwrap();
}

#[test]
fn topology_requires_an_operator_pin() {
    let (service, _budget, _weights, _file) = aliased();
    let pin = format!("sha256:{}", "ab".repeat(32));
    let service = service.with_operators(vec![pin.clone()]);
    let request = |spki: Option<&str>| {
        let request = Request::new(());
        if let Some(spki) = spki {
            let transport = Extensions::new();
            transport.insert(ClientSpki(spki.to_owned()));
            request.extensions().insert(TransportIngress(transport));
        }
        request
    };
    assert_eq!(
        service.federation(&request(None)),
        FederationTopology::default()
    );
    assert_eq!(
        service.federation(&request(Some("sha256:00"))),
        FederationTopology::default()
    );
    // Request metadata does not count.
    let mut spoofed = Request::new(());
    spoofed
        .metadata_mut()
        .insert("x-client-spki", pin.parse().unwrap());
    assert_eq!(service.federation(&spoofed), FederationTopology::default());
    let operator = service.federation(&request(Some(&pin)));
    assert!(
        operator.authorized && !operator.enabled,
        "operators learn federation is off here: {operator:?}"
    );
}

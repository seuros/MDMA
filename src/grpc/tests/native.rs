//! Two native architectures on one mTLS listener. Heavy gates are opt-in.
use super::transport::Fixture;
use super::*;
use crate::backend::Backend;
use crate::grpc::client::Client;
use crate::memory::NativeMemoryBudget;
use crate::registry::{Options, Source};
use rama::graceful::Shutdown;
use rama::http::grpc::Code;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

fn request(model: &str, prompt: &str, max_tokens: u32) -> proto::SynthesizeRequest {
    proto::SynthesizeRequest {
        model: model.into(),
        prompt: prompt.into(),
        max_tokens,
        sampler: Some(proto::SamplerConfig {
            temperature: Some(0.0),
            repetition_penalty: Some(1.0),
            ..Default::default()
        }),
        emit_events: false,
        chat: false,
        system: String::new(),
    }
}

async fn tokens(client: &Client, request: proto::SynthesizeRequest) -> Vec<u32> {
    let mut stream = client.synthesize(request).await.unwrap().into_inner();
    let mut tokens = Vec::new();
    while let Some(token) = stream.next().await {
        tokens.push(token.unwrap().token_id);
    }
    tokens
}

async fn lifecycle<B>(backend: B, name: &'static str, paths: [&str; 2], real: bool)
where
    B: Backend + Send + Sync + 'static,
    B::Buf: Send + Sync,
{
    let budget = NativeMemoryBudget::new(8 * 1024 * 1024 * 1024);
    let models = vec![
        ("tiny".into(), Source::Gguf(paths[0].into())),
        ("gemma".into(), Source::Gguf(paths[1].into())),
    ];
    let registry = Registry::new(
        Arc::new(backend),
        models,
        Some("tiny"),
        Options {
            budget: Some(budget.clone()),
            ..Options::new(64)
        },
    )
    .unwrap();
    assert_eq!(registry.preload(), Vec::new());
    let weights = budget.used();
    assert!(weights > 0);
    let fixture = Fixture::new();
    let trigger = CancellationToken::new();
    let shutdown = Shutdown::new(trigger.clone().cancelled_owned());
    let service = MdmaService::new(registry, name).with_shutdown_guard(shutdown.guard_weak());
    let mut node = fixture
        .start_on(Executor::graceful(shutdown.guard()), |_| service)
        .await;
    let client = fixture.client(&node.url);
    let health = client
        .introspect(proto::IntrospectRequest {})
        .await
        .unwrap()
        .into_inner();
    assert!(health.ready);
    assert_eq!(health.default_model, "tiny");
    assert_eq!(health.models.len(), 2);
    for (entry, model_name, architecture, context) in [
        (
            &health.models[0],
            "tiny",
            "llama",
            if real { 64 } else { 16 },
        ),
        (
            &health.models[1],
            "gemma",
            "gemma4",
            if real { 64 } else { 16 },
        ),
    ] {
        assert!(entry.ready);
        assert_eq!(
            (
                &*entry.name,
                &*entry.architecture,
                &*entry.backend,
                entry.n_ctx
            ),
            (model_name, architecture, name, context)
        );
    }
    let prompt = if real {
        "The capital of France is"
    } else {
        "ab"
    };
    let tiny = tokens(&client, request("tiny", prompt, 3)).await;
    let gemma = tokens(&client, request("gemma", prompt, 3)).await;
    assert_eq!(tokens(&client, request("", prompt, 3)).await, tiny);
    if real {
        assert_eq!(tiny, [3681, 29889, 13]);
        assert_eq!(gemma, [9079, 236761]); // End-of-turn is not streamed.
        assert_eq!(
            tokens(
                &client,
                request("tiny", "<|user|>\nWhat is 2 + 2?</s>\n<|assistant|>\n", 16)
            )
            .await,
            [29906, 718, 29871, 29906, 353, 29871, 29946]
        ); // EOS terminates.
        assert_eq!(
            tokens(
                &client,
                request(
                    "gemma",
                    "<|turn>user\nWhat is 2 + 2? Answer briefly.<turn|>\n<|turn>model\n",
                    16
                )
            )
            .await,
            [236812]
        );
        let multi = "<|turn>user\nHello.<turn|>\n<|turn>model\nHello!<turn|>\n<|turn>user\nWhat is 2 + 2?<turn|>\n<|turn>model\n";
        let answer = tokens(&client, request("gemma", multi, 4)).await;
        assert!(!answer.is_empty());
        assert_eq!(tokens(&client, request("gemma", multi, 4)).await, answer);
    }
    let (a, b) = tokio::join!(
        tokens(&client, request("tiny", prompt, 3)),
        tokens(&client, request("gemma", prompt, 3)),
    );
    assert_eq!((a, b), (tiny.clone(), gemma.clone()));
    assert_eq!(
        client
            .synthesize(request("missing", prompt, 1))
            .await
            .unwrap_err()
            .code(),
        Code::NotFound
    );
    for model in ["tiny", "gemma"] {
        let error = client
            .synthesize(request(model, &" ab".repeat(100), 1))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::ResourceExhausted);
    }
    // Disconnect an active native response, then ensure fresh KV and routing.
    let mut cancelled = client
        .synthesize(request("tiny", prompt, 40))
        .await
        .unwrap()
        .into_inner();
    assert!(cancelled.next().await.unwrap().is_ok());
    drop(cancelled);
    assert_eq!(tokens(&client, request("tiny", prompt, 3)).await, tiny);
    assert_eq!(tokens(&client, request("gemma", prompt, 3)).await, gemma);
    // The graceful executor closes admission/listener and drains an admitted stream.
    let mut stream = client
        .synthesize(request("tiny", prompt, 3))
        .await
        .unwrap()
        .into_inner();
    let first = stream.next().await.unwrap().unwrap().token_id;
    trigger.cancel();
    let mut drained = vec![first];
    while let Some(token) = stream.next().await {
        drained.push(token.unwrap().token_id);
    }
    assert_eq!(drained, tiny);
    shutdown
        .shutdown_with_limit(Duration::from_secs(120))
        .await
        .unwrap();
    (&mut node.task).await.unwrap();
    assert!(tokio::net::TcpStream::connect(node.addr).await.is_err());
    assert_eq!(
        budget.used(),
        0,
        "drained workers and service must release all tensors"
    );
}

#[tokio::test]
async fn two_architecture_compat_mtls_lifecycle() {
    let tiny = crate::model::compat_tests::decoder_fixture(false);
    let gemma = crate::model::compat_tests::decoder_fixture(true);
    tokio::time::timeout(
        Duration::from_secs(30),
        lifecycle(
            CpuBackend,
            "cpu",
            [tiny.0.to_str().unwrap(), gemma.0.to_str().unwrap()],
            false,
        ),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn models_that_do_not_fit_keep_the_mtls_listener_alive_and_drain_cleanly() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let model = crate::gguf::tests::Fixture::new(1, 8)
            .with_tokenizer()
            .write();
        let budget = NativeMemoryBudget::new(0);
        let options = Options {
            budget: Some(budget.clone()),
            warmup_rounds: 2,
            ..Options::new(64)
        };
        let registry = Registry::new(
            Arc::new(CpuBackend),
            vec![("blocked".into(), Source::Gguf(model.path().into()))],
            None,
            options,
        )
        .unwrap();
        let fixture = Fixture::new();
        let trigger = CancellationToken::new();
        let shutdown = Shutdown::new(trigger.clone().cancelled_owned());
        let service = MdmaService::new(registry, "cpu").with_shutdown_guard(shutdown.guard_weak());
        let mut node = fixture
            .start_on(Executor::graceful(shutdown.guard()), |_| service)
            .await;
        let client = fixture.client(&node.url);
        for _ in 0..2 {
            let error = client
                .synthesize(request("", "hello", 1))
                .await
                .unwrap_err();
            assert_eq!(error.code(), Code::ResourceExhausted);
            assert!(
                error.message().contains("does not fit"),
                "{}",
                error.message()
            );
            let health = client
                .introspect(proto::IntrospectRequest {})
                .await
                .unwrap()
                .into_inner();
            assert!(health.ready, "a refusal does not take the node down");
            assert_eq!(
                (health.server_state.as_str(), health.default_model.as_str()),
                ("serving", "blocked")
            );
            assert_eq!(health.models[0].residency, "available");
            assert!(!node.task.is_finished());
        }
        trigger.cancel();
        shutdown
            .shutdown_with_limit(Duration::from_secs(2))
            .await
            .unwrap();
        (&mut node.task).await.unwrap();
        assert!(tokio::net::TcpStream::connect(node.addr).await.is_err());
        assert_eq!(budget.used(), 0);
    })
    .await
    .unwrap();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[tokio::test]
async fn two_architecture_metal_compat_mtls_lifecycle() {
    let tiny = crate::model::compat_tests::decoder_fixture(false);
    let gemma = crate::model::compat_tests::decoder_fixture(true);
    tokio::time::timeout(
        Duration::from_secs(30),
        lifecycle(
            crate::backend::metal::MetalBackend::new().unwrap(),
            "metal",
            [tiny.0.to_str().unwrap(), gemma.0.to_str().unwrap()],
            false,
        ),
    )
    .await
    .unwrap();
}

fn checkpoint_paths() -> [String; 2] {
    ["MDMA_TEST_TINYLLAMA_GGUF", "MDMA_TEST_GEMMA4_GGUF"].map(|variable| {
        std::env::var(variable)
            .unwrap_or_else(|_| panic!("set {variable} to the pinned checkpoint"))
    })
}

#[tokio::test]
#[ignore = "requires MDMA_TEST_TINYLLAMA_GGUF and MDMA_TEST_GEMMA4_GGUF; mTLS lifecycle"]
async fn two_model_cpu_checkpoint_lifecycle() {
    let paths = checkpoint_paths();
    tokio::time::timeout(
        Duration::from_secs(600),
        lifecycle(CpuBackend, "cpu", [&paths[0], &paths[1]], true),
    )
    .await
    .unwrap();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[tokio::test]
#[ignore = "requires MDMA_TEST_TINYLLAMA_GGUF and MDMA_TEST_GEMMA4_GGUF; mTLS lifecycle"]
async fn two_model_metal_checkpoint_lifecycle() {
    let paths = checkpoint_paths();
    tokio::time::timeout(
        Duration::from_secs(600),
        lifecycle(
            crate::backend::metal::MetalBackend::new().unwrap(),
            "metal",
            [&paths[0], &paths[1]],
            true,
        ),
    )
    .await
    .unwrap();
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires an NVIDIA GPU"]
async fn two_architecture_cuda_compat_mtls_lifecycle() {
    let tiny = crate::model::compat_tests::decoder_fixture(false);
    let gemma = crate::model::compat_tests::decoder_fixture(true);
    tokio::time::timeout(
        Duration::from_secs(30),
        lifecycle(
            crate::backend::cuda::CudaBackend::new().unwrap(),
            "cuda",
            [tiny.0.to_str().unwrap(), gemma.0.to_str().unwrap()],
            false,
        ),
    )
    .await
    .unwrap();
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires MDMA_TEST_TINYLLAMA_GGUF, MDMA_TEST_GEMMA4_GGUF and an NVIDIA GPU; mTLS lifecycle"]
async fn two_model_cuda_checkpoint_lifecycle() {
    let paths = checkpoint_paths();
    tokio::time::timeout(
        Duration::from_secs(600),
        lifecycle(
            crate::backend::cuda::CudaBackend::new().unwrap(),
            "cuda",
            [&paths[0], &paths[1]],
            true,
        ),
    )
    .await
    .unwrap();
}

/// Bespoke-MiniCheck through Decide on MiniCheck's README example, which
/// its reference implementation scores 0.984 and 0.011.
async fn minicheck_reference_example<B>(backend: B)
where
    B: Backend + Send + Sync + 'static,
    B::Buf: Send + Sync,
{
    use proto::mdma_server::Mdma;
    use rama::http::grpc::Request;
    let registry = Registry::new(
        Arc::new(backend),
        vec![(
            "minicheck".into(),
            Source::Gguf(crate::backend::test_support::fixture().into()),
        )],
        None,
        Options::new(4096),
    )
    .unwrap();
    let service = MdmaService::new(registry, "native");
    let status = service
        .introspect(Request::new(proto::IntrospectRequest {}))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(status.models[0].decision, "minicheck");
    let reply = service
        .decide(Request::new(proto::DecideRequest {
            model: "minicheck".into(),
            state_json: r#"{"document": "A group of students gather in the school library to study for their upcoming final exams."}"#.into(),
            questions_json: r#"{"exam": {"claim": "The students are preparing for an examination."}, "vacation": {"claim": "The students are on vacation."}}"#.into(),
        }))
        .await
        .unwrap()
        .into_inner();
    let result: serde_json::Value = serde_json::from_str(&reply.result_json).unwrap();
    eprintln!("{result}");
    let answer = |id: &str| {
        let answer = &result["answers"][id];
        (
            answer["supported"].as_bool().unwrap(),
            answer["probability"].as_f64().unwrap(),
        )
    };
    let (exam, vacation) = (answer("exam"), answer("vacation"));
    assert!(
        exam.0 && exam.1 > 0.9 && !vacation.0 && vacation.1 < 0.1,
        "{result}"
    );
    assert_eq!(result["usage"]["output_tokens"], 0, "{result}");
}

#[tokio::test]
#[ignore = "requires MDMA_TEST_GGUF (Bespoke-MiniCheck-7B-Q4_K_M.gguf)"]
async fn minicheck_decides_its_reference_example_on_cpu() {
    minicheck_reference_example(CpuBackend).await;
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[tokio::test]
#[ignore = "requires MDMA_TEST_GGUF (Bespoke-MiniCheck-7B-Q4_K_M.gguf)"]
async fn minicheck_decides_its_reference_example_on_metal() {
    minicheck_reference_example(crate::backend::metal::MetalBackend::new().unwrap()).await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires MDMA_TEST_GGUF (Bespoke-MiniCheck-7B-Q4_K_M.gguf) and an NVIDIA GPU"]
async fn minicheck_decides_its_reference_example_on_cuda() {
    minicheck_reference_example(crate::backend::cuda::CudaBackend::new().unwrap()).await;
}

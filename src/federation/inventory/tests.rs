use super::*;
use crate::backend::cpu::CpuBackend;
use crate::gguf::tests::Fixture;
use crate::grpc::MdmaService;
use crate::registry::{Options, Source};

fn models(file: &std::path::Path, names: &[&str]) -> Vec<(String, Source)> {
    names
        .iter()
        .map(|name| ((*name).to_owned(), Source::Gguf(file.into())))
        .collect()
}

pub(crate) fn native_service() -> MdmaService<CpuBackend> {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let options = Options {
        frozen: true,
        ..Options::new(32)
    };
    let mut registry = Registry::new(
        Arc::new(CpuBackend),
        models(file.path(), &["native", "alias"]),
        None,
        options,
    )
    .unwrap();
    assert_eq!(registry.preload(), Vec::new());
    assert_eq!(registry.capture_execution_profiles().unwrap(), 2);
    // The published inventory must refer to already loaded bytes, never reopen
    // the source after the fixture is destroyed on return.
    MdmaService::new(registry, "not-a-profile-claim")
}

fn exports() -> Vec<String> {
    vec!["native".into(), "alias".into(), "unknown".into()]
}

#[test]
fn only_explicit_loaded_exact_identities_are_visible_with_alias_caps() {
    let service = native_service();
    let inventory = service.federation_inventory();
    assert!(inventory.snapshot(&[]).models.is_empty());
    assert!(
        inventory.snapshot(&[String::new()]).models.is_empty(),
        "no default alias"
    );
    let snapshot = inventory.snapshot(&exports());
    assert_eq!(snapshot.state, ServerState::Serving);
    assert_eq!(snapshot.models.len(), 2);
    let (native, alias) = (&snapshot.models[0], &snapshot.models[1]);
    assert_eq!(native.model, "native");
    assert_eq!(alias.model, "alias");
    assert_eq!(native.artifact_id, alias.artifact_id);
    assert_eq!(native.profile_id, alias.profile_id);
    assert_eq!(
        native.backend, "cpu",
        "never use informational backend label"
    );
    assert!(native.ready && alias.ready);
    assert_eq!(
        native.limits,
        Limits::Synthesize {
            n_ctx: 32,
            max_output_tokens: 32
        }
    );
    assert_eq!(
        alias.limits, native.limits,
        "aliases share the global context"
    );
    assert!(native.resident && alias.resident);

    // The service's output cap is what peers may ask for.
    let capped = native_service()
        .with_max_output_tokens(8)
        .federation_inventory()
        .snapshot(&exports());
    assert_eq!(
        capped.models[0].limits,
        Limits::Synthesize {
            n_ctx: 32,
            max_output_tokens: 8
        }
    );
}

#[test]
fn power_changes_keep_the_offer_critical_and_dead_watchers_withdraw_readiness() {
    use crate::power::ServerState as PowerState;
    let (tx, rx) = watch::channel(PowerStatus::default());
    let service = native_service().with_power(
        rx,
        DegradedLimits {
            n_ctx: 12,
            max_tokens: 3,
        },
    );
    let inventory = service.federation_inventory();
    let serving = inventory.snapshot(&exports());
    tx.send(PowerStatus {
        state: PowerState::Degraded,
        reading: None,
    })
    .unwrap();
    let degraded = inventory.snapshot(&exports());
    assert_eq!(degraded.state, ServerState::Degraded);
    // Degraded caps apply at admission; the catalog keeps the configured
    // maxima so routes selected before the flip stay valid.
    assert_eq!(
        degraded.models[0].limits,
        Limits::Synthesize {
            n_ctx: 32,
            max_output_tokens: 32
        }
    );
    assert_eq!(degraded.models, serving.models);
    assert!(degraded.models.iter().all(|m| m.ready));
    tx.send(PowerStatus {
        state: PowerState::Critical,
        reading: None,
    })
    .unwrap();
    let critical = inventory.snapshot(&exports());
    assert_eq!(critical.state, ServerState::Critical);
    assert!(critical.models.iter().all(|m| !m.ready));
    tx.send(PowerStatus::default()).unwrap();
    // Host figures are live; the offer is what must come back.
    let restored = inventory.snapshot(&exports());
    assert_eq!(
        (restored.state, restored.models),
        (serving.state, serving.models)
    );
    drop(tx);
    let closed = inventory.snapshot(&exports());
    assert_eq!(closed.state, ServerState::Critical);
    assert!(closed.models.iter().all(|m| !m.ready));
}

#[tokio::test]
async fn shutdown_overrides_power_and_marks_all_models_unready() {
    let trigger = tokio_util::sync::CancellationToken::new();
    let shutdown = rama::graceful::Shutdown::new(trigger.clone().cancelled_owned());
    let weak = shutdown.guard_weak();
    let service = native_service().with_shutdown_guard(weak.clone());
    let inventory = service.federation_inventory();
    assert!(
        inventory
            .snapshot(&exports())
            .models
            .iter()
            .all(|m| m.ready)
    );
    trigger.cancel();
    weak.cancelled().await;
    let drained = inventory.snapshot(&exports());
    assert_eq!(drained.state, ServerState::Draining);
    assert!(drained.models.iter().all(|m| !m.ready));
}

#[test]
fn unfrozen_unloaded_or_refused_models_are_not_exportable() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let snapshot = |options: Options, preload: bool| {
        let mut registry = Registry::new(
            Arc::new(CpuBackend),
            models(file.path(), &["native"]),
            None,
            options,
        )
        .unwrap();
        registry.capture_execution_profiles().unwrap();
        if preload {
            registry.preload();
        }
        MdmaService::new(registry, "cpu")
            .federation_inventory()
            .snapshot(&exports())
    };
    assert!(
        snapshot(Options::new(64), true).models.is_empty(),
        "unfrozen loads have no identity"
    );
    assert!(
        snapshot(
            Options {
                frozen: true,
                ..Options::new(64)
            },
            false
        )
        .models
        .is_empty(),
        "nothing is known before the first load"
    );
    let budget = crate::memory::NativeMemoryBudget::new(0);
    let refused = snapshot(
        Options {
            frozen: true,
            budget: Some(budget),
            ..Options::new(64)
        },
        true,
    );
    assert_eq!(
        refused.state,
        ServerState::Serving,
        "a memory refusal is not a degraded node"
    );
    assert!(refused.models.is_empty());
}

#[tokio::test]
async fn hashed_models_are_advertised_unloaded_and_load_on_their_first_federated_use() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let options = Options {
        frozen: true,
        ..Options::new(32)
    };
    let mut registry = Registry::new(
        Arc::new(CpuBackend),
        models(file.path(), &["native", "alias"]),
        None,
        options,
    )
    .unwrap();
    registry.capture_execution_profiles().unwrap();
    registry.hash_artifacts(|| false);
    let service = MdmaService::new(registry, "cpu");
    let before = service.federation_inventory().snapshot(&exports());
    assert_eq!(
        before.models.len(),
        2,
        "a hashed model is exportable before it loads"
    );
    assert!(before.models.iter().all(|m| m.ready && !m.resident));
    let executor = service.federation_executor();
    executor.load("alias").await.unwrap();
    let after = service.federation_inventory().snapshot(&exports());
    assert!(
        after.models.iter().all(|m| m.resident),
        "aliases share the load"
    );
    for (before, after) in before.models.iter().zip(&after.models) {
        assert!(
            before.same_selection(after),
            "loading keeps the advertised identity"
        );
    }
}

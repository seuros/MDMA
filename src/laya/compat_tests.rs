//! Always-on compatibility checks with small generated safetensors.
use super::prompt::{Item, Kind, collate};
use super::weights::Weights;
use super::{AgentConfig, DecisionModel, EncoderConfig};
use crate::backend::cpu::CpuBackend;
use crate::model::arch::modernbert::tests::fixture as encoder_fixture;
use serde_json::json;
use std::assert_matches;

fn fixture(bias: bool) -> (Vec<u8>, EncoderConfig, AgentConfig) {
    let (mut b, encoder) = encoder_fixture(bias);
    let agent = AgentConfig::from_json(
        &json!({
            "encoder": "tiny-modernbert", "head_layers": 2,
            "act_costs": {"answer": 1.0}, "max_len": 16, "head_max_len": 8
        })
        .to_string(),
    )
    .unwrap();
    for i in 0..2 {
        let p = format!("head.layers.{i}");
        b.norm(&format!("{p}.norm1"), true);
        b.tensor(&format!("{p}.self_attn.in_proj_weight"), &[24, 8], false);
        b.tensor(&format!("{p}.self_attn.in_proj_bias"), &[24], false);
        b.linear(&format!("{p}.self_attn.out_proj"), 8, 8, true);
        b.norm(&format!("{p}.norm2"), true);
        b.linear(&format!("{p}.linear1"), 32, 8, true);
        b.linear(&format!("{p}.linear2"), 8, 32, true);
    }
    b.linear("type_emb", 3, 8, false);
    b.norm("scorer.0", true);
    b.linear("scorer.1", 8, 8, true);
    b.linear("scorer.3", 1, 8, true);
    b.linear("act_head.0", 256, 12, true);
    b.linear("act_head.2", 2, 256, true);
    (b.serialize(), encoder, agent)
}

pub(crate) fn checkpoint() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = super::tokenizer::tests::checkpoint();
    std::fs::create_dir(dir.path().join("tokenizer")).unwrap();
    for path in ["tokenizer.json", "tokenizer_config.json"] {
        std::fs::copy(
            tokenizer.path().join(path),
            dir.path().join("tokenizer").join(path),
        )
        .unwrap();
    }
    let (weights, encoder, agent) = fixture(true);
    std::fs::write(dir.path().join("model.safetensors"), weights).unwrap();
    std::fs::create_dir(dir.path().join("encoder")).unwrap();
    std::fs::write(dir.path().join("encoder/config.json"), json!({
        "vocab_size": encoder.vocab_size, "hidden_size": encoder.hidden_size,
        "intermediate_size": encoder.intermediate_size, "num_hidden_layers": encoder.num_hidden_layers,
        "num_attention_heads": encoder.num_attention_heads,
        "norm_bias": encoder.norm_bias, "attention_bias": encoder.attention_bias, "mlp_bias": encoder.mlp_bias,
        "local_attention": encoder.local_attention, "global_attn_every_n_layers": encoder.global_attn_every_n_layers,
        "max_position_embeddings": encoder.max_position_embeddings
    }).to_string()).unwrap();
    std::fs::write(dir.path().join("rl_agent_config.json"), json!({
        "encoder": agent.encoder, "head_layers": agent.head_layers, "act_costs": agent.act_costs,
        "max_len": agent.max_len, "head_max_len": agent.head_max_len,
        "temperature": [0.8, 1.2, 1.4], "temperature_by_options": {"choice:2": 0.75}
    }).to_string()).unwrap();
    dir
}

#[cfg(any(feature = "vulkan", all(feature = "metal", target_os = "macos")))]
fn native_laya_profile_is_cpu_even_with_an_unprofiled_decoder_backend<B>(backend: B)
where
    B: crate::backend::Backend + Send + Sync + 'static,
    B::Buf: Send + Sync,
{
    let dir = checkpoint();
    let mut registry = laya_registry(std::sync::Arc::new(backend), dir.path(), true);
    assert_eq!(registry.capture_execution_profiles().unwrap(), 1);
    let profile = registry.execution_profile("").unwrap();
    assert!(profile.is_current());
    let manifest = profile.manifest().as_object().unwrap();
    assert_eq!(manifest["backend"].as_str(), Some("cpu"));
    assert_eq!(manifest["kv_dtype"].as_str(), Some("none"));
    let inventory = crate::federation::inventory::Native {
        registry: std::sync::Arc::new(registry),
        shutdown: None,
        power: None,
        host: std::sync::Arc::default(),
        max_output_tokens: usize::MAX,
    };
    let snapshot = crate::federation::inventory::Inventory::snapshot(&inventory, &["laya".into()]);
    assert_eq!(snapshot.models.len(), 1);
    assert_eq!(snapshot.models[0].backend, "cpu");
    assert_eq!(
        snapshot.models[0].kind,
        crate::federation::catalog::Kind::Laya
    );
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn laya_cpu_profile_does_not_claim_the_daemons_metal_backend() {
    native_laya_profile_is_cpu_even_with_an_unprofiled_decoder_backend(
        crate::backend::metal::MetalBackend::new().unwrap(),
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn laya_cpu_profile_does_not_claim_the_daemons_vulkan_backend() {
    native_laya_profile_is_cpu_even_with_an_unprofiled_decoder_backend(
        crate::backend::vulkan::VulkanBackend::new().unwrap(),
    );
}

#[test]
fn frozen_laya_registry_identity_is_consumed_bytes_not_canonicalized_tokenizer_or_reopened_paths() {
    use crate::registry::Registry;
    let dir = checkpoint();
    let backend = std::sync::Arc::new(CpuBackend);
    let normal = laya_registry(backend.clone(), dir.path(), false);
    assert!(normal.artifact_id("").is_none());
    let mut frozen = laya_registry(backend.clone(), dir.path(), true);
    assert_eq!(frozen.capture_execution_profiles().unwrap(), 1);
    let original_profile = frozen.execution_profile("").unwrap().as_str().to_owned();
    assert_eq!(
        frozen
            .execution_profile("")
            .unwrap()
            .manifest()
            .as_object()
            .unwrap()["kv_dtype"]
            .as_str(),
        Some("none")
    );
    let original_id = frozen.artifact_id("").unwrap();
    {
        let mut registry = laya_registry(backend.clone(), dir.path(), true);
        registry.capture_execution_profiles().unwrap();
        let service = crate::grpc::MdmaService::new(registry, "cpu");
        let catalog = service.federation_inventory().snapshot(&["laya".into()]);
        assert_eq!(catalog.models.len(), 1);
        assert_eq!(catalog.models[0].artifact_id, original_id.as_str());
        assert_eq!(catalog.models[0].profile_id, original_profile);
        assert_eq!(
            catalog.models[0].limits,
            crate::federation::catalog::Limits::Decide {
                max_questions: crate::federation::catalog::MAX_DECIDE_QUESTIONS,
                max_result_bytes: crate::federation::catalog::MAX_DECIDE_RESULT_BYTES,
            }
        );
        let model = &catalog.models[0];
        assert_eq!(
            (
                model.parameters,
                model.quantization.as_str(),
                model.trained_ctx
            ),
            (normal.info("").unwrap().parameters, "F32", 16)
        );
    }
    assert_eq!(
        &original_id,
        &super::sources::Sources::capture(dir.path()).unwrap().id()
    );
    let predictor = |registry: &Registry<CpuBackend>| {
        registry
            .acquire_blocking("")
            .unwrap()
            .laya()
            .expect("Laya")
            .clone()
    };
    let original = predictor(&frozen);
    assert_eq!(original.temperature_for(Kind::Choice, 2), 0.75);
    let questions =
        json!({"q": {"type": "choice", "instructions": "hello", "criteria": ["world", "hello"]}});
    let questions = questions.as_object().unwrap();
    let expected = predictor(&normal).predict(&json!({}), questions).unwrap();
    assert_eq!(original.predict(&json!({}), questions).unwrap(), expected);

    let tokenizer_path = dir.path().join("tokenizer/tokenizer.json");
    let legacy = std::fs::read(&tokenizer_path).unwrap();
    let canonical = tokenizers::canonicalize_str(std::str::from_utf8(&legacy).unwrap()).unwrap();
    std::fs::write(&tokenizer_path, canonical).unwrap();
    let converted = laya_registry(backend.clone(), dir.path(), true);
    assert_ne!(converted.artifact_id(""), Some(original_id.clone()));
    assert_eq!(
        predictor(&converted)
            .predict(&json!({}), questions)
            .unwrap(),
        expected
    );
    assert_eq!(original.predict(&json!({}), questions).unwrap(), expected);

    std::fs::write(&tokenizer_path, legacy).unwrap();
    let config_path = dir.path().join("rl_agent_config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
    config["temperature_by_options"]["choice:2"] = json!(1.5);
    std::fs::write(config_path, config.to_string()).unwrap();
    let mut recalibrated = laya_registry(backend, dir.path(), true);
    assert_eq!(recalibrated.capture_execution_profiles().unwrap(), 1);
    assert_eq!(
        recalibrated.execution_profile("").unwrap().as_str(),
        original_profile
    );
    assert_ne!(recalibrated.artifact_id(""), Some(original_id.clone()));
    assert_eq!(
        predictor(&recalibrated).temperature_for(Kind::Choice, 2),
        1.5
    );
    assert_eq!(original.temperature_for(Kind::Choice, 2), 0.75);
    std::fs::remove_dir_all(dir.path()).unwrap();
    assert_eq!(original.predict(&json!({}), questions).unwrap(), expected);
    assert_eq!(frozen.artifact_id(""), Some(original_id));
}

/// One Laya checkpoint named `laya`, loaded.
fn laya_registry<B: crate::backend::Backend>(
    backend: std::sync::Arc<B>,
    dir: &std::path::Path,
    frozen: bool,
) -> crate::registry::Registry<B> {
    use crate::registry::{Options, Registry, Source};
    let options = Options {
        frozen,
        ..Options::new(32)
    };
    let registry = Registry::new(
        backend,
        vec![("laya".into(), Source::Laya(dir.into()))],
        None,
        options,
    )
    .unwrap();
    assert_eq!(registry.preload(), Vec::new());
    registry
}

fn close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (&got, &want) in got.iter().zip(want) {
        assert!(
            got.is_finite() && (got - want).abs() < 2e-5,
            "{got} != {want}"
        );
    }
}

#[test]
fn checked_prediction_matches_ordinary_output_and_stops_at_batch_boundaries() {
    let dir = checkpoint();
    let predictor = super::Predictor::load(CpuBackend, dir.path())
        .unwrap()
        .with_batch_size(1);
    let state = json!({});
    let questions = json!({
        "one": {"type": "choice", "instructions": "hello", "criteria": ["world", "hello"]},
        "two": {"type": "noul", "instructions": "hello"}
    });
    let questions = questions.as_object().unwrap();
    let prepared = predictor
        .prepare_checked(&state, questions, || true)
        .unwrap()
        .unwrap();
    let ids: Vec<_> = questions.keys().cloned().collect();
    let expected = predictor.predict(&state, questions).unwrap();
    assert_eq!(
        predictor
            .predict_prepared(&prepared, &ids, || true)
            .unwrap(),
        Some(expected)
    );
    assert!(
        predictor
            .prepare_checked(&state, questions, || false)
            .unwrap()
            .is_none()
    );
    assert!(
        predictor
            .predict_prepared(&prepared, &ids, || false)
            .unwrap()
            .is_none()
    );
    let checks = std::cell::Cell::new(0);
    assert!(
        predictor
            .predict_prepared(&prepared, &ids, || {
                let count = checks.get() + 1;
                checks.set(count);
                count < 3 // Stop after the first actual forward, before answer emission.
            })
            .unwrap()
            .is_none()
    );
    assert_eq!(checks.get(), 3);
}

#[test]
fn laya_workspace_reservation_accounts_for_minimum_two_option_padding_and_checked_overflow() {
    let dir = checkpoint();
    let predictor = super::Predictor::load(CpuBackend, dir.path()).unwrap();
    let questions =
        json!({"one": {"type": "choice", "instructions": "hello", "criteria": ["world"]}});
    let prepared = predictor
        .prepare(&json!({}), questions.as_object().unwrap())
        .unwrap();
    assert_eq!(prepared[0].1.markers.len(), 1);
    let seq = prepared[0].1.ids.len();
    let d = predictor.model.encoder_config().hidden_size;
    let inter = predictor.model.encoder_config().intermediate_size;
    let actions = predictor.model.agent_config().n_act();
    let expected = (16 * seq * d + 3 * seq * inter + 4 * 2 * d + 2 + d + 4 + 512 + actions) * 4;
    assert_eq!(
        predictor.prepared_workspace_bytes(&prepared).unwrap(),
        expected
    );
    assert_eq!(predictor.prepared_workspace_bytes(&[]).unwrap(), 0);
    assert_matches!(
        predictor.model.workspace_bytes(usize::MAX, 16, 2),
        Err(super::Error::Backend(crate::backend::Error::SizeOverflow))
    );
}

#[test]
fn modernbert_laya_compat_batched_heads_and_padding() {
    let backend = CpuBackend;
    let items = [
        Item {
            ids: vec![1, 3, 7, 2, 4, 9],
            markers: vec![1, 4],
            kind: Kind::Choice,
        },
        Item {
            ids: vec![2, 6, 3, 8],
            markers: vec![2],
            kind: Kind::Noul,
        },
    ];
    let batch = collate(&items, 0).unwrap();
    for bias in [false, true] {
        let (bytes, encoder, agent) = fixture(bias);
        let weights = Weights::parse(&bytes).unwrap();
        let model = DecisionModel::load(&backend, &weights, encoder, agent).unwrap();
        let mut ws = model
            .alloc_workspace(&backend, batch.n, batch.seq_len, batch.kmax)
            .unwrap();
        let out = model.forward(&backend, &mut ws, &batch).unwrap();
        // Pinned f32 marker/action logits for the generated checkpoint.
        let (logits, act) = if bias {
            (
                [[-0.12536192, -0.101696126], [-0.12512279, -1e4]],
                [[-1.258848, 1.648989], [-1.2953907, 1.5038197]],
            )
        } else {
            (
                [[-0.10507684, -0.040960923], [-0.10848595, -1e4]],
                [[-0.47215173, 1.050978], [-0.5653026, 1.2019355]],
            )
        };
        for i in 0..batch.n {
            close(&out.logits[i], &logits[i]);
            close(&out.act[i], &act[i]);
        }
        assert_eq!(out.logits[1][1], -1e4);
        for (i, item) in items.iter().enumerate() {
            let single = collate(std::slice::from_ref(item), 0).unwrap();
            let mut ws = model
                .alloc_workspace(&backend, 1, single.seq_len, single.kmax)
                .unwrap();
            let one = model.forward(&backend, &mut ws, &single).unwrap();
            close(&out.logits[i], &one.logits[0]);
            close(&out.act[i], &one.act[0]);
        }
        let repeated = model.forward(&backend, &mut ws, &batch).unwrap();
        for i in 0..batch.n {
            close(&out.logits[i], &repeated.logits[i]);
            close(&out.act[i], &repeated.act[i]);
        }
        let wrong = collate(&items[..1], 0).unwrap();
        assert!(model.forward(&backend, &mut ws, &wrong).is_err());
    }
}

#[test]
fn the_safetensors_header_alone_gives_parameters_and_dtype() {
    use super::weights::Header;
    use crate::registry::{Options, Registry, Source};
    let dir = checkpoint();
    let path = dir.path().join("model.safetensors");
    let bytes = std::fs::read(&path).unwrap();
    let parameters = safetensors::SafeTensors::deserialize(&bytes)
        .unwrap()
        .tensors()
        .iter()
        .map(|(_, view)| view.shape().iter().product::<usize>() as u64)
        .sum();
    assert_eq!(
        Header::read(&path).unwrap(),
        Header {
            parameters,
            dtype: "F32".into()
        }
    );
    let backend = std::sync::Arc::new(CpuBackend);
    let registry = laya_registry(backend.clone(), dir.path(), false);
    let info = registry.info("laya").unwrap();
    assert_eq!(
        (
            info.parameters,
            info.quantization.as_str(),
            info.trained_ctx
        ),
        (parameters, "F32", 16)
    );
    // What a full parse refuses, the header refuses before any load.
    for broken in [&bytes[..bytes.len() - 1], &bytes[..4], &[0xff; 16]] {
        std::fs::write(&path, broken).unwrap();
        assert_matches!(
            Header::read(&path),
            Err(super::weights::Error::Checkpoint(_))
        );
        let listed = vec![("laya".into(), Source::Laya(dir.path().into()))];
        let error = Registry::new(backend.clone(), listed, None, Options::new(32))
            .err()
            .unwrap();
        assert!(error.to_string().contains("safetensors header"), "{error}");
    }
}

#[test]
fn laya_compat_shared_error_mapping() {
    let json_error = EncoderConfig::from_json("{}").unwrap_err();
    assert_matches!(super::Error::from(json_error), super::Error::Json(_));
    let config_error = EncoderConfig::from_json(
        &json!({
            "model_type": "bert", "vocab_size": 32, "hidden_size": 8,
            "intermediate_size": 12, "num_hidden_layers": 3, "num_attention_heads": 2
        })
        .to_string(),
    )
    .unwrap_err();
    assert_matches!(super::Error::from(config_error), super::Error::Config(_));
    let weights_error = Weights::parse(b"not safetensors").err().unwrap();
    assert_matches!(
        super::Error::from(weights_error),
        super::Error::Checkpoint(_)
    );
    let backend_error =
        super::weights::Error::Backend(crate::backend::Error::NotImplemented("test"));
    assert_matches!(super::Error::from(backend_error), super::Error::Backend(_));
}

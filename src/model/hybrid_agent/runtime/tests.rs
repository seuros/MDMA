use super::*;
use crate::backend::{cpu::CpuBackend, test_support::close};
use crate::model::hybrid_agent::{LatentStep, ExitReason};

fn engine() -> Engine<CpuBackend> {
    let mut vocab = (0..128).map(|id| (format!("t{id}"), id)).collect::<std::collections::HashMap<_, _>>();
    for (token, id) in [
        ("<think>", 120), ("</think>", 121), ("<|endoftext|>", 122),
        ("<|im_end|>", 123), ("<latent>", 126), ("</latent>", 127),
    ] {
        vocab.remove(&format!("t{id}"));
        vocab.insert(token.into(), id);
    }
    let tokenizer = test_tokenizer(vocab);
    let controls = Controls::load(&tokenizer).unwrap();
    let backend = Arc::new(CpuBackend);
    let agent = Arc::new(crate::model::hybrid_agent::tests::agent(&*backend));
    Engine { agent, tokenizer: Arc::new(tokenizer), controls, backend }
}

#[test]
fn owned_session_survives_engine_drop_and_resumes_complete_hybrid_kv() {
    let e = engine();
    let mut session = e.session(32, LatentLimits { min_steps: 2, max_steps: 2 }).unwrap();
    session.append("t1 <think>").unwrap();
    let agent_owner = Arc::downgrade(&e.agent);
    let backend_owner = Arc::downgrade(&e.backend);
    drop(e);
    assert_eq!(agent_owner.strong_count(), 1);
    assert_eq!(backend_owner.strong_count(), 1);

    session.token(126).unwrap();
    for steps in 1..=2 {
        assert_eq!(session.step().unwrap(), Some(Step::Latent(LatentStep::Thought { steps })));
    }
    let before = session.position();
    assert!(session.append("t23 t29").is_err());
    assert_eq!(session.position(), before);
    assert_eq!(session.step().unwrap(), Some(Step::Latent(LatentStep::Exit {
        token: 127, steps: 2, reason: ExitReason::MaxSteps,
    })));
    session.append("t23 t29").unwrap();
    session.token(11).unwrap();
    session.finish().unwrap();

    let agent = agent_owner.upgrade().unwrap();
    let backend = backend_owner.upgrade().unwrap();
    let mut reference = agent.decoder(&backend, 32, LatentLimits { min_steps: 2, max_steps: 2 }).unwrap();
    reference.feed_text(&[1, 120]).unwrap();
    reference.token(126).unwrap();
    for _ in 0..3 { reference.latent_step().unwrap(); }
    reference.feed_text(&[23, 29]).unwrap();
    reference.token(11).unwrap();
    assert_eq!(session.position(), reference.position());
    close(&session.logits().unwrap(), &reference.logits().unwrap(), 1e-6);
    drop(reference);
    drop(agent);
    drop(backend);
    drop(session);
    assert!(agent_owner.upgrade().is_none());
    assert!(backend_owner.upgrade().is_none());
}

#[test]
fn eos_is_idempotent_and_external_turn_reopens_generation() {
    let e = engine();
    let mut session = e.session(16, LatentLimits::default()).unwrap();
    session.append("t1").unwrap();
    assert_eq!(session.token(122).unwrap(), Step::Token { token_id: 122, end_of_turn: true });
    let position = session.position();
    assert!(session.is_end_of_turn());
    assert_eq!(session.step().unwrap(), None);
    assert_eq!(session.step().unwrap(), None);
    assert!(session.token(3).is_err());
    assert_eq!(session.position(), position);
    assert!(session.append_tokens(&[]).is_err());
    assert!(session.is_end_of_turn());
    session.append("t23 t29").unwrap();
    assert!(!session.is_end_of_turn());
    assert!(session.step().unwrap().is_some());
}

#[test]
fn sessions_share_only_immutable_weights_and_validate_before_mutating() {
    let e = engine();
    let mut a = e.session(8, LatentLimits::default()).unwrap();
    let mut b = e.session(8, LatentLimits::default()).unwrap();
    for session in [&mut a, &mut b] { session.append("t1").unwrap(); }
    let logits = b.logits().unwrap();
    for id in [121, 126, 127, 128] {
        assert!(a.token(id).is_err());
        assert_eq!(a.position(), 1);
    }
    a.token(120).unwrap();
    assert!(a.token(120).is_err());
    assert!(a.append_tokens(&[1; 8]).is_err());
    assert_eq!(a.position(), 2);
    assert_eq!(b.position(), 1);
    assert_eq!(b.logits().unwrap(), logits);
}

#[test]
fn tokenizer_roundtrips_text_and_rejects_missing_controls() {
    let e = engine();
    let ids = e.encode("t1 <think>").unwrap();
    assert_eq!(ids, [1, 120]);
    assert_eq!(e.decode(&ids).unwrap(), "t1 <think>");
    let tokenizer = test_tokenizer(std::collections::HashMap::from([("t0".into(), 0)]));
    assert!(Controls::load(&tokenizer).is_err());
}

fn test_tokenizer(vocab: std::collections::HashMap<String, u32>) -> Tokenizer {
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "t0"}
    });
    Tokenizer::from_json(&json.to_string()).unwrap()
}

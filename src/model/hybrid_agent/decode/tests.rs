use super::*;
use crate::backend::{cpu::CpuBackend, test_support::close};
use crate::model::hybrid_agent::tests::agent;

mod trace;

fn force_exit(d: &mut HybridDecoder<'_, CpuBackend>) {
    let mut logits = vec![0f32; 128];
    logits[d.agent.exit as usize] = 1.0;
    d.backend.upload(&mut d.ws.logits, 0, bytemuck::cast_slice(&logits)).unwrap();
}

#[test]
fn exit_checks_current_logits_and_minimum_before_injecting_next_vector() {
    let b = CpuBackend;
    let a = agent(&b);
    let mut d = a.decoder(&b, 16, LatentLimits { min_steps: 2, max_steps: 4 }).unwrap();
    d.token(a.entry).unwrap();
    for steps in 1..=2 {
        force_exit(&mut d);
        assert_eq!(d.latent_step().unwrap(), LatentStep::Thought { steps });
    }
    force_exit(&mut d);
    assert_eq!(d.latent_step().unwrap(), LatentStep::Exit { token: a.exit, steps: 2, reason: ExitReason::Anchor });
    assert_eq!(d.position(), 4);
    assert!(!d.in_latent_mode());
    assert!(d.latent_step().is_err());
}

#[test]
fn validation_failures_do_not_mutate_context_or_logits() {
    let b = CpuBackend;
    let a = agent(&b);
    let mut d = a.decoder(&b, 4, LatentLimits::default()).unwrap();
    assert!(d.logits().is_err());
    d.feed_text(&[a.entry]).unwrap();
    assert!(!d.in_latent_mode());
    let want = d.logits().unwrap();
    for tokens in [&[][..], &[3, 128][..], &[1, 2, 3, 4][..]] {
        assert!(d.feed_text(tokens).is_err());
        assert_eq!(d.position(), 1);
        close(&d.logits().unwrap(), &want, 1e-6);
    }
    assert!(d.token(a.exit).is_err());
    d.token(a.entry).unwrap();
    assert!(d.token(3).is_err());
    assert!(d.feed_text(&[5]).is_err());
    assert_eq!(d.position(), 2);
}

#[test]
fn full_context_does_not_overwrite_or_replay_a_latent_position() {
    let b = CpuBackend;
    let a = agent(&b);
    let mut d = a.decoder(&b, 1, LatentLimits::default()).unwrap();
    d.token(a.entry).unwrap();
    let logits = d.logits().unwrap();
    assert!(d.latent_step().is_err());
    assert_eq!(d.position(), 1);
    close(&d.logits().unwrap(), &logits, 1e-6);
}

#[test]
fn invalid_limits_fail_before_allocation_and_nonfinite_logits_poison_trajectory() {
    let b = CpuBackend;
    let a = agent(&b);
    for (context, limits) in [
        (0, LatentLimits::default()), (65, LatentLimits::default()),
        (8, LatentLimits { min_steps: 2, max_steps: 1 }),
        (8, LatentLimits { min_steps: 0, max_steps: 0 }),
    ] { assert!(a.decoder(&b, context, limits).is_err()); }
    let mut d = a.decoder(&b, 8, LatentLimits::default()).unwrap();
    d.token(a.entry).unwrap();
    d.backend.upload(&mut d.ws.logits, 0, &f32::NAN.to_ne_bytes()).unwrap();
    assert!(d.latent_step().is_err());
    assert!(d.logits().is_err());
    assert!(d.feed_text(&[1]).is_err());
}

#[test]
fn continuous_steps_only_download_one_index_without_allocations_or_uploads() {
    let b = trace::Trace::default();
    let a = agent(&b);
    let mut d = a.decoder(&b, 8, LatentLimits::default()).unwrap();
    d.token(a.entry).unwrap();
    b.downloads.borrow_mut().clear();
    let (allocations, uploads) = (b.allocations.get(), b.uploads.get());
    assert_eq!(d.latent_step().unwrap(), LatentStep::Thought { steps: 1 });
    assert_eq!(b.downloads.borrow().as_slice(), &[4]);
    assert_eq!(b.allocations.get(), allocations);
    assert_eq!(b.uploads.get(), uploads);
    assert_eq!(d.position(), 2);
}

#[test]
fn partially_written_layer_cache_is_never_reused_after_backend_failure() {
    let b = trace::Trace::default();
    let a = agent(&b);
    let mut d = a.decoder(&b, 8, LatentLimits::default()).unwrap();
    d.token(a.entry).unwrap();
    b.fail_attention.set(true);
    assert!(d.latent_step().is_err());
    b.fail_attention.set(false);
    assert!(d.latent_step().is_err());
    assert!(d.token(1).is_err());
    assert!(d.feed_text(&[1]).is_err());
    assert!(d.logits().is_err());
}

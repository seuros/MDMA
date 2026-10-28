use mdma::{backend::cpu::CpuBackend, hybrid_agent::{Engine, Session, Step}};

#[test]
fn public_embedding_types_are_owned_and_thread_safe_on_cpu() {
    fn owned_send_sync<T: Send + Sync + 'static>() {}
    owned_send_sync::<Engine<CpuBackend>>();
    owned_send_sync::<Session<CpuBackend>>();
    owned_send_sync::<Step>();
    owned_send_sync::<mdma::Error>();
}

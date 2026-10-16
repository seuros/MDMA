#![expect(
    clippy::arc_with_non_send_sync,
    reason = "the loader fakes count with Cell; a registry owns its backend"
)]
use super::*;
use crate::backend::cpu::CpuBackend;
use crate::backend::{self, AttnSpec, KvView, Pool, QWeight, RopeSpec};
use crate::gguf::tests::Fixture;
use crate::sampler::SamplerConfig;
use crate::session::Session;
use std::assert_matches;
use std::cell::{Cell, RefCell};
use std::fs::File;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::time::Duration;
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

// Loader-only backend: tracks buffer lifetime and injects failures; no kernels.
#[derive(Default)]
struct LoadBackend {
    allocations: Cell<usize>,
    uploads: Cell<usize>,
    live_bytes: Arc<AtomicUsize>,
    fail_upload: Cell<Option<usize>>,
    fail_memory_upload: Cell<Option<usize>>,
    free_device_bytes: Cell<Option<usize>>,
    on_upload: RefCell<Option<Box<dyn FnOnce()>>>,
}

struct Buffer {
    bytes: Vec<u8>,
    live_bytes: Arc<AtomicUsize>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.live_bytes.fetch_sub(self.bytes.len(), SeqCst);
    }
}

impl Backend for LoadBackend {
    type Buf = Buffer;

    fn device_label(&self) -> Option<String> {
        Some("CPU".into())
    }

    fn available_device_bytes(&self) -> backend::Result<Option<usize>> {
        Ok(self
            .free_device_bytes
            .get()
            .map(|bytes| bytes.saturating_sub(self.live_bytes.load(SeqCst))))
    }

    fn alloc(&self, bytes: usize, pool: Pool) -> backend::Result<Buffer> {
        assert_matches!(pool, Pool::Weights);
        self.allocations.set(self.allocations.get() + 1);
        self.live_bytes.fetch_add(bytes, SeqCst);
        Ok(Buffer {
            bytes: vec![0; bytes],
            live_bytes: self.live_bytes.clone(),
        })
    }
    fn upload(&self, dst: &mut Buffer, offset: usize, src: &[u8]) -> backend::Result<()> {
        let count = self.uploads.get() + 1;
        self.uploads.set(count);
        if self.fail_memory_upload.get() == Some(count) {
            return Err(backend::Error::OutOfMemory(
                "injected allocation race".into(),
            ));
        }
        if self.fail_upload.get() == Some(count) {
            return Err(backend::Error::Backend("injected upload failure".into()));
        }
        dst.bytes[offset..offset + src.len()].copy_from_slice(src);
        if let Some(hook) = self.on_upload.borrow_mut().take() {
            hook();
        }
        Ok(())
    }
    fn download(&self, _: &Buffer, _: usize, _: &mut [u8]) -> backend::Result<()> {
        unreachable!("loader-only backend")
    }
    fn copy(
        &self,
        _: &Buffer,
        _: usize,
        _: &mut Buffer,
        _: usize,
        _: usize,
    ) -> backend::Result<()> {
        unreachable!("loader-only backend")
    }
    fn qmatmul(&self, _: &QWeight<Buffer>, _: &Buffer, _: &mut Buffer) -> backend::Result<()> {
        unreachable!("loader-only backend")
    }
    fn embed(&self, _: &QWeight<Buffer>, _: &[u32], _: &mut Buffer) -> backend::Result<()> {
        unreachable!("loader-only backend")
    }
    fn rmsnorm(&self, _: &Buffer, _: &Buffer, _: &mut Buffer, _: f32) -> backend::Result<()> {
        unreachable!("loader-only backend")
    }
    fn add(&self, _: &Buffer, _: &Buffer, _: &mut Buffer) -> backend::Result<()> {
        unreachable!("loader-only backend")
    }
    fn rope(
        &self,
        _: &mut Buffer,
        _: &mut Buffer,
        _: u32,
        _: usize,
        _: &RopeSpec,
    ) -> backend::Result<()> {
        unreachable!("loader-only backend")
    }
    fn attn(
        &self,
        _: &Buffer,
        _: &KvView<Buffer>,
        _: &AttnSpec,
        _: &mut Buffer,
    ) -> backend::Result<()> {
        unreachable!("loader-only backend")
    }
    fn silu_mul(&self, _: &Buffer, _: &Buffer, _: &mut Buffer) -> backend::Result<()> {
        unreachable!("loader-only backend")
    }
}

/// `m0`, `m1`, … in order.
fn models(paths: &[&Path]) -> Vec<(String, Source)> {
    paths
        .iter()
        .enumerate()
        .map(|(i, path)| (format!("m{i}"), Source::Gguf(path.to_path_buf())))
        .collect()
}

fn registry<B: Backend>(backend: &Arc<B>, paths: &[&Path], options: Options) -> Registry<B> {
    Registry::new(backend.clone(), models(paths), None, options).unwrap()
}

fn frozen(n_ctx: usize) -> Options {
    Options {
        frozen: true,
        ..Options::new(n_ctx)
    }
}

fn budgeted(budget: &NativeMemoryBudget) -> Options {
    Options {
        budget: Some(budget.clone()),
        ..Options::new(32)
    }
}

fn decoder<B: Backend>(registry: &Registry<B>, name: &str) -> Decoder<B> {
    registry
        .acquire_blocking(name)
        .unwrap()
        .decoder()
        .unwrap()
        .clone()
}

fn phases<B: Backend>(registry: &Registry<B>) -> Vec<Phase> {
    registry.entries().map(|entry| entry.phase).collect()
}

fn weights(path: &Path) -> usize {
    crate::model::memory::weight_bytes(&GgufModel::open(path).unwrap()).unwrap()
}

fn change_stamp(file: &File) {
    let modified = file.metadata().unwrap().modified().unwrap();
    file.set_modified(modified + Duration::from_secs(1))
        .unwrap();
}

fn forward(model: &Model<CpuBackend>) -> Vec<f32> {
    let mut kv = model.new_kv_cache(&CpuBackend, 1).unwrap();
    let mut ws = model.alloc_workspace(&CpuBackend, 1).unwrap();
    model.forward(&CpuBackend, &mut kv, &mut ws, 1, 0).unwrap()
}

#[test]
fn preflight_reads_metadata_once_per_file_and_uploads_nothing() {
    let fixture = Fixture::new(1, 8).with_tokenizer();
    let file = fixture.write();
    let backend = Arc::new(LoadBackend::default());
    let registry = Registry::new(
        backend.clone(),
        models(&[file.path(), file.path(), file.path()]),
        Some("m2"),
        Options::new(16),
    )
    .unwrap();
    assert_eq!(registry.names(), ["m0", "m1", "m2"]);
    assert_eq!(
        (registry.default_name(), registry.resolve("")),
        (Some("m2"), Some("m2"))
    );
    assert!(
        registry
            .entries()
            .all(|entry| entry.upload == 0 && entry.phase == Phase::Available)
    );
    let info = registry.info("m1").unwrap();
    assert_eq!(
        (info.kind, info.architecture.as_str(), info.n_ctx),
        (Kind::Gguf, "internlm2", 16)
    );
    assert_eq!(info.disk_bytes, file.as_file().metadata().unwrap().len());
    assert_eq!(
        (
            info.parameters,
            info.quantization.as_str(),
            info.trained_ctx
        ),
        (fixture.parameters(), "F32", 64),
        "from the header, before the cap"
    );
    // Uncapped, a model gets its trained context; a cap never raises it.
    for cap in [usize::MAX, 1000] {
        let uncapped = Registry::new(
            backend.clone(),
            models(&[file.path()]),
            None,
            Options::new(cap),
        )
        .unwrap();
        assert_eq!(uncapped.info("m0").unwrap().n_ctx, 64, "cap {cap}");
    }
    assert_eq!((backend.allocations.get(), backend.uploads.get()), (0, 0));
    assert!(registry.resident("m0").is_none());
}

#[test]
fn parameter_counts_read_like_model_names() {
    for (n, expected) in [
        (0, ""),
        (999, "999"),
        (2648, "2.6K"),
        (149_655_232, "150M"),
        (494_032_768, "494M"),
        (999_600_000, "1.0B"),
        (1_100_048_384, "1.1B"),
        (7_241_732_096, "7.2B"),
        (9_960_000_000, "10B"),
        (70_553_706_496, "71B"),
        (1_500_000_000_000, "1.5T"),
        (u64::MAX, "18446744T"),
    ] {
        assert_eq!(parameters_label(n), expected, "{n}");
    }
}

#[test]
fn listing_errors_are_reported_before_any_load() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let backend = Arc::new(CpuBackend);
    let path = || Source::Gguf(file.path().into());
    for (models, default, needle) in [
        (vec![], None, "no models"),
        (
            vec![("a".to_owned(), path()), ("a".to_owned(), path())],
            None,
            "twice",
        ),
        (vec![(" ".to_owned(), path())], None, "empty name"),
        (vec![("a".to_owned(), path())], Some("b"), "not listed"),
    ] {
        let error = Registry::new(backend.clone(), models, default, Options::new(32))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(needle), "{error}");
    }
}

#[test]
fn aliases_share_one_upload_and_tokenizer_until_the_last_use_drops() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let backend = Arc::new(LoadBackend::default());
    let registry = registry(
        &backend,
        &[file.path(), file.path(), file.path()],
        Options::new(16),
    );
    assert_eq!(registry.preload(), Vec::new());
    let first = decoder(&registry, "m0");
    for name in ["m1", "m2"] {
        let alias = decoder(&registry, name);
        assert!(Arc::ptr_eq(&first.model, &alias.model));
        assert!(Arc::ptr_eq(&first.tokenizer, &alias.tokenizer));
    }
    assert_eq!((backend.allocations.get(), backend.uploads.get()), (12, 12));
    let live_bytes = backend.live_bytes.load(SeqCst);
    assert!(live_bytes > 0);
    drop(registry);
    assert_eq!(
        backend.live_bytes.load(SeqCst),
        live_bytes,
        "a use outlives the registry"
    );
    drop(first);
    assert_eq!(backend.live_bytes.load(SeqCst), 0);
}

#[cfg(unix)]
#[test]
fn symlinks_hardlinks_and_dotted_paths_share_the_open_file() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let dir = tempfile::tempdir().unwrap();
    let hardlink = dir.path().join("hard.gguf");
    let symlink = dir.path().join("symbolic.gguf");
    std::fs::hard_link(file.path(), &hardlink).unwrap();
    std::os::unix::fs::symlink(file.path(), &symlink).unwrap();
    let dotted = dir.path().join(".").join("hard.gguf");
    let backend = Arc::new(LoadBackend::default());
    let registry = registry(
        &backend,
        &[file.path(), &hardlink, &symlink, &dotted],
        Options::new(32),
    );
    assert_eq!(registry.preload(), Vec::new());
    for name in ["m1", "m2", "m3"] {
        assert!(Arc::ptr_eq(
            &decoder(&registry, "m0").model,
            &decoder(&registry, name).model
        ));
    }
    assert_eq!((backend.allocations.get(), backend.uploads.get()), (12, 12));
}

#[test]
fn copies_and_same_sized_different_weights_are_not_aliases() {
    let first = Fixture::new(1, 8).with_tokenizer().write();
    let copy = tempfile::NamedTempFile::new().unwrap();
    std::fs::copy(first.path(), copy.path()).unwrap();
    let mut different = Fixture::new(1, 8).with_tokenizer();
    different.replace("output.weight", &[128, 8], crate::gguf::GgmlDType::F32);
    let different = different.write();
    assert_eq!(
        first.as_file().metadata().unwrap().len(),
        different.as_file().metadata().unwrap().len()
    );
    let backend = Arc::new(LoadBackend::default());
    let registry = registry(
        &backend,
        &[first.path(), copy.path(), different.path()],
        Options::new(32),
    );
    assert_eq!(registry.preload(), Vec::new());
    let first = decoder(&registry, "m0");
    for name in ["m1", "m2"] {
        assert!(!Arc::ptr_eq(&first.model, &decoder(&registry, name).model));
        assert!(!Arc::ptr_eq(
            &first.tokenizer,
            &decoder(&registry, name).tokenizer
        ));
    }
    assert_eq!((backend.allocations.get(), backend.uploads.get()), (36, 36));
    drop((registry, first));
    assert_eq!(backend.live_bytes.load(SeqCst), 0);
}

#[test]
fn separate_registries_do_not_share_backend_buffers() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let (a, b) = (
        Arc::new(LoadBackend::default()),
        Arc::new(LoadBackend::default()),
    );
    let first = registry(&a, &[file.path(), file.path()], Options::new(32));
    let second = registry(&b, &[file.path(), file.path()], Options::new(32));
    assert!(!Arc::ptr_eq(
        &decoder(&first, "m0").model,
        &decoder(&second, "m0").model
    ));
    assert_eq!((a.uploads.get(), b.uploads.get()), (12, 12));
    drop(first);
    assert_eq!(a.live_bytes.load(SeqCst), 0);
    assert!(b.live_bytes.load(SeqCst) > 0);
}

#[test]
fn a_bad_tokenizer_is_a_listing_error_before_any_upload() {
    let first = Fixture::new(1, 8).with_tokenizer().write();
    let second = Fixture::new(1, 8).write();
    let backend = Arc::new(LoadBackend::default());
    let error = Registry::new(
        backend.clone(),
        models(&[first.path(), first.path(), second.path()]),
        None,
        Options::new(32),
    )
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("model m2: failed to build tokenizer"),
        "{error}"
    );
    assert_eq!((backend.allocations.get(), backend.uploads.get()), (0, 0));
}

#[test]
fn budget_charges_alias_weights_once_until_the_last_use() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let bytes = weights(file.path());
    let budget = NativeMemoryBudget::new(bytes);
    let backend = Arc::new(LoadBackend::default());
    let registry = registry(&backend, &[file.path(), file.path()], budgeted(&budget));
    assert_eq!(registry.preload(), Vec::new());
    assert_eq!(budget.used(), bytes);
    assert_eq!(budget.used(), backend.live_bytes.load(SeqCst));
    let held = decoder(&registry, "m1").model;
    drop(registry);
    assert_eq!(budget.used(), bytes);
    drop(held);
    assert_eq!(budget.used(), 0);
}

#[test]
fn device_pressure_refuses_before_upload_and_a_later_use_retries() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let bytes = weights(file.path());
    let backend = Arc::new(LoadBackend::default());
    backend
        .free_device_bytes
        .set(Some(bytes + DEVICE_HEADROOM - 1));
    let registry = Registry::new(
        backend.clone(),
        models(&[file.path(), file.path()]),
        Some("m1"),
        Options::new(32),
    )
    .unwrap();
    let failures = registry.preload();
    assert_eq!(failures.len(), 1);
    assert_matches!(&failures[0].1, LoadError::Memory(e) if e.contains("insufficient device memory"), "{failures:?}");
    for entry in registry.entries() {
        assert_eq!(entry.phase, Phase::Available, "a refusal is not a failure");
        assert!(
            entry
                .error
                .as_deref()
                .is_some_and(|e| e.contains("insufficient device memory"))
        );
    }
    assert_eq!((backend.allocations.get(), backend.uploads.get()), (0, 0));
    backend.free_device_bytes.set(None);
    assert!(
        registry.acquire_blocking("m1").is_ok(),
        "the next use loads it"
    );
    assert_eq!(registry.default_name(), Some("m1"));
}

#[test]
fn a_model_that_does_not_fit_leaves_the_others_loading() {
    let large = Fixture::new(2, 8).with_tokenizer().write();
    let small = Fixture::new(1, 8).with_tokenizer().write();
    let bytes = weights(small.path());
    let backend = Arc::new(LoadBackend::default());
    backend.free_device_bytes.set(Some(bytes + DEVICE_HEADROOM));
    let registry = registry(&backend, &[large.path(), small.path()], Options::new(32));
    registry.preload();
    assert_eq!(phases(&registry), [Phase::Available, Phase::Resident]);
    assert_eq!(backend.live_bytes.load(SeqCst), bytes);
    assert_eq!(
        registry.default_name(),
        Some("m0"),
        "never silently change the requested default"
    );
}

#[test]
fn unique_weights_are_admitted_one_model_at_a_time() {
    let a = Fixture::new(1, 8).with_tokenizer().write();
    let b = Fixture::new(1, 8).with_tokenizer().write();
    let bytes = weights(a.path());
    let budget = NativeMemoryBudget::new(bytes);
    let backend = Arc::new(LoadBackend::default());
    let registry = registry(&backend, &[a.path(), b.path(), a.path()], budgeted(&budget));
    assert!(
        registry.preload().is_empty(),
        "each load evicts the idle one before it"
    );
    assert_eq!(
        phases(&registry),
        [Phase::Available, Phase::Resident, Phase::Available]
    );
    assert_eq!(
        (
            budget.used(),
            backend.live_bytes.load(SeqCst),
            backend.uploads.get()
        ),
        (bytes, bytes, 24)
    );
    let alias = decoder(&registry, "m2");
    assert!(
        Arc::ptr_eq(&alias.model, &decoder(&registry, "m0").model),
        "aliases share one upload"
    );
    assert_eq!(
        (
            budget.used(),
            backend.live_bytes.load(SeqCst),
            backend.uploads.get()
        ),
        (bytes, bytes, 36)
    );
    drop(alias);
    // A budget is shared across registries; file identity is not.
    let other = self::registry(&backend, &[b.path()], budgeted(&budget));
    assert_matches!(other.acquire_blocking("m0").map(|_| ()), Err(LoadError::Memory(e)) if e.contains("budget exhausted"));
    drop(registry);
    assert_eq!((budget.used(), backend.live_bytes.load(SeqCst)), (0, 0));
}

#[test]
fn a_partial_oom_upload_is_refunded_and_retried_later() {
    let a = Fixture::new(1, 8).with_tokenizer().write();
    let b = Fixture::new(1, 8).with_tokenizer().write();
    let bytes = weights(a.path());
    let budget = NativeMemoryBudget::new(bytes);
    let backend = Arc::new(LoadBackend::default());
    backend.fail_memory_upload.set(Some(3));
    let registry = registry(&backend, &[a.path(), a.path(), b.path()], budgeted(&budget));
    registry.preload();
    assert_eq!(
        phases(&registry),
        [Phase::Available, Phase::Available, Phase::Resident]
    );
    assert!(
        registry
            .entries()
            .next()
            .unwrap()
            .error
            .unwrap()
            .contains("injected allocation race")
    );
    assert_eq!(
        (
            budget.used(),
            backend.live_bytes.load(SeqCst),
            backend.uploads.get()
        ),
        (bytes, bytes, 15)
    );
    drop(registry);
    assert_eq!((budget.used(), backend.live_bytes.load(SeqCst)), (0, 0));
}

#[test]
fn driver_failures_are_sticky_and_release_their_partial_upload() {
    let a = Fixture::new(1, 8).with_tokenizer().write();
    let b = Fixture::new(1, 8).with_tokenizer().write();
    let budget = NativeMemoryBudget::new(weights(a.path()) * 2);
    let backend = Arc::new(LoadBackend::default());
    backend.fail_upload.set(Some(15));
    let registry = registry(&backend, &[a.path(), b.path(), a.path()], budgeted(&budget));
    let failures = registry.preload();
    assert_matches!(&failures[..], [(name, LoadError::Failed(e))] if name == "m1" && e.contains("injected upload failure"), "{failures:?}");
    assert_eq!(
        phases(&registry),
        [Phase::Resident, Phase::Failed, Phase::Resident]
    );
    assert_eq!(
        budget.used(),
        weights(a.path()),
        "only the loaded model is charged"
    );
    backend.fail_upload.set(None);
    let before = backend.uploads.get();
    assert_matches!(
        registry.acquire_blocking("m1").map(|_| ()),
        Err(LoadError::Failed(_)),
        "a broken model is not retried"
    );
    assert_eq!(backend.uploads.get(), before);
}

#[test]
fn a_panicking_load_puts_the_model_back_and_refunds() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let budget = NativeMemoryBudget::new(100_000);
    let backend = Arc::new(LoadBackend::default());
    let registry = registry(&backend, &[file.path()], budgeted(&budget));
    *backend.on_upload.borrow_mut() = Some(Box::new(|| panic!("injected load panic")));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        registry.acquire_blocking("m0").is_ok()
    }));
    assert!(result.is_err());
    assert_eq!((budget.used(), backend.live_bytes.load(SeqCst)), (0, 0));
    assert_eq!(phases(&registry), [Phase::Available]);
    assert!(registry.acquire_blocking("m0").is_ok());
}

#[test]
fn a_source_change_after_preflight_fails_before_upload() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let backend = Arc::new(LoadBackend::default());
    let registry = registry(&backend, &[file.path(), file.path()], Options::new(32));
    change_stamp(file.as_file());
    let Err(LoadError::Failed(error)) = registry.acquire_blocking("m1") else {
        panic!("a changed file must not load")
    };
    assert!(error.contains("source changed since preflight"), "{error}");
    assert_eq!((backend.allocations.get(), backend.uploads.get()), (0, 0));
    assert_eq!(phases(&registry), [Phase::Failed, Phase::Failed]);
}

#[test]
fn a_source_change_during_upload_releases_the_partial_load() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let backend = Arc::new(LoadBackend::default());
    let source = file.reopen().unwrap();
    *backend.on_upload.borrow_mut() = Some(Box::new(move || change_stamp(&source)));
    let registry = registry(&backend, &[file.path(), file.path()], Options::new(32));
    let Err(LoadError::Failed(error)) = registry.acquire_blocking("m0") else {
        panic!("a changed file must not load")
    };
    assert!(error.contains("source changed"), "{error}");
    assert!(backend.uploads.get() > 0);
    assert_eq!(backend.live_bytes.load(SeqCst), 0);
}

#[cfg(unix)]
#[test]
fn replacing_the_path_does_not_switch_the_preflighted_payload() {
    let first = Fixture::new(1, 8).with_tokenizer().write();
    let mut different = Fixture::new(1, 8).with_tokenizer();
    different.replace("output.weight", &[128, 8], crate::gguf::GgmlDType::F32);
    let second = different.write();
    let dir = tempfile::tempdir().unwrap();
    let link = dir.path().join("model.gguf");
    std::os::unix::fs::symlink(first.path(), &link).unwrap();
    let backend = Arc::new(CpuBackend);
    for options in [Options::new(32), frozen(32)] {
        let registry = registry(&backend, &[&link, &link], options);
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(second.path(), &link).unwrap();
        let loaded = decoder(&registry, "m1");
        assert!(Arc::ptr_eq(&loaded.model, &decoder(&registry, "m0").model));
        let original =
            Model::load(&CpuBackend, &mut GgufModel::open(first.path()).unwrap()).unwrap();
        let replacement = Model::load(&CpuBackend, &mut GgufModel::open(&link).unwrap()).unwrap();
        assert_eq!(forward(&loaded.model), forward(&original));
        assert_ne!(forward(&loaded.model), forward(&replacement));
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(first.path(), &link).unwrap();
    }
}

#[test]
fn frozen_aliases_keep_the_exact_digest_when_the_original_changes_during_upload() {
    use sha2::{Digest, Sha256};
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let expected = format!(
        "sha256:{}",
        hex::encode(Sha256::digest(std::fs::read(file.path()).unwrap()))
    );
    let backend = Arc::new(LoadBackend::default());
    let path = file.path().to_owned();
    *backend.on_upload.borrow_mut() =
        Some(Box::new(move || std::fs::write(path, b"replaced").unwrap()));
    let registry = registry(&backend, &[file.path(), file.path()], frozen(32));
    assert!(
        registry.artifact_id("").is_none(),
        "no identity before a load"
    );
    assert_eq!(registry.preload(), Vec::new());
    assert_eq!(registry.artifact_id("").unwrap().as_str(), expected);
    assert_eq!(registry.artifact_id("m1"), registry.artifact_id("m0"));
    assert!(registry.artifact_id("missing").is_none());
    assert!(Arc::ptr_eq(
        &decoder(&registry, "m0").model,
        &decoder(&registry, "m1").model
    ));
}

#[test]
fn frozen_loads_refuse_a_file_changed_in_place_since_preflight() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let registry = registry(&Arc::new(CpuBackend), &[file.path()], frozen(32));
    let mut replacement = Fixture::new(1, 8).with_tokenizer();
    replacement.replace("output.weight", &[128, 8], crate::gguf::GgmlDType::F32);
    std::fs::copy(replacement.write().path(), file.path()).unwrap();
    assert_matches!(registry.acquire_blocking("m0").map(|_| ()), Err(LoadError::Failed(e)) if e.contains("changed"));
    assert!(registry.artifact_id("m0").is_none());
}

#[test]
fn the_complete_gguf_digest_includes_unconsumed_trailing_bytes() {
    use std::io::{Seek, Write};
    let mut file = Fixture::new(1, 8).with_tokenizer().write();
    let backend = Arc::new(CpuBackend);
    let first = registry(&backend, &[file.path()], frozen(32));
    first.preload();
    file.as_file_mut().seek(std::io::SeekFrom::End(0)).unwrap();
    file.write_all(b"whole-artifact-trailer").unwrap();
    let second = registry(&backend, &[file.path()], frozen(32));
    second.preload();
    assert_ne!(first.artifact_id(""), second.artifact_id(""));
}

#[test]
fn refused_models_have_no_publishable_identity() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let budget = NativeMemoryBudget::new(0);
    let registry = registry(
        &Arc::new(CpuBackend),
        &[file.path(), file.path()],
        Options {
            budget: Some(budget),
            ..frozen(32)
        },
    );
    registry.preload();
    assert!(registry.artifact_id("m0").is_none() && registry.artifact_id("m1").is_none());
}

#[test]
fn native_cpu_profiles_cover_loaded_frozen_models_only() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let backend = Arc::new(CpuBackend);
    let mut registry = registry(&backend, &[file.path(), file.path()], frozen(32));
    assert_eq!(registry.capture_execution_profiles().unwrap(), 2);
    assert!(
        registry.execution_profile("").is_none(),
        "no profile before the artifact is known"
    );
    registry.preload();
    let profile = registry.execution_profile("").unwrap();
    assert!(profile.is_current());
    assert_eq!(
        profile.as_str(),
        registry.execution_profile("m1").unwrap().as_str()
    );
    assert_eq!(
        profile.manifest().as_object().unwrap()["kv_dtype"].as_str(),
        Some("f32")
    );
    assert!(registry.execution_profile("missing").is_none());
    let mut single = Registry::single(
        backend.clone(),
        "library",
        "internlm2",
        32,
        decoder(&registry, ""),
    );
    assert_eq!(single.capture_execution_profiles().unwrap(), 0);
    assert!(single.execution_profile("").is_none());
    let mut normal = self::registry(&backend, &[file.path()], Options::new(32));
    normal.preload();
    assert_eq!(normal.capture_execution_profiles().unwrap(), 0);
    assert!(normal.execution_profile("").is_none());
}

#[test]
fn a_backend_calling_itself_cpu_does_not_get_the_native_cpu_profile() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let backend = Arc::new(LoadBackend::default());
    assert_eq!(backend.device_label().as_deref(), Some("CPU"));
    let mut registry = registry(&backend, &[file.path()], frozen(32));
    registry.preload();
    assert!(registry.artifact_id("").is_some());
    assert_eq!(registry.capture_execution_profiles().unwrap(), 0);
    assert!(registry.execution_profile("").is_none());
}

#[test]
fn a_failed_recapture_clears_profiles_without_unloading_models() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let mut registry = registry(
        &Arc::new(CpuBackend),
        &[file.path(), file.path()],
        frozen(32),
    );
    registry.preload();
    assert_eq!(registry.capture_execution_profiles().unwrap(), 2);
    let error = registry
        .capture_execution_profiles_with(|| {
            Err(std::io::Error::other("running image is unverifiable"))
        })
        .unwrap_err();
    assert!(error.to_string().contains("unverifiable"));
    for name in ["m0", "m1"] {
        assert!(registry.artifact_id(name).is_some());
        assert!(registry.execution_profile(name).is_none());
    }
    assert_eq!(phases(&registry), [Phase::Resident, Phase::Resident]);
}

#[cfg(any(feature = "vulkan", all(feature = "metal", target_os = "macos")))]
fn frozen_gpu_payload_matches_cpu_after_original_is_removed<B>(backend: B)
where
    B: Backend + Send + Sync + 'static,
    B::Buf: Send + Sync,
{
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let cpu = registry(
        &Arc::new(CpuBackend),
        &[file.path(), file.path()],
        frozen(32),
    );
    let backend = Arc::new(backend);
    let mut gpu = registry(&backend, &[file.path(), file.path()], frozen(32));
    assert!(cpu.preload().is_empty() && gpu.preload().is_empty());
    assert_eq!(gpu.capture_execution_profiles().unwrap(), 2);
    assert_eq!(
        gpu.execution_profile("m0").unwrap().as_str(),
        gpu.execution_profile("m1").unwrap().as_str()
    );
    assert_ne!(
        gpu.execution_profile("m0")
            .unwrap()
            .manifest()
            .as_object()
            .unwrap()["backend"]
            .as_str(),
        Some("cpu")
    );
    assert_eq!(cpu.artifact_id(""), gpu.artifact_id(""));
    std::fs::remove_file(file.path()).unwrap();
    let expected = forward(&decoder(&cpu, "").model);
    let model = decoder(&gpu, "").model;
    let mut kv = model.new_kv_cache(&*backend, 1).unwrap();
    let mut ws = model.alloc_workspace(&*backend, 1).unwrap();
    let actual = model.forward(&*backend, &mut kv, &mut ws, 1, 0).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(&expected) {
        assert!(
            actual.is_finite() && (actual - expected).abs() < 2e-3,
            "{actual} != {expected}"
        );
    }
    let inventory = crate::federation::inventory::Native {
        registry: Arc::new(gpu),
        shutdown: None,
        power: None,
        host: Arc::default(),
        max_output_tokens: usize::MAX,
    };
    let snapshot =
        crate::federation::inventory::Inventory::snapshot(&inventory, &["m0".into(), "m1".into()]);
    assert_eq!(
        snapshot.models.len(),
        2,
        "actual current GPU profiles can be cataloged"
    );
    assert!(snapshot.models.iter().all(|m| m.backend != "cpu"));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn frozen_metal_payload_matches_cpu_after_source_removal() {
    frozen_gpu_payload_matches_cpu_after_original_is_removed(
        crate::backend::metal::MetalBackend::new().unwrap(),
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn frozen_vulkan_payload_matches_cpu_after_source_removal() {
    frozen_gpu_payload_matches_cpu_after_original_is_removed(
        crate::backend::vulkan::VulkanBackend::new()
            .unwrap()
            .with_f16_weights(false),
    );
}

#[cfg(feature = "vulkan")]
#[test]
fn vulkan_profile_is_withdrawn_on_mode_change_even_when_restored() {
    let backend = Arc::new(crate::backend::vulkan::VulkanBackend::new().unwrap());
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let mut registry = registry(&backend, &[file.path()], frozen(32));
    registry.preload();
    registry.capture_execution_profiles().unwrap();
    let before = registry.execution_profile("").unwrap().as_str().to_string();
    let mode = backend.fused_norm();
    backend.set_fused_norm(!mode);
    assert!(registry.execution_profile("").is_none());
    backend.set_fused_norm(mode);
    assert!(registry.execution_profile("").is_none());
    registry.capture_execution_profiles().unwrap();
    assert_eq!(registry.execution_profile("").unwrap().as_str(), before);
}

#[test]
#[ignore = "local MiniCheck file; metadata and tokenizer only, no weight uploads"]
fn real_minicheck_alias_preflight() {
    let path = crate::backend::test_support::fixture();
    let backend = Arc::new(LoadBackend::default());
    let registry = registry(&backend, &[path, path], Options::new(4096));
    assert!(registry.entries().all(|entry| entry.upload == 0));
    assert_eq!(registry.info("m0").unwrap().class, Class::Decision);
    assert!(weights(path) > u32::MAX as usize);
    assert_eq!((backend.allocations.get(), backend.uploads.get()), (0, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn alias_sessions_run_concurrently_with_independent_kv() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let backend = Arc::new(CpuBackend);
    let registry = registry(&backend, &[file.path(), file.path()], Options::new(8));
    let session = |name| {
        let d = decoder(&registry, name);
        Session::new(
            backend.clone(),
            d.model,
            d.tokenizer,
            SamplerConfig {
                temperature: 0.0,
                ..Default::default()
            },
            8,
        )
    };
    let (a, b) = (session("m0"), session("m1"));
    let collect = |s: &Session<CpuBackend>, prompt: &str, count| {
        let stream = s.generate(prompt, count, CancellationToken::new()).unwrap();
        async move {
            stream
                .map(|item| item.unwrap().token_id)
                .collect::<Vec<_>>()
                .await
        }
    };
    assert!(
        a.generate("ABCDEFGHI", 1, CancellationToken::new())
            .is_err(),
        "the global n_ctx caps every alias"
    );
    let expected_a = collect(&a, "AB", 6).await;
    let expected_b = collect(&b, "ABCDE", 6).await;
    let (actual_a, actual_b) = tokio::join!(collect(&a, "AB", 6), collect(&b, "ABCDE", 6));
    assert_eq!((actual_a, actual_b), (expected_a, expected_b));
}

#[test]
fn hashing_names_models_before_they_load_with_the_digest_a_load_reports() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let laya = crate::laya::compat_tests::checkpoint();
    let backend = Arc::new(CpuBackend);
    let models = vec![
        ("m0".into(), Source::Gguf(file.path().into())),
        ("m1".into(), Source::Gguf(file.path().into())),
        ("laya".into(), Source::Laya(laya.path().into())),
    ];
    let registry = Registry::new(backend.clone(), models.clone(), None, frozen(32)).unwrap();
    let mut changes = registry.subscribe();
    registry.hash_artifacts(|| false);
    assert!(
        changes.has_changed().unwrap(),
        "Watch learns of new identities"
    );
    let hashed: Vec<_> = ["m0", "m1", "laya"]
        .map(|name| registry.artifact_id(name).unwrap())
        .into();
    assert_eq!(hashed[0], hashed[1], "aliases are hashed once");
    assert_eq!(
        phases(&registry),
        [Phase::Available; 3],
        "hashing loads nothing"
    );
    let loaded = Registry::new(backend, models, None, frozen(32)).unwrap();
    assert_eq!(loaded.preload(), Vec::new());
    assert_eq!(
        ["m0", "m1", "laya"]
            .map(|name| loaded.artifact_id(name).unwrap())
            .to_vec(),
        hashed
    );
    changes.mark_unchanged();
    assert_eq!(registry.preload(), Vec::new());
    assert!(changes.has_changed().unwrap(), "Watch learns of loads");
    assert_eq!(registry.artifact_id("m0"), Some(hashed[0].clone()));
}

#[test]
fn only_frozen_registries_hash_and_hashing_stops_when_cancelled() {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let backend = Arc::new(CpuBackend);
    let plain = registry(&backend, &[file.path()], Options::new(32));
    plain.hash_artifacts(|| false);
    assert!(plain.artifact_id("m0").is_none());
    let frozen = registry(&backend, &[file.path()], frozen(32));
    frozen.hash_artifacts(|| true);
    assert!(frozen.artifact_id("m0").is_none());
}

/// Three different files of one size, under a budget that fits `fit` of them.
fn crowded(
    fit: usize,
) -> (
    Registry<CpuBackend>,
    NativeMemoryBudget,
    Vec<tempfile::NamedTempFile>,
) {
    let files: Vec<_> = (0..3)
        .map(|_| Fixture::new(1, 8).with_tokenizer().write())
        .collect();
    let budget = NativeMemoryBudget::new(weights(files[0].path()) * fit);
    let paths: Vec<_> = files.iter().map(tempfile::NamedTempFile::path).collect();
    (
        registry(&Arc::new(CpuBackend), &paths, budgeted(&budget)),
        budget,
        files,
    )
}

#[test]
fn a_new_load_evicts_the_least_recently_used_idle_model() {
    let (registry, budget, files) = crowded(2);
    registry.acquire_blocking("m0").unwrap();
    std::thread::sleep(Duration::from_millis(5));
    registry.acquire_blocking("m1").unwrap();
    std::thread::sleep(Duration::from_millis(5));
    // A request on a loaded model counts as use, as on the server's fast path.
    drop(registry.use_resident("m0").unwrap());
    assert!(registry.acquire_blocking("m2").is_ok());
    assert_eq!(
        phases(&registry),
        [Phase::Resident, Phase::Available, Phase::Resident],
        "m1 was used least recently"
    );
    assert_eq!(budget.used(), weights(files[0].path()) * 2);
}

#[test]
fn leased_or_busy_models_are_never_evicted() {
    let (registry, _budget, _files) = crowded(1);
    registry.acquire_blocking("m0").unwrap();
    let lease = registry
        .lease(Holder::Client("pin".into()), "m0", None)
        .unwrap();
    let Err(LoadError::Memory(error)) = registry.acquire_blocking("m1") else {
        panic!("m0 is leased")
    };
    assert!(error.contains("held by m0 (1 lease)"), "{error}");
    registry
        .release(&Holder::Client("pin".into()), &lease.id)
        .unwrap();
    let busy = registry.acquire_blocking("m0").unwrap();
    let Err(LoadError::Memory(error)) = registry.acquire_blocking("m1") else {
        panic!("m0 serves a request")
    };
    assert!(error.contains("held by m0 (in use)"), "{error}");
    drop(busy);
    assert!(registry.acquire_blocking("m1").is_ok());
    assert_eq!(
        phases(&registry),
        [Phase::Available, Phase::Resident, Phase::Available]
    );
}

#[test]
fn requests_evict_idle_models_but_never_their_own_or_a_leased_one() {
    let (registry, _budget, _files) = crowded(3);
    registry.acquire_blocking("m0").unwrap();
    registry.acquire_blocking("m1").unwrap();
    assert!(registry.evict_for("m1").unwrap());
    assert_eq!(
        phases(&registry),
        [Phase::Available, Phase::Resident, Phase::Available]
    );
    assert!(!registry.evict_for("m1").unwrap(), "only m1 is left");
    registry.acquire_blocking("m0").unwrap();
    let holder = Holder::Client("pin".into());
    let lease = registry.lease(holder.clone(), "m0", None).unwrap();
    assert!(!registry.evict_for("m1").unwrap(), "m0 is leased");
    registry.release(&holder, &lease.id).unwrap();
    assert!(!registry.evict_for("nope").unwrap());
}

#[test]
fn expired_leases_release_and_idle_models_unload_when_asked() {
    let (registry, budget, _files) = crowded(3);
    let holder = Holder::Peer("periscope.mdma.lan".into());
    registry.acquire_blocking("m0").unwrap();
    let lease = registry
        .lease(holder.clone(), "m0", Some(lease::MIN_TTL))
        .unwrap();
    assert_eq!(registry.entries().next().unwrap().leases, 1);
    let changes = registry.subscribe();
    registry.sweep(Instant::now(), Some(Duration::ZERO));
    assert_eq!(
        phases(&registry)[0],
        Phase::Resident,
        "a leased model stays loaded"
    );
    registry.sweep(Instant::now() + lease::MIN_TTL, None);
    assert!(changes.has_changed().unwrap());
    assert_eq!(
        (
            registry.entries().next().unwrap().leases,
            phases(&registry)[0]
        ),
        (0, Phase::Resident),
        "an expired lease leaves the model evictable"
    );
    assert_eq!(
        registry.renew(&holder, &lease.id, None),
        Err(LeaseError::Unknown)
    );
    registry.sweep(Instant::now(), Some(Duration::ZERO));
    assert_eq!(
        phases(&registry)[0],
        Phase::Available,
        "idle unload frees it"
    );
    assert_eq!(budget.used(), 0);
}

#[test]
fn leases_name_listed_models_and_resolve_the_default() {
    let (registry, _budget, _files) = crowded(1);
    assert_eq!(
        registry
            .lease(Holder::Client("pin".into()), "", None)
            .unwrap()
            .model,
        "m0"
    );
    assert_eq!(
        registry.lease(Holder::Client("pin".into()), "nope", None),
        Err(LeaseError::Missing("nope".into()))
    );
    assert_eq!(
        phases(&registry),
        [Phase::Available; 3],
        "a lease alone loads nothing"
    );
}

#![expect(
    clippy::arc_with_non_send_sync,
    reason = "the loader fakes count with Cell; a registry owns its backend"
)]
use super::*;
use crate::backend::cpu::{CpuBackend, CpuBuf};
use crate::backend::{AttnSpec, KvView, Pool, QWeight, RopeSpec};
use crate::gguf::tests::Fixture;
use crate::kv_cache::KvCache;
use crate::memory::NativeMemoryBudget;
use crate::model::{Model, Workspace};
use crate::registry::{LoadError, Options, Phase, Registry, Source};
use std::assert_matches;
use std::cell::{Cell, RefCell};
use std::path::Path;
use std::sync::Arc;

#[derive(Default)]
struct CountingBackend {
    allocations: Cell<usize>,
    fail_alloc_at: Cell<Option<usize>>,
    batches: RefCell<Vec<usize>>,
    downloads: Cell<usize>,
    factories: RefCell<Vec<(usize, usize)>>,
    positions: RefCell<Vec<u32>>,
    fail_factory: Cell<bool>,
    fail_decode_at: Cell<Option<usize>>,
    fail_download_at: Cell<Option<usize>>,
    nonfinite_at: Cell<Option<usize>>,
}

struct CountingDecoder<'a> {
    backend: &'a CountingBackend,
    model: &'a Model<CountingBackend>,
    kv: &'a mut KvCache<CountingBackend>,
    ws: &'a mut Workspace<CountingBackend>,
}

impl crate::model::decoder::Decoder for CountingDecoder<'_> {
    fn forward(&mut self, token: u32, pos: u32) -> backend::Result<Vec<f32>> {
        self.backend.positions.borrow_mut().push(pos);
        if self.backend.fail_decode_at.get() == Some(self.backend.positions.borrow().len()) {
            return Err(backend::Error::Backend("injected decode failure".into()));
        }
        self.model
            .forward(self.backend, self.kv, self.ws, token, pos)
    }
}

impl Backend for CountingBackend {
    type Buf = CpuBuf;

    fn decoder<'a>(
        &'a self,
        model: &'a Model<Self>,
        kv: &'a mut KvCache<Self>,
        ws: &'a mut Workspace<Self>,
    ) -> backend::Result<Box<dyn crate::model::decoder::Decoder + 'a>> {
        self.factories.borrow_mut().push((kv.capacity(), kv.len()));
        if self.fail_factory.get() {
            return Err(backend::Error::Backend("injected factory failure".into()));
        }
        Ok(Box::new(CountingDecoder {
            backend: self,
            model,
            kv,
            ws,
        }))
    }

    fn alloc(&self, bytes: usize, pool: Pool) -> backend::Result<CpuBuf> {
        self.allocations.set(self.allocations.get() + 1);
        if self.fail_alloc_at.get() == Some(self.allocations.get()) {
            return Err(backend::Error::OutOfMemory(
                "injected warm-up allocation race".into(),
            ));
        }
        CpuBackend.alloc(bytes, pool)
    }
    fn upload(&self, dst: &mut CpuBuf, offset: usize, src: &[u8]) -> backend::Result<()> {
        CpuBackend.upload(dst, offset, src)
    }
    fn download(&self, src: &CpuBuf, offset: usize, dst: &mut [u8]) -> backend::Result<()> {
        let count = self.downloads.get() + 1;
        self.downloads.set(count);
        if self.fail_download_at.get() == Some(count) {
            return Err(backend::Error::Backend("injected readback failure".into()));
        }
        CpuBackend.download(src, offset, dst)?;
        if self.nonfinite_at.get() == Some(count) {
            bytemuck::cast_slice_mut::<u8, f32>(dst)[0] = f32::NAN;
        }
        Ok(())
    }
    fn copy(
        &self,
        src: &CpuBuf,
        a: usize,
        dst: &mut CpuBuf,
        b: usize,
        n: usize,
    ) -> backend::Result<()> {
        CpuBackend.copy(src, a, dst, b, n)
    }
    fn qmatmul(&self, w: &QWeight<CpuBuf>, x: &CpuBuf, y: &mut CpuBuf) -> backend::Result<()> {
        CpuBackend.qmatmul(w, x, y)
    }
    fn embed(&self, w: &QWeight<CpuBuf>, ids: &[u32], y: &mut CpuBuf) -> backend::Result<()> {
        self.batches.borrow_mut().push(ids.len());
        CpuBackend.embed(w, ids, y)
    }
    fn rmsnorm(&self, w: &CpuBuf, x: &CpuBuf, y: &mut CpuBuf, eps: f32) -> backend::Result<()> {
        CpuBackend.rmsnorm(w, x, y, eps)
    }
    fn add(&self, a: &CpuBuf, b: &CpuBuf, y: &mut CpuBuf) -> backend::Result<()> {
        CpuBackend.add(a, b, y)
    }
    fn rope(
        &self,
        q: &mut CpuBuf,
        k: &mut CpuBuf,
        pos: u32,
        n: usize,
        spec: &RopeSpec,
    ) -> backend::Result<()> {
        CpuBackend.rope(q, k, pos, n, spec)
    }
    fn attn(
        &self,
        q: &CpuBuf,
        kv: &KvView<CpuBuf>,
        spec: &AttnSpec,
        y: &mut CpuBuf,
    ) -> backend::Result<()> {
        CpuBackend.attn(q, kv, spec, y)
    }
    fn silu_mul(&self, a: &CpuBuf, b: &CpuBuf, y: &mut CpuBuf) -> backend::Result<()> {
        CpuBackend.silu_mul(a, b, y)
    }
}

fn model(backend: &CountingBackend, budget: Option<&NativeMemoryBudget>) -> Model<CountingBackend> {
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let mut gguf = crate::gguf::GgufModel::open(file.path()).unwrap();
    let memory = budget.map(|budget| {
        budget
            .reserve(
                crate::model::memory::weight_bytes(&gguf).unwrap(),
                crate::memory::MemoryCategory::Weights,
            )
            .unwrap()
    });
    Model::load_reserved(backend, &mut gguf, memory).unwrap()
}

/// `names` all point at `file`; warm-up runs as each loads.
fn registry(
    backend: &Arc<CountingBackend>,
    files: &[(&str, &Path)],
    rounds: u32,
    budget: Option<&NativeMemoryBudget>,
    frozen: bool,
) -> Registry<CountingBackend> {
    let models = files
        .iter()
        .map(|(name, path)| ((*name).to_owned(), Source::Gguf(path.to_path_buf())))
        .collect();
    let options = Options {
        n_ctx: 64,
        warmup_rounds: rounds,
        frozen,
        budget: budget.cloned(),
    };
    Registry::new(backend.clone(), models, None, options).unwrap()
}

fn weights(path: &Path) -> usize {
    crate::model::memory::weight_bytes(&crate::gguf::GgufModel::open(path).unwrap()).unwrap()
}

#[test]
fn bounded_rounds_join_prefill_and_use_the_request_decoder_factory() {
    let backend = CountingBackend::default();
    let model = model(&backend, None);
    assert!(warm_decoder(&backend, &model, 4096, 2, &mut || false).unwrap());
    assert_eq!(*backend.batches.borrow(), [16, 1, 1, 1, 1, 16, 1, 1, 1, 1]);
    assert_eq!(*backend.factories.borrow(), [(20, 16), (20, 16)]);
    assert_eq!(
        *backend.positions.borrow(),
        [16, 17, 18, 19, 16, 17, 18, 19]
    );
    assert_eq!(backend.downloads.get(), 10); // Every operation returns completed host logits.
}

#[test]
fn small_contexts_keep_one_prefill_position_and_never_overrun() {
    for capacity in [1, 2, 5, 19, 20, 64] {
        let backend = CountingBackend::default();
        let model = model(&backend, None);
        assert!(warm_decoder(&backend, &model, capacity, 1, &mut || false).unwrap());
        let cap = capacity.min(20);
        let steps = 4.min(cap - 1);
        assert_eq!(backend.batches.borrow()[0], cap - steps);
        assert_eq!(backend.downloads.get(), 1 + steps);
        assert_eq!(
            *backend.positions.borrow(),
            ((cap - steps)..cap).map(|p| p as u32).collect::<Vec<_>>()
        );
        assert_eq!(backend.factories.borrow().len(), usize::from(steps > 0));
    }
}

#[test]
fn aliases_load_and_warm_once() {
    let backend = Arc::new(CountingBackend::default());
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let registry = registry(
        &backend,
        &[("fixture", file.path()), ("alias", file.path())],
        2,
        None,
        false,
    );
    assert_eq!(registry.preload(), Vec::new());
    assert_eq!(backend.downloads.get(), 10);
    assert_eq!(*backend.factories.borrow(), [(20, 16), (20, 16)]);
}

#[test]
fn disabled_and_precancelled_warmup_allocate_nothing() {
    let backend = CountingBackend::default();
    let model = model(&backend, None);
    let before = backend.allocations.get();
    assert!(warm_decoder(&backend, &model, 64, 0, &mut || false).unwrap());
    assert!(!warm_decoder(&backend, &model, 64, 2, &mut || true).unwrap());
    assert!(!warm_decoder(&backend, &model, 64, 0, &mut || true).unwrap());
    assert_eq!(backend.allocations.get(), before);
    assert!(backend.batches.borrow().is_empty());
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let options = Options {
        warmup_rounds: MAX_WARMUP_ROUNDS + 1,
        ..Options::new(64)
    };
    assert!(
        Registry::new(
            Arc::new(CountingBackend::default()),
            vec![("m".into(), Source::Gguf(file.path().into()))],
            None,
            options
        )
        .is_err()
    );
}

#[test]
fn cancellation_between_prefill_and_decode_does_not_construct_a_decoder() {
    let backend = CountingBackend::default();
    let budget = NativeMemoryBudget::new(1_000_000);
    let model = model(&backend, Some(&budget));
    let weights = budget.used();
    assert!(!warm_decoder(&backend, &model, 64, 2, &mut || backend.downloads.get() > 0).unwrap());
    assert_eq!(backend.downloads.get(), 1);
    assert!(backend.factories.borrow().is_empty());
    assert_eq!(budget.used(), weights);
}

#[test]
fn cancellation_during_decode_stops_before_the_next_step_and_refunds_buffers() {
    let backend = CountingBackend::default();
    let budget = NativeMemoryBudget::new(1_000_000);
    let model = model(&backend, Some(&budget));
    let weights = budget.used();
    assert!(
        !warm_decoder(&backend, &model, 64, 2, &mut || !backend
            .positions
            .borrow()
            .is_empty())
        .unwrap()
    );
    assert_eq!(*backend.positions.borrow(), [16]);
    assert_eq!(backend.downloads.get(), 2);
    assert_eq!(budget.used(), weights);
}

#[test]
fn warmup_failures_fail_the_load_and_refund_everything() {
    for failure in [
        "readback",
        "factory",
        "decode",
        "nonfinite_prefill",
        "nonfinite_decode",
    ] {
        let backend = Arc::new(CountingBackend::default());
        let budget = NativeMemoryBudget::new(1_000_000);
        let file = Fixture::new(1, 8).with_tokenizer().write();
        let registry = registry(
            &backend,
            &[("fixture", file.path())],
            2,
            Some(&budget),
            false,
        );
        match failure {
            "readback" => backend.fail_download_at.set(Some(1)),
            "factory" => backend.fail_factory.set(true),
            "decode" => backend.fail_decode_at.set(Some(2)),
            "nonfinite_prefill" => backend.nonfinite_at.set(Some(1)),
            "nonfinite_decode" => backend.nonfinite_at.set(Some(2)),
            _ => unreachable!(),
        }
        let Err(LoadError::Failed(error)) = registry.acquire_blocking("fixture") else {
            panic!("{failure} must fail the load")
        };
        assert!(
            error.contains("failure") || error.contains("nonfinite"),
            "{error}"
        );
        assert_eq!(budget.used(), 0, "{failure}");
        assert_eq!(
            registry.entries().next().unwrap().phase,
            Phase::Failed,
            "a broken model stays failed"
        );
    }
}

#[test]
fn warmup_memory_refusal_releases_every_alias_and_can_retry() {
    let backend = Arc::new(CountingBackend::default());
    let file = Fixture::new(1, 8).with_tokenizer().write();
    let budget = NativeMemoryBudget::new(weights(file.path()) + 1);
    let registry = registry(
        &backend,
        &[("fixture", file.path()), ("alias", file.path())],
        2,
        Some(&budget),
        true,
    );
    let Err(LoadError::Memory(error)) = registry.acquire_blocking("alias") else {
        panic!("warm-up exceeds the budget")
    };
    assert!(error.contains("memory budget exhausted"), "{error}");
    assert_eq!(
        (budget.used(), backend.downloads.get()),
        (0, 0),
        "no alias keeps rejected weights loaded"
    );
    for entry in registry.entries() {
        assert_eq!(entry.phase, Phase::Available, "{}", entry.name);
        assert!(
            entry
                .error
                .as_deref()
                .is_some_and(|e| e.contains("memory budget exhausted"))
        );
    }
    assert!(
        registry.artifact_id("fixture").is_none() && registry.artifact_id("alias").is_none(),
        "no identity without a loaded model"
    );
    assert_matches!(
        registry.acquire_blocking("fixture").map(|_| ()),
        Err(LoadError::Memory(_)),
        "a refusal is retried, not remembered"
    );
}

#[test]
fn a_refused_model_does_not_stop_the_others() {
    let backend = Arc::new(CountingBackend::default());
    let a = Fixture::new(1, 8).with_tokenizer().write();
    let b = Fixture::new(1, 8).with_tokenizer().write();
    let budget = NativeMemoryBudget::new(1_000_000);
    let registry = registry(
        &backend,
        &[("a", a.path()), ("alias", a.path()), ("b", b.path())],
        2,
        Some(&budget),
        false,
    );
    backend
        .fail_alloc_at
        .set(Some(backend.allocations.get() + 15));
    let failures = registry.preload();
    assert_eq!(
        failures
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["a"]
    );
    let phases: Vec<_> = registry.entries().map(|entry| entry.phase).collect();
    assert_eq!(
        phases,
        [Phase::Available, Phase::Available, Phase::Resident]
    );
    assert_eq!(budget.used(), weights(b.path()));
    assert_eq!(
        backend.downloads.get(),
        10,
        "the remaining model still completes both rounds"
    );
    drop(registry);
    assert_eq!(budget.used(), 0);
}

fn continuation(backend: &CountingBackend, model: &Model<CountingBackend>) -> Vec<Vec<f32>> {
    let mut kv = model.new_kv_cache(backend, 16).unwrap();
    let mut prefill_ws = model.alloc_workspace(backend, 3).unwrap();
    let mut ws = model.alloc_workspace(backend, 1).unwrap();
    let mut logits = model
        .prefill(backend, &mut kv, &mut prefill_ws, &[1, 2, 3], 0)
        .unwrap();
    let mut outputs = vec![logits.clone()];
    for pos in 3..8 {
        logits = model
            .forward(backend, &mut kv, &mut ws, greedy(&logits), pos)
            .unwrap();
        outputs.push(logits.clone());
    }
    outputs
}

#[test]
fn warmup_keeps_weights_resident_without_seeding_user_kv_or_changing_logits() {
    let backend = CountingBackend::default();
    let budget = NativeMemoryBudget::new(1_000_000);
    let model = model(&backend, Some(&budget));
    let weights = budget.used();
    let baseline = continuation(&backend, &model);
    assert!(warm_decoder(&backend, &model, 64, 2, &mut || false).unwrap());
    assert_eq!(budget.used(), weights);
    assert_eq!(continuation(&backend, &model), baseline);
}

#[test]
fn empty_or_nonfinite_logits_are_not_ready() {
    for logits in [
        vec![],
        vec![f32::NAN],
        vec![f32::INFINITY],
        vec![f32::NEG_INFINITY],
    ] {
        assert!(next_token(&logits).is_err());
    }
    assert_eq!(next_token(&[-2.0, 3.0, 1.0]).unwrap(), 1);
}

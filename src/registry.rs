//! The models in one folder: GGUF decoders and CPU Laya checkpoints, loaded on use.
use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::federation::profile::ExecutorIdentity;
use crate::gguf::GgufModel;
use crate::laya::Predictor;
use crate::memory::{MemoryCategory, NativeMemoryBudget, Reservation};
use crate::model::Model;
use crate::tokenizer::{ChatFormat, Tokenizer};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::watch;

mod append;
pub mod class;
pub(crate) mod lease;
mod native;
mod scan;
mod slot;
mod warmup;

pub use crate::artifact::ArtifactId;
pub use crate::federation::profile::ExecutionProfile;
pub use class::{Capability, Class};
pub use lease::{Holder, Lease, LeaseError, requested_ttl};
pub use scan::scan;
pub use slot::{Decoder, LoadError, Phase, Use};
pub use warmup::MAX_WARMUP_ROUNDS;

use append::Append;
use slot::{Resident, Slot};

/// Free device memory a load leaves for driver state and request buffers.
const DEVICE_HEADROOM: usize = 256 << 20;

/// Names one registry lists, counting models added while it serves.
pub const MAX_MODELS: usize = 1024;

#[cfg(test)]
mod tests;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("models: {0}")]
    Config(String),
    #[error("model {name}: {source}")]
    Load {
        name: String,
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Gguf,
    Laya,
}

impl Kind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
            Self::Laya => "laya",
        }
    }
}

/// Where a model's files are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Gguf(PathBuf),
    /// A checkpoint folder.
    Laya(PathBuf),
}

/// How models load.
#[derive(Debug, Clone)]
pub struct Options {
    /// Per-request KV cap, bounded by each model's trained context.
    pub n_ctx: usize,
    pub warmup_rounds: u32,
    /// Copy and hash each artifact as it loads, so federation can prove what runs.
    pub frozen: bool,
    pub budget: Option<NativeMemoryBudget>,
}

impl Options {
    pub const fn new(n_ctx: usize) -> Self {
        Self {
            n_ctx,
            warmup_rounds: 0,
            frozen: false,
            budget: None,
        }
    }
}

/// What is known about a model without loading its weights.
#[derive(Debug, Clone)]
pub struct Info {
    pub kind: Kind,
    pub architecture: String,
    pub n_ctx: usize,
    /// The model's own context, before `Options::n_ctx` caps it.
    pub trained_ctx: usize,
    /// On disk; aliases repeat it.
    pub disk_bytes: u64,
    /// Elements across the weights' tensors, from the file header; 0 when unknown.
    pub parameters: u64,
    /// "Q4_K_M", "F16"; for Laya the safetensors dtype. Empty when unknown.
    pub quantization: String,
    pub class: Class,
    pub capabilities: Vec<Capability>,
    pub chat_format: Option<ChatFormat>,
    /// What it takes through `Decide`; `None` for text generators.
    pub decision: Option<crate::decision::Contract>,
}

/// A parameter count as people write it: "494M", "1.1B", "7.2B"; empty for 0.
pub fn parameters_label(n: u64) -> String {
    const UNITS: [&str; 5] = ["", "K", "M", "B", "T"];
    if n == 0 {
        return String::new();
    }
    let mut value = n as f64;
    let mut unit = 0;
    while value >= 999.5 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    match unit {
        0 => n.to_string(),
        _ if value < 9.95 => format!("{value:.1}{}", UNITS[unit]),
        _ => format!("{value:.0}{}", UNITS[unit]),
    }
}

/// One name and its state, for status reports.
pub struct Entry<'a> {
    pub name: &'a str,
    pub info: &'a Info,
    pub phase: Phase,
    /// Why the last load failed or was refused.
    pub error: Option<String>,
    /// Shared by every alias of one file.
    pub upload: usize,
    pub leases: u32,
}

/// The execution profiles of this backend, one per workload.
#[derive(Default)]
struct Profiles {
    decoder: Option<ExecutionProfile>,
    laya: Option<ExecutionProfile>,
}

pub struct Registry<B: Backend> {
    backend: Arc<B>,
    options: Options,
    /// The models folder, when listed from one.
    dir: Option<PathBuf>,
    /// Names in order, each with its slot; aliases share one. Both only grow.
    names: Append<(String, usize)>,
    slots: Append<Slot<B>>,
    /// Each slot's file or folder, so an alias finds its slot. Held while
    /// listing, which serializes additions.
    identities: Mutex<HashMap<same_file::Handle, usize>>,
    default: usize,
    profiles: Profiles,
    /// Every model listed at startup has been hashed (or failed to).
    hashed: AtomicBool,
    /// Ticks when a model starts or finishes loading, or its artifact is known.
    changes: watch::Sender<u64>,
    /// One load admits memory, and evicts for it, at a time.
    admission: Mutex<()>,
    leases: lease::Table,
}

impl<B: Backend> Registry<B> {
    /// Every `*.gguf` and Laya checkpoint folder in `dir`, named by file stem or
    /// folder. Files that do not preflight are skipped with a log line. The
    /// default is `default`, else the first name. The list may be empty; the
    /// caller decides whether a server without local models is useful.
    pub fn open(
        backend: Arc<B>,
        dir: &Path,
        default: Option<&str>,
        options: Options,
    ) -> Result<Self> {
        let found = scan::scan(dir)?;
        let mut registry = Self::build(backend, found, default, options, true)?;
        registry.dir = Some(dir.to_path_buf());
        Ok(registry)
    }

    /// Exactly these models, in this order; any preflight failure is an error.
    pub fn new(
        backend: Arc<B>,
        models: Vec<(String, Source)>,
        default: Option<&str>,
        options: Options,
    ) -> Result<Self> {
        Self::build(backend, models, default, options, false)
    }

    /// One decoder built by the caller, always loaded and never exported.
    pub fn single(
        backend: Arc<B>,
        name: &str,
        architecture: &str,
        n_ctx: usize,
        decoder: Decoder<B>,
    ) -> Self {
        let trained_ctx = decoder.model.config().n_ctx;
        let info = Info {
            kind: Kind::Gguf,
            architecture: architecture.to_owned(),
            n_ctx: n_ctx.min(trained_ctx),
            trained_ctx,
            disk_bytes: 0,
            parameters: 0,
            quantization: String::new(),
            class: Class::Generative,
            capabilities: Vec::new(),
            chat_format: decoder.tokenizer.chat_format(),
            decision: None,
        };
        let registry = Self::empty(backend, Options::new(info.n_ctx));
        registry
            .slots
            .push(Slot::fixed(info, decoder))
            .unwrap_or_else(|_| unreachable!("an empty registry has room"));
        registry
            .names
            .push((name.to_owned(), 0))
            .unwrap_or_else(|_| unreachable!("an empty registry has room"));
        registry
    }

    fn empty(backend: Arc<B>, options: Options) -> Self {
        Self {
            backend,
            options,
            dir: None,
            names: Append::with_capacity(MAX_MODELS),
            slots: Append::with_capacity(MAX_MODELS),
            identities: Mutex::new(HashMap::new()),
            default: 0,
            profiles: Profiles::default(),
            hashed: AtomicBool::new(false),
            changes: watch::Sender::new(0),
            admission: Mutex::new(()),
            leases: lease::Table::default(),
        }
    }

    fn build(
        backend: Arc<B>,
        models: Vec<(String, Source)>,
        default: Option<&str>,
        options: Options,
        skip_invalid: bool,
    ) -> Result<Self> {
        if models.is_empty() && !skip_invalid {
            return Err(Error::Config("no models listed".into()));
        }
        if options.warmup_rounds > MAX_WARMUP_ROUNDS {
            return Err(Error::Config(format!(
                "warm-up rounds must be between 0 and {MAX_WARMUP_ROUNDS}"
            )));
        }
        if options.frozen
            && models
                .iter()
                .any(|(_, source)| matches!(source, Source::Gguf(_)))
        {
            crate::cache::require_original_source().map_err(|e| Error::Config(e.to_string()))?;
        }
        if models.len() > MAX_MODELS {
            return Err(Error::Config(format!(
                "{} models listed; at most {MAX_MODELS}",
                models.len()
            )));
        }
        let mut registry = Self::empty(backend, options);
        {
            let mut identities = registry.identities.lock().expect("identity lock poisoned");
            for (name, source) in models {
                if name.trim().is_empty() {
                    return Err(Error::Config("a model has an empty name".into()));
                }
                if registry.names.iter().any(|(listed, _)| *listed == name) {
                    return Err(Error::Config(format!("model {name:?} is listed twice")));
                }
                match registry.preflight(&source, &mut identities) {
                    Ok(slot) => {
                        registry
                            .names
                            .push((name, slot))
                            .unwrap_or_else(|_| unreachable!("checked against MAX_MODELS"));
                    }
                    Err(e) if skip_invalid => eprintln!("mdma: skipping model {name}: {e}"),
                    Err(source) => return Err(Error::Load { name, source }),
                }
            }
        }
        registry.default = match default {
            Some(default) => registry
                .names
                .iter()
                .position(|(name, _)| name == default)
                .ok_or_else(|| Error::Config(format!("default model {default:?} is not listed")))?,
            None => 0,
        };
        Ok(registry)
    }

    /// Reads metadata only; aliases of one file or folder share a slot.
    /// `identities` is the held lock, so slots are pushed one at a time.
    fn preflight(
        &self,
        source: &Source,
        identities: &mut HashMap<same_file::Handle, usize>,
    ) -> std::result::Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let full = |_| "the registry lists the most models it can";
        match source {
            Source::Gguf(path) => {
                let (pinned, handle) = native::Pinned::open(path)?;
                if let Some(&index) = identities.get(&handle) {
                    let slot::Source::Gguf { pinned: first, .. } = &self.slots[index].source else {
                        return Err("same file listed as a GGUF and a Laya checkpoint".into());
                    };
                    first.same_as(&pinned)?;
                    return Ok(index);
                }
                let gguf = pinned.gguf()?;
                let config = Model::validate(&*self.backend, &gguf)?;
                let tokenizer = Tokenizer::from_gguf(&gguf)
                    .ok_or("failed to build tokenizer from GGUF metadata")?;
                let (class, capabilities) = class::detect(&gguf);
                let info = Info {
                    kind: Kind::Gguf,
                    architecture: architecture(&gguf),
                    n_ctx: self.options.n_ctx.min(config.n_ctx),
                    trained_ctx: config.n_ctx,
                    disk_bytes: pinned.len(),
                    parameters: gguf.parameters(),
                    quantization: gguf.quantization(),
                    class,
                    capabilities,
                    chat_format: tokenizer.chat_format(),
                    decision: crate::decision::Contract::detect(&gguf, &tokenizer, class),
                };
                let index = self
                    .slots
                    .push(Slot::new(
                        info,
                        slot::Source::Gguf {
                            pinned,
                            tokenizer: Arc::new(tokenizer),
                        },
                    ))
                    .map_err(full)?;
                identities.insert(handle, index);
                Ok(index)
            }
            Source::Laya(dir) => {
                let handle = same_file::Handle::from_path(dir)?;
                if let Some(&index) = identities.get(&handle) {
                    return Ok(index);
                }
                let checkpoint = crate::laya::config::Checkpoint::open(dir)?;
                let weights = dir.join("model.safetensors");
                let header = crate::model::weights::Header::read(&weights)?;
                let info = Info {
                    kind: Kind::Laya,
                    architecture: checkpoint.encoder.model_type,
                    n_ctx: checkpoint.agent.max_len,
                    trained_ctx: checkpoint.agent.max_len,
                    disk_bytes: std::fs::metadata(weights)?.len(),
                    parameters: header.parameters,
                    quantization: header.dtype,
                    class: Class::Decision,
                    capabilities: Vec::new(),
                    chat_format: None,
                    decision: Some(crate::decision::Contract::Laya),
                };
                let index = self
                    .slots
                    .push(Slot::new(info, slot::Source::Laya(dir.clone())))
                    .map_err(full)?;
                identities.insert(handle, index);
                Ok(index)
            }
        }
    }

    /// Lists `<name>.gguf` from the models folder while serving, as startup
    /// would have. `artifact` is the digest the caller took of the file it
    /// wrote; every load re-checks it. A link to a listed file shares its slot.
    pub fn add(&self, name: &str, artifact: Option<ArtifactId>) -> Result<()> {
        crate::federation::config::validate_model(name)
            .map_err(|e| Error::Config(format!("{name:?}: {e}")))?;
        let dir = self
            .dir
            .as_deref()
            .ok_or_else(|| Error::Config("models are not listed from a folder".into()))?;
        if self.options.frozen {
            crate::cache::require_original_source().map_err(|e| Error::Config(e.to_string()))?;
        }
        let mut identities = self.identities.lock().expect("identity lock poisoned");
        if self.names.iter().any(|(listed, _)| listed == name) {
            return Err(Error::Config(format!("model {name:?} is already listed")));
        }
        let source = Source::Gguf(dir.join(format!("{name}.gguf")));
        let index = self
            .preflight(&source, &mut identities)
            .map_err(|source| Error::Load {
                name: name.to_owned(),
                source,
            })?;
        if let Some(artifact) = artifact {
            same_artifact(&self.slots[index], &artifact).map_err(|e| Error::Load {
                name: name.to_owned(),
                source: e.into(),
            })?;
            self.slots[index].set_artifact(artifact);
        }
        self.names
            .push((name.to_owned(), index))
            .map_err(|_| Error::Config(format!("at most {MAX_MODELS} models")))?;
        drop(identities);
        self.changed();
        Ok(())
    }

    /// The first name serving these exact bytes.
    pub fn find_artifact(&self, artifact_id: &str) -> Option<&str> {
        self.names
            .iter()
            .find(|(_, index)| {
                self.slots[*index]
                    .artifact()
                    .is_some_and(|known| known.as_str() == artifact_id)
            })
            .map(|(name, _)| name.as_str())
    }

    /// Whether [`Self::hash_artifacts`] has finished, so every listed model's
    /// artifact that can be known is.
    pub fn hashed(&self) -> bool {
        self.hashed.load(Ordering::Acquire)
    }

    pub const fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    pub fn dir(&self) -> Option<&Path> {
        self.dir.as_deref()
    }

    /// `host` with live memory and the models folder's disk.
    pub fn live_host(&self, host: &crate::grpc::Host) -> crate::grpc::Host {
        let device = (
            self.backend.available_device_bytes().ok().flatten(),
            self.backend.total_device_bytes(),
        );
        crate::host::live(host, self.dir(), device)
    }

    pub fn names(&self) -> Vec<&str> {
        self.names.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// `None` when nothing is listed.
    pub fn default_name(&self) -> Option<&str> {
        self.names.get(self.default).map(|(name, _)| name.as_str())
    }

    /// `""` means the default model.
    fn slot(&self, name: &str) -> Option<&Slot<B>> {
        self.index(name).map(|index| &self.slots[index])
    }

    fn index(&self, name: &str) -> Option<usize> {
        if name.is_empty() {
            return self.names.get(self.default).map(|(_, index)| *index);
        }
        self.names
            .iter()
            .find(|(listed, _)| listed == name)
            .map(|(_, index)| *index)
    }

    /// The listed name, resolving `""` to the default.
    pub fn resolve<'a>(&'a self, name: &'a str) -> Option<&'a str> {
        if name.is_empty() {
            return self.default_name();
        }
        self.names
            .iter()
            .find(|(listed, _)| listed == name)
            .map(|(listed, _)| listed.as_str())
    }

    pub fn info(&self, name: &str) -> Option<&Info> {
        self.slot(name).map(|slot| &slot.info)
    }

    /// Every name in order, with its state.
    pub fn entries(&self) -> impl Iterator<Item = Entry<'_>> {
        self.names.iter().map(|(name, index)| {
            let slot = &self.slots[*index];
            let (phase, error) = slot.phase();
            Entry {
                name,
                info: &slot.info,
                phase,
                error,
                upload: *index,
                leases: slot.leases(),
            }
        })
    }

    /// The model if it is loaded, without loading it.
    pub fn resident(&self, name: &str) -> Option<Use<B>> {
        self.slot(name)?.resident()
    }

    /// The model for a request if it is loaded, without loading it. Unlike
    /// [`Self::resident`] this counts as a use for idle unload and eviction.
    pub fn use_resident(&self, name: &str) -> Option<Use<B>> {
        self.slot(name)?.try_use()
    }

    /// The model, loading it first if needed. Blocks while it loads.
    pub fn acquire_blocking(&self, name: &str) -> std::result::Result<Use<B>, LoadError> {
        let slot = self
            .slot(name)
            .ok_or_else(|| LoadError::Failed(format!("no model named {name:?}")))?;
        let mut loaded = false;
        let result = slot.acquire(|| {
            loaded = true;
            self.changed();
            self.load(slot)
        });
        if loaded {
            self.changed();
        }
        result
    }

    /// Keeps `name` loaded for `holder` until the lease is released or expires.
    /// Taking a lease does not load the model; acquiring one does.
    pub fn lease(
        &self,
        holder: Holder,
        name: &str,
        ttl: Option<Duration>,
    ) -> std::result::Result<Lease, LeaseError> {
        let listed = self
            .resolve(name)
            .ok_or_else(|| LeaseError::Missing(name.to_owned()))?
            .to_owned();
        let index = self.index(&listed).expect("resolved names are listed");
        let (lease, new) = self.leases.take(holder, listed, index, ttl)?;
        if new {
            self.slots[index].add_lease();
            self.changed();
        }
        Ok(lease)
    }

    /// The model a holder's lease keeps loaded.
    pub fn lease_model(
        &self,
        holder: &Holder,
        id: &str,
    ) -> std::result::Result<String, LeaseError> {
        self.leases.model(holder, id)
    }

    pub fn renew(
        &self,
        holder: &Holder,
        id: &str,
        ttl: Option<Duration>,
    ) -> std::result::Result<Lease, LeaseError> {
        self.leases.renew(holder, id, ttl)
    }

    pub fn release(&self, holder: &Holder, id: &str) -> std::result::Result<(), LeaseError> {
        let index = self.leases.release(holder, id)?;
        self.slots[index].drop_lease();
        self.changed();
        Ok(())
    }

    /// Ends expired leases and, with `idle`, unloads models unused that long.
    pub fn sweep(&self, now: Instant, idle: Option<Duration>) {
        let expired = self.leases.expire(now);
        for index in &expired {
            self.slots[*index].drop_lease();
        }
        let mut unloaded = false;
        if let Some(idle) = idle {
            for slot in self
                .slots
                .iter()
                .filter(|slot| !matches!(slot.source, slot::Source::Fixed))
            {
                if slot
                    .idle_since()
                    .is_some_and(|used| now.saturating_duration_since(used) >= idle)
                {
                    unloaded |= slot.unload_if_idle();
                }
            }
        }
        if !expired.is_empty() || unloaded {
            self.changed();
        }
    }

    /// Ticks on every residency or provenance change.
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.changes.subscribe()
    }

    fn changed(&self) {
        self.changes
            .send_modify(|tick| *tick = tick.wrapping_add(1));
    }

    /// Hash every model's files, so federation can advertise a model before
    /// it loads. Reads every byte; run it off the request path. Stops early
    /// when `cancelled`. Only frozen registries publish identities.
    pub fn hash_artifacts(&self, cancelled: impl Fn() -> bool) {
        if !self.options.frozen {
            return;
        }
        for (name, index) in self.names.iter() {
            let slot = &self.slots[*index];
            if cancelled() {
                return;
            }
            if slot.artifact().is_some() {
                continue;
            }
            let id = match &slot.source {
                slot::Source::Gguf { pinned, .. } => pinned.hash(),
                slot::Source::Laya(dir) => crate::laya::artifact_id(dir),
                slot::Source::Fixed => continue,
            };
            match id {
                Ok(id) => {
                    slot.set_artifact(id);
                    self.changed();
                }
                Err(e) => eprintln!("mdma: cannot hash model {name}: {e}"),
            }
        }
        self.hashed.store(true, Ordering::Release);
    }

    /// Load every model now, for a server that preloads. Returns the ones that did not load.
    pub fn preload(&self) -> Vec<(String, LoadError)> {
        let mut failures = Vec::new();
        let mut done = Vec::new();
        for (name, index) in self.names.iter() {
            if done.contains(index) {
                continue;
            }
            done.push(*index);
            if let Err(error) = self.acquire_blocking(name) {
                failures.push((name.clone(), error));
            }
        }
        failures
    }

    fn load(&self, slot: &Slot<B>) -> std::result::Result<Resident<B>, LoadError> {
        let failed = |e: &dyn std::fmt::Display| LoadError::Failed(e.to_string());
        let backend = |e: crate::backend::Error| {
            if e.is_memory_exhausted() {
                LoadError::Memory(e.to_string())
            } else {
                failed(&e)
            }
        };
        match &slot.source {
            slot::Source::Gguf { pinned, tokenizer } => {
                pinned.check_unchanged().map_err(|e| failed(&e))?;
                let (mut gguf, artifact) = if self.options.frozen {
                    let (frozen, id) =
                        crate::artifact::freeze_file(pinned.reader().map_err(|e| failed(&e))?)
                            .map_err(|e| failed(&e))?;
                    same_artifact(slot, &id)?;
                    (
                        GgufModel::from_private(frozen).map_err(|e| failed(&e))?,
                        Some(id),
                    )
                } else {
                    let mut gguf = pinned.gguf().map_err(|e| failed(&e))?;
                    Model::prepare_source(&*self.backend, &mut gguf).map_err(backend)?;
                    (gguf, None)
                };
                let bytes = crate::model::memory::weight_bytes(&gguf).map_err(backend)?;
                let memory = self.admit(slot, bytes).map_err(|e| match backend(e) {
                    LoadError::Memory(reason) => LoadError::Memory(self.held(reason)),
                    failed @ LoadError::Failed(_) => failed,
                })?;
                let model =
                    Model::load_reserved(&*self.backend, &mut gguf, memory).map_err(backend)?;
                if artifact.is_none() {
                    pinned.check_unchanged().map_err(|e| failed(&e))?;
                }
                warmup::warm_decoder(
                    &*self.backend,
                    &model,
                    slot.info.n_ctx,
                    self.options.warmup_rounds,
                    &mut || false,
                )
                .map_err(backend)?;
                if let Some(id) = artifact {
                    slot.set_artifact(id);
                }
                Ok(Resident::Decoder(Decoder {
                    model: Arc::new(model),
                    tokenizer: tokenizer.clone(),
                }))
            }
            slot::Source::Laya(dir) => {
                let predictor = if self.options.frozen {
                    let (predictor, id, _) =
                        Predictor::load_frozen(CpuBackend, dir).map_err(|e| failed(&e))?;
                    same_artifact(slot, &id)?;
                    slot.set_artifact(id);
                    predictor
                } else {
                    Predictor::load(CpuBackend, dir).map_err(|e| failed(&e))?
                };
                Ok(Resident::Laya(Arc::new(predictor)))
            }
            slot::Source::Fixed => unreachable!("caller-built decoders are always loaded"),
        }
    }

    /// Room for `bytes` of weights, unloading the least recently used idle
    /// models (unleased, no request in flight) until they fit.
    fn admit(&self, slot: &Slot<B>, bytes: usize) -> crate::backend::Result<Option<Reservation>> {
        let _one = self.admission.lock().expect("admission lock poisoned");
        loop {
            match self.reserve(bytes) {
                Err(error) if error.is_memory_exhausted() => {
                    if !self.evict_one(slot)? {
                        return Err(error);
                    }
                }
                result => return result,
            }
        }
    }

    /// Makes room for a request on `name` that did not fit, by unloading the
    /// least recently used idle model other than `name`. Returns whether one
    /// was unloaded, so the caller can retry.
    pub fn evict_for(&self, name: &str) -> crate::backend::Result<bool> {
        let Some(slot) = self.slot(name) else {
            return Ok(false);
        };
        let _one = self.admission.lock().expect("admission lock poisoned");
        self.evict_one(slot)
    }

    /// Unloads the least recently used idle model other than `keep`; false
    /// when none can go. Leased and busy models never do.
    fn evict_one(&self, keep: &Slot<B>) -> crate::backend::Result<bool> {
        let victim = self
            .slots
            .iter()
            .filter(|other| {
                !std::ptr::eq(*other, keep)
                    && other.info.kind == Kind::Gguf
                    && !matches!(other.source, slot::Source::Fixed)
            })
            .filter_map(|other| other.idle_since().map(|used| (used, other)))
            .min_by_key(|(used, _)| *used);
        let Some((_, victim)) = victim else {
            return Ok(false);
        };
        if victim.unload_if_idle() {
            self.backend.join_execution()?;
            self.changed();
        }
        Ok(true)
    }

    /// A refusal, naming what holds the memory.
    pub fn held(&self, reason: String) -> String {
        let holders: Vec<_> = self
            .names
            .iter()
            .filter(|(_, index)| self.slots[*index].phase().0 == Phase::Resident)
            .map(|(name, index)| match self.slots[*index].leases() {
                0 => format!("{name} (in use)"),
                1 => format!("{name} (1 lease)"),
                n => format!("{name} ({n} leases)"),
            })
            .collect();
        if holders.is_empty() {
            reason
        } else {
            format!("{reason}; held by {}", holders.join(", "))
        }
    }

    /// Free device memory and the native budget for `bytes` of weights.
    fn reserve(&self, bytes: usize) -> crate::backend::Result<Option<Reservation>> {
        if let Some(available) = self.backend.available_device_bytes()? {
            let requested = crate::memory::sum([bytes, DEVICE_HEADROOM])?;
            if requested > available {
                return Err(crate::backend::Error::InsufficientMemory {
                    requested,
                    available,
                });
            }
        }
        self.options
            .budget
            .as_ref()
            .map(|budget| budget.reserve(bytes, MemoryCategory::Weights))
            .transpose()
    }

    /// Exact bytes of a model loaded from a frozen copy. `""` resolves the default.
    pub fn artifact_id(&self, name: &str) -> Option<ArtifactId> {
        self.slot(name)?.artifact()
    }

    /// One profile per workload for this backend. Only concrete implemented
    /// backends (not labels or caller claims) and native CPU Laya have a
    /// defined vocabulary. Returns how many models can export now; a folder
    /// with none still gets profiles for the models [`Self::add`] lists later.
    /// Recapture clears old snapshots before any fallible work.
    pub fn capture_execution_profiles(&mut self) -> std::io::Result<usize>
    where
        B: 'static,
    {
        self.capture_execution_profiles_with(ExecutorIdentity::capture)
    }

    fn capture_execution_profiles_with(
        &mut self,
        capture: impl FnOnce() -> std::io::Result<ExecutorIdentity>,
    ) -> std::io::Result<usize>
    where
        B: 'static,
    {
        self.profiles = Profiles::default();
        if !self.options.frozen {
            return Ok(0);
        }
        let cpu = (&*self.backend as &dyn std::any::Any).is::<CpuBackend>();
        let gpu = crate::federation::profile::gpu::capture(&*self.backend)?;
        let exportable = self
            .names
            .iter()
            .filter(|(_, index)| match self.slots[*index].info.kind {
                Kind::Laya => true,
                Kind::Gguf => cpu || gpu.is_some(),
            })
            .count();
        if exportable == 0 && self.dir.is_none() {
            return Ok(0);
        }
        let executor = Arc::new(capture()?);
        self.profiles = Profiles {
            decoder: if cpu {
                Some(ExecutionProfile::cpu(executor.clone(), false))
            } else {
                gpu.map(|gpu| ExecutionProfile::gpu(executor.clone(), gpu))
            },
            laya: Some(ExecutionProfile::cpu(executor, true)),
        };
        Ok(exportable)
    }

    /// The current profile of a model with a known artifact. Deployment
    /// changes and unverifiable image identity withdraw it. Neither provenance
    /// getter confers federation export or execute permission.
    pub fn execution_profile(&self, name: &str) -> Option<&ExecutionProfile> {
        let slot = self.slot(name)?;
        slot.artifact()?;
        let profile = match slot.info.kind {
            Kind::Gguf => self.profiles.decoder.as_ref(),
            Kind::Laya => self.profiles.laya.as_ref(),
        }?;
        profile.is_current().then_some(profile)
    }
}

/// A load must run the bytes federation already advertised.
fn same_artifact<B: Backend>(
    slot: &Slot<B>,
    id: &ArtifactId,
) -> std::result::Result<(), LoadError> {
    match slot.artifact() {
        Some(known) if known != *id => Err(LoadError::Failed(
            "model file changed since it was hashed".into(),
        )),
        _ => Ok(()),
    }
}

fn architecture(gguf: &GgufModel) -> String {
    gguf.metadata_str("general.architecture")
        .unwrap_or_default()
        .to_owned()
}

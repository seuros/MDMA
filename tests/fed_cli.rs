//! Federation set up with nothing but mdma-fed: a real executor serves its
//! signed catalog to the origin, whose routes then pass mdma-serve's startup
//! checks, and takes models the origin pushes. One daemon runs per machine:
//! where both sides serve, the receiving one runs in this process.
#[path = "runtime_cli/fixture.rs"]
mod fixture;
#[path = "runtime_cli/process.rs"]
mod process;

use process::{Process, command, free_port};
use rustls_pki_types::pem::PemObject;
use std::path::{Path, PathBuf};
use std::process::Output;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// One mdma-serve per machine: tests that start one take turns.
static DAEMON: Mutex<()> = Mutex::new(());

fn daemon_turn() -> MutexGuard<'static, ()> {
    DAEMON.lock().unwrap_or_else(PoisonError::into_inner)
}

fn fed(args: &[&str]) -> Output {
    command(env!("CARGO_BIN_EXE_mdma-fed"))
        .args(args)
        .output()
        .unwrap()
}

fn ok(output: &Output) -> String {
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "{text}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    text
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

struct Node {
    dir: PathBuf,
    models: PathBuf,
    client: u16,
}

impl Node {
    /// A node whose models folder holds `model` as tiny.gguf, or nothing.
    fn new(root: &Path, name: &str, model: Option<&Path>) -> Self {
        let dir = root.join(name);
        let id = format!("{name}.fleet.test");
        ok(&fed(&[
            "init",
            path(&dir),
            "--id",
            &id,
            "--endpoint",
            &format!("https://127.0.0.1:{}", free_port()),
            "--days",
            "1",
        ]));
        let models = root.join(format!("{name}-models"));
        std::fs::create_dir(&models).unwrap();
        if let Some(model) = model {
            std::fs::copy(model, models.join("tiny.gguf")).unwrap();
        }
        let bundle = root.join(format!("{name}.json"));
        std::fs::write(&bundle, ok(&fed(&["export", path(&dir)]))).unwrap();
        Self {
            dir,
            models,
            client: free_port(),
        }
    }

    /// Enrolls the bundle `root/<peer>.json` with `grants`; what enroll printed.
    fn enroll(&self, root: &Path, peer: &str, grants: &[&str]) -> String {
        let bundle = root.join(format!("{peer}.json"));
        let mut args = vec!["enroll", path(&self.dir), path(&bundle)];
        args.extend_from_slice(grants);
        ok(&fed(&args))
    }

    /// Enrolls the bundle `root/<peer>.json` for control only.
    fn trust(&self, root: &Path, peer: &str) {
        self.enroll(root, peer, &[]);
    }

    fn serve(&self, certs: &Path, log: &Path) -> Process {
        self.serve_with(certs, log, &[])
    }

    fn serve_with(&self, certs: &Path, log: &Path, env: &[(&str, &str)]) -> Process {
        Process(
            command(env!("CARGO_BIN_EXE_mdma-serve"))
                .envs(env.iter().copied())
                .env("MDMA_MODELS_DIR", &self.models)
                .env("MDMA_N_CTX", "64")
                .env("MDMA_BACKEND", "cpu")
                .env("MDMA_LISTEN", format!("127.0.0.1:{}", self.client))
                .env("MDMA_POWER", "off")
                .env("MDMA_FEDERATION_CONFIG", self.dir.join("runtime.toml"))
                .env("MDMA_TLS_CERT", certs.join("server.pem"))
                .env("MDMA_TLS_KEY", certs.join("server-key.pem"))
                .env("MDMA_TLS_CA", certs.join("ca.pem"))
                .stdout(std::fs::File::create(log).unwrap())
                .stderr(std::fs::File::create(log.with_extension("err")).unwrap())
                .spawn()
                .unwrap(),
        )
    }
}

impl Node {
    /// This node's federation listener and models, served the way
    /// mdma-serve serves them, in this process. Runs until the runtime drops.
    fn serve_in_process(&self, certs: &Path) -> InProcess {
        use mdma::backend::cpu::CpuBackend;
        use mdma::federation::runtime::{Config, Prepared};
        use mdma::registry::{Options, Registry};
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (dir, models, client) = (self.dir.clone(), self.models.clone(), self.client);
        let (ca, chain) = (pem(certs, "ca.pem"), pem(certs, "server.pem"));
        let registry = runtime.block_on(async move {
            let config = Config::load(&dir.join("runtime.toml")).unwrap();
            let client_listen = format!("127.0.0.1:{client}").parse().unwrap();
            let prepared = Prepared::load(config, client_listen, &ca, &chain).unwrap();
            let options = Options {
                frozen: true,
                ..Options::new(64)
            };
            let mut registry =
                Registry::open(Arc::new(CpuBackend), &models, None, options).unwrap();
            registry.capture_execution_profiles().unwrap();
            let mut service = mdma::grpc::MdmaService::new(registry, "cpu");
            service.registry().hash_artifacts(|| false);
            let registry = service.registry().clone();
            let executor = rama::rt::Executor::default();
            let prepared = prepared
                .with_native_service(&mut service, executor.clone())
                .unwrap();
            let listener = prepared.bind(executor.clone()).await.unwrap();
            tokio::spawn(async move {
                prepared.serve(listener, executor).await.unwrap();
                drop(service);
            });
            registry
        });
        InProcess {
            _runtime: runtime,
            registry,
        }
    }
}

/// A node served in this process, until dropped.
struct InProcess {
    _runtime: tokio::runtime::Runtime,
    registry: Arc<mdma::registry::Registry<mdma::backend::cpu::CpuBackend>>,
}

impl InProcess {
    /// Each model's phase and last load error, for failure messages.
    fn models(&self) -> String {
        self.registry
            .entries()
            .map(|entry| format!("{} {:?} {:?}", entry.name, entry.phase, entry.error))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Waits until the server's stderr log mentions `line`; returns the log.
fn logged(log: &Path, line: &str) -> String {
    let deadline = Instant::now() + Duration::from_mins(1);
    loop {
        let text = std::fs::read_to_string(log.with_extension("err")).unwrap_or_default();
        if text.contains(line) {
            return text;
        }
        assert!(Instant::now() < deadline, "waiting for {line:?}:\n{text}");
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn pem(certs: &Path, name: &str) -> Vec<rustls_pki_types::CertificateDer<'static>> {
    rustls_pki_types::CertificateDer::pem_file_iter(certs.join(name))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

/// The client and server certificates both nodes' mdma-serve use.
fn certs(root: &Path) -> PathBuf {
    let certs = root.join("certs");
    assert!(
        command(env!("CARGO_BIN_EXE_mdma-certs"))
            .arg(&certs)
            .output()
            .unwrap()
            .status
            .success()
    );
    certs
}

/// The origin routing the executor's catalog for the test client.
fn routes(origin: &Node, certs: &Path) -> Output {
    fed(&[
        "routes",
        path(&origin.dir),
        "--peer",
        "executor.fleet.test",
        "--client-cert",
        path(&certs.join("client.pem")),
    ])
}

/// The names in `dir`, sorted.
fn files(dir: &Path) -> Vec<String> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    files.sort();
    files
}

/// Asserts `output` failed, saying `why`.
fn refused(output: &Output, why: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && stderr.contains(why),
        "expected a refusal for {why:?}: {stderr}"
    );
}

/// Retries `attempt` until it succeeds, for things that wait on a starting node.
fn eventually(what: &str, log: &Path, mut attempt: impl FnMut() -> Output) -> String {
    let deadline = Instant::now() + Duration::from_mins(1);
    loop {
        let output = attempt();
        if output.status.success() {
            return String::from_utf8_lossy(&output.stdout).into_owned();
        }
        assert!(
            Instant::now() < deadline,
            "{what}: {}\n{}",
            String::from_utf8_lossy(&output.stderr),
            std::fs::read_to_string(log.with_extension("err")).unwrap_or_default()
        );
        std::thread::sleep(Duration::from_millis(250));
    }
}

#[test]
fn an_origin_routes_an_executors_model_set_up_with_mdma_fed() {
    let _turn = daemon_turn();
    let model = fixture::model(8);
    let root = tempfile::tempdir().unwrap();
    let certs = certs(root.path());
    let (origin, executor) = (
        Node::new(root.path(), "origin", Some(model.path())),
        Node::new(root.path(), "executor", Some(model.path())),
    );

    let enrolled = executor.enroll(
        root.path(),
        "origin",
        &["--execute", "all", "--models", path(&executor.models)],
    );
    assert!(
        enrolled.contains("enrolled origin.fleet.test") && enrolled.contains("runs tiny"),
        "{enrolled}"
    );
    origin.trust(root.path(), "executor");

    let executor_log = root.path().join("executor.log");
    let running = executor.serve(&certs, &executor_log);
    // The executor lists its model once it has hashed it.
    let routed = eventually("routes", &executor_log, || routes(&origin, &certs));
    assert_eq!(routed.trim(), "executor-tiny -> executor.fleet.test tiny");
    let status = ok(&fed(&["status", path(&origin.dir)]));
    assert!(
        status.contains(
            "route    executor-tiny -> executor.fleet.test tiny (synthesize, 1 client(s))"
        ),
        "{status}"
    );

    // Startup on the origin checks exactly this: routes authorized, polling on.
    let config = mdma::federation::runtime::Config::load(&origin.dir.join("runtime.toml")).unwrap();
    assert!(config.enable_routing && config.poll_catalogs && !config.enable_execution);
    let client_listen = format!("127.0.0.1:{}", origin.client).parse().unwrap();
    drop(
        mdma::federation::runtime::Prepared::load(
            config,
            client_listen,
            &pem(&certs, "ca.pem"),
            &pem(&certs, "server.pem"),
        )
        .unwrap(),
    );

    // A running node shows its state but refuses changes.
    let shown = ok(&fed(&["status", path(&executor.dir)]));
    assert!(
        shown.contains("peer     origin.fleet.test Active")
            && shown.contains("runs     tiny (synthesize)"),
        "{shown}"
    );
    refused(
        &fed(&["export", path(&executor.dir)]),
        "stop mdma-serve first",
    );
    drop(running);

    // Renewing: the origin runs init again under its id. The executor keeps
    // the old credentials until told to replace them, then serves it again.
    std::fs::remove_dir_all(&origin.dir).unwrap();
    std::fs::remove_dir_all(&origin.models).unwrap();
    let origin = Node::new(root.path(), "origin", Some(model.path()));
    let grants = ["--execute", "all", "--models", path(&executor.models)];
    let bundle = root.path().join("origin.json");
    refused(
        &fed(&[
            &["enroll", path(&executor.dir), path(&bundle)][..],
            &grants[..],
        ]
        .concat()),
        "enroll --replace if it ran init again",
    );
    executor.enroll(
        root.path(),
        "origin",
        &[&grants[..], &["--replace"][..]].concat(),
    );
    origin.trust(root.path(), "executor");
    let running = executor.serve(&certs, &executor_log);
    let routed = eventually("routes after renewal", &executor_log, || {
        routes(&origin, &certs)
    });
    assert_eq!(routed.trim(), "executor-tiny -> executor.fleet.test tiny");
    drop(running);

    // Only a node that routes may start without local models.
    let empty = root.path().join("empty-models");
    std::fs::create_dir(&empty).unwrap();
    let mut plain = command(env!("CARGO_BIN_EXE_mdma-serve"))
        .env("MDMA_MODELS_DIR", &empty)
        .env("MDMA_BACKEND", "cpu")
        .env("MDMA_LISTEN", format!("127.0.0.1:{}", free_port()))
        .env("MDMA_POWER", "off")
        .env("MDMA_TLS_CERT", certs.join("server.pem"))
        .env("MDMA_TLS_KEY", certs.join("server-key.pem"))
        .env("MDMA_TLS_CA", certs.join("ca.pem"))
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_mins(1);
    while plain.try_wait().unwrap().is_none() {
        assert!(
            Instant::now() < deadline,
            "a node without models kept running"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
    let output = plain.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && stderr.contains("no usable models"),
        "{stderr}"
    );
    std::fs::remove_file(origin.models.join("tiny.gguf")).unwrap();
    let origin_log = root.path().join("origin.log");
    let _origin = origin.serve(&certs, &origin_log);
    let text = logged(&origin_log, "mdma listening on");
    assert!(text.contains("serving routes only"), "{text}");
}

#[test]
fn an_empty_executor_takes_the_models_its_peers_push() {
    let _turn = daemon_turn();
    let (tiny, check, other) = (fixture::model(8), fixture::minicheck(8), fixture::model(16));
    let root = tempfile::tempdir().unwrap();
    let certs = certs(root.path());
    let (origin, executor) = (
        Node::new(root.path(), "origin", None),
        Node::new(root.path(), "executor", None),
    );
    let enroll =
        |floor: &str| executor.enroll(root.path(), "origin", &["--install", floor, "--installed"]);
    let enrolled = enroll("0");
    assert!(
        enrolled.contains("runs every installed model") && enrolled.contains("installs models"),
        "{enrolled}"
    );
    origin.trust(root.path(), "executor");

    let log = root.path().join("executor.log");
    let running = executor.serve(&certs, &log);
    logged(&log, "waiting for peers to install some");
    let push = |file: &Path, name: &str| {
        fed(&[
            "push",
            path(&origin.dir),
            path(file),
            "--peer",
            "executor.fleet.test",
            "--name",
            name,
        ])
    };
    let pushed = eventually("push", &log, || push(tiny.path(), "tiny"));
    assert!(
        pushed.starts_with("pushed tiny to executor.fleet.test"),
        "{pushed}"
    );
    // The same bytes under another name are not sent again.
    assert_eq!(
        ok(&push(tiny.path(), "copy")).trim(),
        "executor.fleet.test already holds these bytes as tiny"
    );
    // A taken name and a file that is no model are refused, leaving nothing behind.
    refused(&push(other.path(), "tiny"), "model name taken");
    let junk = root.path().join("junk.gguf");
    std::fs::write(&junk, b"not a model").unwrap();
    refused(&push(&junk, "junk"), "not a model this node serves");
    let files = || files(&executor.models);
    assert_eq!(files(), ["tiny.gguf"]);
    // A decision model is installed as one.
    ok(&push(check.path(), "check"));
    assert_eq!(files(), ["check.gguf", "tiny.gguf"]);
    let status = ok(&fed(&["status", path(&executor.dir)]));
    assert!(
        status.contains("model    check (decide) installed by origin.fleet.test")
            && status.contains("model    tiny (synthesize) installed by origin.fleet.test"),
        "{status}"
    );

    // The origin sees and routes what it pushed, without a grant naming it.
    let route = || routes(&origin, &certs);
    let routed =
        "executor-check -> executor.fleet.test check\nexecutor-tiny -> executor.fleet.test tiny";
    assert_eq!(ok(&route()).trim(), routed);

    // Installs outlive a restart, and a node keeps the disk it is told to.
    drop(running);
    enroll("1000000T");
    let _restarted = executor.serve(&certs, &log);
    assert_eq!(
        eventually("routes after restart", &log, route).trim(),
        routed
    );
    refused(
        &push(other.path(), "other"),
        "bytes would leave less than the 1099511627776000000 bytes this node keeps free",
    );
    assert_eq!(files(), ["check.gguf", "tiny.gguf"]);
}

#[test]
fn a_serving_origin_pushes_a_model_for_a_pinned_client() {
    let _turn = daemon_turn();
    let model = fixture::model(8);
    let root = tempfile::tempdir().unwrap();
    let certs = certs(root.path());
    let (origin, executor) = (
        Node::new(root.path(), "origin", Some(model.path())),
        Node::new(root.path(), "executor", None),
    );
    executor.enroll(root.path(), "origin", &["--install", "0", "--installed"]);
    origin.trust(root.path(), "executor");
    let _executor = executor.serve_in_process(&certs);

    let push = |args: &[&str]| {
        command(env!("CARGO_BIN_EXE_mdma-client"))
            .env("MDMA_ADDR", format!("localhost:{}", origin.client))
            .env("MDMA_TLS_CERT", certs.join("client.pem"))
            .env("MDMA_TLS_KEY", certs.join("client-key.pem"))
            .env("MDMA_TLS_CA", certs.join("ca.pem"))
            .arg("--push")
            .arg("executor.fleet.test")
            .args(args)
            .output()
            .unwrap()
    };
    let log = root.path().join("origin.log");
    // Being a client of the origin is not enough to push from it.
    let running = origin.serve(&certs, &log);
    logged(&log, "mdma listening on");
    refused(&push(&["tiny"]), "MDMA_ADMIN_SPKI");
    drop(running);

    let pin = pin(&certs);
    let _running = origin.serve_with(&certs, &log, &[("MDMA_ADMIN_SPKI", pin.as_str())]);
    logged(&log, "mdma listening on");
    let pushed = push(&["tiny"]);
    assert_eq!(ok(&pushed).trim(), "pushed tiny to executor.fleet.test");
    assert!(
        String::from_utf8_lossy(&pushed.stderr).contains("mdma-client: sent 0.00 of"),
        "{pushed:?}"
    );
    // Bytes the peer holds are not sent again, under any name.
    assert_eq!(
        ok(&push(&["tiny", "copy"])).trim(),
        "executor.fleet.test already holds these bytes as tiny"
    );
    // Only a name from the origin's models folder is pushed, never a path.
    refused(&push(&["../origin.json"]), "file, without .gguf,");
    refused(&push(&["missing"]), "missing.gguf");
    let status = ok(&fed(&["status", path(&executor.dir)]));
    assert!(
        status.contains("model    tiny (synthesize) installed by origin.fleet.test"),
        "{status}"
    );
}

/// mdma-client against `origin` as the test client, with extra env.
fn client(
    origin: &Node,
    certs: &Path,
    env: &[(&str, &str)],
    args: &[&str],
) -> std::process::Command {
    let mut command = command(env!("CARGO_BIN_EXE_mdma-client"));
    command
        .env("MDMA_ADDR", format!("localhost:{}", origin.client))
        .env("MDMA_TLS_CERT", certs.join("client.pem"))
        .env("MDMA_TLS_KEY", certs.join("client-key.pem"))
        .env("MDMA_TLS_CA", certs.join("ca.pem"))
        .envs(env.iter().copied())
        .args(args);
    command
}

/// The test client's pin, as MDMA_ADMIN_SPKI takes it.
fn pin(certs: &Path) -> String {
    ok(&command(env!("CARGO_BIN_EXE_mdma-certs"))
        .arg("--spki")
        .arg(certs.join("client.pem"))
        .output()
        .unwrap())
    .trim()
    .to_owned()
}

#[test]
fn a_serving_origin_imports_routes_without_a_restart() {
    let _turn = daemon_turn();
    let (tiny, other) = (fixture::model(8), fixture::model(16));
    let root = tempfile::tempdir().unwrap();
    let certs = certs(root.path());
    let (origin, executor) = (
        Node::new(root.path(), "origin", None),
        Node::new(root.path(), "executor", Some(tiny.path())),
    );
    std::fs::copy(other.path(), origin.models.join("other.gguf")).unwrap();
    executor.enroll(
        root.path(),
        "origin",
        &[
            "--execute",
            "all",
            "--models",
            path(&executor.models),
            "--install",
            "0",
            "--installed",
        ],
    );
    origin.trust(root.path(), "executor");
    let executor_node = executor.serve_in_process(&certs);
    let log = root.path().join("origin.log");
    // Routing starts the way it always has, with the origin stopped.
    let first = eventually("routes", &log, || routes(&origin, &certs));
    assert_eq!(first.trim(), "executor-tiny -> executor.fleet.test tiny");

    let pin = pin(&certs);
    let admin = [("MDMA_ADMIN_SPKI", pin.as_str())];
    let running = origin.serve_with(&certs, &log, &admin);
    logged(&log, "mdma listening on");
    let run = |args: &[&str]| client(&origin, &certs, &[], args).output().unwrap();
    assert_eq!(
        ok(&run(&["--push", "executor.fleet.test", "other"])).trim(),
        "pushed other to executor.fleet.test"
    );
    let routed =
        "executor-tiny -> executor.fleet.test tiny\nexecutor-other -> executor.fleet.test other";
    assert_eq!(
        ok(&run(&["--routes", "executor.fleet.test"])).trim(),
        routed
    );
    let generate = || {
        client(
            &origin,
            &certs,
            &[
                ("MDMA_MODEL", "executor-other"),
                ("MDMA_RPC_TIMEOUT_MS", "60000"),
                ("MDMA_MAX_TOKENS", "4"),
                ("MDMA_READY_TIMEOUT_SECS", "30"),
            ],
            &["hello"],
        )
        .output()
        .unwrap()
    };
    // The new route serves at once, from the running origin.
    let generated = generate();
    assert!(
        generated.status.success(),
        "{}\nexecutor: {}",
        String::from_utf8_lossy(&generated.stderr),
        executor_node.models()
    );
    let status = ok(&fed(&["status", path(&origin.dir)]));
    assert!(
        status.contains(
            "route    executor-other -> executor.fleet.test other (synthesize, 1 client(s))"
        ),
        "{status}"
    );

    // What the running origin imported is what it starts with; a route
    // lists once its peer's catalog is polled, just after start.
    drop(running);
    let _restarted = origin.serve_with(&certs, &log, &admin);
    logged(&log, "mdma listening on");
    eventually("the imported route after a restart", &log, generate);
}

#[test]
fn an_interrupted_push_leaves_nothing_behind() {
    use std::io::BufRead;
    let _turn = daemon_turn();
    let tiny = fixture::model(8);
    let root = tempfile::tempdir().unwrap();
    let certs = certs(root.path());
    let (origin, executor) = (
        Node::new(root.path(), "origin", Some(tiny.path())),
        Node::new(root.path(), "executor", None),
    );
    // Long enough to stop halfway; sparse, so it costs no disk to send.
    let size = 1u64 << 30;
    std::fs::File::create(origin.models.join("big.gguf"))
        .unwrap()
        .set_len(size)
        .unwrap();
    executor.enroll(root.path(), "origin", &["--install", "0", "--installed"]);
    origin.trust(root.path(), "executor");
    let _executor = executor.serve_in_process(&certs);
    let log = root.path().join("origin.log");
    let pin = pin(&certs);
    let _running = origin.serve_with(&certs, &log, &[("MDMA_ADMIN_SPKI", pin.as_str())]);
    logged(&log, "mdma listening on");

    let mut push = client(
        &origin,
        &certs,
        &[],
        &["--push", "executor.fleet.test", "big"],
    )
    .stderr(std::process::Stdio::piped())
    .stdout(std::process::Stdio::piped())
    .spawn()
    .unwrap();
    let progress = std::io::BufReader::new(push.stderr.take().unwrap());
    let mut midway = None;
    for line in progress.lines() {
        let line = line.unwrap();
        if let Some(sent) = line.strip_prefix("mdma-client: sent ")
            && !sent.starts_with("0.00 ")
        {
            midway = Some(line);
            break;
        }
    }
    push.kill().unwrap();
    push.wait().unwrap();
    let midway = midway.expect("the push reported progress");
    assert!(!midway.contains("sent 1.00 of"), "{midway}");

    // The receiver drops what it took; neither side holds a turn.
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let left = files(&executor.models);
        if left.is_empty() {
            break;
        }
        assert!(Instant::now() < deadline, "left behind: {left:?}");
        std::thread::sleep(Duration::from_millis(100));
    }
    let next = client(
        &origin,
        &certs,
        &[],
        &["--push", "executor.fleet.test", "tiny"],
    )
    .output()
    .unwrap();
    assert_eq!(ok(&next).trim(), "pushed tiny to executor.fleet.test");
}

/// A real model end to end: `MDMA_TEST_PUSH_MODEL=/path/model.gguf`, pushed
/// from a serving origin, routed live and run through it.
#[test]
#[ignore = "needs a multi-GB GGUF in MDMA_TEST_PUSH_MODEL"]
fn a_full_size_model_pushes_routes_and_runs() {
    let _turn = daemon_turn();
    let big = PathBuf::from(std::env::var("MDMA_TEST_PUSH_MODEL").unwrap());
    let size = std::fs::metadata(&big).unwrap().len();
    let tiny = fixture::model(8);
    let root = tempfile::tempdir().unwrap();
    let certs = certs(root.path());
    let (origin, executor) = (
        Node::new(root.path(), "origin", None),
        Node::new(root.path(), "executor", Some(tiny.path())),
    );
    std::os::unix::fs::symlink(&big, origin.models.join("big.gguf")).unwrap();
    executor.enroll(
        root.path(),
        "origin",
        &[
            "--execute",
            "all",
            "--models",
            path(&executor.models),
            "--install",
            "0",
            "--installed",
        ],
    );
    origin.trust(root.path(), "executor");
    let executor_node = executor.serve_in_process(&certs);
    let log = root.path().join("origin.log");
    eventually("routes", &log, || routes(&origin, &certs));
    let pin = pin(&certs);
    let _running = origin.serve_with(&certs, &log, &[("MDMA_ADMIN_SPKI", pin.as_str())]);
    logged(&log, "mdma listening on");

    let started = Instant::now();
    let pushed = client(
        &origin,
        &certs,
        &[],
        &["--push", "executor.fleet.test", "big"],
    )
    .output()
    .unwrap();
    let took = started.elapsed();
    assert_eq!(ok(&pushed).trim(), "pushed big to executor.fleet.test");
    eprintln!(
        "pushed {:.2} GiB in {:.1}s ({:.0} MiB/s, hashing included)",
        size as f64 / f64::from(1u32 << 30),
        took.as_secs_f64(),
        size as f64 / f64::from(1u32 << 20) / took.as_secs_f64()
    );
    let imported = ok(
        &client(&origin, &certs, &[], &["--routes", "executor.fleet.test"])
            .output()
            .unwrap(),
    );
    assert!(
        imported.contains("executor-big -> executor.fleet.test big"),
        "{imported}"
    );
    let started = Instant::now();
    let generated = client(
        &origin,
        &certs,
        &[
            ("MDMA_MODEL", "executor-big"),
            ("MDMA_RPC_TIMEOUT_MS", "300000"),
            ("MDMA_MAX_TOKENS", "8"),
            ("MDMA_READY_TIMEOUT_SECS", "60"),
        ],
        &["The capital of France is"],
    )
    .output()
    .unwrap();
    assert!(
        generated.status.success(),
        "{}\nexecutor: {}",
        String::from_utf8_lossy(&generated.stderr),
        executor_node.models()
    );
    eprintln!(
        "routed reply in {:.1}s, load included: {:?}",
        started.elapsed().as_secs_f64(),
        String::from_utf8_lossy(&generated.stdout).trim()
    );
}

//! mdma-web against a real mdma-serve: capability login, live status over SSE,
//! and a playground generation streamed as Datastar events.
#[path = "runtime_cli/fixture.rs"]
mod fixture;
#[path = "runtime_cli/process.rs"]
mod process;

use process::{Process, command, free_port};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

impl Process {
    fn terminate(mut self) -> std::process::ExitStatus {
        assert!(
            Command::new("kill")
                .args(["-TERM", &self.0.id().to_string()])
                .status()
                .unwrap()
                .success()
        );
        let deadline = Instant::now() + Duration::from_secs(35);
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                return status;
            }
            assert!(Instant::now() < deadline, "shutdown timed out");
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// Sends one HTTP/1.1 request and reads until `done` matches or the peer closes.
fn http(port: u16, request: &str, done: impl Fn(&str) -> bool) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let (deadline, mut response, mut buf) = (
        Instant::now() + Duration::from_mins(1),
        Vec::new(),
        [0_u8; 8192],
    );
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => response.extend_from_slice(&buf[..n]),
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(e) => panic!("read: {e}"),
        }
        let text = String::from_utf8_lossy(&response);
        if done(&text) {
            break;
        }
        assert!(Instant::now() < deadline, "no complete response: {text}");
    }
    String::from_utf8(response).unwrap()
}

fn status_line(response: &str) -> &str {
    response.lines().next().unwrap_or_default()
}

#[test]
fn browser_gateway_serves_live_views_and_streams_generation() {
    let model = fixture::model(8);
    let dir = tempfile::tempdir().unwrap();
    let certs = dir.path().join("certs");
    assert!(
        Command::new(env!("CARGO_BIN_EXE_mdma-certs"))
            .arg(&certs)
            .output()
            .unwrap()
            .status
            .success()
    );
    // Keys are private, and running again never replaces them.
    for key in ["server-key.pem", "client-key.pem"] {
        let mode = std::fs::metadata(certs.join(key))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600, "{key}");
    }
    let ca = std::fs::read(certs.join("ca.pem")).unwrap();
    let again = Command::new(env!("CARGO_BIN_EXE_mdma-certs"))
        .arg(&certs)
        .output()
        .unwrap();
    assert!(
        !again.status.success()
            && String::from_utf8_lossy(&again.stderr).contains("never replaces credentials"),
        "{again:?}"
    );
    assert_eq!(std::fs::read(certs.join("ca.pem")).unwrap(), ca);
    let models = dir.path().join("models");
    std::fs::create_dir(&models).unwrap();
    std::fs::copy(model.path(), models.join("tiny.gguf")).unwrap();
    // Listed after tiny, so tiny stays the default.
    std::fs::copy(fixture::minicheck(8).path(), models.join("verify.gguf")).unwrap();

    // The gateway's certificate is an operator.
    let pin = Command::new(env!("CARGO_BIN_EXE_mdma-certs"))
        .arg("--spki")
        .arg(certs.join("client.pem"))
        .output()
        .unwrap();
    assert!(pin.status.success());
    let pin = String::from_utf8(pin.stdout).unwrap().trim().to_owned();
    assert!(pin.starts_with("sha256:") && pin.len() == 71, "{pin}");

    let serve_port = free_port();
    let serve_log = dir.path().join("serve.log");
    let server = Process(
        command(env!("CARGO_BIN_EXE_mdma-serve"))
            .env("MDMA_MODELS_DIR", &models)
            .env("MDMA_MAX_OUTPUT_TOKENS", "4")
            .env("MDMA_BACKEND", "cpu")
            .env("MDMA_LISTEN", format!("127.0.0.1:{serve_port}"))
            .env("MDMA_POWER", "off")
            .env("MDMA_OPERATOR_SPKI", &pin)
            .env("MDMA_NODE_NAME", "e2e-node")
            .env("MDMA_TLS_CERT", certs.join("server.pem"))
            .env("MDMA_TLS_KEY", certs.join("server-key.pem"))
            .env("MDMA_TLS_CA", certs.join("ca.pem"))
            .stdout(Stdio::null())
            .stderr(std::fs::File::create(&serve_log).unwrap())
            .spawn()
            .unwrap(),
    );

    let mut web = Process(
        command(env!("CARGO_BIN_EXE_mdma-web"))
            .env("MDMA_WEB_LISTEN", "127.0.0.1:0")
            .env("MDMA_WEB_WATCH_MS", "250")
            .env("MDMA_ADDR", format!("localhost:{serve_port}"))
            .env("MDMA_TLS_CERT", certs.join("client.pem"))
            .env("MDMA_TLS_KEY", certs.join("client-key.pem"))
            .env("MDMA_TLS_CA", certs.join("ca.pem"))
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut stderr = BufReader::new(web.0.stderr.take().unwrap()).lines();
    let url = stderr
        .by_ref()
        .map_while(Result::ok)
        .find_map(|line| {
            line.strip_prefix("mdma-web listening on ")
                .map(str::to_owned)
        })
        .expect("mdma-web prints its capability URL");
    std::thread::spawn(move || stderr.for_each(drop)); // keep the pipe drained
    let (origin, token) = url.split_once("/?token=").unwrap();
    let port: u16 = origin.rsplit(':').next().unwrap().parse().unwrap();
    let host = format!("127.0.0.1:{port}");
    let cookie = format!("mdma_web_{port}={token}");

    let get = |path: &str, extra: &str| {
        http(
            port,
            &format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n{extra}\r\n"),
            |_| false,
        )
    };
    assert!(
        status_line(&get("/", "")).contains("403"),
        "no capability, no page"
    );
    assert!(status_line(&http(port, &format!("GET / HTTP/1.1\r\nHost: evil.example:{port}\r\nCookie: {cookie}\r\nConnection: close\r\n\r\n"), |_| false)).contains("403"));
    let login = get(&format!("/playground?token={token}"), "");
    assert!(status_line(&login).contains("303"), "{login}");
    assert!(
        login.to_ascii_lowercase().contains(
            &format!("set-cookie: {cookie}; path=/; httponly; samesite=strict")
                .to_ascii_lowercase()
        ),
        "{login}"
    );
    let page = get("/playground", &format!("Cookie: {cookie}\r\n"));
    assert!(
        status_line(&page).contains("200") && page.contains("data-ignore-morph"),
        "{page}"
    );
    assert!(
        page.to_ascii_lowercase()
            .contains("content-security-policy:"),
        "{page}"
    );
    let scene = get("/assets/scene.js", &format!("Cookie: {cookie}\r\n"));
    assert!(
        scene.contains("mdmaScene"),
        "the committed bundle is served"
    );

    // Live status: the hub reaches the server and the models panel morphs in.
    let status = http(
        port,
        &format!(
            "GET /sse/status?view=memory HTTP/1.1\r\nHost: {host}\r\nCookie: {cookie}\r\n\r\n"
        ),
        |text| text.contains("state-serving") && text.contains("models-panel"),
    );
    assert!(status.contains("event: datastar-patch-signals"), "{status}");
    assert!(status.contains("tiny"), "{status}");
    // Watch snapshots: the live budget breakdown; models load on first use.
    let memory = |until: &str| {
        let until = until.to_owned();
        http(
            port,
            &format!(
                "GET /sse/status?view=memory HTTP/1.1\r\nHost: {host}\r\nCookie: {cookie}\r\n\r\n"
            ),
            move |text| text.contains(&until),
        )
    };
    let status = memory("native budget");
    assert!(
        status.contains("0 B of ") && status.contains("KV cache"),
        "a CPU server budgets its RAM: {status}"
    );
    assert!(
        status.contains(r#"\"residency\":\"available\""#),
        "nothing loads before use: {status}"
    );

    // Operator view: the enrollment hint is not shown to an operator.
    let federation = http(
        port,
        &format!(
            "GET /sse/status?view=federation HTTP/1.1\r\nHost: {host}\r\nCookie: {cookie}\r\n\r\n"
        ),
        |text| text.contains("remote-panel") && text.contains("No peers."),
    );
    assert!(!federation.contains("MDMA_OPERATOR_SPKI="), "{federation}");
    assert!(
        federation.contains(&format!(
            r#"<strong class="os {}">e2e-node</strong>"#,
            std::env::consts::OS
        )) && federation.contains(&mdma::host::os_name()),
        "this server names its host: {federation}"
    );

    let run = |origin: &str, body: &str| {
        http(
            port,
            &format!(
                "POST /playground/run HTTP/1.1\r\nHost: {host}\r\nCookie: {cookie}\r\nOrigin: {origin}\r\nDatastar-Request: true\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            ),
            |_| false,
        )
    };
    assert!(
        status_line(&run("http://attacker.example", "{}")).contains("403"),
        "cross-origin POST is refused"
    );
    // Asks for 8; the server's output cap admits 4.
    let generated = run(
        &format!("http://{host}"),
        r#"{"prompt":"hi","model":"tiny","max_tokens":8,"temperature":0}"#,
    );
    assert!(status_line(&generated).contains("200"), "{generated}");
    assert!(
        generated.contains("event: datastar-patch-elements"),
        "{generated}"
    );
    assert!(
        generated.contains(r#""phase":"loading""#),
        "the first use loads the model: {generated}"
    );
    assert!(generated.contains(r#""phase":"prefill""#), "{generated}");
    assert!(
        generated.contains(r#""max_tokens":4,"n_ctx":64"#),
        "the trained context, the capped output: {generated}"
    );
    assert!(
        generated.contains(r#""emitted":4"#),
        "four tokens decode: {generated}"
    );
    assert!(generated.contains(r#""phase":"done""#), "{generated}");
    // Server lifecycle events (emit_events): the admitted prompt and how it ended.
    assert!(
        generated.contains(r#""prompt_tokens":2"#),
        "two byte tokens for \"hi\": {generated}"
    );
    assert!(
        generated.contains(r#""finish_reason":"max_tokens""#),
        "{generated}"
    );
    let status = memory(r#"\"residency\":\"resident\""#);
    assert!(
        status.contains(r#"\"n_layers\":2"#),
        "a loaded model's geometry reaches the scene JSON: {status}"
    );
    // Choosing a model sizes the scene to it and, without a template, turns chat off.
    let choose = |body: &str| {
        http(
            port,
            &format!(
                "POST /playground/model HTTP/1.1\r\nHost: {host}\r\nCookie: {cookie}\r\nOrigin: http://{host}\r\n\
                 Datastar-Request: true\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            ),
            |_| false,
        )
    };
    let chosen = choose(r#"{"model":"tiny"}"#);
    assert!(
        chosen.contains(r#""chat":false"#)
            && chosen.contains(r#""n_ctx":64"#)
            && chosen.contains("no chat template")
            && chosen.contains(r#"<pre id="output"></pre>"#)
            && chosen.contains(r#"<div id="markers""#),
        "{chosen}"
    );
    // The fixture has no chat template.
    let refused = run(
        &format!("http://{host}"),
        r#"{"prompt":"hi","model":"tiny","max_tokens":4,"chat":true}"#,
    );
    assert!(
        refused.contains("has no known chat template") && refused.contains(r#""phase":"error""#),
        "{refused}"
    );
    let invalid = run(&format!("http://{host}"), r#"{"prompt":"","max_tokens":4}"#);
    assert!(
        invalid.contains("prompt is empty") && invalid.contains(r#""phase":"error""#),
        "{invalid}"
    );

    // A claim checker opens what it takes, a document and claims, and its
    // run answers through Decide: one verdict per claim.
    let chosen = choose(r#"{"model":"verify"}"#);
    assert!(
        chosen.contains(r#"<textarea id="document""#)
            && chosen.contains(r#"<textarea id="claims""#)
            && !chosen.contains(r#"id="markers""#)
            && chosen.contains(r#"<div id="output" class="verdicts"></div>"#),
        "{chosen}"
    );
    let checked = run(
        &format!("http://{host}"),
        r#"{"model":"verify","document":"The sky is blue.","claims":"The sky is blue.\n\nThe sky is green."}"#,
    );
    let shares: Vec<f64> = checked
        .split("width: ")
        .skip(1)
        .map(|rest| rest.split('%').next().unwrap().parse().unwrap())
        .collect();
    assert!(
        checked.contains(r#"class="verdicts""#)
            && checked.matches(r#"<td class="verdict "#).count() == 2
            && shares.len() == 2
            && shares.iter().all(|share| (0.0..=100.0).contains(share))
            && checked.contains(r#""phase":"done","prompt_tokens":"#)
            && !checked.contains(r#""phase":"done","prompt_tokens":0,"#),
        "{checked}"
    );
    let missing = run(
        &format!("http://{host}"),
        r#"{"model":"verify","document":"The sky is blue.","claims":" \n"}"#,
    );
    assert!(missing.contains("write 1 to 128 claims"), "{missing}");
    // The same contract straight over gRPC; a text generator refuses it.
    let request = dir.path().join("decide.json");
    std::fs::write(
        &request,
        r#"{"state": {"document": "The sky is blue."}, "questions": {"a": {"claim": "The sky is blue."}}}"#,
    )
    .unwrap();
    let decide = |model: &str| {
        command(env!("CARGO_BIN_EXE_mdma-client"))
            .env("MDMA_ADDR", format!("localhost:{serve_port}"))
            .env("MDMA_MODEL", model)
            .env("MDMA_TLS_CERT", certs.join("client.pem"))
            .env("MDMA_TLS_KEY", certs.join("client-key.pem"))
            .env("MDMA_TLS_CA", certs.join("ca.pem"))
            .arg("--decide")
            .arg(&request)
            .output()
            .unwrap()
    };
    let answered = decide("verify");
    let result: serde_json::Value =
        serde_json::from_slice(&answered.stdout).unwrap_or_else(|e| panic!("{e}: {answered:?}"));
    let answer = &result["answers"]["a"];
    assert!(
        result["model"] == "minicheck"
            && answer["supported"].is_boolean()
            && (0.0..=1.0).contains(&answer["probability"].as_f64().unwrap())
            && result["usage"]["output_tokens"] == 0,
        "{result}"
    );
    let refused = decide("tiny");
    assert!(
        !refused.status.success()
            && String::from_utf8_lossy(&refused.stderr).contains("generates text; use Synthesize"),
        "{refused:?}"
    );

    // Load keeps a model loaded under the gateway's lease until Release.
    let lease = |action: &str, model: &str| {
        http(
            port,
            &format!(
                "POST /models/{action}?model={model} HTTP/1.1\r\nHost: {host}\r\nCookie: {cookie}\r\nOrigin: http://{host}\r\n\
         Datastar-Request: true\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            ),
            |_| false,
        )
    };
    let held = lease("hold", "tiny");
    assert!(
        held.contains("model=tiny")
            && held.contains("Release")
            && held.contains(r#"<p id="lease-error" class="error"></p>"#),
        "{held}"
    );
    memory(r#"\"leases\":1"#);
    assert!(
        lease("hold", "nope").contains("no model named"),
        "unknown models are refused"
    );
    assert!(lease("release", "tiny").contains("Load"));
    memory(r#"\"leases\":0"#);

    assert!(
        web.terminate().success(),
        "mdma-web exits cleanly on SIGTERM"
    );
    assert!(
        server.terminate().success(),
        "serve: {}",
        std::fs::read_to_string(&serve_log).unwrap()
    );
}
